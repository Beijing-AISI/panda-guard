import sys
import os
import yaml
import typer
import logging
import time
import json
import glob
import pandas as pd
from copy import deepcopy

from tqdm import tqdm
from typing import Optional, Iterator, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich import box

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

app = typer.Typer(help="Run evaluation pipeline with Command Line", invoke_without_command=True)
console = Console()


def is_iterator(obj):
    """Check if an object is an iterator but not a list or other sequence."""
    return (
            hasattr(obj, "__iter__") and
            not isinstance(obj, (list, dict, str, bytes, tuple)) and
            hasattr(obj, "__next__")
    )


def get_package_config_path(model_type: str) -> Path:
    """Get the path to a default config file within the package."""
    try:
        package_path = Path(__file__).parent.parent
        config_path = package_path / "config" / f"{model_type}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Default config file for {model_type} not found at {config_path}")

        return config_path
    except Exception as e:
        console.print(f"[bold red]Error finding default config: {str(e)}[/bold red]")
        raise typer.Exit(1)

def fill_llms_configs(d, parent_key='', llm_configs=None):
    for key, value in d.items():
        full_key = f'{parent_key}.{key}' if parent_key else key
        if isinstance(value, dict):
            fill_llms_configs(value, full_key, llm_configs)
        else:
            if key.endswith("llm_config") and not value:
                d[key] = llm_configs


def override_config(config_dict):
    # Override parameters from command line arguments
    fill_llms_configs(config_dict, "", llm_configs=config_dict["defender"]["target_llm_config"])
    return config_dict


def load_json_files_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        return config.get('json_files', [])


def get_input_files(input_dir):
    if input_dir.endswith('.yaml') or input_dir.endswith('.yml'):
        console.print(f"Input is a YAML file: {input_dir}")
        return load_json_files_from_yaml(input_dir)
    else:
        console.print(f"Input is a directory: {input_dir}")
        return glob.glob(os.path.join(input_dir, '**', '*.json'), recursive=True)


def run_inference(pipe, messages, goal):
    result = pipe.parallel_judging(messages, goal)
    pipe.reset()
    return result


def process_file(json_file, input_dir, output_dir, attacker_config, defender_config, judge_configs, config_dict):
    # try:
    # Construct output file path
    generation_dict = yaml.safe_load(open(json_file.replace('results.json', 'config.yaml'), 'r'))
    config_to_save = generation_dict['judges'] = config_dict['judges']

    output_file = os.path.join(output_dir, os.path.relpath(json_file, input_dir))

    # Check if output file exists
    if os.path.exists(output_file):
        console.print(f"Output file {output_file} already exists, skipping.")
        return

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize pipeline
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config,
            defender_config=defender_config,
            judge_configs=judge_configs  # Can be 0, 1, or multiple judges
        ),
        verbose=False
    )

    if 'results' in data:
        data = data['results']

    # Process data
    for i, item in enumerate(tqdm(data, desc=json_file.split('/')[-4])):
        jailbroken = None
        goal = item['goal']

        for x in item['data']:
            messages = deepcopy(x['messages'])
            result = run_inference(pipe, messages, goal)
            x['judged'] = result

            if jailbroken is None:
                jailbroken = deepcopy(result)
            else:  # Select Max of each judge
                jailbroken = {k: max(jailbroken[k], result[k]) for k in jailbroken}

        item['jailbroken'] = jailbroken
        # print(item)

    # Save modified data to the output directory with the same relative file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    yaml.dump(config_to_save, open(output_file.replace('results.json', 'config.yaml'), 'w'))
    with open(output_file, 'w') as f:
        json.dump({
            "config": config_to_save,
            "results": data
        }, f, indent=4)


@app.callback(invoke_without_command=True)
def start(
        config: Optional[str] = typer.Argument(None, help="Path to YAML task file"),
        input_dir: str = typer.Option("./results", "--input-dir", "-i", help="Path to file where inference is required"),
        output_dir: str = typer.Option("./judged", "--output-dir", "-o", help="Save results to file"),
        log_level: str = typer.Option("WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
        num_workers: int = typer.Option(8, "--num-workers", "-n", help="Number of workers for parallel processing"),
):
    """
    Start an eval using configuration from a YAML file or a predefined model type.

    If config is a file path ending with .yaml, it will load configuration from that file.
    If config is one of 'openai', 'gemini', or 'claude', it will load a default configuration and
    apply relevant environment variables.
    """
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    try:
        config_dict = {}

        if config is None:
            config_path = get_package_config_path('tasks/eval')
            config_dict = load_yaml(config_path)
        else:
            config_path = Path(config)
            if not config_path.exists():
                typer.echo(f"Error: Config file {config} not found", err=True)
                raise typer.Exit(1)

            config_dict = load_yaml(config_path)

        config_dict = override_config(config_dict)
        print("Configuration loaded successfully:", config_dict)

        # Convert YAML dictionary into attacker, defender, and judge configurations
        attacker_config, defender_config, judge_configs = parse_configs_from_dict(config_dict)

        # Get input files (from a directory or a YAML file)
        json_files = get_input_files(input_dir)

        console.print(f"[bold green]Evaluation starts[/bold green]")

        # Use ThreadPoolExecutor to process files concurrently
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for json_file in json_files:
                future = executor.submit(process_file, json_file, input_dir, output_dir, attacker_config, defender_config, judge_configs,
                                         config_dict)
                futures.append(future)

            # Display progress bar and handle any exceptions
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    future.result()  # This will raise any exceptions caught in the threads
                except Exception as e:
                    logging.error(f"Error in processing: {e}")

        console.print(f"[bold green]Results saved to {output_dir}[/bold green]")

    except Exception as e:
        typer.echo(f"Error during evaluation: {str(e)}", err=True)
        logging.exception("Exception during evaluation")
        raise typer.Exit(1)


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    app()