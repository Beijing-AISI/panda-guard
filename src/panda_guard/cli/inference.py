import sys
import os
import yaml
import typer
import logging
import time
import json
import pandas as pd
from tqdm import tqdm
from typing import Optional, Iterator, Dict, Any, List
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich import box

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

app = typer.Typer(help="Large Language Model inference for given attacks, defenses, and judgments", invoke_without_command=True)
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


def apply_env_vars_to_config(config_dict: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Apply environment variables to the config if needed."""
    if model_type == "openai":
        if "base_url" in config_dict["defender"]["target_llm_config"] and os.environ.get("OPENAI_BASE_URL"):
            config_dict["defender"]["target_llm_config"]["base_url"] = os.environ["OPENAI_BASE_URL"]

        if os.environ.get("OPENAI_API_KEY"):
            if "api_key" not in config_dict["defender"]["target_llm_config"]:
                config_dict["defender"]["target_llm_config"]["api_key"] = os.environ["OPENAI_API_KEY"]

    elif model_type == "gemini":
        if os.environ.get("GEMINI_API_KEY"):
            if "api_key" not in config_dict["defender"]["target_llm_config"]:
                config_dict["defender"]["target_llm_config"]["api_key"] = os.environ["GEMINI_API_KEY"]

    elif model_type == "claude":
        if os.environ.get("ANTHROPIC_API_KEY"):
            if "api_key" not in config_dict["defender"]["target_llm_config"]:
                config_dict["defender"]["target_llm_config"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]

        if "base_url" in config_dict["defender"]["target_llm_config"] and os.environ.get("ANTHROPIC_BASE_URL"):
            config_dict["defender"]["target_llm_config"]["base_url"] = os.environ["ANTHROPIC_BASE_URL"]

    return config_dict


def display_token_info(usage, response_time=None):
    """Display token usage information in a less intrusive format."""
    # Calculate totals
    total_prompt = sum(role["prompt_tokens"] for role in usage.values())
    total_completion = sum(role["completion_tokens"] for role in usage.values())
    total_tokens = total_prompt + total_completion

    # Display in a more subtle format
    console.print("[dim]Token usage:[/dim] " +
                  f"Prompt: {total_prompt} | " +
                  f"Completion: {total_completion} | " +
                  f"Total: {total_tokens}")

    # Add speed info if response_time is provided
    if response_time is not None and response_time > 0:
        tokens_per_second = total_completion / response_time
        console.print(f"[dim]Response time: {response_time:.2f}s ({tokens_per_second:.2f} tokens/sec)[/dim]")


def display_judge_results(results):
    """Display judge evaluation results in a less intrusive format."""
    if not results:
        return

    console.print("[dim]Judge evaluations:[/dim]")

    for judge_name, result in results.items():
        # Handle different result formats
        if isinstance(result, dict):
            # score = result.get("score", "N/A")
            verdict = result.get("verdict", str(result))
        else:
            # score = "N/A"
            verdict = str(result)

        # Use more subtle formatting
        console.print(f"[dim]{judge_name}:[/dim] {'⚠️ ' if str(verdict) == '10' else ''}{verdict} ")


@app.callback(invoke_without_command=True)
def start(
        config: Optional[str] = typer.Argument(None, help="Path to YAML task file"),
        input_file: Optional[str] = typer.Option(None, "--input-file", "-i", help="Path to file where inference is required"),
        attack: Optional[Path] = typer.Option(None, "--attack", "-a",
                                               help="Path to attack configuration file or attack method (#todo)"),
        defense: Optional[Path] = typer.Option(None, "--defense", "-d",
                                               help="Path to defense configuration file or defense type (goal_priority/icl/none/rpo/self_reminder/smoothllm)"),
        endpoint: Optional[Path] = typer.Option(None, "--endpoint", "-e",
                                                help="Path to endpoint configuration file or endpoint type (openai/gemini/claude/hf)"),
        model: Optional[Path] = typer.Option(None, "--model", "-m", help="model name"),
        device: Optional[str] = typer.Option(None, "--device", help="Device to run the model on (e.g., 'cuda:0')"),
        log_level: str = typer.Option("WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
        output_dir: str = typer.Option("./results", "--output-dir", "-o", help="Save results to file"),
        verbose: bool = typer.Option(False, "--verbose/--no-verbose",
                                     help="Enable/disable verbose mode with token usage info"),
):
    """
    Start an inference session using configuration from a YAML file or a predefined model type.

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
            config_path = get_package_config_path('tasks/inference')
            config_dict = load_yaml(config_path)
        else:
            config_path = Path(config)
            if not config_path.exists():
                typer.echo(f"Error: Config file {config} not found", err=True)
                raise typer.Exit(1)

            config_dict = load_yaml(config_path)

        if attack:
            if not attack.exists():
                attack_path = get_package_config_path(f'attacks/{attack}')

                if not attack_path.exists():
                    typer.echo(f"Error: Defense config file {attack} not found", err=True)
                    raise typer.Exit(1)

                attack = attack_path
            config_dict["attacker"] = load_yaml(attack)

        if defense:
            if not defense.exists():
                defense_path = get_package_config_path(f'defenses/{defense}')

                if not defense_path.exists():
                    typer.echo(f"Error: Defense config file {defense} not found", err=True)
                    raise typer.Exit(1)

                defense = defense_path
            config_dict["defender"] = load_yaml(defense)

        endpoint_type = None

        print(config_dict)

        if endpoint is None:
            endpoint_type = "openai"
            endpoint_path = get_package_config_path(f"endpoints/{endpoint_type}")
            config_dict["defender"]["target_llm_config"] = load_yaml(endpoint_path)
        elif str(endpoint).lower() in ["openai", "gemini", "claude", "hf"]:
            endpoint_type = str(endpoint).lower()
            endpoint_path = get_package_config_path(f"endpoints/{endpoint_type}")
            config_dict["defender"]["target_llm_config"] = load_yaml(endpoint_path)
        elif config and isinstance(config, str) and config.endswith(".yaml"):
            endpoint_path = Path(config)
            if not endpoint_path.exists():
                typer.echo(f"Error: Config file {endpoint_path} not found", err=True)
                raise typer.Exit(1)
            config_dict["defender"]["target_llm_config"] = load_yaml(endpoint_path)
        else:
            typer.echo(
                f"Error: Invalid config option '{config}'. Must be a .yaml file or one of 'openai', 'gemini', 'claude'",
                err=True)
            raise typer.Exit(1)

        if endpoint_path:
            config_dict = apply_env_vars_to_config(config_dict, endpoint_type)

        # Load and merge model config if provided
        if model:
            config_dict["defender"]["target_llm_config"]['model_name'] = str(model)
            if "attacker" in config_dict and "target_llm_config" in config_dict.get("attacker", {}):
                config_dict["attacker"]["target_llm_config"]['llm_type'] = load_yaml(endpoint_path)['llm_type']
                config_dict["attacker"]["target_llm_config"]['model_name'] = str(model)

        # Override device if specified
        if device:
            config_dict["defender"]["target_llm_config"]["device_map"] = device

        print("Configuration loaded successfully:", config_dict)

        output_file = os.path.join(
            output_dir,
            config_dict['defender']['target_llm_config']['model_name'].replace("/", "_"),
            f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}',
            f'{config_dict["defender"]["defender_cls"]}',
            "results.json",
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if os.path.exists(output_file):
            logging.warning(f"Output file {output_file} already exists. Skipping.")
            return

        yaml.dump(
            config_dict, open(output_file.replace("results.json", "config.yaml"), "w")
        )

        # Initialize the pipeline
        attacker_config, defender_config, judge_config = parse_configs_from_dict(config_dict)

        pipe = InferPipeline(
            InferPipelineConfig(
                attacker_config=attacker_config,
                defender_config=defender_config,
                judge_configs=judge_config,
            ),
            verbose=verbose  # Pass verbose flag to pipeline
        )

        console.print(
            f"[bold green]Inference on {config_dict['defender']['target_llm_config']['model_name']}[/bold green]")

        # Chat loop
        df = pd.read_csv(input_file)
        iterator = tqdm(df.iterrows(), total=len(df))
        results = []

        for _, row in iterator:
            messages = [{"role": "user", "content": row["Goal"]}]
            result = [pipe(
                    messages,
                    request_reformulated=row.get(attacker_config.attacker_name, None),
            )]
            results.append({"goal": row["Goal"], "data": result})
            pipe.reset()

        # Save inference history if requested
        with open(output_file, 'w') as f:
            json.dump({"config": config_dict, "results": results}, f, indent=4)

        console.print(f"[bold green]Results saved to {output_file}[/bold green]")

    except Exception as e:
        typer.echo(f"Error during inference: {str(e)}", err=True)
        logging.exception("Exception during inference")
        raise typer.Exit(1)


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    app()