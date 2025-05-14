import sys
import os
import yaml
import typer
import logging
import time
from typing import Optional, Iterator, Dict, Any, List
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich import box

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

app = typer.Typer(help="Interactive language model jailbreak attacks")
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


def display_help():
    """Display available commands."""
    help_table = Table(title="Available Commands", box=box.ROUNDED)
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="green")

    help_table.add_row("/exit", "Exit the attack")
    help_table.add_row("/reset", "Reset the conversation")
    help_table.add_row("/help", "Display this help message")
    help_table.add_row("/verbose", "Toggle verbose mode (show token usage)")
    help_table.add_row("/save [filename]", "Save conversation to file")
    help_table.add_row("/stats", "Display current conversation statistics")

    console.print(help_table)


@app.command()
def start(
        config: Optional[str] = typer.Argument(None, help="Path to YAML configuration file"),
        attacker: Optional[Path] = typer.Option(None, "--attacker", "-a",
                                               help="Path to attack configuration file or attack method (#todo)"),
        endpoint: Optional[Path] = typer.Option(None, "--endpoint", "-e",
                                                help="Path to endpoint configuration file or endpoint type (openai/gemini/claude/hf)"),
        model: Optional[Path] = typer.Option(None, "--model", "-m", help="model name"),
        device: Optional[str] = typer.Option(None, "--device", help="Device to run the model on (e.g., 'cuda:0')"),
        log_level: str = typer.Option("WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save attack history to file"),
        stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable/disable streaming output"),
        verbose: bool = typer.Option(False, "--verbose/--no-verbose",
                                     help="Enable/disable verbose mode with token usage info"),
):
    """
    Start an interactive attack session using configuration from a YAML file or a predefined model type.

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
            config_path = get_package_config_path('tasks/attack')
            config_dict = load_yaml(config_path)
        else:
            config_path = Path(config)
            if not config_path.exists():
                typer.echo(f"Error: Config file {config} not found", err=True)
                raise typer.Exit(1)

            config_dict = load_yaml(config_path)

        if attacker:
            if not attacker.exists():
                attacker_path = get_package_config_path(f'attacks/{attacker}')

                if not attacker_path.exists():
                    typer.echo(f"Error: Defense config file {attacker} not found", err=True)
                    raise typer.Exit(1)

                attacker = attacker_path
            config_dict["attacker"] = load_yaml(attacker)

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
                config_dict["attacker"]["target_llm_config"] = load_yaml(endpoint_path)
                config_dict["attacker"]["target_llm_config"]['model_name'] = str(model)

        # Override device if specified
        if device:
            config_dict["defender"]["target_llm_config"]["device_map"] = device

        # Enable or disable streaming in the configuration
        config_dict["defender"]["target_llm_gen_config"]["stream"] = stream

        print("Configuration loaded successfully:", config_dict)

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
            f"[bold green]Jailbreak attacks on {config_dict['defender']['target_llm_config']['model_name']}[/bold green]")
        console.print("[bold]Type your message, and the model will output the corresponding result after the given attack[/bold]")

        # Initialize attack history and variables for tracking statistics
        messages = []
        total_usage = {"attacker": {"prompt_tokens": 0, "completion_tokens": 0},
                       "defender": {"prompt_tokens": 0, "completion_tokens": 0}}
        total_response_time = 0
        total_responses = 0

        # Chat loop
        while True:
            user_input = console.input("[bold blue]User:[/bold blue] ")

            if user_input.lower().startswith("/"):
                # Handle commands
                if user_input.lower() == "/exit":
                    break

                elif user_input.lower() == "/reset":
                    messages = []
                    pipe.reset()
                    total_usage = {"attacker": {"prompt_tokens": 0, "completion_tokens": 0},
                                   "defender": {"prompt_tokens": 0, "completion_tokens": 0}}
                    total_response_time = 0
                    total_responses = 0
                    console.print("[bold yellow]Conversation reset[/bold yellow]")
                    continue

                elif user_input.lower() == "/help":
                    display_help()
                    continue

                elif user_input.lower() == "/verbose":
                    verbose = not verbose
                    console.print(f"[bold yellow]Verbose mode {'enabled' if verbose else 'disabled'}[/bold yellow]")
                    continue

                elif user_input.lower().startswith("/save"):
                    parts = user_input.split(maxsplit=1)
                    filename = parts[1] if len(parts) > 1 else "conversation.json"
                    import json
                    with open(filename, 'w') as f:
                        json.dump(messages, f, indent=2)
                    console.print(f"[bold]Chat history saved to {filename}[/bold]")
                    continue

                elif user_input.lower() == "/stats":
                    # Display cumulative stats in the new, less intrusive style
                    console.print("[bold]Cumulative Conversation Statistics:[/bold]")

                    # Get the most up-to-date token usage
                    if messages:
                        current_usage = pipe.calc_tokens()

                        # Calculate totals from current usage
                        total_prompt = sum(role["prompt_tokens"] for role in current_usage.values())
                        total_completion = sum(role["completion_tokens"] for role in current_usage.values())
                        total_all = total_prompt + total_completion
                    else:
                        total_prompt = 0
                        total_completion = 0
                        total_all = 0

                    console.print(
                        f"Token usage: Prompt: {total_prompt} | Completion: {total_completion} | Total: {total_all}")

                    if total_responses > 0:
                        avg_response_time = total_response_time / total_responses
                        console.print(f"Average response time: {avg_response_time:.2f}s")
                    continue

                else:
                    display_help()
                    continue

            # Create or update messages
            if not messages:
                messages = [{"role": "user", "content": user_input}]
            else:
                messages.append({"role": "user", "content": user_input})

            try:
                # Record start time for response speed calculation
                start_time = time.time()

                # Process through pipeline
                result = pipe(messages)

                # Flag to track if we need to run judges
                run_judges = len(pipe.judges) > 0

                # Check if the result is streaming or regular
                if stream and isinstance(result, dict) and "messages" in result and is_iterator(result["messages"]):
                    # Handle streaming response
                    console.print("[bold green]Assistant:[/bold green]")

                    full_response = ""

                    try:
                        generator = result["messages"]

                        for text_chunk in generator:
                            full_response += text_chunk
                            console.print(text_chunk, end="")
                            sys.stdout.flush()

                        console.print()

                    except Exception as e:
                        console.print(f"\n[bold red]Error during streaming: {str(e)}[/bold red]")
                        logging.exception("Exception during streaming")
                else:
                    # Handle regular non-streaming response
                    if isinstance(result, dict) and "messages" in result:
                        messages = result["messages"]
                    else:
                        # If result is not a dict with messages, it might be the messages directly
                        messages = result

                    # Display assistant response
                    assistant_response = messages[-1]["content"]
                    console.print("[bold green]Assistant:[/bold green]")
                    console.print(Markdown(assistant_response))

                # Within the attack loop where response time is calculated and usage is displayed:

                # Calculate response time
                response_time = time.time() - start_time
                total_response_time += response_time
                total_responses += 1

                # Get the most up-to-date token usage directly from the pipeline
                current_usage = pipe.calc_tokens()

                # Update total usage with the current values (not incremental)
                for role in current_usage:
                    for token_type in current_usage[role]:
                        if role in total_usage:
                            if token_type not in total_usage[role]:
                                total_usage[role][token_type] = 0
                            # Replace with current value instead of adding
                            total_usage[role][token_type] = current_usage[role][token_type]

                # Display token usage information if verbose mode is enabled
                if verbose:
                    console.print("")  # Add an empty line for better readability
                    display_token_info(current_usage, response_time)  # Pass the actual response_time value

            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                logging.exception("Exception during attack")
                continue

        # Save attack history if requested
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(messages, f, indent=2)
            console.print(f"[bold]Attack history saved to {output}[/bold]")

    except Exception as e:
        typer.echo(f"Error during attack: {str(e)}", err=True)
        logging.exception("Exception during attack")
        raise typer.Exit(1)


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    app()