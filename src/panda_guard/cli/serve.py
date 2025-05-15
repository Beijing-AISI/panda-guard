import sys
import os
import yaml
import typer
import logging
import time
from typing import Optional, Iterator, Dict, Any, List, Union
from pathlib import Path
from rich.console import Console
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import json
import asyncio
from pydantic import BaseModel, Field

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

app = typer.Typer(help="Language model server and chat interface", invoke_without_command=True)
console = Console()

# Global variables to store the pipeline instance
global_pipeline = None
verbose_mode = False


# Define API models based on OpenAI API format
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "panda-guard"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


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


def create_fastapi_app(pipeline_instance):
    """Create a FastAPI application with OpenAI-compatible endpoints."""
    api_app = FastAPI(title="PandaGuard API Server",
                      description="API server compatible with OpenAI API format")

    # Add CORS middleware
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define the chat completions endpoint
    @api_app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        nonlocal pipeline_instance
        try:
            # Convert request to the format expected by the pipeline
            messages = request.messages

            # Update config based on request parameters
            if request.temperature is not None:
                pipeline_instance.defender.target_llm_gen_config.temperature = request.temperature

            # Set streaming mode
            pipeline_instance.defender.target_llm_gen_config.stream = request.stream

            # Process through pipeline
            start_time = time.time()

            if request.stream:
                # For streaming, return a StreamingResponse
                async def stream_generator():
                    try:
                        # Call the pipeline and get the result
                        result = pipeline_instance(messages)

                        # Debug logging
                        logging.debug(f"Stream result type: {type(result)}")
                        if isinstance(result, dict):
                            logging.debug(f"Stream result keys: {result.keys()}")

                        id_value = f"chatcmpl-{int(time.time())}"
                        created_time = int(time.time())
                        model_name = pipeline_instance.defender.target_llm._NAME

                        # Create and yield the initial response
                        header = {
                            "id": id_value,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(header)}\n\n"

                        # Process the streaming content
                        content_so_far = ""

                        # Handle different types of streaming results
                        if isinstance(result, dict) and "messages" in result:
                            stream_content = result["messages"]

                            # Check if it's an iterator
                            if hasattr(stream_content, "__iter__") and hasattr(stream_content, "__next__"):
                                for chunk in stream_content:
                                    content_so_far += chunk
                                    chunk_data = {
                                        "id": id_value,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": chunk},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                                    await asyncio.sleep(0.01)
                            else:
                                # If it's not an iterator, treat it as a complete message
                                if isinstance(stream_content, list) and len(stream_content) > 0:
                                    final_content = stream_content[-1]["content"]
                                    content_so_far = final_content
                                    chunk_data = {
                                        "id": id_value,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": final_content},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                        elif isinstance(result, list):
                            # If the result is a list of messages
                            if len(result) > 0:
                                final_content = result[-1]["content"]
                                content_so_far = final_content
                                chunk_data = {
                                    "id": id_value,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": final_content},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"

                        # Run judge evaluation if judges are available
                        if hasattr(pipeline_instance, 'judges') and pipeline_instance.judges:
                            try:
                                user_msg = messages[-1]["content"]

                                # Create messages for judge evaluation
                                judge_messages = [
                                    {"role": "user", "content": user_msg},
                                    {"role": "assistant", "content": content_so_far}
                                ]

                                # Run judges
                                judge_results = pipeline_instance.parallel_judging(judge_messages, user_msg)

                                # Send judge results as a special chunk if they exist
                                if judge_results:
                                    judge_chunk = {
                                        "id": id_value,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": model_name,
                                        "judge_results": judge_results
                                    }
                                    yield f"data: {json.dumps(judge_chunk)}\n\n"
                            except Exception as judge_error:
                                logging.exception(f"Error during judge evaluation: {str(judge_error)}")

                        # Send completion signal
                        final_chunk = {
                            "id": id_value,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        logging.exception(f"Error during streaming: {str(e)}")
                        error_data = {
                            "error": {
                                "message": str(e),
                                "type": "server_error"
                            }
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                # For non-streaming, return a standard JSON response
                result = pipeline_instance(messages)

                if isinstance(result, dict) and "messages" in result:
                    response_messages = result["messages"]
                else:
                    response_messages = result

                # Get the assistant's response
                if isinstance(response_messages, list) and len(response_messages) > 0:
                    assistant_message = response_messages[-1]["content"]
                else:
                    assistant_message = "No response generated"

                # Get usage data if available
                usage = result.get("usage", {})
                token_usage = {
                    "prompt_tokens": sum(role.get("prompt_tokens", 0) for role in usage.values()),
                    "completion_tokens": sum(role.get("completion_tokens", 0) for role in usage.values()),
                    "total_tokens": sum(
                        role.get("prompt_tokens", 0) + role.get("completion_tokens", 0) for role in usage.values())
                }

                # Run judge evaluation if judges are available
                judge_results = None
                if hasattr(pipeline_instance, 'judges') and pipeline_instance.judges:
                    try:
                        user_msg = messages[-1]["content"]

                        # Create messages for judge evaluation
                        judge_messages = [
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_message}
                        ]

                        # Run judges
                        judge_results = pipeline_instance.parallel_judging(judge_messages, user_msg)
                    except Exception as judge_error:
                        logging.exception(f"Error during judge evaluation: {str(judge_error)}")

                # Create OpenAI-compatible response
                response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": pipeline_instance.defender.target_llm._NAME,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": assistant_message
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": token_usage
                }

                # Add judge results if available
                if judge_results:
                    response["judge_results"] = judge_results

                return response

        except Exception as e:
            logging.exception("Error in chat completions endpoint")
            raise HTTPException(status_code=500, detail=str(e))

    # Define the models endpoint
    @api_app.get("/v1/models")
    async def list_models():
        nonlocal pipeline_instance
        try:
            model_name = pipeline_instance.defender.target_llm._NAME

            # Create a response in OpenAI format
            models_data = [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "panda-guard"
                }
            ]

            return {
                "object": "list",
                "data": models_data
            }

        except Exception as e:
            logging.exception("Error in models endpoint")
            raise HTTPException(status_code=500, detail=str(e))

    # Add health check endpoint
    @api_app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "1.0.0"}

    return api_app


@app.callback(invoke_without_command=True)
def start(
        config: Optional[str] = typer.Argument(None, help="Path to YAML configuration file"),
        defense: Optional[Path] = typer.Option(None, "--defense", "-d",
                                               help="Path to defense configuration file or defense type (goal_priority/icl/none/rpo/self_reminder/smoothllm)"),
        judge: Optional[str] = typer.Option(None, "--judge", "-j",
                                            help="Path to judge configuration file or defense type (llm_based/rule_based). Multiple judges can be specified using comma separation."),
        endpoint: Optional[Path] = typer.Option(None, "--endpoint", "-e",
                                                help="Path to endpoint configuration file or endpoint type (openai/gemini/claude)"),
        model: Optional[Path] = typer.Option(None, "--model", "-m", help="model name"),
        temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Override temperature setting"),
        device: Optional[str] = typer.Option(None, "--device", help="Device to run the model on (e.g., 'cuda:0')"),
        log_level: str = typer.Option("WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
        port: int = typer.Option(8000, "--port", "-p", help="Port to run the server on"),
        host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server to"),
        verbose: bool = typer.Option(False, "--verbose/--no-verbose",
                                     help="Enable/disable verbose mode with token usage info"),
):
    """
    Start an API server compatible with OpenAI API format.

    Accepts the same configuration options as the chat interface, plus host and port settings.
    """
    global global_pipeline, verbose_mode
    verbose_mode = verbose

    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    try:
        config_dict = {}

        if config is None:
            config_path = get_package_config_path('tasks/chat')
            config_dict = load_yaml(config_path)

        # Load and merge defense config if provided
        if defense:
            if not defense.exists():
                defense_path = get_package_config_path(f'defenses/{defense}')

                if not defense_path.exists():
                    typer.echo(f"Error: Defense config file {defense} not found", err=True)
                    raise typer.Exit(1)

                defense = defense_path
            config_dict["defender"] = load_yaml(defense)

        # Parse multiple judges if provided
        judge_configs = []
        if judge:
            judge_names = [j.strip() for j in judge.split(',')]
            for judge_name in judge_names:
                judge_path = Path(judge_name)
                if not judge_path.exists():
                    judge_path = get_package_config_path(f'judges/{judge_name}')

                    if not judge_path.exists():
                        typer.echo(f"Error: Judge config file {judge_name} not found", err=True)
                        raise typer.Exit(1)

                judge_config = load_yaml(judge_path)
                if 'judge_llm_config' in judge_config:
                    if judge_config["judge_llm_config"].get("base_url", None) is None:
                        if os.environ.get("OPENAI_BASE_URL"):
                            judge_config["judge_llm_config"]["base_url"] = os.environ["OPENAI_BASE_URL"]
                        else:
                            judge_config["judge_llm_config"]["base_url"] = "https://api.openai.com/v1"

                    if judge_config["judge_llm_config"].get("api_key", None) is None:
                        if os.environ.get("OPENAI_API_KEY"):
                            judge_config["judge_llm_config"]["api_key"] = os.environ["OPENAI_API_KEY"]
                        else:
                            raise ValueError("API key not found for judge LLM config")

                judge_configs.append(judge_config)

            config_dict["judges"] = judge_configs

        endpoint_type = None

        if endpoint is None:
            endpoint_type = "openai"
            endpoint_path = get_package_config_path(f"endpoints/{endpoint_type}")
            config_dict["defender"]["target_llm_config"] = load_yaml(endpoint_path)
        elif config and config.lower() in ["openai", "gemini", "claude"]:
            endpoint_type = config.lower()
            endpoint_path = get_package_config_path(f"endpoints/{endpoint_type}")
            config_dict["defender"]["target_llm_config"] = load_yaml(endpoint_path)
        elif config and config.endswith(".yaml"):
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

        # Override device if specified
        if device:
            config_dict["defender"]["target_llm_config"]["device_map"] = device

        # Override temperature if specified
        if temperature is not None:
            config_dict["defender"]["target_llm_gen_config"]["temperature"] = temperature

        # Initialize the pipeline
        attacker_config, defender_config, judge_config = parse_configs_from_dict(config_dict)

        pipeline = InferPipeline(
            InferPipelineConfig(
                attacker_config=attacker_config,
                defender_config=defender_config,
                judge_configs=judge_config,
            ),
            verbose=verbose  # Pass verbose flag to pipeline
        )

        # Store in global variable for access from the API
        global_pipeline = pipeline

        # Create the FastAPI app
        api_app = create_fastapi_app(pipeline)

        # Print server information
        model_name = config_dict["defender"]["target_llm_config"]["model_name"]
        console.print(f"[bold green]Starting API server with model: {model_name}[/bold green]")
        console.print(f"[bold green]Server running at http://{host}:{port}[/bold green]")
        console.print("[bold green]API is compatible with OpenAI API format[/bold green]")

        # Start the server
        uvicorn.run(api_app, host=host, port=port)

    except Exception as e:
        typer.echo(f"Error starting server: {str(e)}", err=True)
        logging.exception("Exception during server startup")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()