# src/panda_guard/cli/main.py
import typer
from panda_guard.cli import chat, serve, attack, inference, eval

app = typer.Typer(help="Panda Guard: An Open Pipeline for Jailbreaking Language Models")

# Add subcommands
app.add_typer(attack.app, name="attack")
app.add_typer(inference.app, name="inference")
app.add_typer(eval.app, name="eval")
app.add_typer(chat.app, name="chat")
app.add_typer(serve.app, name="serve")

# Add version info callback
def version_callback(value: bool):
    if value:
        typer.echo("Panda Guard version 0.1.0")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit", callback=version_callback),
):
    """
    Panda Guard CLI for chatting with and serving language models.
    """
    pass

if __name__ == "__main__":
    app()