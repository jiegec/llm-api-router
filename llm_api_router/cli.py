"""CLI for LLM API Router."""

import json
import sys
from pathlib import Path

import click
import uvicorn

from .config import RouterConfig, load_default_config
from .server import create_app


@click.group()
def cli():
    """LLM API Router - Route OpenAI and Anthropic API requests with priority fallback."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file (JSON)",
)
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind the server to",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    show_default=True,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for development",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    show_default=True,
    type=int,
    help="Number of worker processes",
)
def serve(config: Path | None, host: str, port: int, reload: bool, workers: int):
    """Start the LLM API Router server."""
    try:
        # Load configuration
        if config:
            click.echo(f"Loading configuration from: {config}")
            router_config = RouterConfig.from_json_file(str(config))
        else:
            click.echo("Loading default configuration...")
            router_config = load_default_config()
            if router_config is None:
                click.echo(
                    "❌ No configuration found. Create a config file or use --config option.",
                    err=True,
                )
                click.echo(
                    "   Example: cp llm_router_config.example.json llm_router_config.json",
                    err=True,
                )
                sys.exit(1)

        # Validate configuration
        errors = router_config.validate()
        if errors:
            click.echo("❌ Configuration errors:", err=True)
            for error in errors:
                click.echo(f"   - {error}", err=True)
            sys.exit(1)

        # Create FastAPI app
        app = create_app(router_config)

        # Display server info
        click.echo("=" * 60)
        click.echo("🚀 LLM API Router Server")
        click.echo("=" * 60)
        click.echo(f"📁 Config: {config or 'default'}")
        click.echo(f"🌐 Host: {host}")
        click.echo(f"🔌 Port: {port}")
        click.echo(f"🔄 Reload: {reload}")
        click.echo(f"👷 Workers: {workers}")
        click.echo()

        # Display provider info
        click.echo("🤖 Providers:")
        if router_config.openai_providers:
            click.echo(f"  • OpenAI: {len(router_config.openai_providers)} provider(s)")
            for i, provider in enumerate(router_config.openai_providers, 1):
                click.echo(
                    f"    {i}. Priority {provider.priority}: {provider.base_url or 'default'}"
                )
        else:
            click.echo("  • OpenAI: No providers configured")

        if router_config.anthropic_providers:
            click.echo(
                f"  • Anthropic: {len(router_config.anthropic_providers)} provider(s)"
            )
            for i, provider in enumerate(router_config.anthropic_providers, 1):
                click.echo(
                    f"    {i}. Priority {provider.priority}: {provider.base_url or 'default'}"
                )
        else:
            click.echo("  • Anthropic: No providers configured")

        click.echo()
        click.echo("🔗 Endpoints:")
        click.echo(f"  • OpenAI: http://{host}:{port}/openai/chat/completions")
        click.echo(f"  • Anthropic: http://{host}:{port}/anthropic/chat/completions")
        click.echo(f"  • Health: http://{host}:{port}/health")
        click.echo(f"  • Docs: http://{host}:{port}/docs")
        click.echo()
        click.echo("📝 Example usage with OpenAI client:")
        click.echo(f'  openai.OpenAI(base_url="http://{host}:{port}/openai")')
        click.echo()
        click.echo("=" * 60)
        click.echo("Starting server... Press Ctrl+C to stop")
        click.echo("=" * 60)

        # Start server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
        )

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default="llm_router_config.json",
    show_default=True,
    help="Output path for configuration file",
)
def init(output: Path):
    """Create a new configuration file with examples."""
    if output.exists():
        if not click.confirm(f"File {output} already exists. Overwrite?"):
            click.echo("Cancelled.")
            return

    example_config = {
        "openai": [
            {
                "api_key": "sk-your-openai-api-key-here",
                "priority": 1,
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
                "max_retries": 3,
            },
            {
                "api_key": "sk-your-backup-openai-api-key",
                "priority": 2,
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
                "max_retries": 3,
            },
        ],
        "anthropic": [
            {
                "api_key": "sk-ant-your-anthropic-api-key",
                "priority": 1,
                "base_url": "https://api.anthropic.com",
                "timeout": 30,
                "max_retries": 3,
            }
        ],
    }

    try:
        with open(output, "w") as f:
            json.dump(example_config, f, indent=2)

        click.echo(f"✅ Created configuration file: {output}")
        click.echo()
        click.echo("Next steps:")
        click.echo(f"  1. Edit {output} and add your API keys")
        click.echo("  2. Start server: poetry run llm-api-router serve")
        click.echo(f"  3. Or: poetry run llm-api-router serve --config {output}")

    except Exception as e:
        click.echo(f"❌ Error creating config file: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file (JSON)",
)
def check(config: Path | None):
    """Check configuration and provider connectivity."""
    try:
        click.echo("🔍 Checking configuration...")

        # Load config
        if config:
            router_config = RouterConfig.from_json_file(str(config))
        else:
            router_config = load_default_config()
            if router_config is None:
                click.echo("❌ No configuration file found.")
                click.echo("   Run: poetry run llm-api-router init")
                sys.exit(1)

        # Validate
        errors = router_config.validate()
        if errors:
            click.echo("❌ Configuration errors:")
            for error in errors:
                click.echo(f"   - {error}")
            sys.exit(1)

        click.echo("✅ Configuration is valid")
        click.echo()

        # Show provider info
        click.echo("📋 Provider configuration:")
        if router_config.openai_providers:
            click.echo(f"  OpenAI: {len(router_config.openai_providers)} provider(s)")
            for i, provider in enumerate(router_config.openai_providers, 1):
                click.echo(f"    {i}. Priority {provider.priority}")
                click.echo(
                    f"       API Key: {'*' * 8}{provider.api_key[-4:] if len(provider.api_key) > 4 else '****'}"
                )
                click.echo(f"       Base URL: {provider.base_url or 'default'}")
                click.echo(f"       Timeout: {provider.timeout}s")
                click.echo(f"       Max Retries: {provider.max_retries}")
        else:
            click.echo("  OpenAI: No providers configured")

        if router_config.anthropic_providers:
            click.echo(
                f"  Anthropic: {len(router_config.anthropic_providers)} provider(s)"
            )
            for i, provider in enumerate(router_config.anthropic_providers, 1):
                click.echo(f"    {i}. Priority {provider.priority}")
                click.echo(
                    f"       API Key: {'*' * 8}{provider.api_key[-4:] if len(provider.api_key) > 4 else '****'}"
                )
                click.echo(f"       Base URL: {provider.base_url or 'default'}")
                click.echo(f"       Timeout: {provider.timeout}s")
                click.echo(f"       Max Retries: {provider.max_retries}")
        else:
            click.echo("  Anthropic: No providers configured")

        click.echo()
        click.echo("✅ Configuration check passed!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
