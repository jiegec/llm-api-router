"""CLI for LLM API Router."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import uvicorn

from .config import RouterConfig, load_default_config
from .log_compression import compress_old_logs
from .server import create_app


@click.group()
def cli() -> None:
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
    "--workers",
    "-w",
    default=1,
    show_default=True,
    type=int,
    help="Number of worker processes",
)
def serve(config: Path | None, host: str, port: int, workers: int) -> None:
    """Start the LLM API Router server."""
    try:
        # Load configuration
        router_config: RouterConfig | None
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

        # At this point, router_config is not None
        assert router_config is not None

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
        click.echo(f"  • Anthropic: http://{host}:{port}/anthropic/v1/messages")
        click.echo(f"  • Health: http://{host}:{port}/health")
        click.echo(f"  • Status: http://{host}:{port}/status")
        click.echo(f"  • Docs: http://{host}:{port}/docs")
        click.echo()
        click.echo("📝 Example usage:")
        click.echo(f'  OpenAI: openai.OpenAI(base_url="http://{host}:{port}/openai")')
        click.echo(
            f'  Anthropic: anthropic.Anthropic(base_url="http://{host}:{port}/anthropic")'
        )
        click.echo()
        click.echo("=" * 60)
        click.echo("Starting server... Press Ctrl+C to stop")
        click.echo("=" * 60)

        # Start server
        uvicorn.run(
            app,
            host=host,
            port=port,
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
def init(output: Path) -> None:
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
                "provider_name": "openai-1",
                "timeout": 30,
                "max_retries": 3,
            },
            {
                "api_key": "sk-your-backup-openai-api-key",
                "priority": 2,
                "base_url": "https://api.openai.com/v1",
                "provider_name": "openai-2",
                "timeout": 30,
                "max_retries": 3,
            },
        ],
        "anthropic": [
            {
                "api_key": "sk-ant-your-anthropic-api-key",
                "priority": 1,
                "base_url": "https://api.anthropic.com",
                "provider_name": "anthropic-1",
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
    "--log-dir",
    "-l",
    default="logs",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory containing log files",
)
def compress_logs(log_dir: Path) -> None:
    """Compress log files that are at least 7 days old with zstd."""
    try:
        compressed = compress_old_logs(log_dir)
        if compressed:
            for f in compressed:
                click.echo(f"Compressed: {f}")
            click.echo(f"Compressed {len(compressed)} file(s)")
        else:
            click.echo("No files to compress")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file (JSON)",
)
def check(config: Path | None) -> None:
    """Check configuration and provider connectivity."""
    try:
        click.echo("🔍 Checking configuration...")

        # Load config
        router_config: RouterConfig | None
        if config:
            router_config = RouterConfig.from_json_file(str(config))
        else:
            router_config = load_default_config()
            if router_config is None:
                click.echo("❌ No configuration file found.")
                click.echo("   Run: poetry run llm-api-router init")
                sys.exit(1)

        # At this point, router_config is not None
        assert router_config is not None

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


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file (JSON) to pass to the serve command",
)
@click.option(
    "--unit-name",
    default="llm-api-router",
    show_default=True,
    help="systemd user unit name (without .service suffix)",
)
def install(config: Path | None, unit_name: str) -> None:
    """Install llm-api-router as a systemd user service."""
    try:
        poetry_path = shutil.which("poetry")
        if poetry_path is None:
            click.echo("❌ poetry not found in PATH.", err=True)
            sys.exit(1)

        project_dir = Path.cwd().resolve()
        serve_cmd = f"{poetry_path} run llm-api-router serve"
        if config:
            serve_cmd += f" --config {config}"

        service_content = f"""[Unit]
Description=LLM API Router
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={project_dir}
ExecStart={serve_cmd}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target"""

        xdg_config = Path.home() / ".config" / "systemd" / "user"
        service_file = xdg_config / f"{unit_name}.service"

        if service_file.exists():
            click.echo(f"⚠️  Service file already exists: {service_file}")
            if not click.confirm("Overwrite?"):
                click.echo("Cancelled.")
                return

        click.echo(f"This will install a systemd user unit: {unit_name}.service")
        click.echo(f"Working directory: {project_dir}")
        click.echo(f"Service file:      {service_file}")
        click.echo()
        click.echo("The following steps will be performed:")
        click.echo("  1. Write the systemd user unit file")
        click.echo("  2. Reload systemd daemon")
        click.echo("  3. Enable lingering for the current user")
        click.echo("  4. Enable the service to start on boot")
        click.echo("  5. Start the service now")
        click.echo()
        if not click.confirm("Proceed?"):
            click.echo("Cancelled.")
            return

        xdg_config.mkdir(parents=True, exist_ok=True)
        service_file.write_text(service_content)
        click.echo(f"✅ Wrote service file: {service_file}")

        click.echo("Reloading systemd daemon...")
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
        )
        click.echo("✅ Reloaded systemd daemon")

        click.echo("Enabling lingering...")
        loginctl_result = subprocess.run(
            ["sudo", "loginctl", "enable-linger", os.getlogin()],
            capture_output=True,
            text=True,
        )
        if loginctl_result.returncode != 0:
            click.echo(
                f"⚠️  Could not enable lingering (sudo required): {loginctl_result.stderr.strip()}",
                err=True,
            )
            click.echo(
                "   The service will not start at boot until lingering is enabled."
            )
            click.echo(f"   Run manually: sudo loginctl enable-linger {os.getlogin()}")
            if not click.confirm("Continue without lingering?"):
                service_file.unlink()
                click.echo("Rolled back. Cancelled.")
                return
        else:
            click.echo("✅ Enabled lingering")

        click.echo(f"Enabling service {unit_name}...")
        subprocess.run(
            ["systemctl", "--user", "enable", f"{unit_name}.service"],
            check=True,
        )
        click.echo(f"✅ Enabled service {unit_name}")

        click.echo(f"Starting service {unit_name}...")
        subprocess.run(
            ["systemctl", "--user", "start", f"{unit_name}.service"],
            check=True,
        )
        click.echo(f"✅ Started service {unit_name}")

        click.echo()
        click.echo("=" * 60)
        click.echo("🎉 Installation complete!")
        click.echo("=" * 60)
        click.echo()
        click.echo("Useful commands:")
        click.echo(f"  Status:   systemctl --user status {unit_name}")
        click.echo(f"  Logs:     journalctl --user -u {unit_name} -f")
        click.echo(f"  Stop:     systemctl --user stop {unit_name}")
        click.echo(f"  Restart:  systemctl --user restart {unit_name}")
        click.echo(f"  Disable:  systemctl --user disable {unit_name}")
        click.echo(f"  Remove:   rm {service_file}")

    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Command failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
