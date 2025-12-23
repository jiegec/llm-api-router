#!/usr/bin/env python3
"""Example usage of LLM API Router."""

from llm_api_router.config import RouterConfig
from llm_api_router.models import (
    ChatCompletionRequest,
    Message,
    Role,
)
from llm_api_router.router import LLMRouter
from llm_api_router.server import create_app


async def example_programmatic_usage():
    """Example of using the router programmatically."""
    print("=== Programmatic Usage Example ===")

    # Create configuration
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-demo-openai-key-1",
                    "priority": 1,
                },
                {
                    "api_key": "sk-demo-openai-key-2",
                    "priority": 2,
                },
            ],
            "anthropic": [
                {
                    "api_key": "sk-ant-demo-anthropic-key-1",
                    "priority": 1,
                }
            ],
        }
    )

    # Create router for OpenAI providers
    _ = LLMRouter(config.openai_providers)  # Demo only

    # Create a chat completion request
    request = ChatCompletionRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What is the capital of France?"),
        ],
        model="gpt-4o-mini",
        max_tokens=100,
    )

    print(f"Request: {request.model} with {len(request.messages)} messages")
    print("Note: This is a demo. Replace API keys with real ones to test.")
    print()


def example_server_usage():
    """Example of using the server."""
    print("=== Server Usage Example ===")

    # Create configuration
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-your-openai-api-key-here",
                    "priority": 1,
                }
            ],
            "anthropic": [
                {
                    "api_key": "sk-ant-your-anthropic-api-key-here",
                    "priority": 1,
                }
            ],
        }
    )

    # Create FastAPI app
    _ = create_app(config)  # Demo only

    print("Server created with endpoints:")
    print("  POST /openai/chat/completions")
    print("  POST /anthropic/chat/completions")
    print("  GET  /health")
    print("  GET  /")
    print()
    print("To run the server:")
    print("  poetry run python main.py --config your-config.json")
    print()


def example_config_file():
    """Example configuration file content."""
    print("=== Example Configuration File ===")

    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-your-primary-openai-api-key",
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
                    "api_key": "sk-ant-your-primary-anthropic-api-key",
                    "priority": 1,
                    "base_url": "https://api.anthropic.com",
                    "timeout": 30,
                    "max_retries": 3,
                }
            ],
        }
    )

    print("Save as llm_router_config.json:")
    print(config.to_json(indent=2))
    print()


def main():
    """Run all examples."""
    print("LLM API Router Examples")
    print("=" * 50)
    print()

    example_programmatic_usage()
    example_server_usage()
    example_config_file()

    print("For more details, see:")
    print("  - README.md for documentation")
    print("  - tests/ for usage examples")
    print("  - client_example.py for client usage")
    print()
    print("To test the server:")
    print("  1. cp llm_router_config.example.json llm_router_config.json")
    print("  2. Edit llm_router_config.json with your API keys")
    print("  3. poetry run python main.py")
    print("  4. In another terminal: poetry run python client_example.py")


if __name__ == "__main__":
    main()
