#!/usr/bin/env python3
"""Main entry point for LLM API Router server."""

import argparse

import uvicorn

from llm_api_router.config import RouterConfig
from llm_api_router.server import create_app


def main():
    parser = argparse.ArgumentParser(description="LLM API Router Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="llm_router_config.json",
                       help="Path to configuration JSON file")

    args = parser.parse_args()

    # Load configuration
    try:
        config = RouterConfig.from_json_file(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Creating example configuration...")
        config = RouterConfig.from_dict({
            "openai": [
                {
                    "api_key": "sk-demo-openai-key-1",
                    "priority": 1,
                    "base_url": "https://api.openai.com/v1",
                }
            ],
            "anthropic": [
                {
                    "api_key": "sk-ant-demo-anthropic-key-1",
                    "priority": 1,
                    "base_url": "https://api.anthropic.com",
                }
            ]
        })
        print("Using demo configuration. Replace with real API keys in config file.")

    app = create_app(config)

    print(f"Starting server on {args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /              - Server info")
    print("  GET  /health        - Health check")
    print("  POST /openai/chat/completions    - OpenAI chat completion")
    print("  POST /anthropic/chat/completions - Anthropic chat completion")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
