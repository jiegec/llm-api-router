# LLM API Router

A FastAPI-based LLM API router server that provides separate endpoints for OpenAI and Anthropic APIs, each with automatic fallback when APIs fail or are rate limited.

## Features

- **Separate endpoints**: `/openai/chat/completions` and `/anthropic/chat/completions`
- **Multi-provider support per endpoint**: Each endpoint can route to multiple API keys/servers
- **Priority-based routing**: Configure priority order for API providers within each endpoint
- **Automatic fallback**: When a high-priority provider fails or is rate-limited, automatically switches to the next available provider
- **Error handling**: Comprehensive error detection and HTTP error responses
- **Async support**: Built with async/await for high performance
- **Type safety**: Full type hints with Pydantic models
- **FastAPI server**: Production-ready web server with OpenAPI documentation

## Installation

```bash
poetry install
```

## Running the Server

```bash
# Basic server
poetry run python main.py

# With custom host/port
poetry run python main.py --host 0.0.0.0 --port 8080

# With auto-reload for development
poetry run python main.py --reload
```

## API Endpoints

- `GET /` - Server information and available endpoints
- `GET /health` - Health check endpoint
- `POST /openai/chat/completions` - OpenAI-compatible chat completion
- `POST /anthropic/chat/completions` - Anthropic-compatible chat completion

## Configuration

### JSON Configuration File

Create a JSON configuration file (default: `llm_router_config.json`):

```json
{
  "openai": [
    {
      "api_key": "sk-your-primary-openai-api-key-here",
      "priority": 1,
      "base_url": "https://api.openai.com/v1",
      "timeout": 30,
      "max_retries": 3
    },
    {
      "api_key": "sk-your-backup-openai-api-key-here",
      "priority": 2,
      "base_url": "https://api.openai.com/v1",
      "timeout": 30,
      "max_retries": 3
    }
  ],
  "anthropic": [
    {
      "api_key": "sk-ant-your-primary-anthropic-api-key-here",
      "priority": 1,
      "base_url": "https://api.anthropic.com",
      "timeout": 30,
      "max_retries": 3
    },
    {
      "api_key": "sk-ant-your-backup-anthropic-api-key-here",
      "priority": 2,
      "base_url": "https://api.anthropic.com",
      "timeout": 30,
      "max_retries": 3
    }
  ]
}
```

### Running with Custom Config

```bash
# Use default config file (llm_router_config.json)
poetry run python main.py

# Specify custom config file
poetry run python main.py --config /path/to/config.json

# Create config from example
cp llm_router_config.example.json llm_router_config.json
# Edit llm_router_config.json with your API keys
poetry run python main.py
```

### Programmatic Configuration

You can also configure providers programmatically:

```python
from llm_api_router.server import create_app
from llm_api_router.config import RouterConfig

# Create configuration
config = RouterConfig.from_dict({
    "openai": [
        {
            "api_key": "your-openai-key-1",
            "priority": 1,
        },
        {
            "api_key": "your-openai-key-2",
            "priority": 2,
        }
    ],
    "anthropic": [
        {
            "api_key": "your-anthropic-key",
            "priority": 1,
        }
    ]
})

# Create app with configuration
app = create_app(config)
```

## Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=llm_api_router

# Format code
poetry run black llm_api_router tests

# Lint code
poetry run ruff check llm_api_router tests
```

## License

MIT