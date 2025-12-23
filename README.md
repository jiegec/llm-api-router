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

### Using the CLI (Recommended)

```bash
# Show help
poetry run llm-api-router --help

# Initialize a new configuration file
poetry run llm-api-router init

# Check configuration
poetry run llm-api-router check

# Start the server
poetry run llm-api-router serve

# Use specific configuration file
poetry run llm-api-router serve --config /path/to/config.json
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

### CLI Commands

| Command                 | Description                                   |
|-------------------------|-----------------------------------------------|
| `llm-api-router --help` | Show all available commands                   |
| `llm-api-router init`   | Create a new configuration file with examples |
| `llm-api-router check`  | Validate configuration and show provider info |
| `llm-api-router serve`  | Start the LLM API Router server               |

## Logging

The router provides comprehensive logging of all requests and responses:

- **Structured JSON logs**: All requests, responses, errors, and retries are logged in structured JSON format
- **DateTime in filenames**: Log files include timestamp in filename format: `router_YYYYMMDD_HHMMSS_<session_id>.jsonl`
- **Full request/response logging**: Complete request and response data is logged for debugging and analysis
- **Separate error logs**: Error logs are written to separate files with `_errors.jsonl` suffix
- **Console output**: Summary information is also logged to console for real-time monitoring

Log files are stored in the `logs/` directory by default.

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