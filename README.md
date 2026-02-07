# LLM API Router

A FastAPI-based LLM API router server that provides separate endpoints for OpenAI and Anthropic APIs, each with automatic fallback when APIs fail or are rate limited.

> **Note**: This project is designed to be simple and easy to deploy for basic use cases. For more complex requirements, consider using [litellm](https://github.com/BerriAI/litellm) or other advanced LLM routing solutions.

## Features

- **Separate endpoints**: `/openai/chat/completions` and `/anthropic/v1/messages`
- **No API format translation**: Each endpoint maintains the native API format. No translation between formats is performed.
- **Streaming support**: Both endpoints support streaming responses. Streaming responses are passed through transparently while collecting usage statistics.
- **Multi-provider support per endpoint**: Each endpoint can route to multiple API keys/servers
- **Priority-based routing**: Configure priority order for API providers within each endpoint
- **Automatic fallback**: When a high-priority provider fails or is rate-limited, automatically switches to the next available provider
- **Error handling**: Comprehensive error detection and HTTP error responses
- **Reasoning model support**: OpenAI reasoning models (o1, o3, etc.) with `reasoning_content` in streaming responses

## Installation

```bash
poetry install
```

## Configuration

### JSON Configuration File

Create a JSON configuration file (e.g. `~/.llm_router_config.json`):

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

The server runs on port **8000** by default.

## API Endpoints

- `GET /` - Server information and available endpoints
- `GET /health` - Health check endpoint
- `GET /status` - Status endpoint (JSON)
- `GET /metrics` - Prometheus metrics endpoint
- `GET /web` - Status dashboard (web UI with auto-refreshing analytics)
- `POST /openai/chat/completions` - OpenAI-compatible chat completion
- `POST /anthropic/v1/messages` - Anthropic-compatible chat completion

### Analytics API

The router provides analytics endpoints for monitoring usage (used by the web dashboard):

- `GET /analytics/requests?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Request count over time
- `GET /analytics/tokens?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Token usage over time
- `GET /analytics/latency?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Latency percentiles over time
- `GET /analytics/summary?hours={1-720}` - Provider summary statistics

Parameters:
- `interval`: Time bucket size (`minute`, `hour`, `day`)
- `hours`: Lookback period in hours (1-720, default 24)
- `provider_type`: Filter by provider type (`openai`, `anthropic`, or omit for all)

## Client Configuration

### Qwen Code (~/.qwen/settings.json)

Configure Qwen Code to use this router:

```json
{
  "security": {
    "auth": {
      "selectedType": "openai",
      "apiKey": "RANDOM",
      "baseUrl": "http://127.0.0.1:8000/openai"
    }
  },
  "model": "example"
}
```

### Claude Code (~/.claude/settings.json)

Configure Claude Code to use this router:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "RANDOM",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000/anthropic"
  },
  "permissions": {
    "allow": [],
    "deny": [],
    "ask": []
  }
}
```


## Web Dashboard

The `/web` endpoint provides a real-time analytics dashboard with:

- **Status tab**: Current provider health and status (auto-refreshes every 5 seconds)
- **Analytics tab**: Time-series charts for requests, tokens, and latency with animated updates
- **Auto-refresh**: Dashboard automatically refreshes data every 5 seconds when viewing either tab

Data is sourced from `logs/request_stats.csv` and aggregated using DuckDB.

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
