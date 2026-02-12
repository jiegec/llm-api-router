# LLM API Router

A FastAPI-based LLM API router server that provides separate endpoints for OpenAI and Anthropic APIs, each with automatic fallback when APIs fail or are rate limited.

> **Note**: This project is designed to be simple and easy to deploy for basic use cases. For more complex requirements, consider using [litellm](https://github.com/BerriAI/litellm) or other advanced LLM routing solutions.

## Features

- **Separate endpoints**: `/openai/chat/completions` and `/anthropic/v1/messages` (also `/anthropic/v1/messages/count_tokens`)
- **No API format translation**: Each endpoint maintains the native API format. No translation between formats is performed.
- **Streaming support**: Both endpoints support streaming responses. Streaming responses are passed through transparently while collecting usage statistics.
- **Multi-provider support per endpoint**: Each endpoint can route to multiple API keys/servers.
- **Priority-based routing**: Configure priority order for API providers within each endpoint.
- **Automatic fallback**: When a high-priority provider fails or is rate-limited, automatically switches to the next available provider.
- **Error handling**: Comprehensive error detection and HTTP error responses.

## Installation

```bash
git clone https://github.com/jiegec/llm-api-router
cd llm-api-router
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
      "provider_name": "openai-1",
      "timeout": 30,
      "max_retries": 3,
      "model_mapping": {
        "abc": "def"
      }
    },
    {
      "api_key": "sk-your-backup-openai-api-key-here",
      "priority": 2,
      "base_url": "https://api.openai.com/v1",
      "provider_name": "openai-2",
      "timeout": 30,
      "max_retries": 3
    }
  ],
  "anthropic": [
    {
      "api_key": "sk-ant-your-primary-anthropic-api-key-here",
      "priority": 1,
      "base_url": "https://api.anthropic.com",
      "provider_name": "anthropic-1",
      "timeout": 30,
      "max_retries": 3
    },
    {
      "api_key": "sk-ant-your-backup-anthropic-api-key-here",
      "priority": 2,
      "base_url": "https://api.anthropic.com",
      "provider_name": "anthropic-2",
      "timeout": 30,
      "max_retries": 3
    }
  ]
}
```

## Running the Server

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

The server listens on **127.0.0.1:8000** by default. You can change it via `--host` and `--port` command line arguments.

## API Endpoints

- `GET /` - Server information and available endpoints
- `GET /health` - Health check endpoint
- `GET /status` - Status endpoint (JSON)
- `GET /metrics` - Prometheus metrics endpoint
- `GET /web` - Status dashboard (web UI with auto-refreshing analytics)
- `GET /analytics/*` - Analytics API for dashboard, see below
- `POST /openai/chat/completions` - OpenAI-compatible chat completion
- `POST /anthropic/v1/messages` - Anthropic-compatible chat completion
- `POST /anthropic/v1/count_tokens` - Anthropic-compatible count tokens

### Analytics API

The router provides analytics endpoints for monitoring usage (used by the web dashboard):

- `GET /analytics/requests?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Request count over time
- `GET /analytics/tokens?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Token usage over time
- `GET /analytics/summary?hours={1-720}` - Provider summary statistics

Parameters:

- `interval`: Time bucket size (`minute`, `hour`, `day`)
- `hours`: Lookback period in hours (1-720, default 24)
- `provider_type`: Filter by provider type (`openai`, `anthropic`, or omit for all)

## Client Configuration

### Qwen Code (~/.qwen/settings.json)

Configure Qwen Code to use this router:

Via OpenAI endpoint:

```json
{
  "env": {
    "OPENAI_API_KEY": "RANDOM"
  },
  "modelProviders": {
    "openai": [
      {
        "id": "model-name-here",
        "name": "Some Model From Some Vendor",
        "envKey": "OPENAI_API_KEY",
        "baseUrl": "http://127.0.0.1:8000/openai"
      }
    ]
  },
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  },
  "model": "model-name-here"
}
```

Via Anthropic endpoint:

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "RANDOM"
  },
  "modelProviders": {
    "anthropic": [
      {
        "id": "model-name-here",
        "name": "Some Model From Some Vendor",
        "envKey": "ANTHROPIC_API_KEY",
        "baseUrl": "http://127.0.0.1:8000/anthropic"
      }
    ]
  },
  "security": {
    "auth": {
      "selectedType": "anthropic"
    }
  },
  "model": "model-name-here"
}
```

### Claude Code (~/.claude/settings.json)

Configure Claude Code to use this router:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "RANDOM",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000/anthropic"
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

# Format code
poetry run black llm_api_router tests

# Lint code
poetry run ruff check llm_api_router tests
```

## License

MIT
