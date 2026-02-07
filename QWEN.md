# LLM API Router

LLM API router with unified OpenAI/Anthropic endpoints, priority-based fallback, and transparent pass-through.

## Endpoints

| Path | Format | Client |
|------|--------|--------|
| `/openai/chat/completions` | OpenAI | `openai` Python client |
| `/anthropic/v1/messages` | Anthropic | `anthropic` Python client |
| `/anthropic/v1/messages/count_tokens` | Anthropic | `anthropic` Python client |
| `/web` | HTML | Browser (analytics dashboard with auto-refresh) |
| `/status` | JSON | Monitoring |
| `/analytics/*` | JSON | Analytics API |
| `/metrics` | Prometheus | Monitoring |

## Architecture

```
llm_api_router/
├── cli.py              # CLI entry point
├── config.py           # Configuration loading
├── exceptions.py       # Custom exceptions
├── logging.py          # JSON logging
├── models.py           # Pydantic models
├── server.py           # FastAPI server
├── providers/          # Provider implementations
│   ├── base.py         # Base provider (ABC)
│   ├── openai.py       # OpenAI provider
│   └── anthropic.py   # Anthropic provider
├── router.py           # Router with fallback
├── stats.py            # Statistics collection
└── analytics.py        # Analytics queries with DuckDB
```

**Stack**: Python 3.10+, Poetry, FastAPI/uvicorn, httpx, openai/anthropic clients, Pydantic, pytest, black, ruff, mypy

## Configuration

```json
{
  "openai": [{
    "api_key": "...",
    "priority": 1,
    "provider_name": "openai-primary",
    "base_url": "https://api.openai.com/v1",
    "timeout": 30,
    "max_retries": 3,
    "model_mapping": {"gpt-4": "gpt-4", "gpt-3.5-turbo": "gpt-3.5-turbo"}
  }],
  "anthropic": [{
    "api_key": "...",
    "priority": 2,
    "provider_name": "anthropic-primary",
    "base_url": "https://api.anthropic.com",
    "timeout": 30,
    "max_retries": 3,
    "model_mapping": {
      "claude-3-opus": "claude-3-opus-20240229",
      "claude-3-haiku": "claude-3-haiku-20240307"
    }
  }]
}
```

## Key Principles

1. **No API format translation** - Each endpoint handles its own format only
2. **Model name mapping** - Transform model names via config
3. **Transparent pass-through** - Don't modify responses except model name
4. **Streaming support** - Stream chunks transparently
5. **Automatic fallback** - Retry with next provider on failure (HTTP error, 429, 401)

## Provider-Specific Methods

BaseProvider abstract methods (implemented by OpenAIProvider/AnthropicProvider):
- `merge_streaming_chunk()` - Merge streaming chunks into response dict
- `postprocess_response()` - Postprocess after all chunks merged
- `extract_tokens_from_response()` - Extract (input, output, cached) tokens

## Development

**Pre-commit checks**:
```bash
poetry run black --check llm_api_router tests
poetry run ruff check llm_api_router tests
poetry run mypy llm_api_router
poetry run pytest tests/
```

**Auto-fix**:
```bash
poetry run black llm_api_router tests
poetry run ruff check --fix llm_api_router tests
```

## Logging

- **Format**: JSON lines
- **Location**: `logs/` directory
- **Per request**: request body, response body, usage stats, response time, provider, retry attempts, errors
- **CSV logging**: `logs/request_stats.csv` with request statistics for analytics

## Analytics

The `/web` endpoint provides an analytics dashboard with auto-refreshing time-series charts powered by DuckDB.

**Analytics API endpoints**:
- `/analytics/requests?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Request count over time
- `/analytics/tokens?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Token usage over time
- `/analytics/latency?interval={minute,hour,day}&hours={1-720}&provider_type={openai,anthropic}` - Latency percentiles over time
- `/analytics/summary?hours={1-720}` - Provider summary statistics

**Data source**: Reads from `logs/request_stats.csv` using DuckDB's `read_csv_auto()`

**Features**:
- Time-series queries with configurable intervals (minute, hour, day)
- Provider filtering by type (OpenAI, Anthropic, or all)
- SQL injection protection via prepared statements and whitelist validation
- Complete time series generation - missing intervals show as zeros (requests/tokens) or NULL (latency)

## Deployment

```bash
llm-api-router --config config.json
```
