# LLM API Router Project

This is an LLM API router that provides a unified interface for multiple OpenAI and Anthropic API backends with automatic fallback.

## Overview

The router acts as a proxy layer that:
- Exposes standard OpenAI and Anthropic API endpoints
- Routes requests to multiple backend providers with priority-based fallback
- Transparently passes through API requests and responses without modification
- Collects usage statistics for monitoring
- Logs all requests, responses, and metrics

## API Endpoints

### OpenAI Compatible Endpoint
- **Path**: `/openai/chat/completions`
- **Format**: Accepts and returns OpenAI API format
- **Client**: Compatible with vanilla `openai` Python client

### Anthropic Compatible Endpoint
- **Path**: `/anthropic/v1/messages`
- **Format**: Accepts and returns Anthropic API format
- **Client**: Compatible with vanilla `anthropic` Python client

## Core Requirements

### Request/Response Handling
1. **No API format translation** - Each endpoint only handles its own format
2. **Model name mapping** - Transform model names based on JSON config mapping
3. **Transparent pass-through** - Do not modify request/response JSON except for:
   - Model name translation
   - Usage statistics extraction (for logging only)
4. **Streaming support** - Both endpoints support `stream: true` responses with transparent chunk pass-through

### Backend Fallback
- Each endpoint supports multiple API backend providers
- Providers have a configured priority order
- Automatic fallback when:
  - Request fails (HTTP error, timeout)
  - Rate limit encountered (HTTP 429)
  - Authentication error (HTTP 401)

### Statistics Collection
- Extract and log usage statistics:
  - OpenAI: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`
  - Anthropic: `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`
- Track retry attempts and failures
- Measure response times per provider

## Architecture

### Project Structure
```
llm_api_router/
├── __init__.py
├── cli.py           # CLI entry point
├── config.py        # Configuration loading and validation
├── logging.py       # JSON logging to files
├── models.py        # Pydantic models for data structures
├── server.py        # FastAPI server and endpoints
├── providers.py     # OpenAI/Anthropic provider implementations
└── router.py        # Router with fallback logic

tests/               # Test suite
.github/workflows/   # GitHub CI configuration
```

### Technology Stack
- **Language**: Python 3.10+
- **Dependency Management**: Poetry
- **Web Framework**: FastAPI with uvicorn
- **HTTP Clients**: httpx
- **API Clients**: openai, anthropic
- **Validation**: Pydantic

## Configuration

Model mappings and provider settings are configured via JSON config:
```json
{
  "model_mapping": {
    "gpt-4": ["provider1-model-name", "provider2-model-name"],
    "claude-3-opus": ["provider1-model-name"]
  },
  "providers": [
    {
      "name": "openai-primary",
      "type": "openai",
      "api_key": "...",
      "base_url": "...",
      "priority": 1
    },
    {
      "name": "anthropic-fallback",
      "type": "anthropic",
      "api_key": "...",
      "base_url": "...",
      "priority": 2
    }
  ]
}
```

## Development Workflow

1. **Implement features iteratively** - Commit after each significant progress
2. **Write tests** - Use pytest with fixtures and clear test organization
3. **Run CI** - GitHub Actions for automated testing
4. **Type checking** - Use mypy with strict mode
5. **Linting** - Use ruff for code quality
6. **Cleanup** - Remove unused code and maintain clean codebase

## Logging Requirements

- **Format**: JSON lines
- **Location**: Files in `logs/` directory
- **Content per request**:
  - Full request body
  - Full response body
  - Usage statistics (tokens, cached tokens)
  - Response time
  - Provider used
  - Retry attempts
  - Error details (if any)

## Key Constraints

1. **DO NOT translate between API formats** - `/openai/chat/completions` only handles OpenAI format, `/anthropic/v1/messages` only handles Anthropic format
2. **Minimal request modification** - Only change model names based on config
3. **No response body modification** - Extract statistics but don't alter response
4. **Pass-through streaming** - Stream chunks transparently, only collect metrics
5. **Automatic fallback** - Retry with next provider without client intervention

## Testing

- Unit tests for individual components (config, logging, providers)
- Integration tests for router with mock backends
- Streaming tests for both endpoints
- Fallback logic tests
- API compatibility tests with vanilla clients

## Deployment

- Run via CLI: `llm-api-router --config config.json`
- Server runs on configurable host/port
- Logs written to rotating files
- Graceful shutdown support
