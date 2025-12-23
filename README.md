# LLM API Router

A Python-based LLM API router that supports both OpenAI and Anthropic APIs with automatic fallback when APIs fail or are rate limited.

## Features

- **Multi-provider support**: OpenAI and Anthropic API compatibility
- **Priority-based routing**: Configure priority order for API providers
- **Automatic fallback**: When a high-priority provider fails or is rate-limited, automatically switches to the next available provider
- **Error handling**: Comprehensive error detection and handling
- **Async support**: Built with async/await for high performance
- **Type safety**: Full type hints with Pydantic models

## Installation

```bash
poetry install
```

## Usage

```python
from llm_api_router import LLMRouter, Message, ProviderConfig

# Configure providers
router = LLMRouter(
    providers=[
        ProviderConfig(
            name="openai",
            api_key="your-openai-key",
            priority=1,
            base_url="https://api.openai.com/v1"
        ),
        ProviderConfig(
            name="anthropic",
            api_key="your-anthropic-key", 
            priority=2,
            base_url="https://api.anthropic.com"
        )
    ]
)

# Send a chat completion request
response = await router.chat_completion(
    messages=[
        Message(role="user", content="Hello, how are you?")
    ],
    model="gpt-4o-mini"  # Will be mapped to appropriate provider model
)
```

## Configuration

The router supports configuration via environment variables or direct Python configuration:

```python
# Environment variables
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
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