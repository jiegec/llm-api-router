"""Test configuration and fixtures."""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, Mock, patch

from llm_api_router.models import (
    ProviderConfig,
    ProviderType,
    Message,
    Role,
    ChatCompletionRequest,
)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well, thank you for asking.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm doing well, thank you for asking.",
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
        },
    }


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="Hello, how are you?"),
    ]


@pytest.fixture
def sample_request(sample_messages):
    """Sample chat completion request."""
    return ChatCompletionRequest(
        messages=sample_messages,
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def openai_config():
    """OpenAI provider configuration."""
    return ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-openai-key",
        priority=1,
    )


@pytest.fixture
def anthropic_config():
    """Anthropic provider configuration."""
    return ProviderConfig(
        name=ProviderType.ANTHROPIC,
        api_key="test-anthropic-key",
        priority=2,
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()