"""Test configuration and fixtures."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.test_models_common import Message, Role
from llm_api_router.models import (
    ProviderConfig,
    ProviderType,
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
    # Convert messages to dict format
    messages_dict = [
        {
            "role": msg.role.value,
            "content": msg.content,
            "name": msg.name,
        }
        for msg in sample_messages
    ]

    return {
        "messages": messages_dict,
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "temperature": 0.7,
    }


@pytest.fixture
def openai_config():
    """OpenAI provider configuration."""
    return ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-openai-key",
        priority=1,
        model_mapping={
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct",
        },
    )


@pytest.fixture
def anthropic_config():
    """Anthropic provider configuration."""
    return ProviderConfig(
        name=ProviderType.ANTHROPIC,
        api_key="test-anthropic-key",
        priority=2,
        model_mapping={
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-2": "claude-2.1",
            "claude-instant": "claude-instant-1.2",
        },
    )


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient."""
    with patch("llm_api_router.providers.base.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
