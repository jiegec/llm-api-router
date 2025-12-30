"""Test models."""

import pytest

from llm_api_router.models import (
    ProviderConfig,
    ProviderType,
    Usage,
)
from tests.test_models_common import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Role,
)


def test_message_model():
    """Test Message model."""
    message = Message(role=Role.USER, content="Hello")
    assert message.role == Role.USER
    assert message.content == "Hello"
    assert message.name is None
    assert message.tool_calls is None
    assert message.tool_call_id is None


def test_message_with_optional_fields():
    """Test Message model with optional fields."""
    message = Message(
        role=Role.ASSISTANT,
        content="Hello",
        name="assistant1",
        tool_calls=[
            {"type": "function", "function": {"name": "test", "arguments": "{}"}}
        ],
        tool_call_id="call_123",
    )
    assert message.name == "assistant1"
    assert message.tool_calls == [
        {"type": "function", "function": {"name": "test", "arguments": "{}"}}
    ]
    assert message.tool_call_id == "call_123"


def test_provider_config():
    """Test ProviderConfig model."""
    config = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-key",
        priority=1,
        base_url="https://api.test.com",
        timeout=30,
        max_retries=3,
        model_mapping={"gpt-4": "gpt-4-turbo"},
    )
    assert config.name == ProviderType.OPENAI
    assert config.api_key == "test-key"
    assert config.priority == 1
    assert config.base_url == "https://api.test.com"
    assert config.timeout == 30
    assert config.max_retries == 3
    assert config.model_mapping == {"gpt-4": "gpt-4-turbo"}


def test_provider_config_defaults():
    """Test ProviderConfig model defaults."""
    config = ProviderConfig(
        name=ProviderType.ANTHROPIC,
        api_key="test-key",
        priority=2,
    )
    assert config.name == ProviderType.ANTHROPIC
    assert config.api_key == "test-key"
    assert config.priority == 2
    assert config.base_url is None
    assert config.timeout == 30
    assert config.max_retries == 3


def test_provider_config_priority_validation():
    """Test ProviderConfig priority validation."""
    # Should not raise
    ProviderConfig(name=ProviderType.OPENAI, api_key="test", priority=1)

    # Should raise ValueError for priority < 1
    with pytest.raises(ValueError):
        ProviderConfig(name=ProviderType.OPENAI, api_key="test", priority=0)


def test_chat_completion_request():
    """Test ChatCompletionRequest model."""
    messages = [Message(role=Role.USER, content="Hello")]
    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n"],
        stream=False,
    )
    assert request.messages == messages
    assert request.model == "gpt-4o-mini"
    assert request.temperature == 0.8
    assert request.max_tokens == 100
    assert request.top_p == 0.9
    assert request.frequency_penalty == 0.5
    assert request.presence_penalty == 0.5
    assert request.stop == ["\n"]
    assert request.stream is False


def test_chat_completion_request_defaults():
    """Test ChatCompletionRequest model defaults."""
    messages = [Message(role=Role.USER, content="Hello")]
    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-mini",
    )
    assert request.temperature == 0.7
    assert request.max_tokens == 1000
    assert request.top_p == 1.0
    assert request.frequency_penalty == 0.0
    assert request.presence_penalty == 0.0
    assert request.stop is None
    assert request.stream is False


def test_chat_completion_response():
    """Test ChatCompletionResponse model."""
    message = Message(role=Role.ASSISTANT, content="Hello there")
    choice = Choice(index=0, message=message, finish_reason="stop")
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    response = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1234567890,
        model="gpt-4o-mini",
        choices=[choice],
        usage=usage,
        provider=ProviderType.OPENAI,
    )

    assert response.id == "chatcmpl-123"
    assert response.created == 1234567890
    assert response.model == "gpt-4o-mini"
    assert response.choices == [choice]
    assert response.usage == usage
    assert response.object == "chat.completion"


def test_usage_model():
    """Test Usage model."""
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_choice_model():
    """Test Choice model."""
    message = Message(role=Role.ASSISTANT, content="Hello")
    choice = Choice(index=0, message=message, finish_reason="stop")
    assert choice.index == 0
    assert choice.message == message
    assert choice.finish_reason == "stop"
