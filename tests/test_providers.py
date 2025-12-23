"""Test providers."""

import json

import pytest
from httpx import Response

from llm_api_router.models import (
    ChatCompletionRequest,
    Message,
    ProviderConfig,
    ProviderType,
    Role,
)
from llm_api_router.providers import AnthropicProvider, OpenAIProvider, create_provider


@pytest.mark.asyncio
async def test_create_provider_openai(openai_config):
    """Test creating OpenAI provider."""
    provider = create_provider(openai_config)
    assert isinstance(provider, OpenAIProvider)
    assert provider.config == openai_config


@pytest.mark.asyncio
async def test_create_provider_anthropic(anthropic_config):
    """Test creating Anthropic provider."""
    provider = create_provider(anthropic_config)
    assert isinstance(provider, AnthropicProvider)
    assert provider.config == anthropic_config


@pytest.mark.asyncio
async def test_create_provider_invalid():
    """Test creating provider with invalid type."""
    # This should fail at the ProviderConfig validation level, not in create_provider
    with pytest.raises(ValueError, match="Input should be 'openai' or 'anthropic'"):
        ProviderConfig(
            name="invalid",  # type: ignore
            api_key="test",
            priority=1,
        )


@pytest.mark.asyncio
async def test_openai_provider_chat_completion(
    openai_config, sample_request, mock_openai_response, mock_httpx_client
):
    """Test OpenAI provider chat completion."""
    # Mock successful response
    mock_response = Response(
        status_code=200,
        content=json.dumps(mock_openai_response),
    )
    mock_httpx_client.post.return_value = mock_response

    provider = OpenAIProvider(openai_config)
    response = await provider.chat_completion(sample_request)

    # Verify request
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == "/chat/completions"

    # Verify response - now returns dict with raw_response and parsed_for_logging
    assert isinstance(response, dict)
    assert "raw_response" in response
    assert "parsed_for_logging" in response

    raw_response = response["raw_response"]
    parsed_response = response["parsed_for_logging"]

    assert raw_response["id"] == "chatcmpl-123"
    assert raw_response["model"] == "gpt-4o-mini"
    assert raw_response["provider"] == ProviderType.OPENAI.value

    # Check parsed response for logging
    assert parsed_response.id == "chatcmpl-123"
    assert parsed_response.model == "gpt-4o-mini"
    assert parsed_response.provider == ProviderType.OPENAI


@pytest.mark.asyncio
async def test_openai_provider_map_model(openai_config):
    """Test OpenAI provider model mapping."""
    provider = OpenAIProvider(openai_config)

    # Test known models
    assert provider._map_model("gpt-4o") == "gpt-4o"
    assert provider._map_model("gpt-4o-mini") == "gpt-4o-mini"
    assert provider._map_model("gpt-3.5-turbo") == "gpt-3.5-turbo"

    # Test unknown model (pass through)
    assert provider._map_model("unknown-model") == "unknown-model"


@pytest.mark.asyncio
async def test_openai_provider_format_messages(openai_config):
    """Test OpenAI provider message formatting."""
    # This test is deprecated since _format_messages method was removed
    # Messages are now passed through unchanged
    pass


@pytest.mark.asyncio
async def test_anthropic_provider_chat_completion(
    anthropic_config, sample_request, mock_anthropic_response, mock_httpx_client
):
    """Test Anthropic provider chat completion."""
    # Mock successful response
    mock_response = Response(
        status_code=200,
        content=json.dumps(mock_anthropic_response),
    )
    mock_httpx_client.post.return_value = mock_response

    provider = AnthropicProvider(anthropic_config)
    response = await provider.chat_completion(sample_request)

    # Verify request
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == "/v1/messages"

    # Verify response - now returns dict with raw_response and parsed_for_logging
    assert isinstance(response, dict)
    assert "raw_response" in response
    assert "parsed_for_logging" in response

    raw_response = response["raw_response"]
    parsed_response = response["parsed_for_logging"]

    assert raw_response["id"] == "msg_123"
    # Model name should be mapped from "gpt-4o-mini" to "claude-3-5-sonnet-20241022"
    assert raw_response["model"] == "claude-3-5-sonnet-20241022"
    assert raw_response["provider"] == ProviderType.ANTHROPIC.value

    # Check parsed response for logging
    assert parsed_response.id == "msg_123"
    assert parsed_response.model == "claude-3-5-sonnet-20241022"
    assert parsed_response.provider == ProviderType.ANTHROPIC


@pytest.mark.asyncio
async def test_anthropic_provider_map_model(anthropic_config):
    """Test Anthropic provider model mapping."""
    provider = AnthropicProvider(anthropic_config)

    # Test known models
    assert provider._map_model("claude-3-opus") == "claude-3-opus-20240229"
    assert provider._map_model("claude-3-sonnet") == "claude-3-5-sonnet-20241022"
    assert provider._map_model("claude-3-haiku") == "claude-3-haiku-20240307"

    # Test unknown model (pass through)
    assert provider._map_model("unknown-model") == "unknown-model"


@pytest.mark.asyncio
async def test_anthropic_provider_format_messages(anthropic_config):
    """Test Anthropic provider message formatting."""
    # This test is deprecated since _format_messages method was removed
    # Messages are now passed through unchanged
    pass


@pytest.mark.asyncio
async def test_anthropic_provider_format_messages_no_system(anthropic_config):
    """Test Anthropic provider message formatting without system message."""
    # This test is deprecated since _format_messages method was removed
    # Messages are now passed through unchanged
    pass


@pytest.mark.asyncio
async def test_provider_authentication_error(openai_config, mock_httpx_client):
    """Test provider authentication error handling."""
    # Mock authentication error
    mock_response = Response(
        status_code=401,
        content=json.dumps(
            {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                }
            }
        ),
    )
    mock_httpx_client.post.return_value = mock_response

    provider = OpenAIProvider(openai_config)
    request = ChatCompletionRequest(
        messages=[Message(role=Role.USER, content="Hello")],
        model="gpt-4o-mini",
    )

    with pytest.raises(Exception) as exc_info:
        await provider.chat_completion(request)

    assert "Authentication failed" in str(exc_info.value)
    assert "openai" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_rate_limit_error(openai_config, mock_httpx_client):
    """Test provider rate limit error handling."""
    # Mock rate limit error
    mock_response = Response(
        status_code=429,
        content=json.dumps(
            {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "retry_after": 60,
                }
            }
        ),
    )
    mock_httpx_client.post.return_value = mock_response

    provider = OpenAIProvider(openai_config)
    request = ChatCompletionRequest(
        messages=[Message(role=Role.USER, content="Hello")],
        model="gpt-4o-mini",
    )

    with pytest.raises(Exception) as exc_info:
        await provider.chat_completion(request)

    assert "Rate limit exceeded" in str(exc_info.value)
    assert "openai" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_retry_logic(openai_config, mock_httpx_client):
    """Test provider retry logic."""
    # Mock first failure, then success
    error_response = Response(
        status_code=500,
        content=json.dumps({"error": {"message": "Internal server error"}}),
    )
    success_response = Response(
        status_code=200,
        content=json.dumps(
            {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ),
    )

    mock_httpx_client.post.side_effect = [error_response, success_response]

    provider = OpenAIProvider(openai_config)
    request = ChatCompletionRequest(
        messages=[Message(role=Role.USER, content="Hello")],
        model="gpt-4o-mini",
    )

    response = await provider.chat_completion(request)

    # Should have retried once
    assert mock_httpx_client.post.call_count == 2
    assert isinstance(response, dict)
    assert "raw_response" in response
    assert response["raw_response"]["id"] == "chatcmpl-123"
