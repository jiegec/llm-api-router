"""Test router."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from httpx import Response

from llm_api_router import LLMRouter, ProviderConfig, ProviderType
from llm_api_router.models import Message, Role, ChatCompletionRequest
from llm_api_router.exceptions import NoAvailableProviderError


@pytest.mark.asyncio
async def test_router_initialization(openai_config, anthropic_config):
    """Test router initialization."""
    router = LLMRouter(providers=[openai_config, anthropic_config])
    assert len(router.providers) == 2
    assert router.providers[0].priority == 1  # OpenAI
    assert router.providers[1].priority == 2  # Anthropic


@pytest.mark.asyncio
async def test_router_initialization_no_providers():
    """Test router initialization with no providers."""
    with pytest.raises(ValueError, match="At least one provider must be configured"):
        LLMRouter(providers=[])


@pytest.mark.asyncio
async def test_router_priority_sorting():
    """Test router sorts providers by priority."""
    configs = [
        ProviderConfig(name=ProviderType.OPENAI, api_key="key1", priority=3),
        ProviderConfig(name=ProviderType.ANTHROPIC, api_key="key2", priority=1),
        ProviderConfig(name=ProviderType.OPENAI, api_key="key3", priority=2),
    ]
    
    router = LLMRouter(providers=configs)
    assert router.providers[0].priority == 1  # Anthropic
    assert router.providers[1].priority == 2  # OpenAI
    assert router.providers[2].priority == 3  # OpenAI


@pytest.mark.asyncio
async def test_router_chat_completion_success(
    openai_config, anthropic_config, sample_request
):
    """Test router successful chat completion."""
    # Mock providers
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_openai_provider = AsyncMock()
        mock_openai_provider.chat_completion.return_value = AsyncMock(
            provider=ProviderType.OPENAI
        )
        
        mock_create.side_effect = lambda config: (
            mock_openai_provider if config.name == ProviderType.OPENAI
            else AsyncMock()
        )
        
        router = LLMRouter(providers=[openai_config, anthropic_config])
        async with router:
            response = await router.chat_completion(sample_request)
        
        # Should use OpenAI (higher priority)
        mock_openai_provider.chat_completion.assert_called_once_with(sample_request)
        assert response.provider == ProviderType.OPENAI


@pytest.mark.asyncio
async def test_router_fallback_on_failure(
    openai_config, anthropic_config, sample_request
):
    """Test router fallback when first provider fails."""
    # Mock providers
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_openai_provider = AsyncMock()
        mock_anthropic_provider = AsyncMock()
        
        # OpenAI fails with a generic error
        mock_openai_provider.chat_completion.side_effect = Exception("OpenAI error")
        mock_anthropic_provider.chat_completion.return_value = AsyncMock(
            provider=ProviderType.ANTHROPIC
        )
        
        mock_create.side_effect = lambda config: (
            mock_openai_provider if config.name == ProviderType.OPENAI
            else mock_anthropic_provider
        )
        
        router = LLMRouter(providers=[openai_config, anthropic_config])
        async with router:
            response = await router.chat_completion(sample_request)
        
        # Should try OpenAI first, then fall back to Anthropic
        mock_openai_provider.chat_completion.assert_called_once_with(sample_request)
        mock_anthropic_provider.chat_completion.assert_called_once_with(sample_request)
        assert response.provider == ProviderType.ANTHROPIC


@pytest.mark.asyncio
async def test_router_fallback_on_rate_limit(
    openai_config, anthropic_config, sample_request
):
    """Test router fallback when first provider is rate limited."""
    from llm_api_router.exceptions import RateLimitError
    
    # Mock providers
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_openai_provider = AsyncMock()
        mock_anthropic_provider = AsyncMock()
        
        # OpenAI rate limited
        mock_openai_provider.chat_completion.side_effect = RateLimitError(
            "Rate limited", "openai", 60
        )
        mock_anthropic_provider.chat_completion.return_value = AsyncMock(
            provider=ProviderType.ANTHROPIC
        )
        
        mock_create.side_effect = lambda config: (
            mock_openai_provider if config.name == ProviderType.OPENAI
            else mock_anthropic_provider
        )
        
        router = LLMRouter(providers=[openai_config, anthropic_config])
        async with router:
            response = await router.chat_completion(sample_request)
        
        # Should try OpenAI first, then fall back to Anthropic
        mock_openai_provider.chat_completion.assert_called_once_with(sample_request)
        mock_anthropic_provider.chat_completion.assert_called_once_with(sample_request)
        assert response.provider == ProviderType.ANTHROPIC


@pytest.mark.asyncio
async def test_router_fallback_on_authentication_error(
    openai_config, anthropic_config, sample_request
):
    """Test router fallback when first provider has authentication error."""
    from llm_api_router.exceptions import AuthenticationError
    
    # Mock providers
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_openai_provider = AsyncMock()
        mock_anthropic_provider = AsyncMock()
        
        # OpenAI authentication error
        mock_openai_provider.chat_completion.side_effect = AuthenticationError(
            "Invalid API key", "openai"
        )
        mock_anthropic_provider.chat_completion.return_value = AsyncMock(
            provider=ProviderType.ANTHROPIC
        )
        
        mock_create.side_effect = lambda config: (
            mock_openai_provider if config.name == ProviderType.OPENAI
            else mock_anthropic_provider
        )
        
        router = LLMRouter(providers=[openai_config, anthropic_config])
        async with router:
            response = await router.chat_completion(sample_request)
        
        # Should try OpenAI first, then fall back to Anthropic
        mock_openai_provider.chat_completion.assert_called_once_with(sample_request)
        mock_anthropic_provider.chat_completion.assert_called_once_with(sample_request)
        assert response.provider == ProviderType.ANTHROPIC


@pytest.mark.asyncio
async def test_router_all_providers_fail(
    openai_config, anthropic_config, sample_request
):
    """Test router when all providers fail."""
    # Mock providers
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_openai_provider = AsyncMock()
        mock_anthropic_provider = AsyncMock()
        
        # Both providers fail
        mock_openai_provider.chat_completion.side_effect = Exception("OpenAI error")
        mock_anthropic_provider.chat_completion.side_effect = Exception("Anthropic error")
        
        mock_create.side_effect = lambda config: (
            mock_openai_provider if config.name == ProviderType.OPENAI
            else mock_anthropic_provider
        )
        
        router = LLMRouter(providers=[openai_config, anthropic_config])
        async with router:
            with pytest.raises(NoAvailableProviderError) as exc_info:
                await router.chat_completion(sample_request)
        
        # Should include error details from both providers
        error_message = str(exc_info.value)
        assert "All providers failed" in error_message
        assert "openai" in error_message.lower()
        assert "anthropic" in error_message.lower()


@pytest.mark.asyncio
async def test_router_context_manager(openai_config):
    """Test router as async context manager."""
    router = LLMRouter(providers=[openai_config])
    async with router as r:
        assert r is router
        # Should be able to access providers
        assert len(r.providers) == 1
    
    # Router should be closed after context exit
    # (We can't easily test this without mocking)


@pytest.mark.asyncio
async def test_router_get_available_providers(openai_config, anthropic_config):
    """Test router get_available_providers method."""
    router = LLMRouter(providers=[openai_config, anthropic_config])
    providers = router.get_available_providers()
    assert set(providers) == {"openai", "anthropic"}


@pytest.mark.asyncio
async def test_router_get_provider_priority(openai_config, anthropic_config):
    """Test router get_provider_priority method."""
    router = LLMRouter(providers=[openai_config, anthropic_config])
    assert router.get_provider_priority("openai") == 1
    assert router.get_provider_priority("anthropic") == 2
    assert router.get_provider_priority("unknown") is None


@pytest.mark.asyncio
async def test_router_schedule_retry(openai_config, sample_request):
    """Test router schedules retry for rate limited providers."""
    from llm_api_router.exceptions import RateLimitError
    
    # Mock provider
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_provider = AsyncMock()
        mock_provider.chat_completion.side_effect = RateLimitError(
            "Rate limited", "openai", 0.1  # Short delay for test
        )
        mock_create.return_value = mock_provider
        
        router = LLMRouter(providers=[openai_config])
        
        # First call should fail with rate limit
        with pytest.raises(NoAvailableProviderError):
            async with router:
                await router.chat_completion(sample_request)
        
        # Wait for retry delay
        await asyncio.sleep(0.2)
        
        # Provider should have been called once
        mock_provider.chat_completion.assert_called_once_with(sample_request)