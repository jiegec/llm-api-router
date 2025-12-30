"""Test streaming functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_api_router.models import ProviderConfig, ProviderType
from llm_api_router.router import LLMRouter


@pytest.mark.asyncio
async def test_router_streaming_response():
    """Test that router correctly handles streaming responses."""
    # Create a mock provider config
    config = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-key",
        priority=1,
    )

    # Create a streaming request
    streaming_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4",
        "stream": True,
    }

    # Mock the provider
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_provider = AsyncMock()

        # Create a mock streaming response
        mock_response = AsyncMock()
        mock_response.status_code = 200

        # Return a streaming response marker
        mock_provider.chat_completion.return_value = {
            "_streaming": True,
            "_provider": "openai-priority-1",
            "_response": mock_response,
        }

        mock_create.return_value = mock_provider

        # Create router
        router = LLMRouter(providers=[config])

        async with router:
            response = await router.chat_completion(streaming_request)

            # Check that we got a streaming response marker
            assert isinstance(response, dict)
            assert response.get("_streaming") is True
            assert response.get("_provider") == "openai-priority-1"
            assert response.get("_response") == mock_response

            print("✓ Router streaming response test passed")


@pytest.mark.asyncio
async def test_router_streaming_fallback():
    """Test that router fallback works with streaming requests."""
    # Create mock provider configs
    config1 = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-key-1",
        priority=1,
    )

    config2 = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-key-2",
        priority=2,
    )

    # Create a streaming request
    streaming_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4",
        "stream": True,
    }

    # Mock the providers
    with patch("llm_api_router.router.create_provider") as mock_create:
        mock_provider1 = AsyncMock()
        mock_provider2 = AsyncMock()

        # First provider fails
        mock_provider1.chat_completion.side_effect = Exception("Provider 1 failed")

        # Second provider returns streaming response
        mock_response = AsyncMock()
        mock_response.status_code = 200

        mock_provider2.chat_completion.return_value = {
            "_streaming": True,
            "_provider": "openai-priority-2",
            "_response": mock_response,
        }

        mock_create.side_effect = lambda cfg: (
            mock_provider1 if cfg.priority == 1 else mock_provider2
        )

        # Create router
        router = LLMRouter(providers=[config1, config2])

        async with router:
            response = await router.chat_completion(streaming_request)

            # Check that we got a streaming response from the second provider
            assert isinstance(response, dict)
            assert response.get("_streaming") is True
            assert response.get("_provider") == "openai-priority-2"

            # Verify both providers were called
            mock_provider1.chat_completion.assert_called_once_with(streaming_request)
            mock_provider2.chat_completion.assert_called_once_with(streaming_request)

            print("✓ Router streaming fallback test passed")


def test_streaming_request_model():
    """Test that ChatCompletionRequest model has stream field."""
    from tests.test_models_common import ChatCompletionRequest, Message, Role

    # Test with stream=True
    request = ChatCompletionRequest(
        messages=[Message(role=Role.USER, content="Hello")], model="gpt-4", stream=True
    )

    assert request.stream is True

    # Test with stream=False (default)
    request = ChatCompletionRequest(
        messages=[Message(role=Role.USER, content="Hello")],
        model="gpt-4",
    )

    assert request.stream is False

    print("✓ Streaming request model test passed")


if __name__ == "__main__":
    # Run tests
    import asyncio

    async def run_all_tests():
        await test_router_streaming_response()
        await test_router_streaming_fallback()
        test_streaming_request_model()

        print("\n✅ All streaming tests passed!")

    asyncio.run(run_all_tests())
