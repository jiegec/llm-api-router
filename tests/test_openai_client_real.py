"""Integration test with real OpenAI Python client using TestClient.

This test uses the real OpenAI Python client to make requests to our server
via TestClient, which simulates HTTP requests without starting a real server.
"""

import json
import time
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI

from llm_api_router.config import RouterConfig
from llm_api_router.server import LLMAPIServer


def create_mock_openai_response() -> dict:
    """Create a mock OpenAI API response."""
    return {
        "id": "chatcmpl-real-client-test-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from real OpenAI client!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 12,
            "total_tokens": 27,
        },
    }


def test_real_openai_client_with_testclient():
    """Test that our server works with the real OpenAI Python client via TestClient."""
    # Create test config
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-test-key-real-client",
                    "priority": 1,
                }
            ]
        }
    )

    # Create FastAPI app
    server = LLMAPIServer(config)
    app = server.app

    # Mock the OpenAI provider's HTTP client
    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock successful response
        mock_openai_response = create_mock_openai_response()
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_openai_response),
        )
        mock_client.post.return_value = mock_response

        # Create OpenAI client pointing to our TestClient
        # We need to use a custom transport to route through TestClient
        class TestClientTransport(httpx.BaseTransport):
            def __init__(self, test_client: TestClient):
                self.test_client = test_client

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                # Convert httpx request to TestClient request
                method = request.method
                url = str(request.url)
                headers = dict(request.headers)
                content = request.content

                # Make request through TestClient
                response = self.test_client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=content,
                )

                # Convert TestClient response to httpx response
                return httpx.Response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                )

        with TestClient(app) as test_client:
            # Create OpenAI client with custom transport
            client = OpenAI(
                api_key="any-key-works-since-we-mock",
                base_url="http://testserver/openai",
                http_client=httpx.Client(transport=TestClientTransport(test_client)),
            )

            # Make a request using the real OpenAI client
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Hello from real OpenAI client!"}
                ],
            )

            # Verify response structure matches OpenAI SDK expectations
            assert response.id == "chatcmpl-real-client-test-123"
            assert response.object == "chat.completion"
            assert response.model == "gpt-4o-mini"
            assert len(response.choices) == 1

            choice = response.choices[0]
            assert choice.index == 0
            assert choice.message.role == "assistant"
            assert choice.message.content == "Hello from real OpenAI client!"
            assert choice.finish_reason == "stop"

            usage = response.usage
            assert usage.prompt_tokens == 15
            assert usage.completion_tokens == 12
            assert usage.total_tokens == 27

            assert response.model_dump(exclude_none=True) == mock_openai_response

            stats = server.openai_router.get_stats()
            assert stats.total_requests == 1
            assert stats.total_input_tokens == 15
            assert stats.total_output_tokens == 12
            assert stats.total_cached_tokens == 0
            assert stats.providers["openai-priority-1"].total_requests == 1
            assert stats.providers["openai-priority-1"].in_progress_requests == 0
            assert stats.providers["openai-priority-1"].successful_requests == 1
            assert stats.providers["openai-priority-1"].total_input_tokens == 15
            assert stats.providers["openai-priority-1"].total_output_tokens == 12
            assert stats.providers["openai-priority-1"].total_tokens == 27
            assert stats.providers["openai-priority-1"].total_cached_tokens == 0


def test_real_openai_client_fallback():
    """Test fallback with real OpenAI client."""
    # Create config with multiple providers
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-primary",
                    "priority": 1,
                },
                {
                    "api_key": "sk-backup",
                    "priority": 2,
                },
            ]
        }
    )

    server = LLMAPIServer(config)
    app = server.app

    # Mock the HTTP client with side effects
    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # First call fails with rate limit
        rate_limit_response = httpx.Response(
            status_code=429,
            content=json.dumps(
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                    }
                }
            ),
        )

        # Second call succeeds
        success_response = httpx.Response(
            status_code=200,
            content=json.dumps(create_mock_openai_response()),
        )

        mock_client.post.side_effect = [rate_limit_response, success_response]

        # Create OpenAI client with custom transport
        class TestClientTransport(httpx.BaseTransport):
            def __init__(self, test_client: TestClient):
                self.test_client = test_client

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                method = request.method
                url = str(request.url)
                headers = dict(request.headers)
                content = request.content

                response = self.test_client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=content,
                )

                return httpx.Response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                )

        with TestClient(app) as test_client:
            client = OpenAI(
                api_key="any-key",
                base_url="http://testserver/openai",
                http_client=httpx.Client(transport=TestClientTransport(test_client)),
            )

            # Make a request - should trigger fallback
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test fallback"}],
            )

            # Should still get successful response
            assert response.id == "chatcmpl-real-client-test-123"
            assert (
                response.choices[0].message.content == "Hello from real OpenAI client!"
            )


if __name__ == "__main__":
    test_real_openai_client_with_testclient()
    print("Test 1 passed!")
    test_real_openai_client_fallback()
    print("Test 2 passed!")
    print("All tests passed!")
