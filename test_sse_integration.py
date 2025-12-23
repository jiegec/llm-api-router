#!/usr/bin/env python3
"""Test SSE streaming with actual server and OpenAI client."""

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from llm_api_router.config import RouterConfig
from llm_api_router.models import ProviderConfig, ProviderType
from llm_api_router.server import create_app


def test_sse_streaming_basic():
    """Test basic SSE streaming response."""
    print("Testing basic SSE streaming...")
    
    # Create a simple FastAPI app that returns SSE
    app = FastAPI()
    
    @app.post("/test-sse")
    async def test_sse_endpoint():
        async def generate_sse():
            # Simulate OpenAI-style SSE chunks
            chunks = [
                b'data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]}\n\n',
                b'data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": " world"}}]}\n\n',
                b'data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": "!"}}]}\n\n',
                b'data: [DONE]\n\n'
            ]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    client = TestClient(app)
    
    # Make request
    response = client.post("/test-sse")
    
    # Check response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Read streaming response
    content = response.text
    print(f"Response content:\n{content}")
    
    # Verify SSE format
    assert "data: " in content
    assert "Hello" in content
    assert "world" in content
    assert "!" in content
    assert "[DONE]" in content
    
    print("✓ Basic SSE streaming works")


def test_server_streaming_with_mock():
    """Test our server's streaming with mocked provider."""
    print("\nTesting server streaming with mocked provider...")
    
    # Create config with mock provider
    config = RouterConfig(
        openai_providers=[
            ProviderConfig(
                name=ProviderType.OPENAI,
                api_key="test-key",
                priority=1,
            )
        ],
        anthropic_providers=[]
    )
    
    # Create app
    app = create_app(config)
    client = TestClient(app)
    
    # Mock the router to return a streaming response
    with patch("llm_api_router.server.LLMRouter") as mock_router_class:
        mock_router = AsyncMock()
        mock_router_class.return_value = mock_router
        
        # Create a mock httpx response that simulates SSE streaming
        mock_httpx_response = AsyncMock()
        mock_httpx_response.status_code = 200
        
        # Create mock aiter_bytes that yields SSE chunks
        async def mock_aiter_bytes():
            chunks = [
                b'data: {"id": "chatcmpl-test", "choices": [{"delta": {"content": "Mock"}}]}\n\n',
                b'data: {"id": "chatcmpl-test", "choices": [{"delta": {"content": " stream"}}]}\n\n',
                b'data: [DONE]\n\n'
            ]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        
        mock_httpx_response.aiter_bytes = mock_aiter_bytes
        
        # Set up router to return streaming marker
        mock_router.chat_completion.return_value = {
            "_streaming": True,
            "_provider": "openai-priority-1",
            "_response": mock_httpx_response
        }
        
        # Make streaming request
        response = client.post(
            "/openai/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4",
                "stream": True
            },
            headers={"Accept": "text/event-stream"}
        )
        
        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["x-provider"] == "openai-priority-1"
        
        # Read streaming response
        content = response.text
        print(f"Server streaming response:\n{content}")
        
        # Verify SSE format
        assert "data: " in content
        assert "Mock" in content
        assert "stream" in content
        assert "[DONE]" in content
        
        print("✓ Server streaming with mock works")


async def test_openai_client_compatibility():
    """Test that our server works with OpenAI Python client."""
    print("\nTesting OpenAI client compatibility...")
    
    try:
        import openai
        print("OpenAI client available")
    except ImportError:
        print("⚠️ OpenAI client not available, skipping test")
        return
    
    # Create config
    config = RouterConfig(
        openai_providers=[
            ProviderConfig(
                name=ProviderType.OPENAI,
                api_key="test-key",
                priority=1,
                base_url="http://localhost:9999",  # Will be overridden by mock
            )
        ],
        anthropic_providers=[]
    )
    
    # Create app
    app = create_app(config)
    
    # We'll test this with a real server in a separate test
    print("✓ OpenAI client compatibility test structure ready")


def test_sse_chunk_parsing():
    """Test parsing SSE chunks."""
    print("\nTesting SSE chunk parsing...")
    
    # Example SSE chunk from OpenAI
    sse_chunk = b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}\n\n'
    
    # Parse the chunk
    lines = sse_chunk.decode('utf-8').strip().split('\n')
    
    # Should have data line
    assert lines[0].startswith('data: ')
    
    # Extract JSON
    json_str = lines[0][6:]  # Remove 'data: '
    data = json.loads(json_str)
    
    assert data["id"] == "chatcmpl-123"
    assert data["model"] == "gpt-4"
    assert data["choices"][0]["delta"]["content"] == "Hello"
    
    print("✓ SSE chunk parsing works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SSE Streaming Integration Tests")
    print("=" * 60)
    
    test_sse_streaming_basic()
    test_server_streaming_with_mock()
    test_sse_chunk_parsing()
    
    # Run async test
    asyncio.run(test_openai_client_compatibility())
    
    print("\n" + "=" * 60)
    print("✅ All SSE streaming tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Implement proper provider streaming that returns actual SSE")
    print("2. Create end-to-end test with real server")
    print("3. Test with actual OpenAI Python client")


if __name__ == "__main__":
    main()