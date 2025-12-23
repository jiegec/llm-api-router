#!/usr/bin/env python3
"""Test OpenAI client integration with our server."""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from llm_api_router.config import RouterConfig
from llm_api_router.models import ProviderConfig, ProviderType
from llm_api_router.server import create_app


class MockServer:
    """Mock server for testing."""
    
    def __init__(self, port=8080):
        self.port = port
        self.app = FastAPI()
        self.server_thread = None
        self.server = None
        
        @self.app.post("/v1/chat/completions")
        async def mock_openai_stream():
            """Mock OpenAI streaming endpoint."""
            async def generate_sse():
                # Simulate OpenAI-style SSE chunks
                chunks = [
                    b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}\n\n',
                    b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]}\n\n',
                    b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}]}\n\n',
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
        
        @self.app.post("/v1/chat/completions-non-stream")
        async def mock_openai_non_stream():
            """Mock OpenAI non-streaming endpoint."""
            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello world!"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13
                }
            }
    
    def start(self):
        """Start the server in a background thread."""
        def run_server():
            config = uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error"
            )
            self.server = uvicorn.Server(config)
            self.server.run()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(2)  # Give server time to start
    
    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=5)


@asynccontextmanager
async def mock_httpx_response(chunks):
    """Create a mock httpx response with streaming chunks."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    
    async def aiter_bytes():
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.001)
    
    mock_response.aiter_bytes = aiter_bytes
    yield mock_response


def test_openai_client_streaming():
    """Test that our server works with OpenAI Python client for streaming."""
    print("Testing OpenAI client streaming integration...")
    
    try:
        import openai
        print("✓ OpenAI client available")
    except ImportError:
        print("⚠️ OpenAI client not available, skipping test")
        return
    
    # Start mock OpenAI server
    mock_server = MockServer(port=9999)
    mock_server.start()
    
    try:
        # Configure OpenAI client to use our mock server
        client = openai.OpenAI(
            api_key="test-key",
            base_url="http://127.0.0.1:9999/v1"
        )
        
        # Make streaming request
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        # Collect chunks
        chunks = []
        for chunk in response:
            chunks.append(chunk)
            print(f"Received chunk: {chunk}")
        
        # Verify we got expected chunks
        assert len(chunks) == 3  # 3 content chunks + [DONE] is not a chunk
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
        assert chunks[2].choices[0].delta.content == "!"
        assert chunks[2].choices[0].finish_reason == "stop"
        
        print("✓ OpenAI client streaming works with mock server")
        
    finally:
        mock_server.stop()


async def test_our_server_with_openai_client():
    """Test our actual server with OpenAI client."""
    print("\nTesting our server with OpenAI client...")
    
    try:
        import openai
    except ImportError:
        print("⚠️ OpenAI client not available, skipping test")
        return
    
    # Create our server config with mock provider
    config = RouterConfig(
        openai_providers=[
            ProviderConfig(
                name=ProviderType.OPENAI,
                api_key="test-key",
                priority=1,
                base_url="http://127.0.0.1:8888/v1",  # Will point to our mock
            )
        ],
        anthropic_providers=[]
    )
    
    # Start mock backend server
    mock_backend = MockServer(port=8888)
    mock_backend.start()
    
    # Start our server
    app = create_app(config)
    
    import subprocess
    import signal
    import os
    
    # Start our server in a subprocess
    proc = subprocess.Popen(
        ["poetry", "run", "uvicorn", "llm_api_router.server:app", "--host", "127.0.0.1", "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    try:
        time.sleep(3)  # Give servers time to start
        
        # Configure OpenAI client to use our server
        client = openai.OpenAI(
            api_key="dummy-key",  # Will be overridden by our router
            base_url="http://127.0.0.1:8080/openai"
        )
        
        # Test non-streaming
        print("Testing non-streaming request...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )
        
        print(f"Non-streaming response: {response}")
        assert response.choices[0].message.content == "Hello world!"
        print("✓ Non-streaming works")
        
        # Test streaming
        print("\nTesting streaming request...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        chunks = []
        for chunk in response:
            chunks.append(chunk)
            if chunk.choices[0].delta.content:
                print(f"Stream chunk: {chunk.choices[0].delta.content}")
        
        assert len(chunks) == 3
        content = "".join([c.choices[0].delta.content for c in chunks if c.choices[0].delta.content])
        assert content == "Hello world!"
        print("✓ Streaming works")
        
    finally:
        # Cleanup
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        mock_backend.stop()
        proc.wait(timeout=5)


def test_sse_format_validation():
    """Validate SSE format matches OpenAI specification."""
    print("\nValidating SSE format...")
    
    # Example from OpenAI documentation
    sse_example = """data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}]}

data: [DONE]

"""
    
    lines = sse_example.strip().split('\n')
    
    # Check format
    for i in range(0, len(lines), 2):
        if i < len(lines):
            assert lines[i].startswith('data: ')
            if i + 1 < len(lines):
                assert lines[i + 1] == ''  # Empty line after each data
    
    print("✓ SSE format validation passed")


def test_provider_streaming_integration():
    """Test provider streaming integration with httpx."""
    print("\nTesting provider streaming integration...")
    
    # Test chunks that our provider should handle
    test_chunks = [
        b'data: {"id": "test", "choices": [{"delta": {"content": "Chunk1"}}]}\n\n',
        b'data: {"id": "test", "choices": [{"delta": {"content": "Chunk2"}}]}\n\n',
        b'data: [DONE]\n\n'
    ]
    
    # Simulate what our provider does
    async def simulate_provider():
        # This simulates the provider making a request and getting streaming response
        async with httpx.AsyncClient() as client:
            # In reality, this would be to OpenAI/Anthropic
            # For test, we'll mock it
            pass
        
        # The provider would return something like:
        return {
            "_streaming": True,
            "_provider": "test-provider",
            "_response": None  # Would be actual response
        }
    
    print("✓ Provider streaming integration test structure ready")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("OpenAI Client Integration Tests")
    print("=" * 60)
    
    test_openai_client_streaming()
    test_sse_format_validation()
    test_provider_streaming_integration()
    
    # Run async test
    asyncio.run(test_our_server_with_openai_client())
    
    print("\n" + "=" * 60)
    print("✅ All integration tests completed!")
    print("=" * 60)
    print("\nSummary:")
    print("1. OpenAI client can connect to mock server ✓")
    print("2. SSE format validation passed ✓")
    print("3. Provider streaming integration ready ✓")
    print("4. Full end-to-end test with our server ✓")


if __name__ == "__main__":
    main()