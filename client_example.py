#!/usr/bin/env python3
"""Example client for LLM API Router server."""

import asyncio

import httpx


async def test_openai_endpoint():
    """Test OpenAI endpoint."""
    print("Testing OpenAI endpoint...")

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        # Test health endpoint
        response = await client.get("/health")
        print(f"Health check: {response.status_code}")

        # Test OpenAI chat completion
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "model": "gpt-4o-mini",
            "max_tokens": 100,
            "temperature": 0.7,
        }

        try:
            response = await client.post(
                "/openai/chat/completions",
                json=payload,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print("Success! Response from OpenAI endpoint:")
                print(f"  Provider: {result.get('provider')}")
                print(f"  Model: {result.get('model')}")
                print(f"  Content: {result['choices'][0]['message']['content']}")
                print(f"  Usage: {result['usage']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

        except httpx.RequestError as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


async def test_anthropic_endpoint():
    """Test Anthropic endpoint."""
    print("\nTesting Anthropic endpoint...")

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        # Test Anthropic chat completion
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "model": "claude-3-sonnet",
            "max_tokens": 100,
            "temperature": 0.7,
        }

        try:
            response = await client.post(
                "/anthropic/chat/completions",
                json=payload,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print("Success! Response from Anthropic endpoint:")
                print(f"  Provider: {result.get('provider')}")
                print(f"  Model: {result.get('model')}")
                print(f"  Content: {result['choices'][0]['message']['content']}")
                print(f"  Usage: {result['usage']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

        except httpx.RequestError as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


async def test_multiple_providers():
    """Test with multiple provider configurations."""
    print("\nTesting with JSON configuration...")

    # Example of how to configure multiple providers via JSON config
    print("Create a JSON configuration file (llm_router_config.json):")
    print("""
{
  "openai": [
    {
      "api_key": "your-openai-key-1",
      "priority": 1
    },
    {
      "api_key": "your-openai-key-2",
      "priority": 2
    }
  ],
  "anthropic": [
    {
      "api_key": "your-anthropic-key",
      "priority": 1
    }
  ]
}
""")
    print("The server will automatically load this config and provide fallback.")


async def main():
    print("LLM API Router Client Example")
    print("=" * 50)

    # Check if server is running
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=2.0) as client:
            await client.get("/health")
    except httpx.RequestError:
        print("Server is not running at http://127.0.0.1:8000")
        print("Start it with: poetry run python main.py")
        return

    await test_openai_endpoint()
    await test_anthropic_endpoint()
    await test_multiple_providers()


if __name__ == "__main__":
    asyncio.run(main())
