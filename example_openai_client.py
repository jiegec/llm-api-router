#!/usr/bin/env python3
"""
Example: Using official OpenAI client with LLM API Router.

This demonstrates how to use the official OpenAI Python client
with our router server. The router provides OpenAI-compatible
endpoints that work seamlessly with the OpenAI SDK.

Usage:
  1. Start the server: poetry run python main.py
  2. Run this example: poetry run python example_openai_client.py
"""

import asyncio
import json

# In a real scenario, you would import openai and use it directly:
# import openai
# client = openai.OpenAI(base_url="http://localhost:8000/openai")

# For this example, we'll simulate the OpenAI client behavior
# to show how it would work without requiring actual API keys.


def create_mock_openai_response() -> dict:
    """Create a mock OpenAI API response."""
    return {
        "id": "chatcmpl-example-123",
        "object": "chat.completion",
        "created": 1734902400,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm responding through the LLM API Router. "
                    "This message came from the OpenAI provider with priority-based routing.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
        "provider": "openai",  # Our custom field
    }


async def simulate_openai_client_usage():
    """Simulate how the official OpenAI client would work with our router."""
    print("=" * 70)
    print("OpenAI Client Integration Example")
    print("=" * 70)
    print()

    print("1. Setting up OpenAI client to point to our router:")
    print()
    print("   ```python")
    print("   import openai")
    print()
    print("   # Create client pointing to our router's OpenAI endpoint")
    print("   client = openai.OpenAI(")
    print("       # API key can be anything - router uses configured keys")
    print("       api_key='any-key-will-work',")
    print("       ")
    print("       # Point to our router server")
    print("       base_url='http://localhost:8000/openai',")
    print("       ")
    print("       # Optional: increase timeout if needed")
    print("       timeout=30.0,")
    print("   )")
    print("   ```")
    print()

    print("2. Making a request (works exactly like normal OpenAI API):")
    print()
    print("   ```python")
    print("   response = client.chat.completions.create(")
    print("       model='gpt-4o-mini',  # Will be routed appropriately")
    print("       messages=[")
    print("           {'role': 'system', 'content': 'You are helpful.'},")
    print("           {'role': 'user', 'content': 'Hello, how does this work?'}")
    print("       ],")
    print("       temperature=0.7,")
    print("       max_tokens=100,")
    print("   )")
    print("   ```")
    print()

    print("3. Accessing the response:")
    print()
    print("   ```python")
    print("   # Standard OpenAI fields")
    print("   print(f'Response ID: {response.id}')")
    print("   print(f'Model: {response.model}')")
    print("   print(f'Content: {response.choices[0].message.content}')")
    print("   print(f'Usage: {response.usage}')")
    print("   ")
    print("   # Our custom field shows which provider was used")
    print("   print(f'Provider: {response.provider}')")
    print("   ```")
    print()

    print("4. Simulated response from our router:")
    print()

    # Simulate the response our router would return
    mock_response = create_mock_openai_response()

    print(f"   ID: {mock_response['id']}")
    print(f"   Model: {mock_response['model']}")
    print(f"   Provider: {mock_response['provider']}")
    print(f"   Content: {mock_response['choices'][0]['message']['content']}")
    print(f"   Usage: {json.dumps(mock_response['usage'], indent=4)}")
    print()

    print("5. For Anthropic models:")
    print()
    print("   ```python")
    print("   # Point to Anthropic endpoint")
    print("   client = openai.OpenAI(")
    print("       base_url='http://localhost:8000/anthropic',")
    print("   )")
    print("   ")
    print("   # Use Anthropic model names")
    print("   response = client.chat.completions.create(")
    print("       model='claude-3-5-sonnet-20241022',")
    print("       messages=[")
    print("           {'role': 'user', 'content': 'Hello Claude!'}")
    print("       ]")
    print("   )")
    print("   ```")
    print()

    print("=" * 70)
    print("Key Benefits:")
    print("=" * 70)
    print()
    print("✅ 100% OpenAI SDK compatible")
    print("✅ No code changes needed for existing OpenAI clients")
    print("✅ Automatic provider fallback when APIs fail")
    print("✅ Multiple API keys with priority-based routing")
    print("✅ Works with both OpenAI and Anthropic models")
    print("✅ JSON configuration (no environment variables needed)")
    print("✅ Custom 'provider' field shows which backend was used")
    print()


def demonstrate_fallback_scenario():
    """Demonstrate how fallback works in practice."""
    print("\n" + "=" * 70)
    print("Fallback Scenario Demonstration")
    print("=" * 70)
    print()

    print("Configuration with multiple providers:")
    print()
    print("```json")
    print('{')
    print('  "openai": [')
    print('    {')
    print('      "api_key": "sk-primary-key",')
    print('      "priority": 1,')
    print('      "base_url": "https://api.openai.com/v1"')
    print('    },')
    print('    {')
    print('      "api_key": "sk-backup-key",')
    print('      "priority": 2,')
    print('      "base_url": "https://api.openai.com/v1"')
    print('    }')
    print('  ]')
    print('}')
    print("```")
    print()

    print("What happens when you make a request:")
    print()
    print("1. Client sends request to: http://localhost:8000/openai/chat/completions")
    print("2. Router tries provider with priority 1 (primary)")
    print("3. If primary fails (rate limit, auth error, etc.):")
    print("   - Router automatically tries provider with priority 2 (backup)")
    print("   - Client gets successful response without knowing about the failure")
    print("4. Response includes 'provider' field showing which one was used")
    print()

    print("Example error scenarios handled automatically:")
    print("   • Rate limiting (429 errors)")
    print("   • Authentication errors (401 errors)")
    print("   • Network timeouts")
    print("   • Server errors (5xx)")
    print()


async def main():
    """Run the example."""
    await simulate_openai_client_usage()
    demonstrate_fallback_scenario()

    print("=" * 70)
    print("Ready to Use!")
    print("=" * 70)
    print()
    print("To get started:")
    print("1. Copy llm_router_config.example.json to llm_router_config.json")
    print("2. Add your API keys to the config file")
    print("3. Start the server: poetry run python main.py")
    print("4. Update your OpenAI client to use base_url='http://localhost:8000/openai'")
    print("5. Make requests as usual!")
    print()
    print("For more details, see README.md")


if __name__ == "__main__":
    asyncio.run(main())
