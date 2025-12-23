#!/usr/bin/env python3
"""Example usage of LLM API Router."""

import asyncio
import os
from llm_api_router import LLMRouter, Message, ProviderConfig, ProviderType


async def main():
    # Example 1: Direct configuration
    print("Example 1: Direct configuration")
    router = LLMRouter(
        providers=[
            ProviderConfig(
                name=ProviderType.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY", "dummy-key-1"),
                priority=1,
            ),
            ProviderConfig(
                name=ProviderType.ANTHROPIC,
                api_key=os.getenv("ANTHROPIC_API_KEY", "dummy-key-2"),
                priority=2,
            ),
        ]
    )
    
    async with router:
        try:
            response = await router.chat_completion(
                messages=[
                    Message(role="user", content="Hello, how are you?")
                ],
                model="gpt-4o-mini",
                max_tokens=50,
            )
            print(f"Response from {response.provider}: {response.choices[0].message.content}")
            print(f"Usage: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
        except Exception as e:
            print(f"Error: {e}")
    
    # Example 2: Environment variable configuration
    print("\nExample 2: Environment variable configuration")
    from llm_api_router import create_router_from_env
    
    # Set environment variables for demo
    os.environ["OPENAI_API_KEY"] = "dummy-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "dummy-anthropic-key"
    
    router = create_router_from_env()
    if router:
        async with router:
            try:
                response = await router.chat_completion(
                    messages=[
                        Message(role="system", content="You are a helpful assistant."),
                        Message(role="user", content="What is the capital of France?")
                    ],
                    model="claude-3-sonnet",
                    max_tokens=100,
                )
                print(f"Response from {response.provider}: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("No providers configured via environment variables")


if __name__ == "__main__":
    asyncio.run(main())