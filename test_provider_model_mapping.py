#!/usr/bin/env python3
"""Test provider model mapping functionality."""

from llm_api_router.models import ProviderConfig, ProviderType
from llm_api_router.providers import create_provider

def test_openai_model_mapping():
    """Test OpenAI provider model mapping."""
    print("Testing OpenAI provider model mapping...")
    
    config = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="sk-test",
        priority=1,
        model_mapping={
            "gpt-4": "gpt-4-turbo",
            "gpt-3.5": "gpt-3.5-turbo"
        }
    )
    
    provider = create_provider(config)
    
    # Test model mapping
    assert provider._map_model("gpt-4") == "gpt-4-turbo"
    assert provider._map_model("gpt-3.5") == "gpt-3.5-turbo"
    assert provider._map_model("unknown-model") == "unknown-model"  # No mapping, returns original
    
    print(f"✓ OpenAI model mapping works correctly")
    print(f"  'gpt-4' -> '{provider._map_model('gpt-4')}'")
    print(f"  'gpt-3.5' -> '{provider._map_model('gpt-3.5')}'")
    print(f"  'unknown-model' -> '{provider._map_model('unknown-model')}'")

def test_anthropic_model_mapping():
    """Test Anthropic provider model mapping."""
    print("\nTesting Anthropic provider model mapping...")
    
    config = ProviderConfig(
        name=ProviderType.ANTHROPIC,
        api_key="sk-ant-test",
        priority=1,
        model_mapping={
            "claude-3": "claude-3-5-sonnet",
            "claude-2": "claude-2.1"
        }
    )
    
    provider = create_provider(config)
    
    # Test model mapping
    assert provider._map_model("claude-3") == "claude-3-5-sonnet"
    assert provider._map_model("claude-2") == "claude-2.1"
    assert provider._map_model("unknown-model") == "unknown-model"  # No mapping, returns original
    
    print(f"✓ Anthropic model mapping works correctly")
    print(f"  'claude-3' -> '{provider._map_model('claude-3')}'")
    print(f"  'claude-2' -> '{provider._map_model('claude-2')}'")
    print(f"  'unknown-model' -> '{provider._map_model('unknown-model')}'")

def test_empty_model_mapping():
    """Test provider with empty model mapping."""
    print("\nTesting empty model mapping...")
    
    config = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="sk-test",
        priority=1,
        model_mapping={}  # Empty mapping
    )
    
    provider = create_provider(config)
    
    # Test that empty mapping returns original model
    assert provider._map_model("gpt-4") == "gpt-4"
    assert provider._map_model("any-model") == "any-model"
    
    print(f"✓ Empty model mapping works correctly")
    print(f"  'gpt-4' -> '{provider._map_model('gpt-4')}' (no mapping)")
    print(f"  'any-model' -> '{provider._map_model('any-model')}' (no mapping)")

if __name__ == "__main__":
    test_openai_model_mapping()
    test_anthropic_model_mapping()
    test_empty_model_mapping()
    print("\nAll provider model mapping tests passed!")