#!/usr/bin/env python3
"""Test model mapping functionality."""

import json
from llm_api_router.config import RouterConfig
from llm_api_router.models import ProviderConfig, ProviderType

def test_model_mapping_config():
    """Test that model_mapping is properly loaded from config."""
    print("Testing model mapping configuration...")
    
    # Create a config with model_mapping
    config_dict = {
        "openai": [
            {
                "api_key": "sk-test-key",
                "priority": 1,
                "model_mapping": {
                    "gpt-4": "gpt-4-turbo",
                    "gpt-3.5": "gpt-3.5-turbo"
                }
            }
        ],
        "anthropic": [
            {
                "api_key": "sk-ant-test-key",
                "priority": 1,
                "model_mapping": {
                    "claude-3": "claude-3-5-sonnet",
                    "claude-2": "claude-2.1"
                }
            }
        ]
    }
    
    # Load config
    config = RouterConfig.from_dict(config_dict)
    
    # Check OpenAI provider
    openai_provider = config.openai_providers[0]
    assert openai_provider.model_mapping == {
        "gpt-4": "gpt-4-turbo",
        "gpt-3.5": "gpt-3.5-turbo"
    }
    print(f"✓ OpenAI model_mapping: {openai_provider.model_mapping}")
    
    # Check Anthropic provider
    anthropic_provider = config.anthropic_providers[0]
    assert anthropic_provider.model_mapping == {
        "claude-3": "claude-3-5-sonnet",
        "claude-2": "claude-2.1"
    }
    print(f"✓ Anthropic model_mapping: {anthropic_provider.model_mapping}")
    
    # Test to_dict includes model_mapping
    config_dict_out = config.to_dict()
    assert "model_mapping" in config_dict_out["openai"][0]
    assert "model_mapping" in config_dict_out["anthropic"][0]
    print("✓ model_mapping included in to_dict()")
    
    # Test JSON serialization
    json_str = config.to_json()
    config_dict_from_json = json.loads(json_str)
    assert config_dict_from_json["openai"][0]["model_mapping"] == {
        "gpt-4": "gpt-4-turbo",
        "gpt-3.5": "gpt-3.5-turbo"
    }
    print("✓ model_mapping properly serialized to JSON")
    
    print("\nAll tests passed!")

def test_provider_config_default():
    """Test that model_mapping defaults to empty dict."""
    print("\nTesting default model_mapping...")
    
    provider = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="sk-test",
        priority=1
    )
    
    assert provider.model_mapping == {}
    print(f"✓ Default model_mapping: {provider.model_mapping}")

if __name__ == "__main__":
    test_model_mapping_config()
    test_provider_config_default()