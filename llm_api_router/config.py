"""Configuration utilities for LLM API Router."""

import os
from typing import List, Optional
from .models import ProviderConfig, ProviderType


def load_providers_from_env() -> List[ProviderConfig]:
    """
    Load provider configurations from environment variables.
    
    Environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - OPENAI_BASE_URL: Optional custom OpenAI base URL
    - ANTHROPIC_BASE_URL: Optional custom Anthropic base URL
    - OPENAI_PRIORITY: Priority for OpenAI (default: 1)
    - ANTHROPIC_PRIORITY: Priority for Anthropic (default: 2)
    
    Returns:
        List of provider configurations
    """
    providers = []
    
    # OpenAI configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        providers.append(ProviderConfig(
            name=ProviderType.OPENAI,
            api_key=openai_key,
            priority=int(os.getenv("OPENAI_PRIORITY", "1")),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ))
    
    # Anthropic configuration
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        providers.append(ProviderConfig(
            name=ProviderType.ANTHROPIC,
            api_key=anthropic_key,
            priority=int(os.getenv("ANTHROPIC_PRIORITY", "2")),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
        ))
    
    return providers


def create_router_from_env() -> Optional["LLMRouter"]:
    """
    Create a router instance from environment variables.
    
    Returns:
        LLMRouter instance if at least one provider is configured, None otherwise
    """
    from .router import LLMRouter
    
    providers = load_providers_from_env()
    if not providers:
        return None
    
    return LLMRouter(providers)