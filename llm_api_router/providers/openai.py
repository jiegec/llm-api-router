"""OpenAI provider implementation."""

import time
from typing import Any

from ..models import (
    ChatCompletionResponse,
    ProviderType,
    Usage,
)
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def _get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def _parse_response(self, response: dict[str, Any], model: str) -> dict[str, Any]:
        """Parse OpenAI response - just add provider field."""
        # Create a copy to avoid modifying the original
        result = dict(response)

        # Ensure model field is correct
        result["model"] = model

        # Add provider field for tracking
        result["provider"] = ProviderType.OPENAI.value

        # Extract usage for logging if present
        usage_data = result.get("usage", {})
        # Use the model from the response (already mapped)
        response_model = result.get("model", model)
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            # Return both the raw response and parsed usage for logging
            return {
                "raw_response": result,
                "parsed_for_logging": ChatCompletionResponse(
                    id=result.get("id", ""),
                    created=result.get("created", int(time.time())),
                    model=response_model,
                    choices=[],  # Would need to parse choices properly
                    usage=usage,
                    provider=ProviderType.OPENAI,
                ),
            }
        else:
            # No usage data
            return {
                "raw_response": result,
                "parsed_for_logging": ChatCompletionResponse(
                    id=result.get("id", ""),
                    created=result.get("created", int(time.time())),
                    model=response_model,
                    choices=[],
                    usage=None,
                    provider=ProviderType.OPENAI,
                ),
            }

    def _get_endpoint(self) -> str:
        return "/chat/completions"
