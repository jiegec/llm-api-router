"""Anthropic provider implementation."""

import time
from typing import Any

from ..models import (
    ChatCompletionResponse,
    ProviderType,
    Usage,
)
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def _get_default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for Anthropic API."""
        headers = super()._get_headers()
        headers.update(
            {
                "anthropic-version": "2023-06-01",
                "x-api-key": self.config.api_key,
            }
        )
        # Remove Bearer prefix for Anthropic
        headers.pop("Authorization", None)
        return headers

    def _parse_response(self, response: dict[str, Any], model: str) -> dict[str, Any]:
        """Parse Anthropic response - just add provider field."""
        # Create a copy to avoid modifying the original
        result = dict(response)

        # Add provider field for tracking
        result["provider"] = ProviderType.ANTHROPIC.value

        # Extract usage for logging if present
        usage_data = result.get("usage", {})
        # Use the model from the response (already mapped)
        response_model = result.get("model", model)
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0),
            )
            return {
                "raw_response": result,
                "parsed_for_logging": ChatCompletionResponse(
                    id=result.get("id", ""),
                    created=int(time.time()),
                    model=response_model,
                    choices=[],  # Would need to parse choices properly
                    usage=usage,
                    provider=ProviderType.ANTHROPIC,
                ),
            }
        else:
            return {
                "raw_response": result,
                "parsed_for_logging": ChatCompletionResponse(
                    id=result.get("id", ""),
                    created=int(time.time()),
                    model=response_model,
                    choices=[],
                    usage=None,
                    provider=ProviderType.ANTHROPIC,
                ),
            }

    def _get_endpoint(self) -> str:
        return "/v1/messages"

    async def chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request to Anthropic."""
        # Use the base implementation which only maps the model name
        return await super().chat_completion(request)
