"""Anthropic provider implementation."""

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

    def _get_endpoint(self) -> str:
        return "/v1/messages"
