"""OpenAI provider implementation."""

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def _get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for OpenAI API."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _get_endpoint(self) -> str:
        return "/chat/completions"
