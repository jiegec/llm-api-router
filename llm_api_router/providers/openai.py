"""OpenAI provider implementation."""

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def _get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"


    def _get_endpoint(self) -> str:
        return "/chat/completions"
