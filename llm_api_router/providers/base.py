"""Base provider interface."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from httpx import AsyncClient, Timeout

from ..exceptions import AuthenticationError, LLMError, ProviderError, RateLimitError
from ..logging import get_logger
from ..models import (
    ProviderConfig,
)


class BaseProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncClient(
            timeout=Timeout(timeout=config.timeout),
            headers=self._get_headers(),
            base_url=config.base_url or self._get_default_base_url(),
        )
        self.logger = get_logger()
        self.provider_name = config.display_name

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for the provider."""
        return {}

    @abstractmethod
    def _get_default_base_url(self) -> str:
        """Get the default base URL for the provider."""
        pass

    def _map_model(self, model: str) -> str:
        """Map generic model name to provider-specific model name using config mapping."""
        return self.config.model_mapping.get(model, model)

    @abstractmethod
    def _get_endpoint(self) -> str:
        """Get the API endpoint for chat completions."""
        pass

    def _handle_error(self, status_code: int, error_data: dict[str, Any]) -> None:
        """Handle provider-specific errors."""
        error_message = error_data.get("error", {}).get("message", "Unknown error")

        if status_code == 401:
            raise AuthenticationError(error_message, self.config.name.value)
        elif status_code == 429:
            retry_after = error_data.get("error", {}).get("retry_after")
            raise RateLimitError(error_message, self.config.name.value, retry_after)
        elif status_code >= 400:
            error_type = error_data.get("error", {}).get("type", "unknown")
            raise ProviderError(
                error_message, self.config.name.value, error_type, error_data
            )
        else:
            raise LLMError(
                f"HTTP {status_code}: {error_message}",
                self.config.name.value,
                status_code,
            )

    async def chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request to the provider."""
        # Convert request to dict if it's a Pydantic model
        if hasattr(request, "model_dump"):
            request = request.model_dump(exclude_none=True)

        # Extract model from request
        model = request.get("model", "")
        provider_model = self._map_model(model)

        # Build payload - start with original request and update model only
        payload = dict(request)
        payload["model"] = provider_model

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Log the API call
        self.logger.logger.debug(
            f"Provider {self.provider_name}: "
            f"Sending request for model {model} -> {provider_model}, "
            f"messages: {len(request.get('messages', []))}, "
            f"max_retries: {self.config.max_retries}"
        )

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.logger.debug(
                    f"Provider {self.provider_name}: "
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1}"
                )

                start_time = time.time()
                response = await self.client.post(self._get_endpoint(), json=payload)
                duration = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    response_data = response.json()
                    self.logger.logger.debug(
                        f"Provider {self.provider_name}: "
                        f"Request successful in {duration:.0f}ms, "
                        f"status: {response.status_code}"
                    )
                    # Just return the raw response
                    return response_data
                else:
                    try:
                        error_data = response.json()
                    except json.JSONDecodeError:
                        error_data = {"error": {"message": response.text}}

                    self.logger.logger.warning(
                        f"Provider {self.provider_name}: "
                        f"Request failed in {duration:.0f}ms, "
                        f"status: {response.status_code}, "
                        f"error: {error_data.get('error', {}).get('message', 'Unknown error')}"
                    )

                    self._handle_error(response.status_code, error_data)

            except (RateLimitError, AuthenticationError) as e:
                # Don't retry auth or rate limit errors
                self.logger.logger.warning(
                    f"Provider {self.provider_name}: "
                    f"Fatal error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}"
                )
                raise e
            except Exception as e:
                self.logger.logger.warning(
                    f"Provider {self.provider_name}: "
                    f"Error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}"
                )

                if attempt == self.config.max_retries:
                    self.logger.logger.error(
                        f"Provider {self.provider_name}: "
                        f"Max retries ({self.config.max_retries}) exceeded"
                    )
                    raise ProviderError(
                        f"Request failed after {self.config.max_retries + 1} attempts: {str(e)}",
                        self.config.name.value,
                    ) from e

                # Exponential backoff
                backoff_time = 2**attempt
                self.logger.logger.debug(
                    f"Provider {self.provider_name}: "
                    f"Retrying after {backoff_time}s backoff"
                )
                await asyncio.sleep(backoff_time)

        # This should never be reached due to the raise statements above
        raise ProviderError(
            "Unexpected error: max retries exhausted without raising exception",
            self.config.name.value,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
