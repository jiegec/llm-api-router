"""Base provider interface."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
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
            base_url=config.base_url or self._get_default_base_url(),
        )
        self.logger = get_logger()
        self.provider_name = config.display_name

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for the provider."""
        pass

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

    @abstractmethod
    def _get_count_tokens_endpoint(self) -> str:
        """Get the API endpoint for counting tokens."""
        pass

    async def count_tokens(
        self, request: dict[str, Any], user_agent: str | None = None
    ) -> dict[str, Any]:
        """Count tokens for a request without actually generating a response.

        Args:
            request: Count tokens request
            user_agent: Optional user agent to pass to upstream API

        Returns:
            Count tokens response
        """
        # Extract model from request
        model = request.get("model", "")
        provider_model = self._map_model(model)

        # Build payload - start with original request and update model only
        payload = dict(request)
        payload["model"] = provider_model

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Build headers with user agent override if provided
        headers = dict(self._get_headers())
        if user_agent:
            headers["User-Agent"] = user_agent

        # Log the API call
        self.logger.logger.debug(
            f"Provider {self.provider_name}: "
            f"Counting tokens for model {model} -> {provider_model}, "
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
                response = await self.client.post(
                    self._get_count_tokens_endpoint(), json=payload, headers=headers
                )
                duration = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    response_data = response.json()
                    self.logger.logger.debug(
                        f"Provider {self.provider_name}: "
                        f"Count tokens successful in {duration:.0f}ms, "
                        f"status: {response.status_code}"
                    )
                    return response_data if isinstance(response_data, dict) else {}
                else:
                    try:
                        error_data = response.json()
                    except json.JSONDecodeError:
                        error_data = {"error": {"message": response.text}}

                    self.logger.logger.warning(
                        f"Provider {self.provider_name}: "
                        f"Count tokens failed in {duration:.0f}ms, "
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
                        f"Count tokens failed after {self.config.max_retries + 1} attempts: {str(e)}",
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

    async def chat_completion(
        self, request: dict[str, Any], user_agent: str | None = None
    ) -> dict[str, Any]:
        """Send a chat completion request to the provider.

        Args:
            request: Chat completion request
            user_agent: Optional user agent to pass to upstream API

        Returns:
            Response dict or streaming dict with generator
        """
        # Check if streaming is requested
        stream = request.get("stream", False)

        if stream:
            return await self._chat_completion_stream(request, user_agent)
        else:
            return await self._chat_completion_non_stream(request, user_agent)

    async def _chat_completion_non_stream(
        self, request: dict[str, Any], user_agent: str | None = None
    ) -> dict[str, Any]:
        """Send a non-streaming chat completion request to the provider."""
        # Extract model from request
        model = request.get("model", "")
        provider_model = self._map_model(model)

        # Build payload - start with original request and update model only
        payload = dict(request)
        payload["model"] = provider_model

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Build headers with user agent override if provided
        headers = dict(self._get_headers())
        if user_agent:
            headers["User-Agent"] = user_agent

        # Log API call
        self.logger.logger.debug(
            f"Provider {self.provider_name}: "
            f"Sending non-streaming request for model {model} -> {provider_model}, "
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
                response = await self.client.post(
                    self._get_endpoint(), json=payload, headers=headers
                )
                duration = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    response_data = response.json()
                    self.logger.logger.debug(
                        f"Provider {self.provider_name}: "
                        f"Request successful in {duration:.0f}ms, "
                        f"status: {response.status_code}"
                    )
                    # Just return the raw response
                    return response_data if isinstance(response_data, dict) else {}
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

        # This should never be reached due to raise statements above
        raise ProviderError(
            "Unexpected error: max retries exhausted without raising exception",
            self.config.name.value,
        )

    async def _chat_completion_stream(
        self, request: dict[str, Any], user_agent: str | None = None
    ) -> dict[str, Any]:
        """Send a streaming chat completion request to the provider."""
        # Extract model from request
        model = request.get("model", "")
        provider_model = self._map_model(model)

        # Build payload - start with original request and update model only
        payload = dict(request)
        payload["model"] = provider_model
        payload["stream"] = True  # Ensure streaming is enabled

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Build headers with user agent override if provided
        headers = dict(self._get_headers())
        if user_agent:
            headers["User-Agent"] = user_agent

        # Log API call
        self.logger.logger.debug(
            f"Provider {self.provider_name}: "
            f"Sending streaming request for model {model} -> {provider_model}, "
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

                # We need to verify provider is available before returning
                # a streaming generator. We'll make request and check
                # initial response, then create a generator that streams.

                # Create events for coordination
                check_complete = asyncio.Event()
                check_error = None
                response_obj = None
                first_chunk = None
                chunk_iterator = None

                async def check_streaming() -> None:
                    """Check if streaming works by making request and reading first chunk."""
                    nonlocal check_error, response_obj, first_chunk, chunk_iterator

                    # Make streaming request
                    async with self.client.stream(
                        "POST",
                        self._get_endpoint(),
                        json=payload,
                        headers={
                            **headers,
                            "Accept": "text/event-stream",
                        },
                    ) as response:
                        response_obj = response
                        duration = (time.time() - start_time) * 1000

                        # Check response status
                        if response.status_code != 200:
                            # Read error response
                            try:
                                response_content = await response.aread()
                                try:
                                    error_data = json.loads(response_content)
                                except json.JSONDecodeError:
                                    error_data = {
                                        "error": {
                                            "message": response_content.decode(
                                                "utf-8", errors="ignore"
                                            )
                                        }
                                    }
                            except Exception:
                                error_data = {
                                    "error": {
                                        "message": f"HTTP {response.status_code}: Failed to read response"
                                    }
                                }

                            self.logger.logger.warning(
                                f"Provider {self.provider_name}: "
                                f"Streaming request failed in {duration:.0f}ms, "
                                f"status: {response.status_code}, "
                                f"error: {error_data.get('error', {}).get('message', 'Unknown error')}"
                            )

                            # Signal check complete with error
                            check_error = self._create_error_from_status(
                                response.status_code, error_data
                            )
                            check_complete.set()
                            return

                        # Response is 200 OK
                        self.logger.logger.debug(
                            f"Provider {self.provider_name}: "
                            f"Streaming request successful in {duration:.0f}ms, "
                            f"status: {response.status_code}"
                        )

                        # Get chunk iterator
                        chunk_iterator = response.aiter_bytes()

                        # Try to read the first chunk to verify streaming works
                        try:
                            first_chunk = await chunk_iterator.__anext__()
                        except StopAsyncIteration:
                            # No chunks available yet, but response was successful
                            # This might happen with some providers
                            first_chunk = None
                        except Exception as e:
                            # Error reading first chunk
                            check_error = ProviderError(
                                f"Failed to read first chunk: {str(e)}",
                                self.config.name.value,
                            )
                            check_complete.set()
                            return

                        # Signal that the check is complete and successful
                        check_complete.set()

                        # Keep the response alive by not exiting the context manager
                        # We'll handle cleanup in the generator
                        await asyncio.Event().wait()  # Wait forever

                # Start the check in a background task
                check_task = asyncio.create_task(check_streaming())

                # Wait for the check to complete (or timeout)
                try:
                    await asyncio.wait_for(
                        check_complete.wait(), timeout=self.config.timeout
                    )
                except asyncio.TimeoutError:
                    check_task.cancel()
                    # Try to close the response if it was opened
                    if response_obj:
                        await response_obj.aclose()
                    raise ProviderError(
                        "Streaming request timeout while checking provider availability",
                        self.config.name.value,
                    ) from None

                # If there was an error during checking, raise it
                if check_error:
                    check_task.cancel()
                    # Try to close the response if it was opened
                    if response_obj:
                        await response_obj.aclose()
                    raise check_error

                # Check was successful, now create a generator that streams
                async def streaming_generator() -> AsyncGenerator[bytes | str, None]:
                    """Generator that streams response chunks."""
                    nonlocal first_chunk, chunk_iterator, response_obj

                    try:
                        # First, yield the first chunk we already read
                        if first_chunk is not None:
                            yield first_chunk

                        # Then yield the rest of the chunks from the iterator
                        if chunk_iterator:
                            async for chunk in chunk_iterator:
                                yield chunk
                    finally:
                        # Clean up: cancel the check task and close the response
                        check_task.cancel()
                        if response_obj:
                            await response_obj.aclose()

                # Generator checked successfully, return it
                return {
                    "_streaming": True,
                    "_provider": self,
                    "_provider_type": self.config.name,
                    "_provider_name": self.provider_name,
                    "_generator": streaming_generator(),
                }

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
                        f"Streaming request failed after {self.config.max_retries + 1} attempts: {str(e)}",
                        self.config.name.value,
                    ) from e

                # Exponential backoff
                backoff_time = 2**attempt
                self.logger.logger.debug(
                    f"Provider {self.provider_name}: "
                    f"Retrying after {backoff_time}s backoff"
                )
                await asyncio.sleep(backoff_time)

        # This should never be reached due to raise statements above
        raise ProviderError(
            "Unexpected error: max retries exhausted without raising exception",
            self.config.name.value,
        )

    def _create_error_from_status(
        self, status_code: int, error_data: dict[str, Any]
    ) -> Exception:
        """Create appropriate exception from status code and error data."""
        error_message = error_data.get("error", {}).get("message", "Unknown error")

        if status_code == 401:
            return AuthenticationError(error_message, self.config.name.value)
        elif status_code == 429:
            retry_after = error_data.get("error", {}).get("retry_after")
            return RateLimitError(error_message, self.config.name.value, retry_after)
        elif status_code >= 400:
            error_type = error_data.get("error", {}).get("type", "unknown")
            return ProviderError(
                error_message, self.config.name.value, error_type, error_data
            )
        else:
            return LLMError(
                f"HTTP {status_code}: {error_message}",
                self.config.name.value,
                status_code,
            )

    @abstractmethod
    def merge_streaming_chunk(
        self, response_dict: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a streaming chunk into a response dict.

        Args:
            response_dict: The current accumulated response dict
            data: The new chunk data to merge

        Returns:
            Updated response dict
        """
        pass

    @abstractmethod
    def postprocess_response(self, response_dict: dict[str, Any]) -> dict[str, Any]:
        """Postprocess a response dict after all chunks are merged.

        Args:
            response_dict: The accumulated response dict

        Returns:
            Postprocessed response dict
        """
        pass

    @abstractmethod
    def extract_tokens_from_response(
        self, response: dict[str, Any]
    ) -> tuple[int, int, int]:
        """Extract token counts from a response.

        Args:
            response: The provider response

        Returns:
            Tuple of (input_tokens, output_tokens, cached_tokens)
        """
        pass

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
