"""Base provider interface."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from httpx import AsyncClient, Timeout

from ..models import (
    ProviderConfig, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    Message,
    ProviderType
)
from ..exceptions import (
    RateLimitError, 
    AuthenticationError, 
    ProviderError,
    LLMError
)


class BaseProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncClient(
            timeout=Timeout(timeout=config.timeout),
            headers=self._get_headers(),
            base_url=config.base_url or self._get_default_base_url()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for the provider."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    @abstractmethod
    def _get_default_base_url(self) -> str:
        """Get the default base URL for the provider."""
        pass
    
    @abstractmethod
    def _map_model(self, model: str) -> str:
        """Map generic model name to provider-specific model name."""
        pass
    
    @abstractmethod
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the provider's API."""
        pass
    
    @abstractmethod
    def _parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse provider response to standard format."""
        pass
    
    @abstractmethod
    def _get_endpoint(self) -> str:
        """Get the API endpoint for chat completions."""
        pass
    
    def _handle_error(self, status_code: int, error_data: Dict[str, Any]) -> None:
        """Handle provider-specific errors."""
        error_message = error_data.get("error", {}).get("message", "Unknown error")
        
        if status_code == 401:
            raise AuthenticationError(error_message, self.config.name.value)
        elif status_code == 429:
            retry_after = error_data.get("error", {}).get("retry_after")
            raise RateLimitError(error_message, self.config.name.value, retry_after)
        elif status_code >= 400:
            error_type = error_data.get("error", {}).get("type", "unknown")
            raise ProviderError(error_message, self.config.name.value, error_type, error_data)
        else:
            raise LLMError(f"HTTP {status_code}: {error_message}", self.config.name.value, status_code)
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send a chat completion request to the provider."""
        provider_model = self._map_model(request.model)
        formatted_messages = self._format_messages(request.messages)
        
        payload = {
            "model": provider_model,
            "messages": formatted_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop,
            "stream": request.stream,
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(
                    self._get_endpoint(),
                    json=payload
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return self._parse_response(response_data, request.model)
                else:
                    try:
                        error_data = response.json()
                    except json.JSONDecodeError:
                        error_data = {"error": {"message": response.text}}
                    
                    self._handle_error(response.status_code, error_data)
                    
            except (RateLimitError, AuthenticationError) as e:
                # Don't retry auth or rate limit errors
                raise e
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise ProviderError(
                        f"Request failed after {self.config.max_retries + 1} attempts: {str(e)}",
                        self.config.name.value
                    )
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()