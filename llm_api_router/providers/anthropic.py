"""Anthropic provider implementation."""

import json
from typing import Any

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def _get_default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for Anthropic API."""
        return {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.config.api_key,
        }

    def _get_endpoint(self) -> str:
        return "/v1/messages"

    def _get_count_tokens_endpoint(self) -> str:
        return "/v1/messages/count_tokens"

    def merge_streaming_chunk(
        self, response_dict: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a streaming chunk from Anthropic into a response dict."""
        # Initialize response dict with metadata from first chunk
        if not response_dict:
            response_dict = {
                "id": data.get("message", {}).get("id", ""),
                "type": "message",
                "role": "assistant",
                "model": data.get("message", {}).get("model", ""),
                "content": [],
            }

        # Merge content from content_block
        if data["type"] == "content_block_start":
            # add new one
            content_idx = data.get("index", 0)
            while len(response_dict["content"]) <= content_idx:
                response_dict["content"].append({})
            existing_content = response_dict["content"][content_idx]
            for k in data["content_block"]:
                existing_content[k] = data["content_block"][k]
        elif data["type"] == "content_block_delta":
            # Update existing content
            content_idx = data.get("index", 0)
            existing_content = response_dict["content"][content_idx]
            if "text" in data["delta"]:
                existing_content["text"] += data["delta"]["text"]
            if "thinking" in data["delta"]:
                existing_content["thinking"] += data["delta"]["thinking"]
            if "partial_json" in data["delta"]:
                if "partial_json" not in existing_content:
                    existing_content["partial_json"] = ""
                existing_content["partial_json"] += data["delta"]["partial_json"]
            if "signature" in data["delta"]:
                existing_content["signature"] += data["delta"]["signature"]
        elif data["type"] == "message_delta":
            if "stop_reason" in data["delta"]:
                response_dict["stop_reason"] = data["delta"]["stop_reason"]
            if "usage" in data:
                response_dict["usage"] = data["usage"]
        return response_dict

    def postprocess_response(self, response_dict: dict[str, Any]) -> dict[str, Any]:
        """Postprocess response dict from Anthropic."""
        # convert partial_json to dict
        if "content" in response_dict:
            for content_idx in range(len(response_dict["content"])):
                existing_content = response_dict["content"][content_idx]
                if "partial_json" in existing_content:
                    try:
                        existing_content["input"] = json.loads(
                            existing_content["partial_json"]
                        )
                    except Exception:
                        existing_content["input"] = {}
                    del existing_content["partial_json"]
        return response_dict

    def extract_tokens_from_response(
        self, response: dict[str, Any]
    ) -> tuple[int, int, int]:
        """Extract token counts from Anthropic response."""
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        return input_tokens, output_tokens, cached_tokens
