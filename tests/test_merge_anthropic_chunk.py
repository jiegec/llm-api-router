"""Test _merge_anthropic_chunk function."""

from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
)

from llm_api_router.server import _merge_anthropic_chunk, _postprocess_anthropic_chunk


def _anthropic_event_to_dict(event: Message | RawMessageStartEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawMessageDeltaEvent) -> dict:
    """Convert Anthropic event to dict."""
    return event.model_dump(mode="json", exclude_none=True)


def _validate_message_dict(data: dict) -> Message:
    """Validate that a dict conforms to Message schema."""
    return Message.model_construct(**data)


def test_merge_anthropic_chunk_initializes_from_first_chunk():
    """Test that _merge_anthropic_chunk initializes response dict from first chunk."""
    message_start = RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_123",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "Hello"}],
            model="claude-3-5-sonnet-20241022",
            stop_reason=None,
            stop_sequence=None,
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
    )

    data = _anthropic_event_to_dict(message_start)
    result = _merge_anthropic_chunk({}, data)

    assert result["id"] == "msg_123"
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["model"] == "claude-3-5-sonnet-20241022"
    assert len(result["content"]) == 0  # content_block_start will populate this

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.id == "msg_123"
    assert message.model == "claude-3-5-sonnet-20241022"


def test_merge_anthropic_chunk_accumulates_content():
    """Test that _merge_anthropic_chunk accumulates content across chunks."""
    # First content block starts
    block_start = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "text", "text": ""},
    )

    # First delta
    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": "Hello"},
    )

    # Second delta
    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": " world"},
    )

    # Third delta
    delta3 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": "!"},
    )

    # Final message delta with stop reason
    message_delta = RawMessageDeltaEvent(
        type="message_delta",
        delta={"type": "delta", "stop_reason": "end_turn"},
        usage={
            "output_tokens": 12,
        },
    )

    data1 = _anthropic_event_to_dict(block_start)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data3)

    data4 = _anthropic_event_to_dict(delta3)
    result = _merge_anthropic_chunk(result, data4)

    data5 = _anthropic_event_to_dict(message_delta)
    result = _merge_anthropic_chunk(result, data5)

    assert result["content"][0]["text"] == "Hello world!"
    assert result["stop_reason"] == "end_turn"

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.content[0].text == "Hello world!"
    assert message.stop_reason == "end_turn"


def test_merge_anthropic_chunk_multiple_content_blocks():
    """Test that _merge_anthropic_chunk handles multiple content blocks correctly."""
    # First content block starts
    block_start1 = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "text", "text": ""},
    )

    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": "First block"},
    )

    # Second content block starts
    block_start2 = RawContentBlockStartEvent(
        type="content_block_start",
        index=1,
        content_block={"type": "text", "text": ""},
    )

    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=1,
        delta={"type": "text_delta", "text": "Second block"},
    )

    data1 = _anthropic_event_to_dict(block_start1)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(block_start2)
    result = _merge_anthropic_chunk(result, data3)

    data4 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data4)

    assert len(result["content"]) == 2
    assert result["content"][0]["text"] == "First block"
    assert result["content"][1]["text"] == "Second block"

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert len(message.content) == 2
    assert message.content[0].text == "First block"
    assert message.content[1].text == "Second block"


def test_merge_anthropic_chunk_usage():
    """Test that _merge_anthropic_chunk stores usage data."""
    message_delta = RawMessageDeltaEvent(
        type="message_delta",
        delta={"type": "delta", "stop_reason": "end_turn"},
        usage={
            "output_tokens": 15,
        },
    )

    data = _anthropic_event_to_dict(message_delta)
    result = _merge_anthropic_chunk({}, data)

    assert "usage" in result
    assert result["usage"]["output_tokens"] == 15

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.usage.output_tokens == 15


def test_merge_anthropic_chunk_stop_reason():
    """Test that _merge_anthropic_chunk stores stop_reason correctly."""
    message_delta = RawMessageDeltaEvent(
        type="message_delta",
        delta={"type": "delta", "stop_reason": "max_tokens"},
        usage={
            "output_tokens": 5,
        },
    )

    data = _anthropic_event_to_dict(message_delta)
    result = _merge_anthropic_chunk({}, data)

    assert result["stop_reason"] == "max_tokens"

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.stop_reason == "max_tokens"


def test_merge_anthropic_chunk_empty_delta():
    """Test that _merge_anthropic_chunk handles empty content correctly."""
    block_start = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "text", "text": ""},
    )

    delta = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": ""},
    )

    data1 = _anthropic_event_to_dict(block_start)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta)
    result = _merge_anthropic_chunk(result, data2)

    assert result["content"][0]["text"] == ""

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.content[0].text == ""


def test_merge_anthropic_chunk_out_of_order_blocks():
    """Test that _merge_anthropic_chunk handles content blocks arriving out of order."""
    # First block with index 1
    block_start1 = RawContentBlockStartEvent(
        type="content_block_start",
        index=1,
        content_block={"type": "text", "text": ""},
    )

    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=1,
        delta={"type": "text_delta", "text": "Second block"},
    )

    # Second block with index 0
    block_start2 = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "text", "text": ""},
    )

    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "text_delta", "text": "First block"},
    )

    data1 = _anthropic_event_to_dict(block_start1)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(block_start2)
    result = _merge_anthropic_chunk(result, data3)

    data4 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data4)

    assert len(result["content"]) == 2
    assert result["content"][0]["text"] == "First block"
    assert result["content"][1]["text"] == "Second block"

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert len(message.content) == 2
    assert message.content[0].text == "First block"
    assert message.content[1].text == "Second block"


def test_merge_anthropic_chunk_with_cached_tokens():
    """Test that _merge_anthropic_chunk stores cached tokens correctly."""
    message_delta = RawMessageDeltaEvent(
        type="message_delta",
        delta={"type": "delta", "stop_reason": "end_turn"},
        usage={
            "output_tokens": 10,
            "cache_read_input_tokens": 5,
        },
    )

    data = _anthropic_event_to_dict(message_delta)
    result = _merge_anthropic_chunk({}, data)

    assert "usage" in result
    assert result["usage"]["output_tokens"] == 10
    assert result["usage"]["cache_read_input_tokens"] == 5

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.usage.output_tokens == 10
    assert message.usage.cache_read_input_tokens == 5


def test_merge_anthropic_chunk_model_from_message_start():
    """Test that model name is extracted from message_start event."""
    message_start = RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_123",
            type="message",
            role="assistant",
            content=[],
            model="claude-3-opus-20240229",
            stop_reason=None,
            stop_sequence=None,
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
    )

    data = _anthropic_event_to_dict(message_start)
    result = _merge_anthropic_chunk({}, data)

    assert result["model"] == "claude-3-opus-20240229"

    # Validate result conforms to Message schema
    message = _validate_message_dict(result)
    assert message.model == "claude-3-opus-20240229"


def test_merge_anthropic_chunk_thinking_delta():
    """Test that _merge_anthropic_chunk accumulates thinking content."""
    # Start a content block with minimal required fields
    block_start = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "thinking", "thinking": "", "signature": ""},
    )

    # Add thinking deltas
    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "thinking_delta", "thinking": "Let me think"},
    )

    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "thinking_delta", "thinking": " about this..."},
    )

    data1 = _anthropic_event_to_dict(block_start)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data3)

    assert result["content"][0]["thinking"] == "Let me think about this..."


def test_merge_anthropic_chunk_signature_delta():
    """Test that _merge_anthropic_chunk accumulates signature content."""
    # Start a content block with minimal required fields
    block_start = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "thinking", "thinking": "", "signature": ""},
    )

    # Add signature delta
    delta = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "signature_delta", "signature": "abc123"},
    )

    data1 = _anthropic_event_to_dict(block_start)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta)
    result = _merge_anthropic_chunk(result, data2)

    assert result["content"][0]["signature"] == "abc123"


def test_merge_anthropic_chunk_partial_json_delta():
    """Test that _merge_anthropic_chunk accumulates partial_json content."""
    # Start a content block with minimal required fields
    block_start = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "tool_use", "name": "search", "id": "tool_123", "input": {}},
    )

    # Add partial_json deltas
    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": '{"query'},
    )

    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": '":"hello"}'},
    )

    data1 = _anthropic_event_to_dict(block_start)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data3)

    assert result["content"][0]["partial_json"] == '{"query":"hello"}'


def test_postprocess_anthropic_chunk_converts_partial_json_to_dict():
    """Test that _postprocess_anthropic_chunk converts partial_json to input dict."""
    response_dict = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "tool_use",
                "name": "search",
                "id": "tool_123",
                "partial_json": '{"query":"hello","limit":10}',
            }
        ],
    }

    result = _postprocess_anthropic_chunk(response_dict)

    assert "partial_json" not in result["content"][0]
    assert "input" in result["content"][0]
    assert result["content"][0]["input"] == {"query": "hello", "limit": 10}


def test_postprocess_anthropic_chunk_handles_invalid_json():
    """Test that _postprocess_anthropic_chunk handles invalid JSON gracefully."""
    response_dict = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "tool_use",
                "name": "search",
                "id": "tool_123",
                "partial_json": "invalid json {",
            }
        ],
    }

    result = _postprocess_anthropic_chunk(response_dict)

    # Invalid JSON should be handled - partial_json is removed but input is not added
    assert "partial_json" not in result["content"][0]
    assert "input" not in result["content"][0]


def test_postprocess_anthropic_chunk_handles_missing_partial_json():
    """Test that _postprocess_anthropic_chunk handles missing partial_json field."""
    response_dict = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "text",
                "text": "Hello world",
            }
        ],
    }

    result = _postprocess_anthropic_chunk(response_dict)

    # No partial_json means no changes
    assert result["content"][0]["text"] == "Hello world"


def test_merge_and_postprocess_tool_use_streaming():
    """Test end-to-end tool use streaming with partial_json conversion."""
    # Start tool use content block
    block_start = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "tool_use", "name": "calculate", "id": "tool_456", "input": {}},
    )

    # Add partial_json deltas
    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": '{"a":1,'},
    )

    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": '"b":2}'},
    )

    # Merge chunks
    data1 = _anthropic_event_to_dict(block_start)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data3)

    # Before postprocessing, partial_json should be present
    assert result["content"][0]["partial_json"] == '{"a":1,"b":2}'

    # Postprocess to convert partial_json to input
    result = _postprocess_anthropic_chunk(result)

    # After postprocessing, input should be a dict
    assert "partial_json" not in result["content"][0]
    assert result["content"][0]["input"] == {"a": 1, "b": 2}


def test_merge_anthropic_chunk_multiple_tool_use_blocks():
    """Test handling multiple tool_use content blocks with partial_json."""
    # First tool use
    block_start1 = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block={"type": "tool_use", "name": "search", "id": "tool_1", "input": {}},
    )

    delta1 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": '{"q":"test"}'},
    )

    # Second tool use
    block_start2 = RawContentBlockStartEvent(
        type="content_block_start",
        index=1,
        content_block={"type": "tool_use", "name": "fetch", "id": "tool_2", "input": {}},
    )

    delta2 = RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=1,
        delta={"type": "input_json_delta", "partial_json": '{"url":"http://example.com"}'},
    )

    # Merge all chunks
    data1 = _anthropic_event_to_dict(block_start1)
    result = _merge_anthropic_chunk({}, data1)

    data2 = _anthropic_event_to_dict(delta1)
    result = _merge_anthropic_chunk(result, data2)

    data3 = _anthropic_event_to_dict(block_start2)
    result = _merge_anthropic_chunk(result, data3)

    data4 = _anthropic_event_to_dict(delta2)
    result = _merge_anthropic_chunk(result, data4)

    # Postprocess
    result = _postprocess_anthropic_chunk(result)

    # Check both tool calls are processed
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "tool_use"
    assert result["content"][0]["name"] == "search"
    assert result["content"][0]["input"] == {"q": "test"}
    assert result["content"][1]["type"] == "tool_use"
    assert result["content"][1]["name"] == "fetch"
    assert result["content"][1]["input"] == {"url": "http://example.com"}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
