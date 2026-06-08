import json

import pytest

from src.core.converters.response_converter import OpenAIToAnthropicConverter


async def _stream_from_lines(lines):
    for line in lines:
        yield line


async def _collect_events(lines):
    raw_events = [
        event
        async for event in OpenAIToAnthropicConverter.convert_openai_stream_to_anthropic_stream(
            _stream_from_lines(lines),
            model="claude-test",
            request_id="test-stream",
        )
    ]
    parsed_events = []
    for raw_event in raw_events:
        assert raw_event.endswith("\n\n")
        event_type = None
        data_parts = []
        for line in raw_event.strip().splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_parts.append(line.removeprefix("data: "))

        assert event_type is not None
        assert data_parts
        data = json.loads("\n".join(data_parts))
        parsed_events.append((event_type, data))

    rendered = "".join(raw_events)
    assert '"choices"' not in rendered
    assert "[DONE]" not in rendered
    return parsed_events


@pytest.mark.asyncio
async def test_text_stream_uses_anthropic_sse_sequence():
    events = await _collect_events(
        [
            'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n',
            "data: [DONE]\n",
        ]
    )

    event_types = [event_type for event_type, _ in events]
    assert event_types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    message_start = events[0][1]
    assert message_start["message"]["content"] == []

    content_start = events[1][1]
    assert content_start["index"] == 0
    assert content_start["content_block"] == {"type": "text", "text": ""}

    content_delta = events[2][1]
    assert content_delta["index"] == 0
    assert content_delta["delta"] == {"type": "text_delta", "text": "Hello"}

    assert events[3][1] == {"type": "content_block_stop", "index": 0}
    assert events[4][1]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_tool_stream_uses_tool_use_block_then_input_json_delta():
    events = await _collect_events(
        [
            'data:{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                '"function":{"name":"run_command","arguments":"{\\"cmd\\""}}]},'
                '"finish_reason":null}]}\n'
            ),
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":":\\"ls\\"}"}}]},"finish_reason":null}]}\n'
            ),
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n',
        ]
    )

    event_types = [event_type for event_type, _ in events]
    assert event_types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    tool_start = events[1][1]
    assert tool_start["index"] == 0
    assert tool_start["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "run_command",
        "input": {},
    }

    first_delta = events[2][1]
    second_delta = events[3][1]
    assert first_delta["index"] == 0
    assert first_delta["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"cmd"',
    }
    assert second_delta["index"] == 0
    assert second_delta["delta"] == {
        "type": "input_json_delta",
        "partial_json": ':"ls"}',
    }

    assert events[4][1] == {"type": "content_block_stop", "index": 0}
    assert events[5][1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_done_without_content_still_closes_a_content_block():
    events = await _collect_events(
        [
            '{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
            "data: [DONE]\n",
        ]
    )

    event_types = [event_type for event_type, _ in events]
    assert event_types == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    assert events[2][1] == {"type": "content_block_stop", "index": 0}
