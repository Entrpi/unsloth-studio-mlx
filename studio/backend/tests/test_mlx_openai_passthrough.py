# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""SSE-frame-shape tests for the MLX OpenAI tool-calling pass-through.

Runs against a mocked backend so no model is needed. The mocked backend
emits canned cumulative-text events plus a metadata event, exactly the
shape ``MlxLmBackend.generate_chat_completion_with_tools`` yields.

Each test drives the async generator to completion with
``asyncio.run`` and collects the SSE frames, then asserts the shapes
match OpenAI's streaming wire format:

- first frame carries ``delta.role = "assistant"``.
- content deltas carry ``delta.content``.
- tool-call deltas carry ``delta.tool_calls[i]`` with
  ``index`` / ``id`` / ``function`` (name on first, arguments on second).
- final frame carries ``finish_reason`` = ``"stop"`` or ``"tool_calls"``.
- stream terminates with ``data: [DONE]``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest import mock

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest  # noqa: E402

from models.inference import ChatCompletionRequest, ChatMessage  # noqa: E402
from routes.inference import (  # noqa: E402
    _mlx_agentic_stream,
    _mlx_openai_passthrough_non_streaming,
    _mlx_openai_passthrough_stream,
)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


def _parse_sse(frames: list[str]) -> list[dict | str]:
    """Parse a list of SSE ``data: ...`` frames into objects/strings."""
    out: list[dict | str] = []
    for frame in frames:
        # Each frame is one ``data: ...\n\n`` block.
        body = frame.removeprefix("data: ").rstrip("\n")
        if not body:
            continue
        if body.strip() == "[DONE]":
            out.append("[DONE]")
            continue
        try:
            out.append(json.loads(body))
        except json.JSONDecodeError:
            out.append(body)
    return out


def _drive_async_gen(agen_coro):
    """Collect every yielded item from an async generator synchronously."""

    async def _collect():
        out: list = []
        async for x in agen_coro:
            out.append(x)
        return out

    return asyncio.run(_collect())


def _make_payload(*, tools=None, tool_choice=None, stream=True) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test",
        messages=[ChatMessage(role="user", content="hi")],
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
    )


# ---------------------------------------------------------------------
# Mock backends — return scripted generator output
# ---------------------------------------------------------------------


def _mock_backend_with_tool_call(script_text: str, *, usage: dict | None = None):
    """Build a mock MLX backend whose
    generate_chat_completion_with_tools yields cumulative text for
    ``script_text`` followed by a metadata event.
    """
    backend = mock.MagicMock()

    def _gen(**_kwargs):
        # Emit cumulative text in two chunks to exercise streaming.
        if script_text:
            mid = max(1, len(script_text) // 2)
            yield script_text[:mid]
            yield script_text
        yield {
            "type": "metadata",
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "timings": {},
        }

    backend.generate_chat_completion_with_tools = _gen
    return backend


def _mock_backend_for_agentic():
    """Build a mock backend whose agentic-loop generator emits a
    tool_start / tool_end pair and a final content turn.
    """
    backend = mock.MagicMock()

    def _gen(**_kwargs):
        # First turn: model calls a tool (event stream mirrors what
        # MlxLmBackend.generate_chat_completion_with_tools yields).
        yield {"type": "status", "text": "Searching: weather"}
        yield {
            "type": "tool_start",
            "tool_name": "web_search",
            "tool_call_id": "call_0",
            "arguments": {"query": "weather"},
        }
        yield {
            "type": "tool_end",
            "tool_name": "web_search",
            "tool_call_id": "call_0",
            "result": "sunny, 22C",
        }
        yield {"type": "status", "text": ""}
        # Second turn: model summarises.
        yield {"type": "content", "text": "It's sunny in Paris."}
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timings": {},
        }

    backend.generate_chat_completion_with_tools = _gen
    return backend


# ---------------------------------------------------------------------
# Agentic streaming (enable_tools=true path)
# ---------------------------------------------------------------------


class TestMlxAgenticStream:
    def test_full_frame_shape(self):
        backend = _mock_backend_for_agentic()

        def run_gen():
            return backend.generate_chat_completion_with_tools()

        frames = _drive_async_gen(
            _mlx_agentic_stream(
                request=_FakeRequest(),
                cancel_event=mock.MagicMock(),
                run_gen=run_gen,
                completion_id="chatcmpl-test",
                created=1000,
                model_name="test-model",
            )
        )

        parsed = _parse_sse(frames)
        # First real frame is the role delta.
        first = parsed[0]
        assert isinstance(first, dict)
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # tool_status / tool_start / tool_end custom events present.
        types = [
            p.get("type") for p in parsed if isinstance(p, dict) and "type" in p
        ]
        assert "tool_status" in types
        assert "tool_start" in types
        assert "tool_end" in types

        # A content delta for the final turn's text.
        contents = [
            p
            for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0]["delta"].get("content")
        ]
        assert any(
            "sunny in Paris" in c["choices"][0]["delta"]["content"]
            for c in contents
        )

        # Finish reason chunk.
        finals = [
            p
            for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0].get("finish_reason") == "stop"
        ]
        assert finals, "missing finish_reason=stop chunk"

        # Terminates with [DONE].
        assert parsed[-1] == "[DONE]"


# ---------------------------------------------------------------------
# Client-side tools pass-through (standard OpenAI tools=[...])
# ---------------------------------------------------------------------


class TestMlxOpenaiPassthroughStream:
    def test_emits_tool_calls_delta_when_model_calls_tool(self):
        script = (
            'thinking...\n'
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
            '</tool_call>'
        )
        backend = _mock_backend_with_tool_call(script)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        payload = _make_payload(tools=tools)

        frames = _drive_async_gen(
            _mlx_openai_passthrough_stream(
                request=_FakeRequest(),
                cancel_event=mock.MagicMock(),
                mlx_backend=backend,
                payload=payload,
                messages=[{"role": "user", "content": "weather?"}],
                stop=None,
                completion_id="chatcmpl-x",
                created=1000,
                model_name="test-model",
            )
        )

        parsed = _parse_sse(frames)

        # Role chunk first.
        assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"

        # tool_calls deltas present (id + function.name header, then args chunk).
        tc_frames = [
            p for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0]["delta"].get("tool_calls")
        ]
        assert tc_frames, "expected at least one tool_calls delta"

        # Extract the tool call payload assembled across the deltas.
        names: list[str] = []
        args_chunks: list[str] = []
        for frame in tc_frames:
            for tc in frame["choices"][0]["delta"]["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name"):
                    names.append(fn["name"])
                if fn.get("arguments"):
                    args_chunks.append(fn["arguments"])
        assert names == ["get_weather"]
        combined = "".join(args_chunks)
        assert json.loads(combined) == {"city": "Paris"}
        # Chunk E (E7): tool arguments must be streamed across multiple
        # fragments (character-by-character in ~8-char chunks) rather
        # than a single blob, matching llama-server's wire shape. The
        # example ``{"city": "Paris"}`` (18 chars) yields 3 fragments
        # at an 8-char chunk size; just assert >= 2 to allow future
        # chunk-size tuning without breaking the test.
        assert len(args_chunks) >= 2, (
            f"Expected argument deltas to be streamed across >=2 fragments, "
            f"got {len(args_chunks)}: {args_chunks!r}"
        )

        # Finish reason.
        finals = [
            p for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0].get("finish_reason") == "tool_calls"
        ]
        assert finals, "missing finish_reason=tool_calls chunk"

        # [DONE] terminator.
        assert parsed[-1] == "[DONE]"

    def test_plain_content_when_no_tool_call(self):
        script = "The answer is 42."
        backend = _mock_backend_with_tool_call(script)

        payload = _make_payload(tools=[{"type": "function", "function": {"name": "x"}}])

        frames = _drive_async_gen(
            _mlx_openai_passthrough_stream(
                request=_FakeRequest(),
                cancel_event=mock.MagicMock(),
                mlx_backend=backend,
                payload=payload,
                messages=[{"role": "user", "content": "q?"}],
                stop=None,
                completion_id="chatcmpl-y",
                created=1000,
                model_name="m",
            )
        )
        parsed = _parse_sse(frames)

        # No tool_calls deltas.
        tc_frames = [
            p for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0]["delta"].get("tool_calls")
        ]
        assert not tc_frames

        # Content delta present.
        contents = [
            p for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0]["delta"].get("content")
        ]
        joined = "".join(c["choices"][0]["delta"]["content"] for c in contents)
        assert "The answer is 42." in joined

        # Finish=stop.
        finals = [
            p for p in parsed
            if isinstance(p, dict)
            and p.get("choices")
            and p["choices"][0].get("finish_reason") == "stop"
        ]
        assert finals
        assert parsed[-1] == "[DONE]"

    def test_usage_chunk_emitted_last(self):
        script = "hi"
        backend = _mock_backend_with_tool_call(
            script,
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )

        payload = _make_payload(tools=[{"type": "function", "function": {"name": "x"}}])

        frames = _drive_async_gen(
            _mlx_openai_passthrough_stream(
                request=_FakeRequest(),
                cancel_event=mock.MagicMock(),
                mlx_backend=backend,
                payload=payload,
                messages=[{"role": "user", "content": "q"}],
                stop=None,
                completion_id="id",
                created=1,
                model_name="m",
            )
        )
        parsed = _parse_sse(frames)

        usage_chunks = [
            p for p in parsed
            if isinstance(p, dict)
            and p.get("choices") == []
            and p.get("usage") is not None
        ]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }


class TestMlxOpenaiPassthroughNonStreaming:
    def test_tool_calls_attached_when_present(self):
        script = (
            '<tool_call>{"name": "ping", "arguments": {"x": 1}}</tool_call>'
        )
        backend = _mock_backend_with_tool_call(script)

        payload = _make_payload(
            tools=[{"type": "function", "function": {"name": "ping"}}],
            stream=False,
        )

        resp = asyncio.run(
            _mlx_openai_passthrough_non_streaming(
                mlx_backend=backend,
                payload=payload,
                messages=[{"role": "user", "content": "q"}],
                stop=None,
                completion_id="id",
                created=1,
                model_name="m",
            )
        )

        # FastAPI JSONResponse exposes the body via the .body bytes.
        body = json.loads(resp.body.decode())
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "ping"

    def test_content_only_when_no_tool_call(self):
        backend = _mock_backend_with_tool_call("plain answer")

        payload = _make_payload(
            tools=[{"type": "function", "function": {"name": "ping"}}],
            stream=False,
        )

        resp = asyncio.run(
            _mlx_openai_passthrough_non_streaming(
                mlx_backend=backend,
                payload=payload,
                messages=[{"role": "user", "content": "q"}],
                stop=None,
                completion_id="id",
                created=1,
                model_name="m",
            )
        )
        body = json.loads(resp.body.decode())
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in body["choices"][0]["message"]
        assert body["choices"][0]["message"]["content"] == "plain answer"
