# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""End-to-end shape tests for the Anthropic Messages API MLX branch.

Uses mocked MLX backends so no model load is required. The route
layer is exercised via a FastAPI TestClient scoped to just the
inference router.
"""

from __future__ import annotations

import json
import os
import sys
from unittest import mock

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _mlx_backend_stub(chat_completion_with_tools_output):
    """Build a mocked MLX backend that satisfies the Anthropic route."""
    backend = mock.MagicMock()
    backend.is_loaded = True
    backend.supports_tools = True
    backend.supports_reasoning = False
    backend.model_identifier = "mlx-test-model"

    def _gen_tools(**_kwargs):
        yield from chat_completion_with_tools_output

    def _gen_plain(**_kwargs):
        yield from chat_completion_with_tools_output

    backend.generate_chat_completion_with_tools = _gen_tools
    backend.generate_chat_completion = _gen_plain
    return backend


@pytest.fixture
def mlx_app():
    from routes import inference as inf_mod

    # Build a minimal FastAPI app that mounts only the inference router
    # without auth and without any backend lifecycle hooks.
    app = FastAPI()

    # Override the auth dependency to a no-op so we don't need JWTs.
    async def _no_auth():
        return "test@local"

    app.include_router(inf_mod.router, prefix = "/inference")

    async def _no_auth_override():
        return "test@local"

    app.dependency_overrides[inf_mod.get_current_subject] = _no_auth_override
    return app


# ---------------------------------------------------------------------
# Server-side tools path (Studio's enable_tools shorthand)
# ---------------------------------------------------------------------


def test_anthropic_mlx_server_tools_streaming(mlx_app, monkeypatch):
    from routes import inference as inf_mod

    # Scripted agentic-loop output with one tool round-trip.
    events = [
        {"type": "status", "text": "Searching"},
        {
            "type": "tool_start",
            "tool_name": "web_search",
            "tool_call_id": "call_0",
            "arguments": {"query": "x"},
        },
        {
            "type": "tool_end",
            "tool_name": "web_search",
            "tool_call_id": "call_0",
            "result": "hit",
        },
        {"type": "status", "text": ""},
        {"type": "content", "text": "final answer"},
        {
            "type": "metadata",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "timings": {},
        },
    ]
    backend = _mlx_backend_stub(events)

    monkeypatch.setattr(inf_mod, "get_mlx_lm_backend", lambda: backend)
    # Make sure llama_backend path is inert.
    gguf = mock.MagicMock()
    gguf.is_loaded = False
    gguf.supports_tools = False
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: gguf)

    client = TestClient(mlx_app)
    with client.stream(
        "POST",
        "/inference/messages",
        json = {
            "model": "mlx-test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "enable_tools": True,
            "tools": [],
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    # Sanity: SSE frames present.
    assert "event: message_start" in body
    assert "content_block_start" in body
    # Tool_use block appears (from tool_start event).
    assert "tool_use" in body
    # Final text appears (Anthropic text_delta).
    assert "final answer" in body
    # Message stop.
    assert "message_stop" in body


def test_anthropic_mlx_server_tools_non_streaming(mlx_app, monkeypatch):
    from routes import inference as inf_mod

    events = [
        {"type": "content", "text": "hello world"},
        {
            "type": "metadata",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "timings": {},
        },
    ]
    backend = _mlx_backend_stub(events)

    monkeypatch.setattr(inf_mod, "get_mlx_lm_backend", lambda: backend)
    gguf = mock.MagicMock()
    gguf.is_loaded = False
    gguf.supports_tools = False
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: gguf)

    client = TestClient(mlx_app)
    resp = client.post(
        "/inference/messages",
        json = {
            "model": "mlx-test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "enable_tools": True,
            "tools": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "assistant"
    assert body["type"] == "message"
    assert any(b.get("type") == "text" for b in body["content"])
    assert body["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------
# Client-side pass-through path (standard Anthropic tools=[...])
# ---------------------------------------------------------------------


def test_anthropic_mlx_client_passthrough_streaming(mlx_app, monkeypatch):
    from routes import inference as inf_mod

    # generate_chat_completion_with_tools is called with
    # max_tool_iterations=0, so the route's pass-through helper sees
    # cumulative text events. Emulate a tool call in the output.
    script = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        "</tool_call>"
    )

    def _gen(**_kwargs):
        mid = len(script) // 2
        yield script[:mid]
        yield script
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            "timings": {},
        }

    backend = mock.MagicMock()
    backend.is_loaded = True
    backend.supports_tools = True
    backend.supports_reasoning = False
    backend.model_identifier = "mlx-test"
    backend.generate_chat_completion_with_tools = _gen

    monkeypatch.setattr(inf_mod, "get_mlx_lm_backend", lambda: backend)
    gguf = mock.MagicMock()
    gguf.is_loaded = False
    gguf.supports_tools = False
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: gguf)

    client = TestClient(mlx_app)
    with client.stream(
        "POST",
        "/inference/messages",
        json = {
            "model": "mlx-test",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    # Anthropic passthrough emitter produces content_block_start with
    # tool_use when tool_calls deltas arrive.
    assert "tool_use" in body
    assert "input_json_delta" in body
    assert "get_weather" in body
    # Must terminate cleanly.
    assert "message_stop" in body


def test_anthropic_mlx_client_passthrough_non_streaming(mlx_app, monkeypatch):
    from routes import inference as inf_mod

    script = (
        '<tool_call>{"name": "ping", "arguments": {"x": 1}}</tool_call>'
    )

    def _gen(**_kwargs):
        yield script
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "timings": {},
        }

    backend = mock.MagicMock()
    backend.is_loaded = True
    backend.supports_tools = True
    backend.supports_reasoning = False
    backend.model_identifier = "mlx-test"
    backend.generate_chat_completion_with_tools = _gen

    monkeypatch.setattr(inf_mod, "get_mlx_lm_backend", lambda: backend)
    gguf = mock.MagicMock()
    gguf.is_loaded = False
    gguf.supports_tools = False
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: gguf)

    client = TestClient(mlx_app)
    resp = client.post(
        "/inference/messages",
        json = {
            "model": "mlx-test",
            "messages": [{"role": "user", "content": "q"}],
            "stream": False,
            "tools": [
                {
                    "name": "ping",
                    "description": "x",
                    "input_schema": {"type": "object"},
                }
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool_use = next(b for b in body["content"] if b["type"] == "tool_use")
    assert tool_use["name"] == "ping"
    assert tool_use["input"] == {"x": 1}


# ---------------------------------------------------------------------
# Plain chat (no tools) — MLX fallback path
# ---------------------------------------------------------------------


def test_anthropic_mlx_plain_chat_non_streaming(mlx_app, monkeypatch):
    from routes import inference as inf_mod

    def _gen(**_kwargs):
        yield "hello"
        yield "hello world"
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "timings": {},
        }

    backend = mock.MagicMock()
    backend.is_loaded = True
    backend.supports_tools = False
    backend.supports_reasoning = False
    backend.model_identifier = "mlx-test"
    backend.generate_chat_completion = _gen

    monkeypatch.setattr(inf_mod, "get_mlx_lm_backend", lambda: backend)
    gguf = mock.MagicMock()
    gguf.is_loaded = False
    gguf.supports_tools = False
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: gguf)

    client = TestClient(mlx_app)
    resp = client.post(
        "/inference/messages",
        json = {
            "model": "mlx-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "assistant"
    assert any("hello" in b.get("text", "") for b in body["content"] if b["type"] == "text")
