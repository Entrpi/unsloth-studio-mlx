# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for Phase-5 schema additions and the tool-history preserving
variant of ``_extract_content_parts``.

Covers:

* Typed ``FunctionCall`` / ``ToolCall`` models accept OpenAI wire format
  (arguments as JSON string and as dict).
* ``ChatMessage`` supports ``reasoning_content`` and validates the
  per-role shape constraints that already existed.
* ``ChoiceDelta.tool_calls`` + ``ToolCallDelta`` round-trip through
  Pydantic into the OpenAI-compatible JSON delta shape.
* ``_extract_content_parts(preserve_tool_history=True)`` keeps
  ``tool_calls`` / ``tool_call_id`` / ``name`` /
  ``reasoning_content`` on the emitted message dicts; the default
  (``False``) preserves legacy Phase-1 behaviour and strips them.
"""

from __future__ import annotations

import json
import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest  # noqa: E402

from models.inference import (  # noqa: E402
    ChatMessage,
    ChoiceDelta,
    ChunkChoice,
    FunctionCall,
    ToolCall,
    ToolCallDelta,
    ToolCallFunctionDelta,
)
from routes.inference import _extract_content_parts  # noqa: E402


# ---------------------------------------------------------------------
# Typed tool-call models
# ---------------------------------------------------------------------


class TestFunctionCall:
    def test_string_arguments(self):
        fc = FunctionCall(name="x", arguments='{"a": 1}')
        assert fc.name == "x"
        assert fc.arguments == '{"a": 1}'

    def test_dict_arguments(self):
        fc = FunctionCall(name="x", arguments={"a": 1})
        # Both shapes are permitted.
        assert fc.arguments == {"a": 1}

    def test_missing_arguments_defaults_empty(self):
        fc = FunctionCall(name="x")
        assert fc.arguments == ""


class TestToolCall:
    def test_minimal(self):
        tc = ToolCall(
            id="call_0",
            function=FunctionCall(name="web_search", arguments='{}'),
        )
        assert tc.type == "function"
        assert tc.id == "call_0"
        assert tc.function.name == "web_search"

    def test_dump_shape_is_openai_compatible(self):
        tc = ToolCall(
            id="call_0",
            function=FunctionCall(name="w", arguments='{"q": 1}'),
        )
        d = tc.model_dump()
        assert d == {
            "id": "call_0",
            "type": "function",
            "function": {"name": "w", "arguments": '{"q": 1}'},
        }


# ---------------------------------------------------------------------
# ChatMessage — tool-use round-trips and reasoning_content
# ---------------------------------------------------------------------


class TestChatMessageToolRoundTrip:
    def test_assistant_with_typed_tool_calls(self):
        # ToolCall instances are normalised to dicts on input so every
        # downstream consumer sees the OpenAI wire shape.
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_0",
                    function=FunctionCall(
                        name="get_weather", arguments='{"city": "Paris"}'
                    ),
                )
            ],
        )
        assert msg.content is None
        assert msg.tool_calls is not None
        tc = msg.tool_calls[0]
        # Normalised to dict form.
        assert isinstance(tc, dict)
        assert tc["function"]["name"] == "get_weather"
        assert tc["id"] == "call_0"

    def test_assistant_with_dict_tool_calls_still_accepted(self):
        # Clients that serialize tool_calls as dicts still work.
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "arguments": "{}",
                    },
                }
            ],
        )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0]["function"]["name"] == "ping"

    def test_full_conversation_round_trip(self):
        # user -> assistant(tool_calls) -> tool(result) -> assistant(answer)
        convo = [
            ChatMessage(role="user", content="What's the weather in Paris?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_0",
                        function=FunctionCall(
                            name="get_weather", arguments='{"city": "Paris"}'
                        ),
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_0",
                name="get_weather",
                content='{"temperature": 22}',
            ),
            ChatMessage(
                role="assistant", content="The weather in Paris is 22°C."
            ),
        ]
        assert convo[0].role == "user"
        assert convo[1].tool_calls is not None
        # Normalised to dict form.
        assert convo[1].tool_calls[0]["id"] == "call_0"
        assert convo[2].tool_call_id == "call_0"
        assert convo[3].content == "The weather in Paris is 22°C."

    def test_reasoning_content_field(self):
        msg = ChatMessage(
            role="assistant",
            content="final answer",
            reasoning_content="the model thought about it...",
        )
        assert msg.reasoning_content == "the model thought about it..."

    def test_reasoning_content_defaults_none(self):
        msg = ChatMessage(role="user", content="hi")
        assert msg.reasoning_content is None


# ---------------------------------------------------------------------
# Streaming delta models
# ---------------------------------------------------------------------


class TestStreamingDeltas:
    def test_tool_calls_delta_dump(self):
        d = ChoiceDelta(
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    id="call_0",
                    type="function",
                    function=ToolCallFunctionDelta(
                        name="get_weather", arguments=""
                    ),
                )
            ],
        )
        dumped = d.model_dump(exclude_none=True)
        assert dumped == {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ],
        }

    def test_arguments_only_delta(self):
        # Arguments-only deltas (no id/name) — mid-stream chunks.
        d = ChoiceDelta(
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    function=ToolCallFunctionDelta(arguments='{"cit'),
                )
            ]
        )
        dumped = d.model_dump(exclude_none=True)
        call = dumped["tool_calls"][0]
        # type defaults to "function" so stays in the dump.
        assert call["function"] == {"arguments": '{"cit'}

    def test_finish_reason_tool_calls_allowed(self):
        ch = ChunkChoice(delta=ChoiceDelta(), finish_reason="tool_calls")
        assert ch.finish_reason == "tool_calls"


# ---------------------------------------------------------------------
# _extract_content_parts with and without preserve_tool_history
# ---------------------------------------------------------------------


def _build_full_convo() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="you are a helpful assistant"),
        ChatMessage(role="user", content="what is the weather in Paris?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    ),
                )
            ],
            reasoning_content="the user wants weather info",
        ),
        ChatMessage(
            role="tool",
            tool_call_id="call_0",
            name="get_weather",
            content='{"temperature": 22}',
        ),
    ]


class TestExtractContentPartsPreserve:
    def test_default_strips_tool_history(self):
        system, msgs, img = _extract_content_parts(_build_full_convo())
        assert system == "you are a helpful assistant"
        # role="tool" dropped entirely; assistant loses tool_calls.
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"]
        assert "tool_calls" not in msgs[1]
        assert "reasoning_content" not in msgs[1]
        assert img is None

    def test_preserve_tool_history_keeps_tool_calls(self):
        system, msgs, img = _extract_content_parts(
            _build_full_convo(), preserve_tool_history=True
        )
        assert system == "you are a helpful assistant"
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool"]
        # Assistant kept tool_calls as plain dicts.
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[1]["reasoning_content"] == "the user wants weather info"
        # Tool message preserved tool_call_id + name.
        assert msgs[2]["tool_call_id"] == "call_0"
        assert msgs[2]["name"] == "get_weather"
        assert msgs[2]["content"] == '{"temperature": 22}'

    def test_preserve_works_with_dict_tool_calls(self):
        convo = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "p", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_1",
                content="ok",
            ),
        ]
        system, msgs, _ = _extract_content_parts(
            convo, preserve_tool_history=True
        )
        assert msgs[1]["tool_calls"][0]["id"] == "call_1"

    def test_preserve_does_not_leak_into_default_path(self):
        # Repeated call with default=False after a preserve=True
        # extraction must not carry any state between calls.
        convo = _build_full_convo()
        _extract_content_parts(convo, preserve_tool_history=True)
        _, msgs, _ = _extract_content_parts(convo)
        assert len(msgs) == 2
        assert "tool_calls" not in msgs[1]
