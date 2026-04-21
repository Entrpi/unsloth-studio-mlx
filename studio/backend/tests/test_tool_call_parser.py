# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for ``core.inference._tool_call_parser``.

Exercises both dialects the GGUF backend supported pre-extraction and
the behaviour-preserving guarantees the MLX backend relies on (single
parameter, multi-parameter, unclosed tags, model-forcing hints).
"""

from __future__ import annotations

import json
import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest  # noqa: E402

from core.inference._tool_call_parser import (  # noqa: E402
    ParsedToolCall,
    TOOL_ALL_PATS,
    TOOL_CLOSED_PATS,
    TOOL_XML_SIGNALS,
    parse_tool_calls_from_text,
    strip_tool_markup,
)


# ---------------------------------------------------------------------
# Dialect 1 — JSON body inside <tool_call>
# ---------------------------------------------------------------------


class TestJsonDialect:
    def test_simple_qwen_tool_call(self):
        text = '<tool_call>{"name": "web_search", "arguments": {"query": "hello"}}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "hello"}
        assert calls[0]["id"].startswith("call_")

    def test_pretty_printed_body(self):
        text = """
<tool_call>
{
  "name": "python",
  "arguments": {
    "code": "print(1+1)"
  }
}
</tool_call>
"""
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "python"
        assert json.loads(calls[0]["function"]["arguments"]) == {"code": "print(1+1)"}

    def test_unclosed_tool_call_tag(self):
        # Models drop </tool_call> routinely — parser must cope.
        text = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"

    def test_embedded_quotes_in_arguments(self):
        # Quotes inside string values must not break the brace balancer.
        text = (
            '<tool_call>{"name": "python", '
            '"arguments": {"code": "print(\\"hi\\")"}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "code": 'print("hi")'
        }

    def test_arguments_passed_as_string(self):
        # OpenAI allows arguments as JSON string; preserve as-is.
        text = '<tool_call>{"name": "web_search", "arguments": "{\\"q\\": 1}"}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == '{"q": 1}'

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
            'foo'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "a"
        assert calls[1]["function"]["name"] == "b"
        # IDs are unique.
        assert calls[0]["id"] != calls[1]["id"]

    def test_malformed_json_is_skipped(self):
        text = '<tool_call>{"name": bad}</tool_call>'
        assert parse_tool_calls_from_text(text) == []

    def test_empty_input(self):
        assert parse_tool_calls_from_text("") == []
        assert parse_tool_calls_from_text(None) == []  # type: ignore[arg-type]

    def test_plain_text_with_no_markup(self):
        assert parse_tool_calls_from_text("Hello, world.") == []

    def test_name_with_missing_arguments(self):
        # Model emits a minimal call with no arguments; parser should
        # still synthesize a call with empty arguments.
        text = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "ping"
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_qwen_bonsai_forced_hint(self):
        text = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        calls = parse_tool_calls_from_text(text, model_family="qwen")
        assert len(calls) == 1


# ---------------------------------------------------------------------
# Dialect 2 — XML <function=name>/<parameter=key> body
# ---------------------------------------------------------------------


class TestXmlDialect:
    def test_single_parameter_function(self):
        text = (
            "<function=python><parameter=code>print(1)</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "python"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "code": "print(1)"
        }

    def test_multi_parameter_function(self):
        text = (
            "<function=web_search>"
            "<parameter=query>python tools</parameter>"
            "<parameter=url>https://x.example</parameter>"
            "</function>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        got = json.loads(calls[0]["function"]["arguments"])
        assert got == {"query": "python tools", "url": "https://x.example"}

    def test_missing_closing_tags(self):
        # Drop the trailing </parameter> and </function> — models do
        # this routinely when the value contains </function>-like
        # substrings.
        text = (
            "<function=python><parameter=code>"
            "print('</function>')"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "python"
        # Single-parameter path: value extends to end-of-body.
        got = json.loads(calls[0]["function"]["arguments"])
        assert "print('" in got["code"]

    def test_xml_wrapped_in_tool_call(self):
        # End-tag </tool_call> acts as a body boundary.
        text = (
            "<tool_call>"
            "<function=ls><parameter=path>/tmp</parameter></function>"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "ls"
        got = json.loads(calls[0]["function"]["arguments"])
        assert got == {"path": "/tmp"}

    def test_xml_not_tried_when_json_succeeded(self):
        # Auto mode should stop at JSON; the XML content after should
        # not produce extra calls.
        text = (
            '<tool_call>{"name": "a", "arguments": {"q": 1}}</tool_call>'
            "<function=b><parameter=x>y</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "a"

    def test_force_xml_hint(self):
        text = '<tool_call>{"name": "a", "arguments": {"q": 1}}</tool_call>'
        # Forcing XML dialect ignores the JSON payload.
        calls = parse_tool_calls_from_text(text, model_family="claude")
        assert calls == []

    def test_multiple_xml_functions(self):
        text = (
            "<function=a><parameter=x>1</parameter></function>"
            "<function=b><parameter=y>2</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "a"
        assert calls[1]["function"]["name"] == "b"


# ---------------------------------------------------------------------
# Auto-heal / strip_tool_markup
# ---------------------------------------------------------------------


class TestStripToolMarkup:
    def test_strip_closed_block(self):
        text = (
            "hello <tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> world"
        )
        assert strip_tool_markup(text) == "hello  world"

    def test_dont_strip_unclosed_in_default_mode(self):
        # final=False leaves trailing unclosed blocks intact (they
        # might still be completing).
        text = 'hello <tool_call>{"name":"a"'
        out = strip_tool_markup(text)
        assert "<tool_call>" in out

    def test_strip_unclosed_when_final(self):
        text = 'hello <tool_call>{"name":"a"'
        assert strip_tool_markup(text, final=True) == "hello"

    def test_strip_function_block(self):
        text = (
            "pre <function=a><parameter=x>1</parameter></function> post"
        )
        assert strip_tool_markup(text) == "pre  post"

    def test_final_also_strips_dangling_function(self):
        text = "pre <function=a><parameter=x>1"
        assert strip_tool_markup(text, final=True) == "pre"

    def test_empty_input(self):
        assert strip_tool_markup("") == ""
        assert strip_tool_markup("", final=True) == ""

    def test_patterns_are_exported(self):
        assert len(TOOL_CLOSED_PATS) == 2
        assert len(TOOL_ALL_PATS) == 4
        assert "<tool_call>" in TOOL_XML_SIGNALS
        assert "<function=" in TOOL_XML_SIGNALS


# ---------------------------------------------------------------------
# ParsedToolCall dataclass surface
# ---------------------------------------------------------------------


class TestParsedToolCallDataclass:
    def test_to_openai_dict_round_trip(self):
        p = ParsedToolCall(
            name="python",
            arguments='{"code": "1"}',
            id="call_0",
            raw="<tool_call>...</tool_call>",
            dialect="json",
        )
        d = p.to_openai_dict()
        assert d == {
            "id": "call_0",
            "type": "function",
            "function": {"name": "python", "arguments": '{"code": "1"}'},
        }


# ---------------------------------------------------------------------
# Behaviour preservation — parity against the old inlined GGUF path
# ---------------------------------------------------------------------


class TestGgufBackendParity:
    def test_llama_cpp_backend_delegates(self):
        # The GGUF backend still exposes _parse_tool_calls_from_text for
        # internal callers; make sure it produces the same result as
        # the shared parser.
        from core.inference.llama_cpp import LlamaCppBackend

        text = '<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        assert LlamaCppBackend._parse_tool_calls_from_text(
            text
        ) == parse_tool_calls_from_text(text)

    def test_ids_renumbered_when_both_dialects_match(self):
        # Forced XML dialect: IDs must still be sequential integers.
        text = (
            "<function=a><parameter=x>1</parameter></function>"
            "<function=b><parameter=y>2</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text, model_family="xml")
        ids = [c["id"] for c in calls]
        assert ids == ["call_0", "call_1"]
