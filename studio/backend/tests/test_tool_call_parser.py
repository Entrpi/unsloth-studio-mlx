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
    extract_channel_thought,
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

    def test_strip_gemma_channel_block(self):
        # Gemma-4 thinking-channel blocks are stripped even mid-stream
        # (closed variant): downstream routes channel content to
        # reasoning_content via ``extract_channel_thought``.
        text = "<|channel>thought\nanalysing...<channel|>prose"
        assert strip_tool_markup(text) == "prose"

    def test_final_strips_orphan_tool_call_close(self):
        # When Gemma-4 resumes mid-tool-call (template anchored the
        # prompt into an open tool_call body), the continuation turn
        # emits only the tail close markers like
        # ``<|"|>}<tool_call|>``. At final-flush time the markup
        # tokens are strippable; residual JSON punctuation (``}``) is
        # harmless debris and fine to leave as-is.
        out = strip_tool_markup('<|"|>}<tool_call|>', final=True)
        assert "<tool_call|>" not in out
        assert '<|"|>' not in out

    def test_final_strips_orphan_tool_response_close(self):
        assert strip_tool_markup("<tool_response|>", final=True) == ""

    def test_final_strips_orphan_channel_close(self):
        assert strip_tool_markup("<channel|>tail", final=True) == "tail"

    def test_final_strips_gemma_quote_token(self):
        assert (
            strip_tool_markup('foo <|"|>bar<|"|> baz', final=True)
            == "foo bar baz"
        )

    def test_patterns_are_exported(self):
        # Closed patterns: JSON-in-<tool_call>, XML <function=>,
        # Gemma-4 <|tool_call>, and Gemma-4 <|channel> (thinking
        # block). ALL_PATS adds one unclosed variant per dialect for
        # the final-flush pass, plus orphan-fragment strippers for
        # stray close markers that leak out of malformed / resume-
        # mid-body continuation turns.
        assert len(TOOL_CLOSED_PATS) == 4
        assert len(TOOL_ALL_PATS) == 13
        assert "<tool_call>" in TOOL_XML_SIGNALS
        assert "<function=" in TOOL_XML_SIGNALS
        assert "<|tool_call>" in TOOL_XML_SIGNALS
        assert "<|channel>" in TOOL_XML_SIGNALS


# ---------------------------------------------------------------------
# Dialect 5 — GLM-4/4.6/4.7 <tool_call>name\n<arg_key>/<arg_value>
# ---------------------------------------------------------------------


class TestGlmDialect:
    def test_simple_glm_tool_call(self):
        text = (
            "<tool_call>web_search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>Ternary Bonsai</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        args = json.loads(calls[0]["function"]["arguments"])
        # Unquoted value → preserved as a string after JSON-decode
        # fails.
        assert args == {"query": "Ternary Bonsai"}

    def test_glm_multiple_args(self):
        text = (
            "<tool_call>web_search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>Ternary Bonsai</arg_value>\n"
            "<arg_key>url</arg_key>\n"
            "<arg_value>None</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        # "None" isn't valid JSON → kept as the string "None".
        assert args == {"query": "Ternary Bonsai", "url": "None"}

    def test_glm_with_think_and_prose_prelude(self):
        # Real-world shape from GLM-4.6V-Flash: a <think> block, then
        # prose, then the tool call. Parser ignores surrounding noise.
        text = (
            "<think>The user is asking about X. I should search.</think>\n"
            "I will search for X now.\n"
            "<tool_call>web_search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>X</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"

    def test_glm_json_encoded_value_decodes(self):
        # When the model emits a JSON-quoted string value the parser
        # should decode it (avoids double-wrapping in the final
        # arguments string).
        text = (
            "<tool_call>python\n"
            "<arg_key>code</arg_key>\n"
            "<arg_value>\"print(1)\"</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"code": "print(1)"}

    def test_glm_nested_object_value_decodes(self):
        text = (
            "<tool_call>configure\n"
            "<arg_key>options</arg_key>\n"
            "<arg_value>{\"depth\": 3, \"verbose\": true}</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"options": {"depth": 3, "verbose": True}}

    def test_glm_multiple_tool_calls(self):
        text = (
            "<tool_call>a\n<arg_key>x</arg_key><arg_value>1</arg_value>"
            "</tool_call>"
            "<tool_call>b\n<arg_key>y</arg_key><arg_value>2</arg_value>"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "a"
        assert calls[1]["function"]["name"] == "b"
        # ids are re-numbered uniquely
        assert calls[0]["id"] != calls[1]["id"]

    def test_glm_empty_body_skipped(self):
        # A <tool_call>name\n</tool_call> with NO arg pairs is not
        # executable — parser drops it rather than emitting a call
        # with empty arguments.
        text = "<tool_call>web_search\n</tool_call>"
        calls = parse_tool_calls_from_text(text)
        assert calls == []

    def test_json_dialect_still_wins_when_body_is_json(self):
        # A proper JSON-in-<tool_call> body must not accidentally be
        # consumed by the GLM dialect (GLM start regex is guarded by
        # ``(?!\{)``).
        text = '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "a"
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}

    def test_glm_forced_family_still_parses(self):
        text = (
            "<tool_call>f\n"
            "<arg_key>k</arg_key><arg_value>v</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text, model_family="glm")
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "f"


# ---------------------------------------------------------------------
# extract_channel_thought — Gemma-4 <|channel>thought...<channel|>
# ---------------------------------------------------------------------


class TestExtractChannelThought:
    def test_no_channel_returns_input_unchanged(self):
        reasoning, remaining = extract_channel_thought("Just plain prose.")
        assert reasoning is None
        assert remaining == "Just plain prose."

    def test_empty_input(self):
        reasoning, remaining = extract_channel_thought("")
        assert reasoning is None
        assert remaining == ""

    def test_gemma_channel_with_thought_prefix(self):
        text = (
            "<|channel>thought\n"
            "The user wants the weather.\n"
            "<channel|>"
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
        )
        reasoning, remaining = extract_channel_thought(text)
        assert reasoning == "The user wants the weather."
        # Channel block gone, tool_call preserved verbatim for the
        # tool-markup stripper to handle.
        assert "<|channel>" not in remaining
        assert "<channel|>" not in remaining
        assert "<|tool_call>call:get_weather" in remaining

    def test_channel_without_thought_prefix(self):
        text = "<|channel>raw analysis text<channel|>after"
        reasoning, remaining = extract_channel_thought(text)
        assert reasoning == "raw analysis text"
        assert remaining == "after"

    def test_multiple_channels_concatenate(self):
        text = (
            "<|channel>thought\nfirst thought<channel|>"
            "prose"
            "<|channel>thought\nsecond thought<channel|>"
        )
        reasoning, remaining = extract_channel_thought(text)
        assert reasoning == "first thought\n\nsecond thought"
        assert remaining == "prose"

    def test_empty_channel_body_yields_no_reasoning(self):
        text = "<|channel>thought\n<channel|>after"
        reasoning, remaining = extract_channel_thought(text)
        # Empty body after trim → contributes nothing. Channel block
        # still removed from remaining.
        assert reasoning is None
        assert remaining == "after"

    def test_unclosed_channel_left_alone(self):
        # Mid-stream state — opener without closer. We don't strip it
        # here; the caller's hold-back logic will buffer until the
        # closer arrives or the stream ends.
        text = "<|channel>thought\nstill thinking..."
        reasoning, remaining = extract_channel_thought(text)
        assert reasoning is None
        assert remaining == text


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


# ---------------------------------------------------------------------
# Dialect 3 — Gemma-4 <|tool_call>call:NAME{…}<tool_call|>
# ---------------------------------------------------------------------


class TestGemmaDialect:
    def test_simple_gemma_tool_call(self):
        text = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
        assert calls[0]["id"].startswith("call_")

    def test_gemma_with_chain_of_thought_prefix(self):
        # Gemma-4 wraps its reasoning in <|channel>thought ... <channel|>
        # and follows it with the tool call. The parser must ignore the
        # reasoning block and pick up the call.
        text = (
            "<|channel>thought\n"
            "Analyse the request. The user wants Paris weather.\n"
            "<channel|>"
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_gemma_multi_arg_with_types(self):
        text = (
            '<|tool_call>call:configure{'
            'name:<|"|>test<|"|>,'
            "enabled:true,"
            "count:42,"
            'tags:[<|"|>a<|"|>,<|"|>b<|"|>],'
            "ratio:0.5,"
            "extra:null"
            "}<tool_call|>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {
            "name": "test",
            "enabled": True,
            "count": 42,
            "tags": ["a", "b"],
            "ratio": 0.5,
            "extra": None,
        }

    def test_gemma_nested_object(self):
        text = (
            '<|tool_call>call:submit{'
            'payload:{user:<|"|>alice<|"|>,ids:[1,2,3]},'
            "timeout:30"
            "}<tool_call|>"
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"payload": {"user": "alice", "ids": [1, 2, 3]}, "timeout": 30}

    def test_gemma_unclosed_call_is_skipped(self):
        # Missing <tool_call|> end marker; balanced-brace walker runs
        # off the end and the call is skipped rather than mis-parsed.
        text = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>'  # no close
        calls = parse_tool_calls_from_text(text)
        assert calls == []

    def test_gemma_multiple_calls_in_one_turn(self):
        text = (
            '<|tool_call>call:first{x:1}<tool_call|>'
            '<|tool_call>call:second{y:<|"|>two<|"|>}<tool_call|>'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2
        assert [c["function"]["name"] for c in calls] == ["first", "second"]
        assert [c["id"] for c in calls] == ["call_0", "call_1"]

    def test_gemma_forced_family_hint(self):
        # model_family="gemma" should force the Gemma dialect only.
        text = '<|tool_call>call:ping{}<tool_call|>'
        calls = parse_tool_calls_from_text(text, model_family="gemma")
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "ping"

    def test_gemma_dialect_not_tried_when_json_forced(self):
        # When family="qwen"/"json", Gemma markup is ignored entirely.
        text = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        assert parse_tool_calls_from_text(text, model_family="qwen") == []

    def test_strip_removes_closed_gemma_block(self):
        text = (
            "Prose before. "
            '<|tool_call>call:x{y:1}<tool_call|>'
            " Prose after."
        )
        assert strip_tool_markup(text) == "Prose before.  Prose after."

    def test_strip_final_removes_unclosed_gemma(self):
        text = (
            "Prose. "
            '<|tool_call>call:x{y:1}'  # no close marker
        )
        assert strip_tool_markup(text, final=True) == "Prose."

    def test_gemma_signal_in_tool_xml_signals(self):
        assert "<|tool_call>" in TOOL_XML_SIGNALS


# ---------------------------------------------------------------------
# Dialect 4 — loose top-level JSON envelope (no wrapper tag)
# ---------------------------------------------------------------------


class TestLooseJsonEnvelope:
    """Some models (notably Gemma-4 E4B under some prompts) drop the
    wrapper tag and emit a bare top-level JSON object with tool-call-
    shaped keys. Dialect 4 recognises these as a fall-through after
    the three wrapper-based dialects produce nothing. Key aliases:
    name / tool_name / tool / function for the name field; arguments /
    params / input / parameters for the args field.
    """

    def test_tool_name_params_shape(self):
        text = '{ "tool_name": "web_search", "params": { "query": "Ternary Bonsai" } }'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "Ternary Bonsai"}

    def test_name_arguments_shape_without_wrapper(self):
        text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_tool_input_shape(self):
        text = '{"tool": "python", "input": {"code": "print(1)"}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "python"
        assert json.loads(calls[0]["function"]["arguments"]) == {"code": "print(1)"}

    def test_tool_calls_wrapper_list(self):
        text = (
            '{"tool_calls": ['
            '{"name": "a", "arguments": {"x": 1}},'
            '{"name": "b", "arguments": {"y": 2}}'
            ']}'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2
        assert [c["function"]["name"] for c in calls] == ["a", "b"]
        assert [c["id"] for c in calls] == ["call_0", "call_1"]

    def test_function_nested_shape(self):
        text = '{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_prose_prefix_before_json(self):
        text = (
            "I will call the tool now.\n"
            '{"tool_name": "web_search", "params": {"query": "x"}}'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"

    def test_string_args_pass_through(self):
        # OpenAI wire: arguments is a JSON string even when the model
        # emitted it as a string. Don't re-quote.
        text = '{"name": "x", "arguments": "raw-string-args"}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == "raw-string-args"

    # ── False-positive defences ────────────────────────────────

    def test_incidental_name_only_is_not_a_tool_call(self):
        # Answer to "what's your name?" shouldn't trigger.
        text = '{"name": "Alice"}'
        assert parse_tool_calls_from_text(text) == []

    def test_incidental_json_content_not_tool_call(self):
        text = '{"temperature": 22, "humidity": 40}'
        assert parse_tool_calls_from_text(text) == []

    def test_non_string_name_is_skipped(self):
        text = '{"name": 42, "params": {}}'
        assert parse_tool_calls_from_text(text) == []

    def test_wrapped_qwen_dialect_takes_precedence(self):
        # When the <tool_call> wrapper is present, dialect 1 fires and
        # dialect 4 doesn't re-parse the inner JSON.
        text = '<tool_call>{"name": "X", "arguments": {"a": 1}}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "X"
        # Only one call emitted — dialect 4 didn't double-add.

    def test_empty_arguments_shape(self):
        text = '{"tool_name": "ping", "params": {}}'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_forced_qwen_dialect_skips_loose(self):
        # When caller forces a specific dialect, dialect 4 should not
        # hijack. 'qwen' implies try_json=True, but loose-JSON only
        # fires when wrappered forms found nothing.
        text = '{"tool_name": "X", "params": {}}'
        assert parse_tool_calls_from_text(text, model_family="qwen") != []
        # XML-forced should NOT match the loose JSON (try_json=False).
        assert parse_tool_calls_from_text(text, model_family="xml") == []


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
