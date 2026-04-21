# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Chunk E (E8): per-model-family tool-call template rendering tests.

Chunk C's ``supports_tools`` detection lights up for Hermes / Qwen /
Mistral / Llama-3.1 / Gemma via their tokenizer chat templates, but
end-to-end tool-call *rendering* had only been exercised against the
Bonsai template. This file probes each locally-cached MLX model family
that advertises tool support and asserts the rendered prompt contains
the tool call payload.

Key contract under test:

    _extract_content_parts(preserve_tool_history=True) emits an
    assistant turn as {"role":"assistant", "content": "<string>",
    "tool_calls": [...]}. Some templates (Bonsai, Qwen, Hermes) render
    the ``tool_calls`` block verbatim. Others (pure chat-only Gemma
    templates that lack a tools branch, for instance) may render an
    empty assistant turn and lose the tool call entirely.

We use the locally-cached models under ``~/.lmstudio/models`` — the
test is SKIPPED when a specific model directory isn't present so the
CI box without the cache doesn't block. Per Chunk E's no-download
rule, new models are never fetched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from models.inference import ChatMessage, FunctionCall, ToolCall  # noqa: E402
from routes.inference import _extract_content_parts  # noqa: E402


# ── Candidate models ────────────────────────────────────────────────
#
# Each entry: (family_label, model_dir, expected_marker_in_prompt).
# ``expected_marker_in_prompt`` is any substring that MUST appear in
# the rendered template output for a tool-call-carrying assistant turn.
# If the marker is missing, the template swallowed the tool_calls and
# the rendering is broken for that family.

_LMSTUDIO_ROOT = Path.home() / ".lmstudio" / "models"

_CANDIDATES = [
    # Qwen 3.5 text family — custom nested XML:
    #   <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...
    # The Qwen template does ``tool_call.arguments|items`` which requires
    # ``arguments`` to be a mapping. OpenAI's wire format delivers a JSON
    # string, and ToolCall.model_dump faithfully preserves that. Bonsai
    # and Gemma's templates branch on ``is string`` / ``is mapping`` so
    # they handle both forms; Qwen's doesn't. This is a documented gap:
    # ``_extract_content_parts`` is faithful to OpenAI's shape, and Qwen's
    # template expects a pre-parsed dict. Tracked as an xfail here so
    # the rendering contract is still exercised and any future extractor
    # update (e.g. opportunistic json.loads on ``arguments``) flips this
    # test green without us silently losing coverage.
    pytest.param(
        "qwen3.5-4b",
        _LMSTUDIO_ROOT / "mlx-community" / "Qwen3.5-4B-MLX-4bit",
        "<tool_call>",
        id = "qwen3.5-4b",
        marks = pytest.mark.xfail(
            reason = (
                "Qwen 3.5 template expects tool_call.arguments to be a "
                "mapping; OpenAI wire format delivers a JSON string. "
                "Extractor-shape vs template-expectation mismatch. "
                "See Chunk E PR_DESCRIPTION: E8 for the documented gap."
            ),
            strict = False,
        ),
    ),
    pytest.param(
        "qwen3.5-35b-a3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Qwen3.5-35B-A3B-4bit",
        "<tool_call>",
        id = "qwen3.5-35b-a3b",
        marks = pytest.mark.xfail(
            reason = (
                "Same template family as qwen3.5-4b — see that xfail for "
                "details. Grouped here for family-coverage clarity."
            ),
            strict = False,
        ),
    ),
    # Gemma 4 — different idiom: ``<|tool_call>call:NAME{...}<tool_call|>``.
    # Template handles both string and dict forms of arguments.
    pytest.param(
        "gemma-4-31b",
        _LMSTUDIO_ROOT / "mlx-community" / "gemma-4-31b-it-4bit",
        "<|tool_call>",
        id = "gemma-4-31b",
    ),
    # Bonsai (2bit) — already the baseline Chunk C tested, include here
    # for regression-protection parity across the whole candidate set.
    pytest.param(
        "bonsai-1.7b",
        _LMSTUDIO_ROOT / "prism-ml" / "Ternary-Bonsai-1.7B-mlx-2bit",
        "<tool_call>",
        id = "bonsai-1.7b",
    ),
]


def _load_tokenizer(model_dir: Path):
    """Load the tokenizer from a local MLX checkpoint. Returns None if
    the directory is missing (the test is skipped in that case)."""
    if not model_dir.is_dir():
        return None
    # Prefer transformers's AutoTokenizer — it handles chat_template.jinja
    # and tokenizer_config.json together the same way mlx_lm does.
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code = False
        )
    except Exception:
        return None


def _build_tool_call_history() -> list[ChatMessage]:
    """A conversation where the assistant issued a tool call and the
    tool returned a result. This is the exact shape Chunk C flagged
    as untested across non-Bonsai templates:

        user -> "What's the weather in Paris?"
        assistant -> content=None, tool_calls=[get_weather({"city": "Paris"})]
        tool -> "Paris: 15C sunny" (tool_call_id matches)
        user -> (follow-up)
    """
    return [
        ChatMessage(
            role = "user",
            content = "What's the weather in Paris?",
        ),
        ChatMessage(
            role = "assistant",
            content = None,
            tool_calls = [
                ToolCall(
                    id = "call_abc123",
                    type = "function",
                    function = FunctionCall(
                        name = "get_weather",
                        arguments = '{"city": "Paris"}',
                    ),
                )
            ],
        ),
        ChatMessage(
            role = "tool",
            content = "Paris: 15C sunny",
            tool_call_id = "call_abc123",
            name = "get_weather",
        ),
        ChatMessage(role = "user", content = "Thanks!"),
    ]


# ── Parametrized rendering test ─────────────────────────────────────


@pytest.mark.parametrize(
    "family,model_dir,expected_marker",
    _CANDIDATES,
)
def test_tool_history_renders_tool_calls_block(
    family: str, model_dir: Path, expected_marker: str
):
    """For each locally-cached candidate MLX model that advertises tool
    support, assert that an assistant-only-tool-calls turn survives
    round-tripping through ``_extract_content_parts(preserve_tool_history=True)``
    AND rendering via the tokenizer's chat template. The rendered
    output must contain the family-specific tool-call marker.
    """
    tokenizer = _load_tokenizer(model_dir)
    if tokenizer is None:
        pytest.skip(
            f"{family}: model dir not present at {model_dir} "
            f"(Chunk E respects the no-download rule)"
        )
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip(f"{family}: tokenizer has no chat_template")

    # Build conversation and run it through the same extractor the
    # production route uses.
    msgs = _build_tool_call_history()
    _sys, chat_messages, _img = _extract_content_parts(
        msgs, preserve_tool_history = True
    )

    # Sanity check: the extractor kept the tool_calls block on the
    # assistant turn (this is the bit Chunk C specifically worried
    # about — if the extractor drops tool_calls when content is None,
    # no template can render what isn't there).
    assistant_entries = [m for m in chat_messages if m.get("role") == "assistant"]
    assert assistant_entries, "extractor dropped the assistant turn"
    assert assistant_entries[0].get("tool_calls"), (
        "extractor dropped tool_calls on assistant turn with content=None"
    )

    # Tools schema that the model was notionally prompted with — pass
    # through to apply_chat_template so templates that require
    # ``tools=...`` to activate their tool-call branch do so.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    # Don't swallow exceptions here — if the template raises, that's a
    # rendering-contract failure worth reporting (or xfail-tracking, per
    # _CANDIDATES marks). The exception-surfacing path lets pytest's
    # xfail mechanism catch AssertionError AND template-raised errors.
    rendered = tokenizer.apply_chat_template(
        chat_messages,
        tools = tools,
        tokenize = False,
        add_generation_prompt = False,
    )

    assert isinstance(rendered, str)
    assert rendered, f"{family}: rendered prompt is empty"
    # The core assertion: the family-specific tool-call marker must
    # appear somewhere in the rendered prompt. If it doesn't, the
    # template silently dropped the tool_calls block — that's the
    # failure mode Chunk C warned about.
    assert expected_marker in rendered, (
        f"{family}: tool-call marker {expected_marker!r} missing from "
        f"rendered prompt. Either the template doesn't recognize the "
        f"tool_calls field on assistant turns, or the extractor shape "
        f"doesn't match what the template expects. Rendered prompt "
        f"(first 1000 chars):\n{rendered[:1000]}"
    )
    # Secondary: the function name should appear verbatim (every
    # in-scope template emits it).
    assert "get_weather" in rendered, (
        f"{family}: function name missing from rendered prompt"
    )


def test_extract_preserves_tool_calls_with_none_content():
    """Fast sanity check with no tokenizer dependency: the extractor
    must retain ``tool_calls`` on the assistant turn even when
    ``content`` is None (not just empty string). Chunk C flagged this
    as the risk path."""
    msgs = _build_tool_call_history()
    _sys, chat_messages, _img = _extract_content_parts(
        msgs, preserve_tool_history = True
    )
    assistant = next(m for m in chat_messages if m.get("role") == "assistant")
    assert assistant.get("tool_calls"), (
        "tool_calls lost when content is None"
    )
    tc = assistant["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert "Paris" in tc["function"]["arguments"]
