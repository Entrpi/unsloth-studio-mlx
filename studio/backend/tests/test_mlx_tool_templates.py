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
    # string, and ToolCall.model_dump faithfully preserves that.
    #
    # Chunk F (F1): ``_extract_content_parts(preserve_tool_history=True)``
    # now opportunistically ``json.loads`` the ``arguments`` string into
    # a dict so Qwen3.5's ``|items`` filter works. Bonsai/Gemma handle
    # both forms so the upgrade is safe across the native-iteration
    # templates. This flipped the two Qwen3.5 xfails green.
    pytest.param(
        "qwen3.5-4b",
        _LMSTUDIO_ROOT / "mlx-community" / "Qwen3.5-4B-MLX-4bit",
        "<tool_call>",
        id = "qwen3.5-4b",
    ),
    pytest.param(
        "qwen3.5-35b-a3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Qwen3.5-35B-A3B-4bit",
        "<tool_call>",
        id = "qwen3.5-35b-a3b",
    ),
    # Gemma 4 — different idiom: ``<|tool_call>call:NAME{...}<tool_call|>``.
    # Template handles both string and dict forms of arguments.
    pytest.param(
        "gemma-4-31b",
        _LMSTUDIO_ROOT / "mlx-community" / "gemma-4-31b-it-4bit",
        "<|tool_call>",
        id = "gemma-4-31b",
    ),
    # Gemma 4 E4B — same template family / marker as 31B. Added on the
    # Gemma-4 matrix-closure pass so both dense variants are locked in
    # for tool-template round-trip regression protection. Inherits the
    # native-iteration branch (template uses ``message['tool_calls']``).
    pytest.param(
        "gemma-4-e4b",
        _LMSTUDIO_ROOT / "mlx-community" / "gemma-4-e4b-it-4bit",
        "<|tool_call>",
        id = "gemma-4-e4b",
    ),
    # Bonsai (2bit) — already the baseline Chunk C tested, include here
    # for regression-protection parity across the whole candidate set.
    pytest.param(
        "bonsai-1.7b",
        _LMSTUDIO_ROOT / "prism-ml" / "Ternary-Bonsai-1.7B-mlx-2bit",
        "<tool_call>",
        id = "bonsai-1.7b",
    ),
    # Chunk H-2 (H2-1): non-Qwen / non-Gemma tool-template families.
    #
    # Hermes-3-Llama-3.2-3B — Nous Research Hermes chat-ML style.
    # Its chat_template is minimal (just <|im_start|>/<|im_end|> role
    # wrapping) and does NOT iterate ``message.tool_calls``. Chunk F1's
    # content-synthesis branch fires: the extractor injects
    # ``<tool_call>...</tool_call>`` JSON into the assistant turn's
    # content so the raw text carries the call. Marker: ``<tool_call>``.
    pytest.param(
        "hermes-3-3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Hermes-3-Llama-3.2-3B-bf16",
        "<tool_call>",
        id = "hermes-3-3b",
    ),
    # Llama-3.2-3B-Instruct — Meta ipython / JSON dialect.
    # The template DOES iterate ``message.tool_calls`` natively (camp
    # (a) in F1 parlance); it emits the tool call as inline
    # ``{"name": "...", "parameters": {...}}`` JSON inside an
    # ``<|start_header_id|>assistant<|end_header_id|>`` block. There is
    # no ``<tool_call>`` literal — the JSON object itself IS the call.
    # Marker: the function-name signature the template bakes in.
    pytest.param(
        "llama-3.2-3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Llama-3.2-3B-Instruct-4bit",
        '{"name": "get_weather"',
        id = "llama-3.2-3b",
    ),
    # Ministral-3-3B-Instruct — Mistral 2512 dialect.
    # Uses ``[AVAILABLE_TOOLS]`` / ``[TOOL_CALLS]`` / ``[TOOL_RESULTS]``.
    # The template DOES iterate ``message.tool_calls`` (camp (a)), but
    # it ALSO executes ``message['content'] | length > 0`` eagerly on
    # the assistant branch even when ``tool_calls`` is present.
    #
    # Chunk H-2 (B1) closure: the extractor's native-iteration branch
    # now coerces ``content=None`` to ``content=""`` when tool_calls is
    # populated, so the Ministral template's ``content|length`` call
    # hits an empty string (safe) instead of None (TypeError). The
    # 4-turn fixture (user→assistant-tc→tool→user) also tripped
    # Ministral's strict alternation check, so the shared fixture is
    # now 3-turn — the canonical OpenAI tool-call round-trip where the
    # next turn would be the final assistant response. Every other
    # family renders identically against the 3-turn shape.
    pytest.param(
        "ministral-3-3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Ministral-3-3B-Instruct-2512-4bit",
        "[TOOL_CALLS]",
        id = "ministral-3-3b",
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

    Chunk H-2 (B1): previously this fixture included a trailing
    ``user="Thanks!"`` turn, but Ministral-3's chat template enforces
    strict user/assistant alternation on content-bearing turns (tool
    results and assistant-only-tool-calls are exempt from the
    alternation count), which means ``user → assistant-tc → tool → user``
    trips its alternation check. The 3-turn shape used here is the
    canonical OpenAI tool-call round-trip — the NEXT thing generation
    should produce is the final assistant response — and it renders
    cleanly across every family in _CANDIDATES including Ministral.
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
    # production route uses. Chunk F (F1): pass the tokenizer's
    # chat_template so the extractor routes between native ``tool_calls``
    # and synthesised ``<tool_call>`` content the same way the route
    # does in production.
    msgs = _build_tool_call_history()
    _sys, chat_messages, _img = _extract_content_parts(
        msgs,
        preserve_tool_history = True,
        chat_template = getattr(tokenizer, "chat_template", None),
    )

    # Sanity check: the extractor kept SOME representation of the tool
    # call on the assistant turn — either as a native ``tool_calls``
    # list (templates that iterate it) or as synthesised ``<tool_call>``
    # content (templates that don't). Chunk C worried specifically
    # about silent drops when ``content`` is None, and Chunk F (F1)
    # added the second branch.
    assistant_entries = [m for m in chat_messages if m.get("role") == "assistant"]
    assert assistant_entries, "extractor dropped the assistant turn"
    _a = assistant_entries[0]
    assert _a.get("tool_calls") or "<tool_call>" in (_a.get("content") or ""), (
        "extractor dropped both tool_calls and synthesised content on "
        "assistant turn with content=None"
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
    as the risk path. No ``chat_template`` passed, so the extractor
    defaults to the native-kwarg shape (camp (a)) — the historical
    behaviour."""
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
    # Chunk F (F1): arguments is now opportunistically decoded into a
    # dict. The resulting shape is dict-or-string; either form must
    # still mention "Paris".
    args = tc["function"]["arguments"]
    if isinstance(args, dict):
        assert args.get("city") == "Paris"
    else:
        assert "Paris" in args


def test_content_synthesis_roundtrips_through_tool_call_parser():
    """Chunk F (F1): templates that don't iterate ``message.tool_calls``
    get ``<tool_call>`` JSON content synthesised into the assistant
    turn. The synthesised block MUST round-trip through the shared
    parser so reconstruction is lossless — callers that later scrape
    ``parse_tool_calls_from_text`` over the prompt recover the same
    ``ToolCall`` shape the extractor fed in."""
    from core.inference._tool_call_parser import parse_tool_calls_from_text

    msgs = _build_tool_call_history()
    # Pass a chat_template that does NOT reference message.tool_calls —
    # the Hermes/Ministral camp — so the extractor takes the content-
    # synthesis branch.
    minimal_template = (
        "{% for message in messages %}"
        "{{ message.role }}:{{ message.content }}\n"
        "{% endfor %}"
    )
    _sys, chat_messages, _img = _extract_content_parts(
        msgs,
        preserve_tool_history = True,
        chat_template = minimal_template,
    )
    assistant = next(m for m in chat_messages if m.get("role") == "assistant")
    # Synthesis branch: no native tool_calls, content carries the block.
    assert "tool_calls" not in assistant, (
        "template doesn't iterate tool_calls — extractor should NOT "
        "pass them natively"
    )
    synth_content = assistant["content"]
    assert "<tool_call>" in synth_content and "</tool_call>" in synth_content

    # Round-trip: parse the synthesised markup back.
    parsed = parse_tool_calls_from_text(synth_content)
    assert len(parsed) == 1, f"expected 1 parsed call, got {len(parsed)}"
    p = parsed[0]
    assert p["function"]["name"] == "get_weather"
    # Parser normalises arguments back to a JSON string; parse that.
    import json as _json
    args = _json.loads(p["function"]["arguments"])
    assert args == {"city": "Paris"}


def test_template_iterates_tool_calls_heuristic():
    """The template-iterates-tool_calls heuristic is a plain substring
    scan but the exact idioms it recognises are load-bearing for F1's
    routing. Lock the expected matches here so a future refactor
    can't silently regress the split between native-kwarg and content-
    synthesis camps."""
    from routes.inference import _template_iterates_tool_calls

    # Attribute access — Qwen3.5 / Bonsai style.
    assert _template_iterates_tool_calls(
        "{% for tc in message.tool_calls %}...{% endfor %}"
    )
    # Bracket-string access — some Llama-3.1 / Gemma-4 variants.
    assert _template_iterates_tool_calls(
        "{% if message['tool_calls'] %}...{% endif %}"
    )
    assert _template_iterates_tool_calls(
        '{% if message["tool_calls"] %}...{% endif %}'
    )
    # Empty / None → False (safer default).
    assert not _template_iterates_tool_calls(None)
    assert not _template_iterates_tool_calls("")
    # Template without tool_calls at all → False.
    assert not _template_iterates_tool_calls(
        "{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}"
    )
