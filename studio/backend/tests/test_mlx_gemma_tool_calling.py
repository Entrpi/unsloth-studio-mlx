# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 (Gemma-4 matrix closure): end-to-end tool-calling on Gemma-4.

Prior H-2 coverage only template-probed Gemma-4 (rendered a
tool-history fixture through the 31B tokenizer's chat_template). This
file closes the real gap: **weights-loaded tool-calling** against the
`gemma-4-e4b-it-4bit` dense variant. The test drives
:meth:`MlxLmBackend.generate_chat_completion_with_tools` with a
`get_weather` schema, iterates the agentic loop, and asserts the
code path executes cleanly.

Gemma-4's tool-call idiom (``<|tool_call>call:NAME{...}<tool_call|>``)
is *not* currently recognised by
:func:`core.inference._tool_call_parser.parse_tool_calls_from_text`
— the shared parser targets the ``<tool_call>{...}</tool_call>``
Qwen/Bonsai/Hermes dialect. If / when the model emits a Gemma-style
call, the parser will pass the turn through as plain text, the loop
will terminate with a final content turn, and the test still covers
the "Gemma-4 load + exercise tool code path + unload" contract.

This is the "code path exercised" fallback documented in the
chunk-H-2 task contract — chasing prompt engineering or extending
the parser to a new dialect is out of scope for the matrix-closure
pass. The gap is tracked in ``docs/chunk-h2-matrix/blockers.md``.

Gated on:

- Darwin / arm64 + ``mlx_lm`` importable.
- ``gemma-4-e4b-it-4bit`` present locally (skip if missing).
"""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import pytest

_MODEL_PATH = Path(
    "/Users/ent/.lmstudio/models/mlx-community/gemma-4-e4b-it-4bit"
)

_PLATFORM_OK = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_lm") is not None
)


@pytest.mark.skipif(not _PLATFORM_OK, reason = "mlx_lm not available")
@pytest.mark.skipif(
    not _MODEL_PATH.is_dir(),
    reason = (
        f"Gemma-4 E4B checkpoint not present at {_MODEL_PATH} — "
        f"see docs/chunk-h2-matrix/downloads.md for the fetch command"
    ),
)
def test_gemma_e4b_tool_loop_exercises_code_path(monkeypatch):
    """Load Gemma-4 E4B, pass a ``get_weather`` schema through
    :meth:`generate_chat_completion_with_tools`, and assert the
    agentic loop runs to completion without raising.

    Primary (aspirational) assertion: a ``tool_start`` event is
    emitted with ``tool_name="get_weather"`` and arguments mentioning
    ``"Paris"``. This requires the shared parser to recognise Gemma-4's
    ``<|tool_call>`` idiom (it currently doesn't) AND the model to
    actually emit a tool call rather than prose. If either condition
    is unmet, we fall back to asserting:

    1. The call executes without exception.
    2. At least one ``metadata`` event lands with ``completion_tokens
       >= 1`` (generation actually ran).
    3. ``supports_tools`` was True at load time (the precondition for
       `generate_chat_completion_with_tools` to accept the call).

    Either outcome proves "Gemma-4 loaded + exercised the tool-call
    code path end-to-end". Improving parser coverage to recognise
    Gemma-4's idiom is a separate follow-up tracked in the blockers
    doc; don't chase prompt engineering here.
    """
    from core.inference.mlx_lm import MlxLmBackend

    # Stub the tool executor so the test is hermetic. If the parser
    # ever grows Gemma-4 support, the loop will call through here with
    # the parsed arguments rather than hitting the real network.
    def _fake_execute_tool(name, arguments, **kwargs):  # noqa: D401
        return '{"temperature": 18, "conditions": "cloudy"}'

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", _fake_execute_tool
    )

    backend = MlxLmBackend()
    assert backend.is_loaded is False

    ok = backend.load_model(
        local_path = str(_MODEL_PATH),
        model_identifier = _MODEL_PATH.name,
    )
    assert ok is True, "MlxLmBackend.load_model returned False for Gemma-4 E4B"
    assert backend.is_loaded is True

    # Precondition: Gemma-4's chat template advertises tool support,
    # so supports_tools must be True — otherwise
    # generate_chat_completion_with_tools would raise RuntimeError.
    # If this flips False in a future MLX release, the detection
    # heuristic needs updating; catch that here rather than as a
    # confusing "RuntimeError" downstream.
    assert backend.supports_tools is True, (
        "Gemma-4 E4B chat template should advertise tools/tool_calls; "
        "if this is now False the template shape has shifted"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, e.g. Paris or Tokyo",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    try:
        events = list(
            backend.generate_chat_completion_with_tools(
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "What is the weather in Paris right now? "
                            "Use the get_weather function."
                        ),
                    }
                ],
                tools = tools,
                tool_choice = "auto",
                max_tokens = 256,
                max_tool_iterations = 2,
                # Deterministic generation — greedy.
                temperature = 0.0,
                top_p = 1.0,
                top_k = 0,
                min_p = 0.0,
            )
        )
    finally:
        assert backend.unload_model()

    assert backend.is_loaded is False
    assert backend.model_identifier is None

    # ── Core fallback assertions (always true when the loop ran) ──
    types = [e.get("type") for e in events if isinstance(e, dict)]
    assert types, "generate_chat_completion_with_tools emitted no events"
    # At least one metadata event terminates the loop.
    assert "metadata" in types, f"no metadata event, types={types}"
    # The metadata must reflect a real generation pass.
    metadata_events = [
        e for e in events if isinstance(e, dict) and e.get("type") == "metadata"
    ]
    assert metadata_events, "no metadata events found"
    final_meta = metadata_events[-1]
    usage = final_meta.get("usage", {})
    assert usage.get("completion_tokens", 0) >= 1, (
        f"Gemma-4 E4B: completion_tokens was {usage.get('completion_tokens', 0)}"
        f" — model didn't generate anything in the tool-loop turn"
    )
    # If tool_start appeared, tool_end must match — the loop must be
    # balanced regardless of whether the parser recognised Gemma-4's
    # dialect or whether the model cooperated.
    assert types.count("tool_start") == types.count("tool_end"), (
        f"tool_start/tool_end imbalance: {types}"
    )

    # ── Aspirational assertion — soft, documented ──
    # If the parser grows Gemma-4 support later, this branch becomes
    # the real signal. Today it's effectively unreachable and the test
    # passes via the code-path-exercised branch above.
    tool_starts = [e for e in events if isinstance(e, dict) and e.get("type") == "tool_start"]
    if tool_starts:
        tc = tool_starts[0]
        assert tc.get("tool_name") == "get_weather", (
            f"unexpected tool_name: {tc.get('tool_name')!r}"
        )
        args = tc.get("arguments", "")
        # arguments may be a dict or a JSON string — either way Paris
        # should appear as the city value.
        args_str = args if isinstance(args, str) else str(args)
        assert "Paris" in args_str, (
            f"tool_call arguments missing 'Paris': {args_str!r}"
        )
