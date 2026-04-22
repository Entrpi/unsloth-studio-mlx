# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 — end-to-end VLM tool-calling against Gemma-4 E4B.

Peers ``test_mlx_gemma_tool_calling.py`` but exercises the VLM
backend's new :meth:`MlxVlmBackend.generate_chat_completion_with_tools`
instead of the MLX-LM method. Gemma-4 E4B is a Gemma-4 VLM — Studio's
backend detector routes it through :class:`MlxVlmBackend`, so the
tool-calling path must work on that backend for the user's real
workflow to function.

The same contract as ``test_mlx_gemma_tool_calling.py``: drive the
agentic loop with a ``get_weather`` schema and assert the model emits
a parseable tool call with ``tool_name == "get_weather"`` and
``"Paris"`` in the arguments.

Gated on:
- Darwin / arm64 + ``mlx_vlm`` importable.
- ``gemma-4-e4b-it-4bit`` present locally.
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
    and importlib.util.find_spec("mlx_vlm") is not None
)


@pytest.mark.skipif(not _PLATFORM_OK, reason = "mlx_vlm not available")
@pytest.mark.skipif(
    not _MODEL_PATH.is_dir(),
    reason = (
        f"Gemma-4 E4B checkpoint not present at {_MODEL_PATH} — "
        f"see docs/chunk-h2-matrix/downloads.md for the fetch command"
    ),
)
def test_gemma_e4b_vlm_tool_loop_exercises_code_path(monkeypatch):
    """Load Gemma-4 E4B through :class:`MlxVlmBackend`, drive
    :meth:`generate_chat_completion_with_tools` with a ``get_weather``
    schema, and assert the agentic loop parses the model's Gemma-dialect
    tool call.

    Mirrors the MlxLm contract test (``test_gemma_e4b_tool_loop_exercises_code_path``)
    but through the VLM backend.
    """
    from core.inference.mlx_vlm import MlxVlmBackend

    # Stub the tool executor so the test is hermetic.
    def _fake_execute_tool(name, arguments, **kwargs):  # noqa: D401
        return '{"temperature": 18, "conditions": "cloudy"}'

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", _fake_execute_tool
    )

    backend = MlxVlmBackend()
    assert backend.is_loaded is False

    ok = backend.load_model(
        local_path = str(_MODEL_PATH),
        model_identifier = _MODEL_PATH.name,
    )
    assert ok is True, (
        "MlxVlmBackend.load_model returned False for Gemma-4 E4B"
    )
    assert backend.is_loaded is True

    # Precondition: Gemma-4's chat template advertises tool support.
    assert backend.supports_tools is True, (
        "Gemma-4 E4B VLM chat template should advertise tools/tool_calls; "
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
    assert types, (
        "generate_chat_completion_with_tools emitted no events"
    )
    assert "metadata" in types, f"no metadata event, types={types}"
    metadata_events = [
        e for e in events
        if isinstance(e, dict) and e.get("type") == "metadata"
    ]
    assert metadata_events, "no metadata events found"
    final_meta = metadata_events[-1]
    usage = final_meta.get("usage", {})
    assert usage.get("completion_tokens", 0) >= 1, (
        f"Gemma-4 E4B VLM: completion_tokens was "
        f"{usage.get('completion_tokens', 0)} — model didn't generate "
        f"anything in the tool-loop turn"
    )
    assert types.count("tool_start") == types.count("tool_end"), (
        f"tool_start/tool_end imbalance: {types}"
    )

    # ── Real signal — parser recognised Gemma-4's dialect ──
    # With the B3 closure the parser handles Gemma-4's
    # <|tool_call>call:...<tool_call|> idiom; the route must surface
    # a tool_start event for the ``get_weather`` call.
    tool_starts = [
        e for e in events
        if isinstance(e, dict) and e.get("type") == "tool_start"
    ]
    assert tool_starts, (
        "expected at least one tool_start event; Gemma-4 E4B VLM "
        f"emitted prose only. event types={types}"
    )
    tc = tool_starts[0]
    assert tc.get("tool_name") == "get_weather", (
        f"unexpected tool_name: {tc.get('tool_name')!r}"
    )
    args = tc.get("arguments", "")
    args_str = args if isinstance(args, str) else str(args)
    assert "Paris" in args_str, (
        f"tool_call arguments missing 'Paris': {args_str!r}"
    )
