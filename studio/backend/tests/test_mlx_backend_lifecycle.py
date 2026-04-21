# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration test for the MLX backend against a real local checkpoint.

Gated behind availability of ``mlx_lm`` and the target model directory.
Runs on Apple Silicon macOS only. A different model may be swapped in
via the ``MLX_TEST_MODEL_PATH`` env var.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path

import pytest

_DEFAULT_MODEL = "/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit"
_MODEL_PATH = os.environ.get("MLX_TEST_MODEL_PATH", _DEFAULT_MODEL)

MLX_LM_AVAILABLE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_lm") is not None
    and Path(_MODEL_PATH).is_dir()
)


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_load_unload_roundtrip():
    """Smoke: load the Bonsai MLX checkpoint, inspect state, unload."""
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    assert backend.is_loaded is False

    ok = backend.load_model(
        local_path = _MODEL_PATH,
        model_identifier = Path(_MODEL_PATH).name,
    )
    assert ok is True
    assert backend.is_loaded is True
    assert backend.is_active is True
    assert backend.model_identifier == Path(_MODEL_PATH).name
    # max_position_embeddings must be populated from config.json
    assert backend.context_length is not None
    assert backend.context_length > 0
    assert backend.native_context_length == backend.context_length
    # Feature flags: nothing but plain text in Phase 1.
    assert backend.is_vision is False
    assert backend.supports_tools is False
    assert backend.supports_reasoning is False
    assert backend.cache_type_kv is None

    assert backend.unload_model() is True
    assert backend.is_loaded is False
    assert backend.model_identifier is None


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_load_chat_unload_bonsai():
    """End-to-end: load, stream tokens, capture metadata, unload."""
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    ok = backend.load_model(
        local_path = _MODEL_PATH,
        model_identifier = Path(_MODEL_PATH).name,
    )
    assert ok and backend.is_loaded

    try:
        messages = [
            {"role": "user", "content": "Reply with exactly one word: ok."}
        ]
        cumulative_last: str = ""
        saw_metadata = False
        for event in backend.generate_chat_completion(
            messages = messages,
            max_tokens = 32,
        ):
            if isinstance(event, dict):
                # Final metadata event
                saw_metadata = True
                assert event.get("type") == "metadata"
                usage = event.get("usage", {})
                assert usage.get("completion_tokens", 0) >= 1
                assert usage.get("prompt_tokens", 0) >= 1
            else:
                # Cumulative text contract: each yield is the full text so far.
                assert isinstance(event, str)
                assert event.startswith(cumulative_last) or cumulative_last == ""
                cumulative_last = event

        assert saw_metadata, "generate must emit a final metadata dict"
        assert len(cumulative_last) > 0
    finally:
        assert backend.unload_model()
        assert not backend.is_loaded


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_stop_string_end_to_end():
    """With ``stop=["END"]``, the backend MUST truncate before the literal
    ``END`` and never emit it (nor anything after)."""
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    ok = backend.load_model(
        local_path = _MODEL_PATH,
        model_identifier = Path(_MODEL_PATH).name,
    )
    assert ok and backend.is_loaded

    try:
        messages = [
            {
                "role": "user",
                "content": (
                    "Respond with exactly this, nothing else: ok END extra"
                ),
            }
        ]
        final_text = ""
        saw_metadata = False
        for event in backend.generate_chat_completion(
            messages = messages,
            max_tokens = 64,
            stop = ["END"],
        ):
            if isinstance(event, dict):
                saw_metadata = True
                assert event.get("type") == "metadata"
                assert event.get("finish_reason") == "stop"
            else:
                final_text = event
        assert saw_metadata
        # Core contract of Phase 2 stop strings: the literal stop sequence
        # must be absent from the emitted response.
        assert "END" not in final_text, (
            f"stop string leaked into output: {final_text!r}"
        )
    finally:
        assert backend.unload_model()


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_repetition_penalty_changes_output():
    """Two greedy (temp=0) runs with identical prompts but different
    repetition_penalty values must produce different outputs — the
    penalty is actually taking effect, not silently dropped.

    Greedy sampling makes the run deterministic so any divergence is
    attributable to the logits processor."""
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    ok = backend.load_model(
        local_path = _MODEL_PATH,
        model_identifier = Path(_MODEL_PATH).name,
    )
    assert ok and backend.is_loaded

    messages = [
        {
            "role": "user",
            "content": (
                "Write a short paragraph about the color blue. "
                "Use the word 'blue' several times."
            ),
        }
    ]

    def _run(rep: float) -> str:
        out = ""
        for event in backend.generate_chat_completion(
            messages = messages,
            temperature = 0.0,  # greedy → deterministic per-processor
            top_p = 1.0,
            top_k = 0,
            min_p = 0.0,
            repetition_penalty = rep,
            max_tokens = 96,
        ):
            if not isinstance(event, dict):
                out = event
        return out

    try:
        baseline = _run(1.0)
        penalized = _run(1.3)
        assert baseline, "empty baseline output"
        assert penalized, "empty penalized output"
        assert baseline != penalized, (
            "repetition_penalty=1.3 produced identical output to 1.0 — "
            "logits processor is not taking effect"
        )
    finally:
        assert backend.unload_model()
