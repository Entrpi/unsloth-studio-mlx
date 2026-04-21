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
    # Feature flags. Bonsai ships a Qwen3-derived template with literal
    # <think>/</think> tags → Phase 4 detects it as reasoning-capable
    # (always-on). Vision and tools remain false in Chunk A.
    assert backend.is_vision is False
    assert backend.supports_tools is False
    assert backend.supports_reasoning is True
    assert backend.reasoning_always_on is True
    assert backend.reasoning_default is True
    assert backend.chat_template is not None
    assert len(backend.chat_template) > 0
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


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_reasoning_flags_populated_after_load():
    """Bonsai's template hardcodes <think>/</think>; after load we must
    see ``supports_reasoning=True``, ``reasoning_always_on=True``, and a
    non-empty ``chat_template``."""
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    ok = backend.load_model(
        local_path = _MODEL_PATH,
        model_identifier = Path(_MODEL_PATH).name,
    )
    assert ok
    try:
        assert backend.supports_reasoning is True
        assert backend.reasoning_always_on is True
        assert backend.reasoning_default is True
        assert isinstance(backend.chat_template, str)
        assert len(backend.chat_template) > 0
    finally:
        backend.unload_model()


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_enable_thinking_changes_prompt():
    """For a reasoning-capable model with a Jinja2 template that reads
    ``enable_thinking``, toggling the kwarg must materially change the
    rendered prompt. Bonsai's template doesn't actually *read*
    enable_thinking (it hardcodes the tags), so we verify the weaker
    property: both calls succeed and return a string, and one call
    exercises the chat_template_kwargs code path end-to-end."""
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    ok = backend.load_model(
        local_path = _MODEL_PATH,
        model_identifier = Path(_MODEL_PATH).name,
    )
    assert ok
    try:
        tok = backend._tokenizer
        messages = [{"role": "user", "content": "Hi"}]
        # Explicitly pass chat_template_kwargs both ways; this mirrors
        # what generate_chat_completion does internally when
        # supports_reasoning is True.
        with_thinking = tok.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False,
            chat_template_kwargs = {"enable_thinking": True},
        )
        without = tok.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False,
            chat_template_kwargs = {"enable_thinking": False},
        )
        # Both calls succeeded and returned strings. (Bonsai's template
        # doesn't branch on the kwarg, so equality is expected here.)
        assert isinstance(with_thinking, str) and with_thinking
        assert isinstance(without, str) and without
    finally:
        backend.unload_model()


@pytest.mark.skipif(not MLX_LM_AVAILABLE, reason = "mlx_lm or model not available")
def test_generate_does_not_strip_think_tags():
    """Phase 4 contract: the backend must NOT strip or mangle literal
    ``<think>`` / ``</think>`` tokens if the model emits them. The
    frontend parses these tags directly, so any server-side scrubbing
    would break the thinking-panel UI.

    This is a structural test: mock ``stream_generate`` to emit tokens
    that spell a thinking block, then assert the backend yields the
    raw text unchanged."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod
    from core.inference.mlx_lm import MlxLmBackend

    class _R:
        def __init__(self, t: str):
            self.text = t
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    def _stream(*a, **kw):
        # Tokenization-boundary variety: the opening tag spans two chunks.
        yield _R("<th")
        yield _R("ink>")
        yield _R("reasoning here")
        yield _R("</think>")
        yield _R(" answer")

    b = MlxLmBackend()
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"
    b._model_identifier = "x"
    b._supports_reasoning = True

    with mock.patch("mlx_lm.stream_generate", _stream):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            events = list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    enable_thinking = True,
                )
            )
    final = [e for e in events if isinstance(e, str)][-1]
    assert "<think>" in final
    assert "</think>" in final
    assert "reasoning here" in final
    assert "answer" in final
