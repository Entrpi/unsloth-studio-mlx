# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the MLX-LM backend skeleton.

These tests do not require ``mlx_lm`` to be installed — the module is
lazy-imported, so constructing a backend and poking its properties must
work on any platform.
"""

from __future__ import annotations

import sys
import threading

import pytest


def _fresh_backend():
    """Return a freshly-constructed ``MlxLmBackend``.

    Imports at call time so a single test can reload the module for the
    "importable without mlx_lm" case.
    """
    from core.inference.mlx_lm import MlxLmBackend

    return MlxLmBackend()


def test_import_does_not_load_mlx_lm() -> None:
    """Importing the module must not pull in ``mlx_lm``."""
    # Ensure ``mlx_lm`` is not in sys.modules just because we imported ours.
    from core.inference import mlx_lm as _mod  # noqa: F401

    # We cannot assert ``mlx_lm`` is absent from sys.modules — another
    # test module may have imported it. Instead, assert that the backend
    # module itself has no ``mlx_lm`` attribute at top level.
    assert not hasattr(_mod, "mlx_lm_module")
    assert not hasattr(_mod, "mx")


def test_backend_not_loaded_at_init() -> None:
    b = _fresh_backend()
    assert b.is_loaded is False
    assert b.is_active is False
    assert b.model_identifier is None


def test_backend_property_defaults() -> None:
    b = _fresh_backend()
    assert b.is_vision is False
    assert b.supports_tools is False
    assert b.supports_reasoning is False
    assert b.reasoning_always_on is False
    # Default matches GGUF's LlamaCppBackend.__init__ at llama_cpp.py:111.
    # When the UI sees ``supports_reasoning=False`` it hides the toggle, so
    # this value is only consulted when a reasoning-capable model is
    # loaded. Start it at True so a newly-detected reasoning model with
    # default semantics ("thinking on") doesn't need to flip a flag.
    assert b.reasoning_default is True
    assert b.hf_variant is None
    assert b.context_length is None
    assert b.max_context_length is None
    assert b.native_context_length is None
    assert b.chat_template is None
    assert b.cache_type_kv is None
    assert b.speculative_type is None
    assert b.detect_audio_type() is None
    assert b.load_progress() is None


def test_backend_has_internal_lock() -> None:
    b = _fresh_backend()
    # Lock type-check: acquiring and releasing must work.
    with b._lock:
        pass


def test_backend_generate_raises_when_cold() -> None:
    b = _fresh_backend()
    with pytest.raises(RuntimeError, match = "not loaded"):
        gen = b.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
        )
        next(gen)


def test_backend_generate_rejects_image() -> None:
    b = _fresh_backend()
    with pytest.raises(ValueError, match = "image"):
        gen = b.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            image_b64 = "abc",
        )
        next(gen)


def test_backend_load_model_rejects_bad_path() -> None:
    """On the platform, load_model should reject a non-existent dir
    with a RuntimeError (not NotImplementedError, not silent False)."""
    import platform as _platform

    b = _fresh_backend()
    if (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
    ):
        with pytest.raises(RuntimeError, match = "not a directory"):
            b.load_model(
                local_path = "/nonexistent/path/to/mlx",
                model_identifier = "dummy",
            )
    else:
        # Off-platform: load_model raises before it even inspects the path.
        with pytest.raises(RuntimeError, match = "not available on this platform"):
            b.load_model(
                local_path = "/nonexistent/path/to/mlx",
                model_identifier = "dummy",
            )


def test_backend_unload_when_cold_returns_false() -> None:
    """Unloading a backend that was never loaded is a no-op that returns False."""
    b = _fresh_backend()
    assert b.unload_model() is False


# ── Phase 2: sampling fidelity ──────────────────────────────────────

def _mlx_lm_available() -> bool:
    import importlib.util
    import platform as _platform

    return (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )


_requires_mlx = pytest.mark.skipif(
    not _mlx_lm_available(), reason = "mlx_lm not importable on this platform"
)


@_requires_mlx
def test_build_sampler_no_processors_for_defaults() -> None:
    """At the Studio defaults (repetition=1.0, presence=0, freq=0) the
    helper returns an empty logits-processor list: nothing to run
    per-token. This is the fast path and is load-bearing for 47 tok/s."""
    from core.inference.mlx_lm import _build_mlx_sampler_and_processors

    sampler, processors = _build_mlx_sampler_and_processors(
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.01,
        repetition_penalty = 1.0,
        presence_penalty = 0.0,
        frequency_penalty = 0.0,
    )
    assert sampler is not None
    assert processors == []


@_requires_mlx
def test_build_sampler_adds_repetition_when_above_1() -> None:
    """repetition_penalty > 1.0 must actually wire a processor."""
    from core.inference.mlx_lm import _build_mlx_sampler_and_processors

    _, processors = _build_mlx_sampler_and_processors(
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.01,
        repetition_penalty = 1.15,
        presence_penalty = 0.0,
        frequency_penalty = 0.0,
    )
    assert len(processors) == 1


@_requires_mlx
def test_build_sampler_repetition_1_is_noop() -> None:
    """repetition_penalty == 1.0 must be treated as disabled (no processor)."""
    from core.inference.mlx_lm import _build_mlx_sampler_and_processors

    _, processors = _build_mlx_sampler_and_processors(
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.01,
        repetition_penalty = 1.0,
    )
    assert processors == []


@_requires_mlx
def test_build_sampler_presence_and_frequency_add_processors() -> None:
    """Non-zero presence/frequency penalties add one processor each."""
    from core.inference.mlx_lm import _build_mlx_sampler_and_processors

    _, processors = _build_mlx_sampler_and_processors(
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.01,
        repetition_penalty = 1.0,
        presence_penalty = 0.5,
        frequency_penalty = 0.3,
    )
    # upstream make_logits_processors adds one callable per non-zero penalty
    assert len(processors) == 2


@_requires_mlx
def test_build_sampler_logit_bias_adds_processor() -> None:
    """logit_bias dict must become a processor."""
    from core.inference.mlx_lm import _build_mlx_sampler_and_processors

    _, processors = _build_mlx_sampler_and_processors(
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.01,
        repetition_penalty = 1.0,
        logit_bias = {1: 0.5, 2: -0.5},
    )
    assert len(processors) == 1


@_requires_mlx
def test_build_sampler_temperature_0_is_greedy() -> None:
    """At temperature=0, upstream make_sampler returns an argmax sampler.
    We verify by constructing deterministic logits and confirming the
    sampler picks the argmax index every time (no randomness)."""
    import mlx.core as mx

    from core.inference.mlx_lm import _build_mlx_sampler_and_processors

    sampler, _ = _build_mlx_sampler_and_processors(
        temperature = 0.0,
        top_p = 0.95,
        top_k = 20,
        min_p = 0.01,
        repetition_penalty = 1.0,
    )
    # (batch=1, vocab=5) — argmax is index 3.
    logits = mx.array([[0.1, 0.2, 0.3, 5.0, 0.5]])
    out = sampler(logits)
    # shape depends on upstream; argmax over axis=-1 gives scalar per batch.
    assert int(out[0]) == 3


@_requires_mlx
def test_stop_string_truncates_cumulative() -> None:
    """With a mocked ``mlx_lm.stream_generate`` emitting tokens that spell
    ``hello END world``, the backend must truncate at ``END`` and yield
    the truncated cumulative once before the metadata event."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    class _FakeResp:
        def __init__(self, text: str, pt: int = 3, gt: int = 1) -> None:
            self.text = text
            self.prompt_tokens = pt
            self.generation_tokens = gt
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    # Tokens intentionally chosen so "END" appears mid-stream and the
    # tokenization boundary crosses through "E" + "ND" to exercise the
    # tail-scan logic.
    def _fake_stream(model, tokenizer, **kw):
        yield _FakeResp("hello ")
        yield _FakeResp("E")
        yield _FakeResp("ND")
        yield _FakeResp(" world")

    b = _fresh_backend()
    # Bypass load_model — plant the required state directly.
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"
    b._model_identifier = "test"
    b._context_length = 8192

    with mock.patch("mlx_lm.stream_generate", _fake_stream):
        # Short-circuit the sampler helper too — it requires a real
        # mlx_lm install and we're testing the stop-string path here,
        # not sampling construction.
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            events = list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    stop = ["END"],
                )
            )

    text_events = [e for e in events if isinstance(e, str)]
    meta_events = [e for e in events if isinstance(e, dict)]

    assert text_events, "expected at least one cumulative text yield"
    # The final cumulative yield must be at "hello " and must NOT contain "END".
    assert text_events[-1] == "hello "
    assert "END" not in text_events[-1]
    assert len(meta_events) == 1
    assert meta_events[0]["type"] == "metadata"
    # We truncate-and-break normally, so finish_reason should be the stop default.
    assert meta_events[0].get("finish_reason") == "stop"


@_requires_mlx
def test_stop_string_list_accepts_string_form() -> None:
    """OpenAI's schema allows ``stop`` as a single string. Exercise that path."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    class _FakeResp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    def _fake_stream(model, tokenizer, **kw):
        yield _FakeResp("alpha")
        yield _FakeResp(" beta ")
        yield _FakeResp("STOP")
        yield _FakeResp(" gamma")

    b = _fresh_backend()
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "p"
    b._model_identifier = "t"

    with mock.patch("mlx_lm.stream_generate", _fake_stream):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            events = list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    stop = "STOP",
                )
            )
    text_events = [e for e in events if isinstance(e, str)]
    assert text_events[-1] == "alpha beta "
    assert "STOP" not in text_events[-1]
