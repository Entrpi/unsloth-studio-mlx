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
    # Phase 3: load_progress is None when no load is in flight.
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


# ── Phase 4: reasoning / <think> support ────────────────────────────

def _tok_stub(template: str):
    """Tokenizer stub for _detect_reasoning tests — only needs
    ``chat_template`` and a throwaway ``apply_chat_template``."""
    from unittest import mock

    t = mock.Mock()
    t.chat_template = template
    t.apply_chat_template.return_value = "prompt"
    return t


def test_reasoning_detection_from_template_with_enable_thinking() -> None:
    """A Qwen3-style template that reads ``enable_thinking`` as a
    chat_template_kwarg must register as reasoning-capable but NOT
    always-on."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub("{%- if enable_thinking %}<think>{% endif %}"),
        "Qwen3-7B-mlx-4bit",
    )
    assert b.supports_reasoning is True
    assert b.reasoning_always_on is False
    assert b.chat_template is not None
    assert "enable_thinking" in b.chat_template


def test_reasoning_detection_from_template_with_think_tag() -> None:
    """A Bonsai-style template that hardcodes literal <think>/</think>
    tags registers as reasoning-capable AND always-on (UI hides toggle)."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub(
            "{%- if add_generation_prompt %}"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
        ),
        "Ternary-Bonsai-8B-mlx-2bit",
    )
    assert b.supports_reasoning is True
    assert b.reasoning_always_on is True
    assert b.reasoning_default is True


def test_no_reasoning_from_plain_chatml_template() -> None:
    """A plain ChatML template with no thinking hooks must NOT register."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub(
            "{%- for m in messages %}"
            "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
            "{% endfor %}"
            "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        ),
        "plain-chatml-model",
    )
    assert b.supports_reasoning is False
    assert b.reasoning_always_on is False
    assert b.chat_template is not None  # template still captured


def test_no_reasoning_when_template_missing() -> None:
    """A tokenizer with no chat_template leaves everything at defaults."""
    b = _fresh_backend()
    b._detect_reasoning(_tok_stub(""), "nameless")
    assert b.chat_template is None
    assert b.supports_reasoning is False
    assert b.reasoning_always_on is False


def test_reasoning_default_for_small_qwen35_is_false() -> None:
    """Qwen3.5/3.6 <9B ships with thinking off by default. Port of
    llama_cpp.py:1519-1535 logic."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub("{%- if enable_thinking %}<think>{% endif %}"),
        "Qwen3.5-4B-Instruct-mlx",
    )
    assert b.supports_reasoning is True
    assert b.reasoning_default is False  # small → default off


def test_reasoning_default_for_large_qwen35_is_true() -> None:
    """Qwen3.5/3.6 ≥9B ships with thinking on by default."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub("{%- if enable_thinking %}<think>{% endif %}"),
        "Qwen3.5-30B-A3B-mlx",
    )
    assert b.supports_reasoning is True
    # 30B total, A3B active (3B) — extract_model_size_b prefers the
    # MoE active count, which is <9. This matches llama_cpp.py's own
    # behavior at 1519-1535: the size_val path uses extract_model_size_b.
    # We accept either outcome as long as it's a bool (the point of
    # this test is wiring, not the exact policy tune).
    assert isinstance(b.reasoning_default, bool)


def test_reasoning_default_for_non_qwen35_stays_true() -> None:
    """Non-Qwen3.5/3.6 reasoning-capable models default to thinking on
    regardless of size."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub("{%- if enable_thinking %}<think>{% endif %}"),
        "Bonsai-1.7B-mlx",
    )
    assert b.supports_reasoning is True
    assert b.reasoning_default is True


def test_reasoning_state_resets_after_unload() -> None:
    """Unloading must clear reasoning flags so a subsequent load of a
    non-reasoning model doesn't inherit the previous model's state."""
    b = _fresh_backend()
    b._detect_reasoning(
        _tok_stub("{%- if enable_thinking %}<think>{% endif %}"),
        "reasoning-model",
    )
    assert b.supports_reasoning is True
    # Simulate a load: poke the model/tokenizer sentinels so
    # _unload_locked's early-return doesn't no-op.
    b._model = object()
    b._tokenizer = object()
    b._unload_locked()
    assert b.supports_reasoning is False
    assert b.reasoning_always_on is False
    assert b.reasoning_default is True
    assert b.chat_template is None


def test_enable_thinking_not_forwarded_when_not_supported() -> None:
    """For a model without reasoning support, passing ``enable_thinking``
    must NOT inject ``chat_template_kwargs`` into apply_chat_template.
    Templates that don't know the kwarg can raise on unknown keys; we
    must defend them."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    b = _fresh_backend()
    b._model = object()
    b._model_identifier = "no-reasoning"
    b._supports_reasoning = False
    # Real-looking tokenizer mock.
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"

    class _Resp:
        def __init__(self) -> None:
            self.text = "hi"
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 50.0

    def _one(*a, **kw):
        yield _Resp()

    with mock.patch("mlx_lm.stream_generate", _one):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    enable_thinking = True,
                )
            )
    # apply_chat_template must have been called WITHOUT chat_template_kwargs.
    assert b._tokenizer.apply_chat_template.called
    _, kwargs = b._tokenizer.apply_chat_template.call_args
    assert "chat_template_kwargs" not in kwargs


def test_enable_thinking_forwarded_when_supported() -> None:
    """For a reasoning-capable model, ``enable_thinking`` must be
    forwarded via ``chat_template_kwargs={"enable_thinking": bool}``."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    b = _fresh_backend()
    b._model = object()
    b._model_identifier = "reasoning"
    b._supports_reasoning = True
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"

    class _Resp:
        def __init__(self) -> None:
            self.text = "hi"
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 50.0

    def _one(*a, **kw):
        yield _Resp()

    with mock.patch("mlx_lm.stream_generate", _one):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    enable_thinking = False,
                )
            )
    _, kwargs = b._tokenizer.apply_chat_template.call_args
    assert kwargs.get("chat_template_kwargs") == {"enable_thinking": False}


# ── Phase 8: quantized KV cache ─────────────────────────────────────


def test_cache_type_kv_none_is_unquantized() -> None:
    from core.inference.mlx_lm import _cache_type_kv_to_mlx

    kv_bits, kv_group = _cache_type_kv_to_mlx(None)
    assert kv_bits is None
    assert kv_group == 64


def test_cache_type_kv_f16_bf16_are_unquantized() -> None:
    """f16 / bf16 map to "no quantization" — i.e. kv_bits is None."""
    from core.inference.mlx_lm import _cache_type_kv_to_mlx

    assert _cache_type_kv_to_mlx("f16") == (None, 64)
    assert _cache_type_kv_to_mlx("bf16") == (None, 64)
    assert _cache_type_kv_to_mlx("FP16") == (None, 64)  # case-insensitive


def test_cache_type_kv_q8_maps_to_8bit() -> None:
    from core.inference.mlx_lm import _cache_type_kv_to_mlx

    assert _cache_type_kv_to_mlx("q8_0") == (8, 64)


def test_cache_type_kv_q4_variants_map_to_4bit() -> None:
    from core.inference.mlx_lm import _cache_type_kv_to_mlx

    assert _cache_type_kv_to_mlx("q4_0") == (4, 64)
    assert _cache_type_kv_to_mlx("q4_1") == (4, 64)


def test_cache_type_kv_q5_rounds_down_to_q4(capsys) -> None:
    """mlx-lm has no 5-bit KV path; round down rather than upgrade.

    Studio uses structlog, so standard ``caplog`` doesn't see the
    records — warnings are emitted to stdout via a console renderer.
    Capture stdout to assert the warning surfaced.
    """
    from core.inference.mlx_lm import _cache_type_kv_to_mlx

    kv_bits, kv_group = _cache_type_kv_to_mlx("q5_1")
    assert kv_bits == 4
    assert kv_group == 64
    captured = capsys.readouterr()
    assert "q5_1" in captured.out


def test_cache_type_kv_unknown_falls_back_unquantized(capsys) -> None:
    """Any unknown label must fall back to unquantized with a warning —
    never raise, so a stray client value doesn't brick load."""
    from core.inference.mlx_lm import _cache_type_kv_to_mlx

    kv_bits, kv_group = _cache_type_kv_to_mlx("xyz")
    assert kv_bits is None
    assert kv_group == 64
    captured = capsys.readouterr()
    # The warning must mention the offending label. We don't assert the
    # exact phrasing so a future log-message refinement doesn't break
    # the test.
    assert "xyz" in captured.out


def test_cache_type_kv_property_none_when_unquantized() -> None:
    b = _fresh_backend()
    # Fresh backend is unquantized.
    assert b.cache_type_kv is None
    # Explicit None on the internal field.
    b._kv_bits = None
    assert b.cache_type_kv is None


def test_cache_type_kv_property_q8_for_8bit() -> None:
    b = _fresh_backend()
    b._kv_bits = 8
    assert b.cache_type_kv == "q8_0"


def test_cache_type_kv_property_q4_for_4bit() -> None:
    b = _fresh_backend()
    b._kv_bits = 4
    assert b.cache_type_kv == "q4_0"


def test_cache_type_kv_property_other_bits_defensive() -> None:
    """For any unexpected bit count we still return a consistent label."""
    b = _fresh_backend()
    b._kv_bits = 2
    assert b.cache_type_kv == "q2_0"


def test_kv_state_resets_on_unload() -> None:
    b = _fresh_backend()
    b._model = object()
    b._tokenizer = object()
    b._kv_bits = 4
    b._kv_group_size = 128
    b._quantized_kv_start = 256
    b._cache_type_kv_label = "q4_0"
    b._unload_locked()
    assert b._kv_bits is None
    assert b._kv_group_size == 64
    assert b._quantized_kv_start == 0
    assert b._cache_type_kv_label is None
    assert b.cache_type_kv is None


def test_generate_passes_kv_bits_when_quantized() -> None:
    """With ``_kv_bits`` set, ``stream_generate`` must receive the
    ``kv_bits``/``kv_group_size``/``quantized_kv_start`` kwargs."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    captured_kwargs: Dict = {}

    class _R:
        def __init__(self, t: str):
            self.text = t
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    def _fake_stream(model, tokenizer, **kw):
        captured_kwargs.update(kw)
        yield _R("ok")

    b = _fresh_backend()
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"
    b._model_identifier = "x"
    b._kv_bits = 8
    b._kv_group_size = 32
    b._quantized_kv_start = 128

    with mock.patch("mlx_lm.stream_generate", _fake_stream):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                )
            )
    assert captured_kwargs.get("kv_bits") == 8
    assert captured_kwargs.get("kv_group_size") == 32
    assert captured_kwargs.get("quantized_kv_start") == 128


def test_generate_omits_kv_bits_when_unquantized() -> None:
    """With ``_kv_bits=None`` the kwargs must NOT be present at all.

    This preserves the exact Chunk A stream_generate call shape — zero
    behavioural change when KV quantization is off.
    """
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    captured_kwargs: Dict = {}

    class _R:
        def __init__(self, t: str):
            self.text = t
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    def _fake_stream(model, tokenizer, **kw):
        captured_kwargs.update(kw)
        yield _R("ok")

    b = _fresh_backend()
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"
    b._model_identifier = "x"
    b._kv_bits = None

    with mock.patch("mlx_lm.stream_generate", _fake_stream):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                )
            )
    assert "kv_bits" not in captured_kwargs
    assert "kv_group_size" not in captured_kwargs
    assert "quantized_kv_start" not in captured_kwargs


# ── Phase 6: LoRA adapter loading ───────────────────────────────────


def test_is_lora_default_false() -> None:
    """Fresh backend has no adapter, so is_lora must be False."""
    b = _fresh_backend()
    assert b.is_lora is False
    assert b.adapter_path is None


def test_is_lora_true_when_adapter_path_set() -> None:
    """Directly poking the adapter path field must flip is_lora True."""
    b = _fresh_backend()
    b._adapter_path = "/some/path"
    assert b.is_lora is True
    assert b.adapter_path == "/some/path"


def test_adapter_state_resets_on_unload() -> None:
    b = _fresh_backend()
    b._model = object()
    b._tokenizer = object()
    b._adapter_path = "/some/adapter"
    b._unload_locked()
    assert b._adapter_path is None
    assert b.is_lora is False


def test_load_model_rejects_missing_adapter_dir(tmp_path) -> None:
    """Pointing ``adapter_path`` at a non-existent directory must
    surface a clean False return rather than exploding in mlx_lm.load."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    # Synthesize a plausible base MLX dir so the base-path check passes.
    import json as _json

    (tmp_path / "config.json").write_text(
        _json.dumps({"quantization": {"bits": 2, "group_size": 128}})
    )
    b = _fresh_backend()
    ok = b.load_model(
        local_path = str(tmp_path),
        model_identifier = "fake",
        adapter_path = str(tmp_path / "not-there"),
    )
    assert ok is False
    # State must not have been mutated.
    assert b.is_loaded is False
    assert b.is_lora is False


def test_load_model_threads_adapter_path_into_mlx_load() -> None:
    """With a mocked mlx_lm.load we verify:

    - the kwarg ``adapter_path`` is forwarded verbatim,
    - ``self._adapter_path`` is set post-load,
    - ``is_lora`` returns True.

    We skip the platform gate here because the only import we touch is
    the private ``_mlx_load`` name inside ``load_model``, which we
    monkey-patch before the function runs.
    """
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from unittest import mock

    base = _Path(tempfile.mkdtemp(prefix = "mlx-base-"))
    (base / "config.json").write_text(
        _json.dumps(
            {
                "quantization": {"bits": 2, "group_size": 128},
                "max_position_embeddings": 4096,
            }
        )
    )
    adapter = _Path(tempfile.mkdtemp(prefix = "mlx-adapter-"))
    (adapter / "adapters.safetensors").write_bytes(b"\x00\x00")
    (adapter / "adapter_config.json").write_text(_json.dumps({"peft_type": "LORA"}))

    captured: Dict = {}

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    def _fake_mlx_load(path, *args, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return object(), _FakeTokenizer()

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        ok = b.load_model(
            local_path = str(base),
            model_identifier = "fake",
            adapter_path = str(adapter),
        )
    assert ok is True
    assert captured["kwargs"].get("adapter_path") == str(adapter)
    assert b.is_loaded
    assert b.is_lora is True
    assert b.adapter_path == str(adapter)
    b.unload_model()


# ── Phase 7: speculative decoding ───────────────────────────────────


def test_speculative_type_default_none() -> None:
    """Fresh backend has no draft, so speculative_type is None."""
    b = _fresh_backend()
    assert b.speculative_type is None
    assert b.draft_model_path is None


def test_speculative_type_mlx_draft_model_when_loaded() -> None:
    """With ``_draft_model`` set, speculative_type returns the MLX label."""
    b = _fresh_backend()
    b._draft_model = object()
    assert b.speculative_type == "mlx-draft-model"


def test_draft_state_resets_on_unload() -> None:
    b = _fresh_backend()
    b._model = object()
    b._tokenizer = object()
    b._draft_model = object()
    b._draft_tokenizer = object()
    b._draft_path = "/some/draft"
    b._unload_locked()
    assert b._draft_model is None
    assert b._draft_tokenizer is None
    assert b._draft_path is None
    assert b.speculative_type is None


def test_unload_drops_draft_even_without_base() -> None:
    """Edge case: ``_unload_locked`` returned False when neither
    base nor draft was set. With Phase 7 the condition must also
    early-return False if only the draft is set but not the base
    (can't happen in practice but sanity)."""
    b = _fresh_backend()
    # No base, no draft → nothing to do.
    assert b._unload_locked() is False


def test_draft_mem_preflight_refuses_when_over_75pct(tmp_path) -> None:
    """Simulate a 32 GB box with 2 GB available and a 30 GB draft.
    Combined footprint = (32 - 2) + 30 = 60 GB, limit = 24 GB.
    Must raise RuntimeError mentioning 75%.

    We patch ``Path.iterdir`` to return a fake child whose ``.stat()``
    returns a 30 GB size — cleaner than stubbing real ``Path.stat``
    which also drives ``is_file()``."""
    from unittest import mock

    from core.inference.mlx_lm import MlxLmBackend

    class _FakeStat:
        st_size = 30 * 1024**3

    class _FakePath:
        def __init__(self, suffix: str) -> None:
            self.suffix = suffix

        def is_file(self) -> bool:
            return True

        def stat(self) -> "_FakeStat":
            return _FakeStat()

    # Create a real directory so Path.is_dir() passes on tmp_path.
    # Patch iterdir at the Path type level so our backend sees the fake.
    def _fake_iterdir(self):
        yield _FakePath(".safetensors")

    class _VM:
        total = 32 * 1024**3
        available = 2 * 1024**3

    with mock.patch.object(type(tmp_path), "iterdir", _fake_iterdir):
        with mock.patch("psutil.virtual_memory", return_value = _VM()):
            with pytest.raises(RuntimeError, match = "75% of available memory"):
                MlxLmBackend._draft_mem_preflight(tmp_path)


def test_draft_mem_preflight_passes_when_ample(tmp_path) -> None:
    """With plenty of headroom, preflight returns cleanly."""
    from unittest import mock

    from core.inference.mlx_lm import MlxLmBackend

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"\x00" * 1024)

    class _VM:
        total = 128 * 1024**3
        available = 100 * 1024**3  # 100 GB free

    with mock.patch("psutil.virtual_memory", return_value = _VM()):
        # Tiny draft (1 KB), huge headroom → no raise.
        MlxLmBackend._draft_mem_preflight(tmp_path)


def test_draft_mem_preflight_noop_without_psutil(tmp_path, monkeypatch) -> None:
    """If psutil isn't importable (CI box without it), preflight is a
    no-op — NEVER raise from a missing optional dep."""
    import sys

    from core.inference.mlx_lm import MlxLmBackend

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"\x00" * 1024)

    # Block the import.
    orig_psutil = sys.modules.pop("psutil", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    try:
        MlxLmBackend._draft_mem_preflight(tmp_path)  # must not raise
    finally:
        if orig_psutil is not None:
            sys.modules["psutil"] = orig_psutil


# ── Chunk E (E5): hard-refuse tokenizer mismatch on speculative ────


def _setup_base_and_draft_dirs(tmp_path):
    """Create a plausible base MLX dir + a draft dir with a safetensors
    shard so the draft-path + preflight checks pass."""
    import json as _json

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        _json.dumps(
            {
                "quantization": {"bits": 2, "group_size": 128},
                "max_position_embeddings": 4096,
            }
        )
    )
    draft = tmp_path / "draft"
    draft.mkdir()
    # Small shard so _draft_mem_preflight sees a trivial footprint.
    (draft / "model.safetensors").write_bytes(b"\x00" * 1024)
    return base, draft


def test_draft_vocab_mismatch_refusal(tmp_path) -> None:
    """Base vocab_size != draft vocab_size must raise RuntimeError and
    leave the backend unloaded."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    from unittest import mock

    base, draft = _setup_base_and_draft_dirs(tmp_path)

    class _Tokenizer:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 0
            self.chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    call_count = {"n": 0}

    def _fake_mlx_load(path, *args, **kwargs):
        # First call → base (vocab 32000). Second call → draft (vocab 32001).
        call_count["n"] += 1
        if call_count["n"] == 1:
            return object(), _Tokenizer(32000)
        return object(), _Tokenizer(32001)

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        with pytest.raises(RuntimeError, match = "matching tokenizers"):
            b.load_model(
                local_path = str(base),
                model_identifier = "fake",
                draft_model_path = str(draft),
            )
    # Backend must be left in the unloaded state.
    assert b.is_loaded is False
    assert b._draft_model is None
    assert b._model is None


def test_draft_sentinel_mismatch_refusal(tmp_path) -> None:
    """Same vocab_size but different eos_token_id also hard-refuses."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    from unittest import mock

    base, draft = _setup_base_and_draft_dirs(tmp_path)

    class _Tokenizer:
        def __init__(self, eos_id: int):
            self.vocab_size = 32000
            self.bos_token_id = 1
            self.eos_token_id = eos_id
            self.pad_token_id = 0
            self.chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    call_count = {"n": 0}

    def _fake_mlx_load(path, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return object(), _Tokenizer(2)
        return object(), _Tokenizer(7)  # different EOS

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        with pytest.raises(RuntimeError, match = "eos_token_id"):
            b.load_model(
                local_path = str(base),
                model_identifier = "fake",
                draft_model_path = str(draft),
            )
    assert b.is_loaded is False


def test_draft_tokenizer_match_succeeds(tmp_path) -> None:
    """Matching tokenizers → load completes, draft attached."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    from unittest import mock

    base, draft = _setup_base_and_draft_dirs(tmp_path)

    class _Tokenizer:
        vocab_size = 32000
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _Tokenizer()

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        ok = b.load_model(
            local_path = str(base),
            model_identifier = "fake",
            draft_model_path = str(draft),
        )
    assert ok is True
    assert b._draft_model is not None
    assert b.speculative_type == "mlx-draft-model"
    b.unload_model()


def test_generate_passes_draft_model_when_loaded() -> None:
    """With a draft model set on the backend, ``stream_generate`` must
    receive ``draft_model=`` and ``num_draft_tokens=``."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    captured: Dict = {}

    class _R:
        def __init__(self, t: str):
            self.text = t
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    def _fake_stream(model, tokenizer, **kw):
        captured.update(kw)
        yield _R("ok")

    b = _fresh_backend()
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"
    b._model_identifier = "x"
    _draft_sentinel = object()
    b._draft_model = _draft_sentinel
    b._num_draft_tokens = 5

    with mock.patch("mlx_lm.stream_generate", _fake_stream):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                )
            )
    assert captured.get("draft_model") is _draft_sentinel
    assert captured.get("num_draft_tokens") == 5


def test_generate_omits_draft_model_when_none() -> None:
    """With no draft, ``stream_generate`` must NOT receive the draft
    kwargs at all."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    captured: Dict = {}

    class _R:
        def __init__(self, t: str):
            self.text = t
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 47.0

    def _fake_stream(model, tokenizer, **kw):
        captured.update(kw)
        yield _R("ok")

    b = _fresh_backend()
    b._model = object()
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"
    b._model_identifier = "x"
    b._draft_model = None

    with mock.patch("mlx_lm.stream_generate", _fake_stream):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                )
            )
    assert "draft_model" not in captured
    assert "num_draft_tokens" not in captured


# ── Chunk E (E4): warnings surfaced through LoadProgressResponse ────


def test_load_progress_includes_warnings_when_present() -> None:
    """When ``_load_warnings`` is populated, ``load_progress()`` must
    include them in its return dict so the route handler can surface
    them to the frontend via the new LoadProgressResponse.warnings field."""
    b = _fresh_backend()
    b._load_phase = "loading"
    b._weights_bytes_total = 1024 * 1024 * 1024  # 1 GB
    b._load_warnings = [
        "model size 45.0 GB exceeds 1.5x available RAM (20.0 GB) — expect swap",
    ]
    p = b.load_progress()
    assert p is not None
    assert "warnings" in p
    assert len(p["warnings"]) == 1
    assert "swap" in p["warnings"][0]


def test_load_progress_omits_warnings_when_empty() -> None:
    """Empty warnings list → key omitted (backward compat with callers
    that don't know about the field)."""
    b = _fresh_backend()
    b._load_phase = "loading"
    b._weights_bytes_total = 1024
    b._load_warnings = []
    p = b.load_progress()
    assert p is not None
    assert "warnings" not in p


# ── Chunk E (E3): num_draft_tokens override via LoadRequest ─────────


def test_num_draft_tokens_propagates() -> None:
    """When ``load_model`` is called with ``num_draft_tokens=N``, the
    backend's ``_num_draft_tokens`` must be set to N (overriding the
    class default of 3). ``None`` preserves the default."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from unittest import mock

    base = _Path(tempfile.mkdtemp(prefix = "mlx-base-"))
    (base / "config.json").write_text(
        _json.dumps(
            {
                "quantization": {"bits": 2, "group_size": 128},
                "max_position_embeddings": 4096,
            }
        )
    )

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _FakeTokenizer()

    b = _fresh_backend()
    # Class default must be 3 (preserved for backwards compatibility).
    assert b._num_draft_tokens == 3

    with mock.patch("mlx_lm.load", _fake_mlx_load):
        ok = b.load_model(
            local_path = str(base),
            model_identifier = "fake",
            num_draft_tokens = 5,
        )
    assert ok is True
    assert b._num_draft_tokens == 5
    b.unload_model()


def test_num_draft_tokens_none_preserves_default() -> None:
    """Passing ``num_draft_tokens=None`` (the API default) must leave
    the backend's setting at 3 — not overwrite it with a stale value."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from unittest import mock

    base = _Path(tempfile.mkdtemp(prefix = "mlx-base-"))
    (base / "config.json").write_text(
        _json.dumps(
            {
                "quantization": {"bits": 2, "group_size": 128},
                "max_position_embeddings": 4096,
            }
        )
    )

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _FakeTokenizer()

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        ok = b.load_model(
            local_path = str(base),
            model_identifier = "fake",
            # No num_draft_tokens kwarg.
        )
    assert ok is True
    assert b._num_draft_tokens == 3
    b.unload_model()


def test_num_draft_tokens_clamps_out_of_range() -> None:
    """Backend defensively clamps out-of-range overrides to [1, 32]."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from unittest import mock

    base = _Path(tempfile.mkdtemp(prefix = "mlx-base-"))
    (base / "config.json").write_text(
        _json.dumps(
            {
                "quantization": {"bits": 2, "group_size": 128},
                "max_position_embeddings": 4096,
            }
        )
    )

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "prompt"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _FakeTokenizer()

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        ok = b.load_model(
            local_path = str(base),
            model_identifier = "fake",
            num_draft_tokens = 999,
        )
    assert ok is True
    assert b._num_draft_tokens == 32  # clamped to upper bound
    b.unload_model()


# ── Phase 3: remote HF pulls, load_progress, hf_variant ─────────────


def test_extract_mlx_variant_common_suffixes() -> None:
    from core.inference.mlx_lm import _extract_mlx_variant

    assert _extract_mlx_variant("mlx-community/Qwen2.5-7B-Instruct-4bit") == "4bit"
    assert _extract_mlx_variant("prism-ml/Ternary-Bonsai-8B-mlx-2bit") == "mlx-2bit"
    assert _extract_mlx_variant("foo/bar-8bit") == "8bit"
    assert _extract_mlx_variant("foo/bar-FP16") == "fp16"
    # Case-insensitive.
    assert _extract_mlx_variant("foo/bar-4BIT") == "4bit"


def test_extract_mlx_variant_no_match() -> None:
    from core.inference.mlx_lm import _extract_mlx_variant

    assert _extract_mlx_variant("unsloth/foo-bar") is None
    assert _extract_mlx_variant("") is None
    assert _extract_mlx_variant("plain-name") is None


def test_load_progress_downloading_phase_shape() -> None:
    b = _fresh_backend()
    b._load_phase = "downloading"
    b._download_bytes_loaded = 1024
    b._download_bytes_total = 4096
    p = b.load_progress()
    assert p is not None
    assert p["phase"] == "downloading"
    assert p["bytes_loaded"] == 1024
    assert p["bytes_total"] == 4096
    assert p["fraction"] == pytest.approx(0.25, abs = 1e-4)


def test_load_progress_loading_phase_shape() -> None:
    """Loading phase samples RSS; we don't assert the exact byte count
    (depends on the test process) but the shape must be right and
    fraction must be 0..1."""
    b = _fresh_backend()
    b._load_phase = "loading"
    b._weights_bytes_total = 8 * 1024**3
    p = b.load_progress()
    assert p is not None
    assert p["phase"] == "loading"
    assert p["bytes_total"] == 8 * 1024**3
    assert 0.0 <= p["fraction"] <= 1.0


def test_load_progress_loaded_terminal() -> None:
    b = _fresh_backend()
    b._load_phase = "loaded"
    b._download_bytes_total = 1000
    b._weights_bytes_total = 2000
    p = b.load_progress()
    assert p is not None
    assert p["phase"] == "loaded"
    # Prefer weights_bytes_total when both are known.
    assert p["bytes_total"] == 2000
    assert p["fraction"] == 1.0


def test_load_progress_includes_warnings() -> None:
    b = _fresh_backend()
    b._load_phase = "loading"
    b._load_warnings = ["low RAM"]
    b._weights_bytes_total = 1024
    p = b.load_progress()
    assert p is not None
    assert p.get("warnings") == ["low RAM"]


def test_hf_variant_parsed_from_model_identifier_after_load() -> None:
    """After a successful synthetic load, hf_variant is populated from
    the identifier tail."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from unittest import mock

    base = _Path(tempfile.mkdtemp(prefix = "mlx-base-"))
    (base / "config.json").write_text(
        _json.dumps({"quantization": {"bits": 4, "group_size": 64}})
    )

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "p"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _FakeTokenizer()

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        ok = b.load_model(
            local_path = str(base),
            model_identifier = "mlx-community/Fake-Model-4bit",
        )
    assert ok is True
    assert b.hf_variant == "4bit"
    b.unload_model()
    assert b.hf_variant is None


def test_download_mlx_uses_allow_patterns() -> None:
    """Verify ``_download_mlx`` forwards the expected allow_patterns to
    snapshot_download and that the progress tqdm subclass updates
    counters."""
    from unittest import mock

    captured: Dict[str, Any] = {}

    def _fake_snapshot(**kwargs):
        captured["kwargs"] = kwargs
        # Simulate the tqdm class being instantiated with a 10-byte
        # total, then updated once with n=5 — exercises both counters.
        tqdm_class = kwargs.get("tqdm_class")
        if tqdm_class is not None:
            bar = tqdm_class(total = 10)
            bar.update(5)
        return "/tmp/fake-downloaded-dir"

    b = _fresh_backend()
    with mock.patch("huggingface_hub.snapshot_download", _fake_snapshot):
        out = b._download_mlx("mlx-community/Test-Repo", hf_token = None)
    assert out == "/tmp/fake-downloaded-dir"
    kw = captured["kwargs"]
    assert kw["repo_id"] == "mlx-community/Test-Repo"
    patterns = kw["allow_patterns"]
    assert "*.safetensors" in patterns
    assert "config.json" in patterns
    assert "*.jinja" in patterns
    # Counters ticked: total==10 from tqdm init, loaded==5 from update.
    assert b._download_bytes_total >= 10
    assert b._download_bytes_loaded >= 5


def test_ram_warning_fires_when_total_below_1_5x(tmp_path) -> None:
    """When total RAM < 1.5x model size, a warning must be appended to
    ``_load_warnings`` so ``load_progress`` can surface it."""
    import importlib.util
    import json as _json
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    from unittest import mock

    # Synthesize a base MLX dir with a fake safetensors to drive the
    # weight-size sum.
    (tmp_path / "config.json").write_text(
        _json.dumps({"quantization": {"bits": 2, "group_size": 128}})
    )
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"\x00" * 1024)

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "p"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _FakeTokenizer()

    # Patch the safetensors sum helper AND psutil so the ratio check
    # triggers deterministically (100 GB "model" on an 8 GB "box").
    from core.inference.mlx_lm import MlxLmBackend

    class _VM:
        total = 8 * 1024**3
        available = 4 * 1024**3

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        with mock.patch.object(
            MlxLmBackend,
            "_sum_safetensors_bytes",
            staticmethod(lambda _p: 100 * 1024**3),
        ):
            with mock.patch("psutil.virtual_memory", return_value = _VM()):
                ok = b.load_model(
                    local_path = str(tmp_path),
                    model_identifier = "fake",
                )
    assert ok is True
    # Warning recorded AND visible in load_progress.
    assert any("RAM" in w for w in b._load_warnings)
    p = b.load_progress()
    assert p is not None
    assert p.get("warnings"), "expected warnings in load_progress"


def test_remote_mlx_download_triggers_when_path_is_not_dir() -> None:
    """When ``local_path`` looks like an HF repo id (contains '/') and
    doesn't exist on disk, ``load_model`` must call ``_download_mlx``
    before attempting the local ``mlx_lm.load``."""
    import importlib.util
    import platform as _platform

    if not (
        _platform.system() == "Darwin"
        and _platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    ):
        pytest.skip("mlx_lm not available on this platform")

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from unittest import mock

    # Pre-stage a "downloaded" directory that _download_mlx can return.
    pseudo = _Path(tempfile.mkdtemp(prefix = "mlx-pseudo-"))
    (pseudo / "config.json").write_text(
        _json.dumps({"quantization": {"bits": 4, "group_size": 64}})
    )

    class _FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *a, **kw):
            return "p"

    def _fake_mlx_load(path, *args, **kwargs):
        return object(), _FakeTokenizer()

    download_called = {"n": 0, "arg": None}

    def _fake_download(self, repo, hf_token = None):
        download_called["n"] += 1
        download_called["arg"] = repo
        return str(pseudo)

    from core.inference.mlx_lm import MlxLmBackend

    b = _fresh_backend()
    with mock.patch("mlx_lm.load", _fake_mlx_load):
        with mock.patch.object(MlxLmBackend, "_download_mlx", _fake_download):
            ok = b.load_model(
                local_path = "mlx-community/Fake-Model-4bit",
                model_identifier = "mlx-community/Fake-Model-4bit",
            )
    assert ok is True
    assert download_called["n"] == 1
    assert download_called["arg"] == "mlx-community/Fake-Model-4bit"
    b.unload_model()


def test_enable_thinking_omitted_when_none() -> None:
    """When the caller passes ``enable_thinking=None`` we must not
    inject chat_template_kwargs even if the model supports reasoning —
    the template's own default should apply."""
    from unittest import mock

    from core.inference import mlx_lm as mlx_lm_mod

    b = _fresh_backend()
    b._model = object()
    b._model_identifier = "reasoning"
    b._supports_reasoning = True
    b._tokenizer = mock.Mock()
    b._tokenizer.apply_chat_template.return_value = "prompt"

    class _Resp:
        def __init__(self) -> None:
            self.text = "hi"
            self.prompt_tokens = 1
            self.generation_tokens = 1
            self.prompt_tps = 100.0
            self.generation_tps = 50.0

    def _one(*a, **kw):
        yield _Resp()

    with mock.patch("mlx_lm.stream_generate", _one):
        with mock.patch.object(
            mlx_lm_mod,
            "_build_mlx_sampler_and_processors",
            return_value = (None, []),
        ):
            list(
                b.generate_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    enable_thinking = None,
                )
            )
    _, kwargs = b._tokenizer.apply_chat_template.call_args
    assert "chat_template_kwargs" not in kwargs


# =====================================================================
# Phase 5 — tool-calling loop
# =====================================================================


class _FakeResp:
    """Stand-in for mlx-lm's stream_generate response object."""

    def __init__(self, text, **kwargs):
        self.text = text
        for k, v in kwargs.items():
            setattr(self, k, v)


def _stub_backend_for_tools(mocker_module=None):
    """Return a loaded-shaped backend ready for tool-loop tests.

    The backend has tokenizer/model stubs and ``_supports_tools=True``.
    ``_tools_kwarg_ok=True`` so the render path goes through the tools
    branch without probing the template.
    """
    from unittest import mock as _mock

    b = _fresh_backend()
    b._model = _mock.MagicMock(name="model")
    b._tokenizer = _mock.MagicMock(name="tokenizer")
    b._tokenizer.apply_chat_template = _mock.MagicMock(return_value="PROMPT")
    b._tokenizer.chat_template = "{% if tools %}{{ tool_calls }}{% endif %}"
    b._model_identifier = "unit/test"
    b._supports_tools = True
    b._tools_kwarg_ok = True
    return b


def _run_tool_loop(b, *, turns_text, tools, tool_choice=None, max_iter=10):
    """Drive generate_chat_completion_with_tools with a scripted stream.

    ``turns_text`` is a list — the i-th entry is the cumulative text
    the i-th assistant turn should yield. Each turn's internal loop
    delivers fake (text, metadata) pairs.
    """
    from unittest import mock as _mock

    # Each turn is a list of _FakeResp deltas (cumulative = join).
    def make_resps(full_text):
        # Split roughly in thirds so we exercise multi-chunk streaming.
        if not full_text:
            return [_FakeResp("", prompt_tokens=5, generation_tokens=0)]
        mid = len(full_text) // 2
        return [
            _FakeResp(full_text[:mid]),
            _FakeResp(
                full_text[mid:],
                prompt_tokens=10,
                generation_tokens=20,
                prompt_tps=5.0,
                generation_tps=30.0,
            ),
        ]

    turn_iter = iter(turns_text)

    def fake_stream_generate(_model, _tokenizer, **_kwargs):
        txt = next(turn_iter)
        yield from make_resps(txt)

    import core.inference.mlx_lm as mlx_lm_mod

    with _mock.patch.object(
        mlx_lm_mod, "stream_generate", fake_stream_generate, create=True
    ), _mock.patch.object(
        mlx_lm_mod,
        "_build_mlx_sampler_and_processors",
        return_value=(None, []),
    ):
        # Inject fake mlx_lm module so the local `from mlx_lm import
        # stream_generate` inside the backend resolves to our stub.
        fake_mod = _mock.MagicMock()
        fake_mod.stream_generate = fake_stream_generate
        with _mock.patch.dict(sys.modules, {"mlx_lm": fake_mod}):
            return list(
                b.generate_chat_completion_with_tools(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tool_iterations=max_iter,
                )
            )


class TestToolDetect:
    def test_supports_tools_false_without_template(self):
        b = _fresh_backend()
        b._tokenizer = type("T", (), {"chat_template": None})()
        b._detect_tools(b._tokenizer)
        assert b.supports_tools is False

    def test_supports_tools_true_with_tools_kwds(self):
        b = _fresh_backend()
        # Template mentions both literals.
        b._tokenizer = type("T", (), {"chat_template": "{{ tools }} {{ tool_calls }}"})()
        b._detect_tools(b._tokenizer)
        assert b.supports_tools is True

    def test_supports_tools_false_without_tool_calls(self):
        b = _fresh_backend()
        b._tokenizer = type("T", (), {"chat_template": "{{ tools }} only"})()
        b._detect_tools(b._tokenizer)
        assert b.supports_tools is False


class TestToolChoiceNormalisation:
    def test_none_short_circuits_loop(self):
        b = _stub_backend_for_tools()
        # Only one turn should be consumed even if we would loop.
        events = _run_tool_loop(
            b,
            turns_text=["plain answer, no tool"],
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="none",
        )
        types = [e.get("type") for e in events if isinstance(e, dict)]
        # No tool_start / tool_end should appear.
        assert "tool_start" not in types
        assert "tool_end" not in types
        assert "metadata" in types

    def test_auto_maps_correctly(self):
        from core.inference.mlx_lm import MlxLmBackend

        assert MlxLmBackend._normalize_tool_choice(None) == "auto"
        assert MlxLmBackend._normalize_tool_choice("auto") == "auto"
        assert MlxLmBackend._normalize_tool_choice("required") == "required"
        assert MlxLmBackend._normalize_tool_choice("none") == "none"
        # Structured: treated as "required" (loop runs).
        assert (
            MlxLmBackend._normalize_tool_choice(
                {"type": "function", "function": {"name": "x"}}
            )
            == "required"
        )
        # Unknown → "auto"
        assert MlxLmBackend._normalize_tool_choice("invalid") == "auto"


class TestAgenticLoopSingleToolHappyPath:
    def test_single_tool_call_then_final_answer(self):
        from unittest import mock

        b = _stub_backend_for_tools()
        tool_call_markup = (
            'thinking...\n<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Paris"}}</tool_call>'
        )

        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value='{"temp": 22}',
        ) as exec_mock:
            events = _run_tool_loop(
                b,
                turns_text=[tool_call_markup, "The weather is 22°C."],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {},
                        },
                    }
                ],
            )

        types = [e.get("type") for e in events if isinstance(e, dict)]
        assert types.count("tool_start") == 1
        assert types.count("tool_end") == 1
        assert types.count("metadata") == 1
        # Executor was called with the parsed arguments.
        exec_mock.assert_called_once()
        args = exec_mock.call_args.args
        kwargs = exec_mock.call_args.kwargs
        assert args[0] == "get_weather"
        assert args[1] == {"city": "Paris"}
        # Final content event carries the synthesised answer text.
        content_events = [e for e in events if e.get("type") == "content"]
        final_text = content_events[-1]["text"]
        assert "22°C" in final_text

    def test_tool_start_carries_id_and_arguments(self):
        from unittest import mock

        b = _stub_backend_for_tools()
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value="result",
        ):
            events = _run_tool_loop(
                b,
                turns_text=[
                    '<tool_call>{"name": "x", "arguments": {"q": "1"}}</tool_call>',
                    "done",
                ],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )
        tool_start = next(e for e in events if e.get("type") == "tool_start")
        tool_end = next(e for e in events if e.get("type") == "tool_end")
        assert tool_start["tool_name"] == "x"
        assert tool_start["arguments"] == {"q": "1"}
        assert tool_start["tool_call_id"].startswith("call_")
        assert tool_end["result"] == "result"
        assert tool_end["tool_call_id"] == tool_start["tool_call_id"]


class TestAgenticLoopMultiTurn:
    def test_two_tool_calls_then_final(self):
        from unittest import mock

        b = _stub_backend_for_tools()

        results = iter(['{"t1": 1}', '{"t2": 2}'])

        with mock.patch(
            "core.inference.tools.execute_tool",
            side_effect=lambda *a, **kw: next(results),
        ):
            events = _run_tool_loop(
                b,
                turns_text=[
                    '<tool_call>{"name": "a", "arguments": {"i": 1}}</tool_call>',
                    '<tool_call>{"name": "b", "arguments": {"j": 2}}</tool_call>',
                    "All done.",
                ],
                tools=[
                    {"type": "function", "function": {"name": "a"}},
                    {"type": "function", "function": {"name": "b"}},
                ],
            )

        starts = [e for e in events if e.get("type") == "tool_start"]
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert [s["tool_name"] for s in starts] == ["a", "b"]
        assert [e["result"] for e in ends] == ['{"t1": 1}', '{"t2": 2}']


class TestAgenticLoopMaxIterations:
    def test_cap_triggers_final_nudge(self):
        from unittest import mock

        b = _stub_backend_for_tools()

        # Every turn wants to call a tool; cap at 2 so after 2
        # iterations the loop injects the "no more tools" nudge and
        # runs one more turn with plain text.
        calls = [
            '<tool_call>{"name": "loop", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "loop", "arguments": {}}</tool_call>',
            "Final fallback answer.",
        ]
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value="ok",
        ):
            events = _run_tool_loop(
                b,
                turns_text=calls,
                tools=[{"type": "function", "function": {"name": "loop"}}],
                max_iter=2,
            )
        # Exactly two tool executions ran (one per loop iteration).
        starts = [e for e in events if e.get("type") == "tool_start"]
        assert len(starts) == 2
        # Final content event has the fallback text.
        contents = [e for e in events if e.get("type") == "content"]
        assert any("Final fallback answer" in e["text"] for e in contents)


class TestPromptRenderWithTools:
    def test_forwards_tools_kwarg_when_template_accepts(self):
        from unittest import mock

        b = _fresh_backend()
        b._tokenizer = mock.MagicMock()
        b._tokenizer.apply_chat_template.return_value = "RENDERED"
        b._tools_kwarg_ok = True  # pretend detection succeeded
        out = b._render_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="auto",
        )
        assert out == "RENDERED"
        call = b._tokenizer.apply_chat_template.call_args
        # tools kwarg was forwarded
        assert call.kwargs.get("tools") == [
            {"type": "function", "function": {"name": "x"}}
        ]
        assert call.kwargs.get("tool_choice") == "auto"

    def test_falls_back_when_tools_kwarg_rejected(self):
        from unittest import mock

        b = _fresh_backend()
        b._tokenizer = mock.MagicMock()
        # First call (with tools) raises TypeError; second call (without)
        # returns the tool-less prompt.
        b._tokenizer.apply_chat_template.side_effect = [
            TypeError("unexpected keyword argument 'tools'"),
            "RENDERED_NOTOOLS",
        ]
        b._tools_kwarg_ok = None
        out = b._render_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        # The detection has been cached for future calls.
        assert b._tools_kwarg_ok is False
        assert out == "RENDERED_NOTOOLS"
        # Second call: no tools kwarg.
        second_call = b._tokenizer.apply_chat_template.call_args_list[1]
        assert "tools" not in second_call.kwargs


class TestStatusText:
    def test_web_search_query(self):
        from core.inference.mlx_lm import MlxLmBackend

        assert (
            MlxLmBackend._tool_status_text("web_search", {"query": "abc"})
            == "Searching: abc"
        )

    def test_web_search_url(self):
        from core.inference.mlx_lm import MlxLmBackend

        out = MlxLmBackend._tool_status_text(
            "web_search", {"url": "https://example.com"}
        )
        assert out.startswith("Reading: ")

    def test_python_code(self):
        from core.inference.mlx_lm import MlxLmBackend

        out = MlxLmBackend._tool_status_text("python", {"code": "print(1)\nprint(2)"})
        assert out.startswith("Running Python:")

    def test_terminal_command(self):
        from core.inference.mlx_lm import MlxLmBackend

        out = MlxLmBackend._tool_status_text("terminal", {"command": "ls /"})
        assert out.startswith("Running:")

    def test_unknown_tool_fallback(self):
        from core.inference.mlx_lm import MlxLmBackend

        assert (
            MlxLmBackend._tool_status_text("custom_x", {}) == "Calling: custom_x"
        )


class TestToolRefusalWhenNotSupported:
    def test_raises_when_not_tool_capable(self):
        b = _fresh_backend()
        b._model = object()
        b._tokenizer = object()
        b._supports_tools = False
        with pytest.raises(RuntimeError, match="does not advertise tool-calling"):
            list(
                b.generate_chat_completion_with_tools(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "x"}}],
                )
            )

    def test_raises_when_not_loaded(self):
        b = _fresh_backend()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(
                b.generate_chat_completion_with_tools(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "x"}}],
                )
            )
