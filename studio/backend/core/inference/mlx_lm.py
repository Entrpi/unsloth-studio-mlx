# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Apple-Silicon MLX backend for Unsloth Studio.

Loads MLX checkpoints via ``mlx_lm.load`` and streams chat completions via
``mlx_lm.stream_generate``. Mirrors the public surface of
``core.inference.llama_cpp.LlamaCppBackend`` so the route can branch on
``is_mlx``/``is_gguf`` without changing the Unsloth / transformers path.

Phase 1 scope:
- Local MLX directories only (no remote-HF MLX download).
- Text-in / text-out only. Vision and audio are rejected.
- No tool calling, no reasoning/<think> wrapping, no speculative decoding.
- No LoRA adapters, no training.

Chunk A (Phase 2 + Phase 4) extensions:
- Phase 2: ``_build_mlx_sampler_and_processors`` helper wires
  ``make_logits_processors`` for repetition/presence/frequency penalties and
  ``make_sampler`` for temperature/top_p/top_k/min_p. Stop-string enforcement
  happens in-loop by scanning the cumulative decoded text.
- Phase 4: reasoning capability is introspected from
  ``tokenizer.chat_template`` at load time. ``enable_thinking`` is forwarded
  to ``apply_chat_template`` via ``chat_template_kwargs`` when the model
  supports it. Literal ``<think>`` tags in the stream pass through
  unmodified — the frontend parses them directly.

The module is importable on any platform: ``mlx_lm`` is lazy-imported inside
``load_model``. Instantiating ``MlxLmBackend`` does not import ``mlx_lm``.
"""

from __future__ import annotations

import gc
import json
import platform
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from loggers import get_logger

logger = get_logger(__name__)

# Sentinel for a metadata event at the end of the generator stream.
MetadataEvent = Dict[str, Any]


# Phase 3 — hf_variant extraction. MLX repos commonly ship with a quant
# suffix: ``-4bit`` / ``-2bit`` / ``-mlx-2bit`` / ``-8bit``. This regex
# matches at the end of the repo / dir name. Mirrors the shape of
# ``llama_cpp.py:_extract_quant_label`` but tuned for MLX naming.
_MLX_VARIANT_RE = re.compile(r"-(\d+bit|mlx-\d+bit|fp16|bf16)$", re.IGNORECASE)


def _extract_mlx_variant(name: str) -> Optional[str]:
    """Return the MLX quant suffix of a repo/dir name, or None.

    Examples:

    - ``"mlx-community/Qwen2.5-7B-Instruct-4bit"`` → ``"4bit"``
    - ``"prism-ml/Ternary-Bonsai-8B-mlx-2bit"`` → ``"mlx-2bit"``
    - ``"unsloth/foo-bar"`` → ``None``
    """
    if not name:
        return None
    tail = name.split("/")[-1]
    m = _MLX_VARIANT_RE.search(tail)
    return m.group(1).lower() if m else None


# ── Phase 8 — Quantized KV cache ─────────────────────────────────────
# Map the frontend's existing KV dtype dropdown strings (same values the
# GGUF path has always accepted — f16 / bf16 / q8_0 / q5_1 / q4_1 / q4_0)
# to the mlx-lm ``(kv_bits, kv_group_size)`` pair. Unquantized modes
# (``f16`` / ``bf16``) map to ``(None, 64)`` which means "don't pass the
# kwargs" — matches the behaviour of omitting ``kv_bits`` in
# ``mlx_lm.generate.generate_step``.
#
# Note: MLX has no native 5-bit KV path; ``q5_1`` is rounded **down** to
# 4 bits rather than up to 8 because that's how the GGUF side behaves
# when its chosen bits-per-value is unavailable, and because rounding up
# would silently give the user a heavier cache than they asked for.
def _cache_type_kv_to_mlx(value: Optional[str]) -> Tuple[Optional[int], int]:
    """Map a UI KV-dtype string to an MLX ``(kv_bits, kv_group_size)`` pair.

    ``None`` and the unquantized labels return ``(None, 64)`` so the
    caller can detect "no quantization" by checking ``kv_bits is None``.
    The group-size default of 64 matches the upstream
    ``quantize_kv_cache`` default.

    Args:
        value: UI dropdown value. Accepts: ``None``, ``"f16"``, ``"bf16"``,
            ``"q8_0"``, ``"q5_1"``, ``"q4_1"``, ``"q4_0"``. Case-insensitive.

    Returns:
        ``(kv_bits, kv_group_size)``. ``kv_bits`` is ``None`` for the
        unquantized dtypes and an ``int`` for the quantized ones.
    """
    if value is None:
        return None, 64
    v = str(value).strip().lower()
    if v in ("", "f16", "bf16", "fp16"):
        return None, 64
    if v == "q8_0":
        return 8, 64
    if v == "q5_1":
        # mlx has no 5-bit KV path; round down to 4 rather than upgrade to
        # 8 so the user's "smaller cache" intent is preserved.
        logger.warning(
            "KV cache dtype 'q5_1' is not natively supported by mlx-lm; "
            "rounding down to 4-bit (q4_0 equivalent)."
        )
        return 4, 64
    if v in ("q4_0", "q4_1"):
        return 4, 64
    # Unknown label: treat as no quantization and log once.
    logger.warning(
        f"Unknown KV cache dtype '{value}'; falling back to unquantized."
    )
    return None, 64


def _build_mlx_sampler_and_processors(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    repetition_context_size: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logit_bias: Optional[Dict[int, float]] = None,
):
    """Build an ``(sampler, logits_processors)`` pair for ``stream_generate``.

    This is the single place in the backend that knows the mlx-lm sampling
    API. Phases 6/7/8 add kwargs here; callers in ``generate_chat_completion``
    never import ``mlx_lm.sample_utils`` directly.

    Semantics:

    - ``temperature <= 0`` → greedy (``make_sampler(temp=0)`` returns an
      argmax sampler upstream, confirmed by reading the source of
      ``mlx_lm.sample_utils.make_sampler`` on 0.31.2).
    - ``top_k <= 0``, ``min_p <= 0``, ``top_p >= 1.0`` are all treated as
      "off" by ``make_sampler`` (upstream guards).
    - A logits processor is added ONLY when its value is non-default, so the
      empty-list fast path is honored in ``stream_generate``. Specifically:

      * ``repetition_penalty > 1.0`` → adds the repetition processor. A
        value of ``1.0`` is a no-op mathematically but the upstream
        ``make_logits_processors`` naively adds it when ``penalty != 0``;
        we gate here to avoid the per-token cost.
      * ``presence_penalty != 0`` / ``frequency_penalty != 0`` → adds the
        corresponding processor.
      * ``logit_bias`` truthy → adds the bias processor.

    - If the installed mlx-lm rejects a kwarg (signature drift between
      patch releases), we drop the offending kwarg with a ``debug`` log
      and retry with a smaller subset. Mirrors the defensive pattern that
      the Phase-1 sampler already used.

    Returns:
        Tuple of (sampler, processors_list). Either may be empty/None to
        signal "no special handling" to ``stream_generate``.
    """
    from mlx_lm.sample_utils import (  # type: ignore
        make_logits_processors,
        make_sampler,
    )

    # ── Sampler ───────────────────────────────────────────────────────
    try:
        sampler = make_sampler(
            temp = float(temperature) if temperature and temperature > 0 else 0.0,
            top_p = float(top_p) if top_p and top_p > 0 else 0.0,
            top_k = int(top_k) if top_k and top_k > 0 else 0,
            min_p = float(min_p) if min_p and min_p > 0 else 0.0,
        )
    except TypeError as e:
        logger.debug(
            f"make_sampler kwargs signature drift ({e}); retrying with temp/top_p only"
        )
        try:
            sampler = make_sampler(
                temp = float(temperature) if temperature and temperature > 0 else 0.0,
                top_p = float(top_p) if top_p and top_p > 0 else 0.0,
            )
        except Exception as e2:
            logger.debug(f"make_sampler fallback also failed ({e2}); using default")
            sampler = None

    # ── Logits processors (penalties + bias) ──────────────────────────
    # Only include a kwarg when it represents an actual change from
    # neutral, so the returned processor list is empty in the common
    # "defaults everywhere" case.
    lp_kwargs: Dict[str, Any] = {}
    if repetition_penalty is not None and float(repetition_penalty) > 1.0:
        lp_kwargs["repetition_penalty"] = float(repetition_penalty)
        if repetition_context_size is not None and repetition_context_size > 0:
            lp_kwargs["repetition_context_size"] = int(repetition_context_size)
    if presence_penalty is not None and float(presence_penalty) != 0.0:
        lp_kwargs["presence_penalty"] = float(presence_penalty)
    if frequency_penalty is not None and float(frequency_penalty) != 0.0:
        lp_kwargs["frequency_penalty"] = float(frequency_penalty)
    if logit_bias:
        lp_kwargs["logit_bias"] = dict(logit_bias)

    if not lp_kwargs:
        return sampler, []

    try:
        processors = make_logits_processors(**lp_kwargs)
    except TypeError as e:
        # Defensive: if an individual kwarg is rejected by an older/newer
        # mlx-lm, drop it and retry. Log at debug to mirror the Phase-1
        # pattern at mlx_lm.py:357-365.
        logger.debug(
            f"make_logits_processors kwargs rejected ({e}); retrying with reduced set"
        )
        _safe: Dict[str, Any] = {}
        for k, v in lp_kwargs.items():
            try:
                make_logits_processors(**{k: v})
                _safe[k] = v
            except TypeError:
                logger.debug(f"  dropping unsupported kwarg: {k}")
        try:
            processors = make_logits_processors(**_safe)
        except Exception as e2:
            logger.debug(f"make_logits_processors fallback failed ({e2}); using empty list")
            processors = []

    return sampler, list(processors or [])


class MlxLmBackend:
    """In-process MLX-LM backend. One instance per Studio process.

    Lifecycle:
        1. ``load_model(local_path, model_identifier)`` — calls
           ``mlx_lm.load`` and parses ``max_position_embeddings``.
        2. ``generate_chat_completion(...)`` — yields cumulative text
           strings followed by a final metadata dict.
        3. ``unload_model()`` — drops refs, ``gc.collect()``,
           ``mx.metal.clear_cache()`` if available.

    Not thread-safe against concurrent ``load_model`` / ``unload_model``
    — guarded by an internal lock. ``generate_chat_completion`` is
    expected to run serially under the FastAPI route.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_identifier: Optional[str] = None
        self._local_path: Optional[str] = None
        self._context_length: Optional[int] = None
        # Phase 6 — LoRA adapter path threaded into ``mlx_lm.load`` via
        # its native ``adapter_path`` kwarg. ``None`` means "base model
        # only"; a non-None value means an adapter is layered on top.
        self._adapter_path: Optional[str] = None
        # Phase 7 — speculative decoding. ``_draft_model`` holds a second
        # ``nn.Module`` loaded from a separate (typically smaller)
        # checkpoint; ``stream_generate(draft_model=...)`` uses it to
        # propose tokens which the base model verifies. The draft
        # tokenizer is kept for vocab-size symmetry checks.
        self._draft_model: Any = None
        self._draft_tokenizer: Any = None
        self._draft_path: Optional[str] = None
        self._num_draft_tokens: int = 3
        # Phase 4 — reasoning / <think> state. Populated at load time by
        # ``_detect_reasoning``; reset by ``_unload_locked``.
        self._chat_template: Optional[str] = None
        self._supports_reasoning: bool = False
        self._reasoning_always_on: bool = False
        self._reasoning_default: bool = True
        # Phase 3 — load progress tracking. The GGUF backend drives its
        # load-progress endpoint off /proc VmRSS; MLX is in-process so
        # we instead track two distinct phases:
        #   downloading → bytes_loaded / bytes_total from a custom tqdm
        #                 class wired through snapshot_download
        #   loading     → VmRSS / weights_bytes_total from psutil
        #   loaded      → final phase (counters frozen at final values)
        self._load_phase: Optional[str] = None
        self._download_bytes_loaded: int = 0
        self._download_bytes_total: int = 0
        self._weights_bytes_total: int = 0
        self._load_warnings: List[str] = []
        # Phase 3 — hf_variant surface. Derived from the tail of the
        # repo/path name at load time. Mirrors the GGUF surface which
        # exposes the quant label on the backend so the UI can tag it.
        self._hf_variant: Optional[str] = None
        # Phase 8 — quantized KV cache settings. Set by ``load_model`` from
        # the ``cache_type_kv`` request field (mirrors GGUF behaviour where
        # KV quantization is a load-time decision because ``stream_generate``
        # only reads the kwargs when building the prompt cache). ``None``
        # means "do not pass kv_bits to stream_generate" (== unquantized).
        self._kv_bits: Optional[int] = None
        self._kv_group_size: int = 64
        self._quantized_kv_start: int = 0
        self._cache_type_kv_label: Optional[str] = None
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def is_active(self) -> bool:
        # Phase 1 has no separate "loading" state — load is fast enough
        # (sub-2s for an 8B 2-bit checkpoint) that is_active mirrors is_loaded.
        return self.is_loaded

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return False

    @property
    def is_lora(self) -> bool:
        """True iff the backend was loaded with an adapter path."""
        return self._adapter_path is not None

    @property
    def adapter_path(self) -> Optional[str]:
        """Absolute path of the loaded LoRA adapter directory, or None."""
        return self._adapter_path

    @property
    def hf_variant(self) -> Optional[str]:
        """MLX quant suffix parsed at load time (e.g. ``"4bit"``,
        ``"mlx-2bit"``). ``None`` when the dir/repo name has no
        recognizable quant tail."""
        return self._hf_variant

    @property
    def context_length(self) -> Optional[int]:
        return self._context_length

    @property
    def max_context_length(self) -> Optional[int]:
        # MLX uses unified memory; we don't run VRAM autofit. Surface the
        # native value so the UI's warning-threshold logic sees a valid
        # ceiling.
        return self._context_length

    @property
    def native_context_length(self) -> Optional[int]:
        return self._context_length

    @property
    def chat_template(self) -> Optional[str]:
        return self._chat_template

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_always_on(self) -> bool:
        return self._reasoning_always_on

    @property
    def reasoning_default(self) -> bool:
        # When a model always reasons the UI hides the toggle but the
        # effective mode is still "thinking on". Mirrors the GGUF logic at
        # llama_cpp.py:1519-1535: when ``reasoning_always_on`` is True the
        # template emits ``<think>`` regardless of the kwarg; the
        # advertised default stays True so no user-facing contradiction.
        return self._reasoning_default

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def cache_type_kv(self) -> Optional[str]:
        """UI-facing KV-cache dtype label.

        - ``None`` when KV quantization is off (the default).
        - ``"q8_0"`` when loaded with ``kv_bits=8``.
        - ``"q4_0"`` when loaded with ``kv_bits=4``.
        - ``f"q{kv_bits}_0"`` for any other bit value (defensive fallback
          for a future mlx-lm that supports, say, 2-bit KV).
        """
        if self._kv_bits is None:
            return None
        if self._kv_bits == 8:
            return "q8_0"
        if self._kv_bits == 4:
            return "q4_0"
        return f"q{int(self._kv_bits)}_0"

    @property
    def speculative_type(self) -> Optional[str]:
        """``"mlx-draft-model"`` when a draft model is loaded, else None.

        Distinct from the GGUF ``"ngram-simple"`` / ``"ngram-mod"`` values
        so the UI / telemetry can tell them apart — MLX uses a real
        draft model, not n-gram speculation.
        """
        return "mlx-draft-model" if self._draft_model is not None else None

    @property
    def draft_model_path(self) -> Optional[str]:
        return self._draft_path

    def detect_audio_type(self) -> Optional[str]:
        """MLX backend never serves audio/TTS codecs. Always None."""
        return None

    def load_progress(self) -> Optional[dict]:
        """Return live MLX load progress (remote HF + local load), or
        ``None`` when no load is in flight.

        Shape mirrors the GGUF progress endpoint
        (``llama_cpp.py:load_progress``):

        - ``phase``: ``"downloading" | "loading" | "loaded"``
        - ``bytes_loaded`` / ``bytes_total`` / ``fraction``: clamp to 0..1

        During ``downloading`` the counters come from a snapshot_download
        tqdm subclass. During ``loading`` we sample ``psutil.rss`` against
        the sum of local ``*.safetensors`` sizes — best-effort (mlx_lm.load
        is a black box) but gives the UI a non-frozen bar.

        The optional ``warnings`` list carries RAM-pressure notes the
        load_model path generated (e.g. "model size exceeds 1.5x
        available RAM — expect swap").
        """
        phase = self._load_phase
        if phase is None:
            return None

        if phase == "downloading":
            bl = int(self._download_bytes_loaded)
            bt = int(self._download_bytes_total)
        elif phase == "loading":
            # Best-effort RSS sampling. If psutil isn't importable, we
            # return bytes_loaded=0 and let the frontend show a spinner.
            bl = 0
            try:
                import os

                import psutil  # type: ignore

                bl = int(psutil.Process(os.getpid()).memory_info().rss)
            except Exception:
                bl = 0
            bt = int(self._weights_bytes_total)
        else:  # "loaded"
            bl = int(self._weights_bytes_total or self._download_bytes_total)
            bt = int(self._weights_bytes_total or self._download_bytes_total)

        fraction = 0.0
        if bt > 0:
            fraction = max(0.0, min(1.0, bl / bt))

        out: Dict[str, Any] = {
            "phase": phase,
            "bytes_loaded": bl,
            "bytes_total": bt,
            "fraction": round(fraction, 4),
        }
        if self._load_warnings:
            out["warnings"] = list(self._load_warnings)
        return out

    # ── Lifecycle ─────────────────────────────────────────────────

    @staticmethod
    def _platform_ok() -> bool:
        """True iff the host can run MLX (Apple Silicon macOS)."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @staticmethod
    def _sum_safetensors_bytes(dir_path: Path) -> int:
        """Best-effort sum of ``*.safetensors`` file sizes in a dir.

        Used to drive both:

        - The ``loading`` phase progress heuristic (sample RSS against
          this total).
        - The RAM-warning check (compare to ``psutil.total``).

        Returns 0 on any OSError to keep the caller's code path simple.
        """
        total = 0
        try:
            for p in dir_path.iterdir():
                if p.is_file() and p.suffix == ".safetensors":
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        except OSError:
            return 0
        return total

    def _download_mlx(
        self,
        repo: str,
        hf_token: Optional[str] = None,
    ) -> str:
        """Download a remote MLX repo via ``huggingface_hub.snapshot_download``
        while feeding per-shard bytes into the backend's progress counters.

        We restrict ``allow_patterns`` to the files MLX actually needs —
        the weights shards (``*.safetensors``), ``config.json``,
        tokenizer artifacts, and any ``*.jinja`` chat-template file.
        This avoids pulling unrelated repo contents (READMEs, images,
        eval tensors) and keeps the progress counters accurate.

        Progress is wired through a custom ``tqdm_class`` that updates
        ``self._download_bytes_loaded`` and ``self._download_bytes_total``
        as huggingface_hub streams each shard. The tqdm subclass is
        strictly additive; if the upstream tqdm API changes, the
        counters just stop updating — the download still succeeds.

        Args:
            repo: HF repo id (e.g. ``mlx-community/Qwen2.5-7B-4bit``).
            hf_token: Optional HF token for gated repos.

        Returns:
            Absolute path to the downloaded snapshot directory.
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"huggingface_hub is required for remote MLX downloads: {e}"
            ) from e

        # Reset counters.
        self._download_bytes_loaded = 0
        self._download_bytes_total = 0

        backend = self

        # Custom tqdm subclass that writes into the backend counters. We
        # fall back to a trivial class if tqdm is unavailable; the
        # download still works, just with frozen counters.
        try:
            from tqdm.auto import tqdm as _base_tqdm  # type: ignore
        except ImportError:
            _base_tqdm = None

        if _base_tqdm is not None:

            class _HFProgress(_base_tqdm):  # type: ignore[misc]
                """tqdm subclass that mirrors bytes into backend counters.

                huggingface_hub instantiates one tqdm per shard. Because
                of that we track a backend-wide total across bars:
                ``total`` accumulates into ``_download_bytes_total`` on
                bar init, and ``update`` bumps ``_download_bytes_loaded``
                by the per-call ``n``. The result is a rolling count
                across the full snapshot download.
                """

                def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
                    super().__init__(*args, **kwargs)
                    total = getattr(self, "total", None)
                    if isinstance(total, (int, float)) and total > 0:
                        backend._download_bytes_total += int(total)

                def update(self, n: int = 1) -> Any:  # noqa: D401
                    backend._download_bytes_loaded += int(n)
                    return super().update(n)

            tqdm_class: Any = _HFProgress
        else:
            tqdm_class = None

        logger.info(f"MLX: downloading {repo} via snapshot_download")
        local_dir = snapshot_download(
            repo_id = repo,
            token = hf_token,
            allow_patterns = [
                "*.safetensors",
                "*.safetensors.index.json",
                "config.json",
                "tokenizer*",
                "special_tokens_map.json",
                "*.json",
                "*.jinja",
                "chat_template*",
                "generation_config.json",
            ],
            tqdm_class = tqdm_class,
        )
        logger.info(
            f"MLX: download complete: {repo} → {local_dir} "
            f"({self._download_bytes_loaded / 1e9:.2f} GB)"
        )
        return local_dir

    @staticmethod
    def _draft_mem_preflight(draft_dir: Path) -> None:
        """Phase 7 — refuse to load a draft that would push combined RAM
        usage past 75% of total unified memory.

        The estimate is deliberately conservative: we sum the safetensors
        shards in the draft directory and treat that as "at least this
        much new allocation" — MLX's lazy loading actually uses less
        peak memory, but for the purpose of refusing-on-low-RAM the
        upper bound is what we want.

        Raises ``RuntimeError`` on refusal. Callers wrap in try/except
        and fail the load gracefully.
        """
        try:
            import psutil  # type: ignore
        except ImportError:
            # Without psutil we can't preflight; skip silently. Preflight
            # is a safety net, not a hard prerequisite.
            return

        if not draft_dir.is_dir():
            return

        draft_bytes = 0
        try:
            for p in draft_dir.iterdir():
                if p.is_file() and p.suffix == ".safetensors":
                    try:
                        draft_bytes += p.stat().st_size
                    except OSError:
                        pass
        except OSError:
            return

        if draft_bytes == 0:
            return

        vm = psutil.virtual_memory()
        total = getattr(vm, "total", 0) or 0
        available = getattr(vm, "available", 0) or 0
        if total <= 0:
            return

        # "Combined footprint" ≈ (total - available) + draft_bytes. If
        # that exceeds 75% of total, refuse.
        combined = (total - available) + draft_bytes
        limit = int(total * 0.75)
        if combined > limit:
            raise RuntimeError(
                "draft model would exceed 75% of available memory; refusing "
                f"to load (draft={draft_bytes / 1e9:.1f} GB, "
                f"combined_footprint={combined / 1e9:.1f} GB, "
                f"limit={limit / 1e9:.1f} GB on a {total / 1e9:.1f} GB box)"
            )

    def _detect_reasoning(
        self, tokenizer: Any, model_identifier: str
    ) -> None:
        """Populate reasoning flags by inspecting the chat template.

        Mirrors the GGUF metadata scan at ``llama_cpp.py:893-921`` and the
        size-based default logic at ``llama_cpp.py:1519-1535``. The rule is:

        * ``enable_thinking`` substring in the template → supports_reasoning
          is True and the template actually reads the kwarg (Qwen3, Bonsai
          derivatives that kept it, etc.).
        * Else if BOTH ``<think>`` AND ``</think>`` appear in the template →
          supports_reasoning is True AND reasoning_always_on is True: the
          template hardcodes the tags, so the model always reasons and the
          toggle has no effect. The UI hides it.
        * Otherwise no reasoning support.

        For always-on-capable reasoning models (``enable_thinking`` path)
        the default is True unless the model identifier says Qwen3.5 /
        Qwen3.6 < 9B — those ship with thinking disabled by default
        per upstream Qwen recommendation.
        """
        self._chat_template = None
        self._supports_reasoning = False
        self._reasoning_always_on = False
        self._reasoning_default = True

        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            return
        self._chat_template = template

        if "enable_thinking" in template:
            self._supports_reasoning = True
            # Size-based default, ported from llama_cpp.py:1519-1535.
            thinking_default = True
            mid = (model_identifier or "").lower()
            if "qwen3.5" in mid or "qwen3.6" in mid:
                try:
                    from utils.models import extract_model_size_b

                    size_val = extract_model_size_b(mid)
                    if size_val is not None and size_val < 9:
                        thinking_default = False
                except Exception as e:
                    logger.debug(
                        f"extract_model_size_b failed ({e}); keeping default=True"
                    )
            self._reasoning_default = thinking_default
            logger.info(
                f"MLX: reasoning detected via enable_thinking kwarg "
                f"(default={thinking_default})"
            )
            return

        if "<think>" in template and "</think>" in template:
            self._supports_reasoning = True
            self._reasoning_always_on = True
            self._reasoning_default = True
            logger.info(
                "MLX: reasoning detected via literal <think> tags (always-on)"
            )

    def load_model(
        self,
        local_path: str,
        model_identifier: str,
        hf_token: Optional[str] = None,
        n_ctx: Optional[int] = None,
        cache_type_kv: Optional[str] = None,
        adapter_path: Optional[str] = None,
        draft_model_path: Optional[str] = None,
    ) -> bool:
        """Load an MLX checkpoint.

        Args:
            local_path: Directory containing ``config.json`` and the MLX
                weight shards. For Phase 1 this must be a local path;
                remote-HF MLX download is not wired.
            model_identifier: Public-facing id the UI and /models/list
                surface. Typically the original ``org/repo`` or local
                path the user selected.
            hf_token: Accepted for interface symmetry with the GGUF
                backend; ignored for local loads.
            n_ctx: Optional cap on the effective context length. If
                provided, ``context_length`` is ``min(config.max_pos, n_ctx)``.
            cache_type_kv: Optional UI dtype label for the KV cache —
                one of ``"f16" | "bf16" | "q8_0" | "q5_1" | "q4_1" | "q4_0"``.
                ``None`` / ``"f16"`` / ``"bf16"`` keep the cache
                unquantized. Phase 8 passes the mapped ``kv_bits`` /
                ``kv_group_size`` into ``stream_generate`` on each
                generation tick; MLX applies the quantization when it
                builds the prompt cache.
            adapter_path: Phase 6 — absolute path to an MLX LoRA adapter
                directory (containing ``adapters.safetensors`` +
                ``adapter_config.json``). When provided, forwarded to
                ``mlx_lm.load(..., adapter_path=...)`` which fuses the
                adapter onto the base model at load time. The base must
                match the adapter's target architecture; otherwise
                mlx-lm raises and this function returns False.
            draft_model_path: Phase 7 — absolute path to a smaller MLX
                checkpoint to use for speculative decoding. When
                provided, the backend loads a second model via
                ``mlx_lm.load`` and keeps it alongside the base; every
                ``stream_generate`` call passes it as ``draft_model=``.
                Memory-preflighted: refused if combined footprint would
                exceed 75% of total unified memory. A tokenizer
                vocab-size mismatch is logged as a warning but not
                fatal — mlx-lm requires same-tokenizer draft models.

        Returns:
            True on success. Returns False on a failed load (and logs
            the exception) so the route can surface a clean 500.
        """
        if not self._platform_ok():
            raise RuntimeError(
                "mlx_lm is not available on this platform "
                "(requires macOS on Apple Silicon)"
            )

        with self._lock:
            if self.is_loaded:
                logger.warning(
                    "MlxLmBackend.load_model called while a model is already "
                    "loaded; unloading first"
                )
                self._unload_locked()

            # Phase 3 — initialize load progress state. Reset from any
            # prior load so a stale terminal "loaded" state doesn't leak.
            self._load_phase = None
            self._download_bytes_loaded = 0
            self._download_bytes_total = 0
            self._weights_bytes_total = 0
            self._load_warnings = []

            # Phase 3 — remote HF repo path. If the local dir doesn't
            # exist but the identifier looks like an HF repo id (has a
            # slash, not a filesystem path), download it. We don't
            # second-guess the caller's identifier: if they said "load
            # this remote", we try. Detection of "is this actually an
            # MLX repo" happens in ModelConfig.from_identifier before
            # we get here.
            path = Path(local_path)
            if not path.is_dir():
                looks_like_repo = (
                    "/" in local_path
                    and not local_path.startswith("/")
                    and not local_path.startswith(".")
                )
                if looks_like_repo:
                    self._load_phase = "downloading"
                    try:
                        downloaded = self._download_mlx(
                            local_path, hf_token = hf_token
                        )
                        path = Path(downloaded)
                    except Exception as e:
                        self._load_phase = None
                        logger.error(
                            f"MLX snapshot_download failed for {local_path}: {e}"
                        )
                        return False
                if not path.is_dir():
                    self._load_phase = None
                    raise RuntimeError(
                        f"MLX model path is not a directory: {local_path}"
                    )

            # Phase 3 — RAM-pressure warning. Compare weight-shard bytes
            # to total unified memory; if the ratio falls below 1.5x we
            # expect macOS to swap.
            self._weights_bytes_total = self._sum_safetensors_bytes(path)
            try:
                import psutil as _ps  # type: ignore

                vm = _ps.virtual_memory()
                if (
                    self._weights_bytes_total > 0
                    and vm.total > 0
                    and vm.total < 1.5 * self._weights_bytes_total
                ):
                    warn = (
                        f"MLX model size ({self._weights_bytes_total / 1e9:.1f} GB) "
                        f"is close to total RAM ({vm.total / 1e9:.1f} GB); "
                        f"expect swap / slow inference."
                    )
                    logger.warning(warn)
                    self._load_warnings.append(warn)
            except ImportError:
                pass

            # Now switch to loading phase.
            self._load_phase = "loading"

            # Lazy import — keeps the module importable on non-Darwin CI.
            try:
                from mlx_lm import load as _mlx_load  # type: ignore
            except ImportError as e:
                self._load_phase = None
                raise RuntimeError(
                    f"mlx_lm is not installed in this Python env: {e}"
                ) from e

            # Phase 6 — when an adapter path is provided, pass it through
            # to mlx_lm.load. Verified adapter_path kwarg exists on 0.31.2.
            load_kwargs: Dict[str, Any] = {}
            resolved_adapter: Optional[str] = None
            if adapter_path:
                adapter_dir = Path(adapter_path)
                if not adapter_dir.is_dir():
                    logger.error(
                        f"MLX adapter path is not a directory: {adapter_path}"
                    )
                    return False
                resolved_adapter = str(adapter_dir)
                load_kwargs["adapter_path"] = resolved_adapter

            t0 = time.time()
            try:
                model, tokenizer = _mlx_load(str(path), **load_kwargs)
            except Exception as e:
                self._load_phase = None
                logger.error(f"mlx_lm.load failed for {local_path}: {e}")
                return False
            load_s = time.time() - t0

            # Read max_position_embeddings from config.json; cap by n_ctx
            # if supplied.
            native_ctx: Optional[int] = None
            try:
                with open(path / "config.json", "r", encoding = "utf-8") as f:
                    cfg = json.load(f)
                val = cfg.get("max_position_embeddings")
                if isinstance(val, int) and val > 0:
                    native_ctx = val
            except (OSError, ValueError):
                pass

            effective_ctx: Optional[int] = native_ctx
            if n_ctx is not None and n_ctx > 0:
                effective_ctx = (
                    min(native_ctx, n_ctx) if native_ctx else n_ctx
                )

            # Phase 7 — optionally load a draft model for speculative
            # decoding. We do this BEFORE setting ``self._model`` so a
            # failed preflight / load doesn't leave the backend in a
            # half-loaded state.
            draft_model: Any = None
            draft_tokenizer: Any = None
            resolved_draft: Optional[str] = None
            if draft_model_path:
                draft_dir = Path(draft_model_path)
                if not draft_dir.is_dir():
                    self._load_phase = None
                    logger.error(
                        f"MLX draft model path is not a directory: "
                        f"{draft_model_path}"
                    )
                    return False
                try:
                    self._draft_mem_preflight(draft_dir)
                except RuntimeError as e:
                    self._load_phase = None
                    logger.error(f"MLX draft memory preflight failed: {e}")
                    return False
                try:
                    t_d = time.time()
                    draft_model, draft_tokenizer = _mlx_load(str(draft_dir))
                    logger.info(
                        f"MLX draft model loaded in {time.time() - t_d:.2f}s "
                        f"from {draft_dir}"
                    )
                except Exception as e:
                    self._load_phase = None
                    logger.error(f"mlx_lm.load failed for draft {draft_dir}: {e}")
                    return False
                resolved_draft = str(draft_dir)

                # Warn on vocab-size mismatch; mlx-lm's speculative decoding
                # requires identical tokenizers. Users typically pair
                # checkpoints in the same family (e.g. Bonsai-8B + Bonsai-1.7B).
                try:
                    base_vocab = getattr(tokenizer, "vocab_size", None)
                    draft_vocab = getattr(draft_tokenizer, "vocab_size", None)
                    if (
                        base_vocab is not None
                        and draft_vocab is not None
                        and base_vocab != draft_vocab
                    ):
                        logger.warning(
                            f"MLX draft/base vocab_size mismatch "
                            f"(base={base_vocab} draft={draft_vocab}); "
                            f"speculative decoding may behave unexpectedly."
                        )
                except Exception:
                    pass

            self._model = model
            self._tokenizer = tokenizer
            self._model_identifier = model_identifier
            self._local_path = str(path)
            self._context_length = effective_ctx
            # Phase 6 — record the adapter path (None for base-only loads).
            self._adapter_path = resolved_adapter
            # Phase 7 — record the draft refs.
            self._draft_model = draft_model
            self._draft_tokenizer = draft_tokenizer
            self._draft_path = resolved_draft
            # Phase 3 — derive hf_variant from the repo/dir name (tail
            # after the last /). Prefer the user-supplied identifier
            # over the local cache path so "-mlx-2bit" from a repo id
            # isn't lost to a hashed cache dir.
            self._hf_variant = _extract_mlx_variant(
                model_identifier
            ) or _extract_mlx_variant(str(path))
            # Phase 3 — load is complete; freeze progress at "loaded".
            self._load_phase = "loaded"

            # Phase 8: map the UI KV-dtype label to mlx-lm's
            # (kv_bits, kv_group_size) pair and stash for the generate
            # loop. Unquantized → kv_bits=None so stream_generate never
            # sees the kwarg (preserves identical behaviour to Chunk A).
            kv_bits, kv_group = _cache_type_kv_to_mlx(cache_type_kv)
            self._kv_bits = kv_bits
            self._kv_group_size = kv_group
            self._quantized_kv_start = 0
            self._cache_type_kv_label = cache_type_kv

            # Phase 4: introspect reasoning support from the tokenizer's
            # chat template. Errors are non-fatal — a model without a
            # template just gets supports_reasoning=False.
            try:
                self._detect_reasoning(tokenizer, model_identifier)
            except Exception as e:
                logger.debug(f"MLX reasoning detection failed ({e}); defaulting to off")

            logger.info(
                f"MLX model loaded in {load_s:.2f}s: "
                f"identifier={model_identifier} path={path} "
                f"context_length={effective_ctx} "
                f"reasoning={self._supports_reasoning} "
                f"always_on={self._reasoning_always_on} "
                f"cache_type_kv={self.cache_type_kv} "
                f"adapter={self._adapter_path or 'none'}"
            )
            return True

    def _unload_locked(self) -> bool:
        """Internal unload. Caller must hold ``self._lock``."""
        if (
            self._model is None
            and self._tokenizer is None
            and self._draft_model is None
        ):
            return False

        self._model = None
        self._tokenizer = None
        self._model_identifier = None
        self._local_path = None
        self._context_length = None
        # Phase 6 — reset LoRA adapter state.
        self._adapter_path = None
        # Phase 7 — drop the draft refs alongside the base. Both live on
        # one backend; one unload pays off both.
        self._draft_model = None
        self._draft_tokenizer = None
        self._draft_path = None
        # Phase 3 — clear load progress + hf_variant so a subsequent
        # load_progress() returns None (== "no load in flight").
        self._load_phase = None
        self._download_bytes_loaded = 0
        self._download_bytes_total = 0
        self._weights_bytes_total = 0
        self._load_warnings = []
        self._hf_variant = None
        # Phase 4: clear reasoning state. A subsequent load_model of a
        # different model must not inherit the previous model's flags.
        self._chat_template = None
        self._supports_reasoning = False
        self._reasoning_always_on = False
        self._reasoning_default = True
        # Phase 8: reset KV-cache state to "unquantized default" so a
        # subsequent load without cache_type_kv doesn't inherit the
        # previous model's quantization.
        self._kv_bits = None
        self._kv_group_size = 64
        self._quantized_kv_start = 0
        self._cache_type_kv_label = None

        gc.collect()
        # mx.metal.clear_cache() may not exist in every MLX build.
        try:
            import mlx.core as mx  # type: ignore

            metal = getattr(mx, "metal", None)
            if metal is not None:
                clear = getattr(metal, "clear_cache", None)
                if callable(clear):
                    try:
                        clear()
                    except Exception as e:
                        logger.debug(f"mx.metal.clear_cache() failed: {e}")
        except ImportError:
            pass

        return True

    def unload_model(self) -> bool:
        with self._lock:
            return self._unload_locked()

    def generate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        stop: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Generator[Union[str, MetadataEvent], None, None]:
        """Stream a chat completion.

        Contract (matches ``LlamaCppBackend.generate_chat_completion``):

        - Yields cumulative text strings: each yield is the FULL text
          generated so far, not just the new delta. The route diffs the
          cumulative string to derive OpenAI ``delta.content`` frames.
        - After the text stream ends, yields exactly one metadata dict
          with ``{"type": "metadata", "usage": {...}, "timings": {...}}``.

        Chunk A (Phase 2 + Phase 4) behavior:

        - ``image_b64`` is rejected (not supported).
        - ``enable_thinking`` is threaded into ``apply_chat_template`` via
          ``chat_template_kwargs={"enable_thinking": bool}`` when the
          model advertises ``supports_reasoning``. When the backend does
          NOT support reasoning (plain chat template) the kwarg is
          dropped silently — no schema breakage, no template errors.
          Literal ``<think>...</think>`` tags emitted by the model pass
          through the stream unmodified; the frontend parses them.
        - ``stop`` strings are enforced **backend-side** by scanning the
          cumulative decoded text after each token tick. When a stop
          match is found, we truncate cumulative at the match boundary,
          yield the truncated string once, and terminate the loop with
          ``finish_reason="stop"`` baked into the metadata event.
          (``mlx-lm`` has no native ``stop`` kwarg on 0.31.2 — confirmed
          against the upstream source.)
        - ``temperature``, ``top_p``, ``top_k``, ``min_p``,
          ``repetition_penalty``, ``presence_penalty``,
          ``frequency_penalty``, and ``logit_bias`` are forwarded through
          :func:`_build_mlx_sampler_and_processors`. ``temperature <= 0``
          yields a greedy / argmax sampler upstream.
        """
        if image_b64:
            raise ValueError(
                "MLX backend does not support image inputs in Phase 1"
            )
        if not self.is_loaded:
            raise RuntimeError("MLX model is not loaded")

        # Lazy imports
        try:
            from mlx_lm import stream_generate  # type: ignore
        except ImportError as e:
            raise RuntimeError(f"mlx_lm is not installed: {e}") from e

        tokenizer = self._tokenizer
        model = self._model

        # Build the prompt via the tokenizer's chat template. Fall back to
        # a minimal ChatML-style prompt if the model has no template.
        # When the model advertises reasoning support AND the caller
        # explicitly set ``enable_thinking``, pass it through
        # ``chat_template_kwargs``. When the model doesn't support
        # reasoning we skip the kwarg entirely: some templates reject
        # unknown kwargs.
        apply_kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if self.supports_reasoning and enable_thinking is not None:
            apply_kwargs["chat_template_kwargs"] = {
                "enable_thinking": bool(enable_thinking)
            }
        try:
            prompt = tokenizer.apply_chat_template(messages, **apply_kwargs)
        except Exception as e:
            logger.warning(
                f"apply_chat_template failed ({e}); falling back to manual prompt"
            )
            parts: list[str] = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            parts.append("<|assistant|>\n")
            prompt = "\n".join(parts)

        # Build the sampler and logits processors via the single helper.
        sampler, processors = _build_mlx_sampler_and_processors(
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            repetition_context_size = repetition_context_size,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
        )

        # Kwargs for stream_generate — filter out Nones.
        sg_kwargs: Dict[str, Any] = {"prompt": prompt}
        if max_tokens is not None and max_tokens > 0:
            sg_kwargs["max_tokens"] = int(max_tokens)
        if sampler is not None:
            sg_kwargs["sampler"] = sampler
        if processors:
            sg_kwargs["logits_processors"] = processors

        # Phase 8 — quantized KV cache. Pass kv_bits / kv_group_size /
        # quantized_kv_start **only when kv_bits is set**. Upstream
        # ``stream_generate`` forwards via ``**kwargs`` into
        # ``generate_step`` (and ``speculative_generate_step``), which
        # accept these kwargs on 0.31.2. Omitting them when None
        # preserves the exact behaviour of Phase 1 / Chunk A.
        if self._kv_bits is not None:
            sg_kwargs["kv_bits"] = int(self._kv_bits)
            sg_kwargs["kv_group_size"] = int(self._kv_group_size)
            sg_kwargs["quantized_kv_start"] = int(self._quantized_kv_start)

        # Phase 7 — speculative decoding. When a draft model was loaded
        # alongside the base, forward it as the first positional
        # equivalent kwarg that stream_generate accepts natively
        # (confirmed on 0.31.2: ``draft_model`` is an explicit parameter
        # on ``stream_generate``). ``num_draft_tokens`` is accepted by
        # ``speculative_generate_step`` via ``**kwargs`` forwarding.
        if self._draft_model is not None:
            sg_kwargs["draft_model"] = self._draft_model
            sg_kwargs["num_draft_tokens"] = int(self._num_draft_tokens)

        # Normalize stop strings: accept str | list[str] | None; strip empties.
        stop_strings: List[str] = []
        if stop:
            if isinstance(stop, str):
                stop_strings = [stop]
            else:
                stop_strings = [s for s in stop if isinstance(s, str) and s]
        max_stop_len = max((len(s) for s in stop_strings), default = 0)

        # Generation loop. stream_generate is a plain Python generator;
        # the caller (route) drives it from a worker thread via
        # asyncio.to_thread(next, gen, sentinel) to keep the event loop
        # free. We accumulate resp.text and yield the cumulative string.
        cumulative = ""
        last_resp: Any = None
        finish_reason = "stop"  # default; may be overridden by cancel path.
        try:
            for resp in stream_generate(model, tokenizer, **sg_kwargs):
                last_resp = resp
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("MLX generation cancelled by client")
                    finish_reason = "cancelled"
                    break
                text = getattr(resp, "text", "") or ""
                if not text:
                    continue
                cumulative += text

                # Stop-string enforcement: scan the tail of the cumulative
                # decoded string for any configured stop sequence. The tail
                # window is "max stop string length + the text we just
                # appended" — this guarantees we catch a match that spans
                # a tokenization boundary without rescanning the full
                # buffer each tick. At 47 tok/s the cost is negligible
                # either way, but keeping it tail-bounded means long
                # transcripts don't drag the tick time up.
                if stop_strings:
                    scan_start = max(
                        0, len(cumulative) - (max_stop_len + len(text))
                    )
                    hay = cumulative[scan_start:]
                    earliest_rel: Optional[int] = None
                    for s in stop_strings:
                        idx = hay.find(s)
                        if idx != -1 and (earliest_rel is None or idx < earliest_rel):
                            earliest_rel = idx
                    if earliest_rel is not None:
                        cut = scan_start + earliest_rel
                        cumulative = cumulative[:cut]
                        yield cumulative
                        break

                yield cumulative
        except Exception as e:
            logger.error(f"MLX stream_generate raised: {e}")
            raise

        # Final metadata. ``last_resp`` carries final counts/tps.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        timings = {"prompt_per_second": None, "predicted_per_second": None}
        if last_resp is not None:
            pt = int(getattr(last_resp, "prompt_tokens", 0) or 0)
            gt = int(getattr(last_resp, "generation_tokens", 0) or 0)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": gt,
                "total_tokens": pt + gt,
            }
            timings = {
                "prompt_per_second": getattr(last_resp, "prompt_tps", None),
                "predicted_per_second": getattr(last_resp, "generation_tps", None),
            }

        yield {
            "type": "metadata",
            "usage": usage,
            "timings": timings,
            "finish_reason": finish_reason,
        }
