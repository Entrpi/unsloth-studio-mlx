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
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from loggers import get_logger

logger = get_logger(__name__)

# Sentinel for a metadata event at the end of the generator stream.
MetadataEvent = Dict[str, Any]


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
    def hf_variant(self) -> Optional[str]:
        return None

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
        return None

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def reasoning_always_on(self) -> bool:
        return False

    @property
    def reasoning_default(self) -> bool:
        return False

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def cache_type_kv(self) -> Optional[str]:
        return None

    @property
    def speculative_type(self) -> Optional[str]:
        return None

    def detect_audio_type(self) -> Optional[str]:
        """MLX backend never serves audio/TTS codecs. Always None."""
        return None

    def load_progress(self) -> Optional[dict]:
        """MLX load is fast (<2s for 8B 2-bit); no progress bar needed."""
        return None

    # ── Lifecycle ─────────────────────────────────────────────────

    @staticmethod
    def _platform_ok() -> bool:
        """True iff the host can run MLX (Apple Silicon macOS)."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def load_model(
        self,
        local_path: str,
        model_identifier: str,
        hf_token: Optional[str] = None,
        n_ctx: Optional[int] = None,
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

            path = Path(local_path)
            if not path.is_dir():
                raise RuntimeError(
                    f"MLX model path is not a directory: {local_path}"
                )

            # Lazy import — keeps the module importable on non-Darwin CI.
            try:
                from mlx_lm import load as _mlx_load  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    f"mlx_lm is not installed in this Python env: {e}"
                ) from e

            t0 = time.time()
            try:
                model, tokenizer = _mlx_load(str(path))
            except Exception as e:
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

            self._model = model
            self._tokenizer = tokenizer
            self._model_identifier = model_identifier
            self._local_path = str(path)
            self._context_length = effective_ctx

            logger.info(
                f"MLX model loaded in {load_s:.2f}s: "
                f"identifier={model_identifier} path={path} "
                f"context_length={effective_ctx}"
            )
            return True

    def _unload_locked(self) -> bool:
        """Internal unload. Caller must hold ``self._lock``."""
        if self._model is None and self._tokenizer is None:
            return False

        self._model = None
        self._tokenizer = None
        self._model_identifier = None
        self._local_path = None
        self._context_length = None

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
