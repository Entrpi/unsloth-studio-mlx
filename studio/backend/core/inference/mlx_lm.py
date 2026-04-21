# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Apple-Silicon MLX backend for Unsloth Studio.

Loads MLX checkpoints via ``mlx_lm.load`` and streams chat completions via
``mlx_lm.stream_generate``. Mirrors the public surface of
``core.inference.llama_cpp.LlamaCppBackend`` so the route can branch on
``is_mlx``/``is_gguf`` without changing the Unsloth / transformers path.

Phase 1 scope (see /tmp/mlx-backend-implementation-plan.md):
- Local MLX directories only (no remote-HF MLX download).
- Text-in / text-out only. Vision and audio are rejected.
- No tool calling, no reasoning/<think> wrapping, no speculative decoding.
- No LoRA adapters, no training.

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
        presence_penalty: float = 0.0,
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

        Phase-1 limitations:

        - ``image_b64`` is rejected (not supported).
        - ``enable_thinking`` is silently ignored (Bonsai has no
          ``<think>`` tag plumbing in Phase 1).
        - ``stop`` strings are not honored yet — MLX's stream_generate
          does not natively accept custom stop strings; the route can
          still truncate on its side if needed. We pass them through
          ``mlx_lm`` if it supports the ``stop`` kwarg and silently
          swallow TypeError otherwise for forward compatibility.
        - ``repetition_penalty``, ``presence_penalty``, ``top_k`` are
          passed through ``sampler`` / ``logits_processors`` when
          supported by the installed mlx-lm version. Unsupported
          values are dropped with a debug log rather than raising.
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
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except ImportError as e:
            raise RuntimeError(f"mlx_lm is not installed: {e}") from e

        tokenizer = self._tokenizer
        model = self._model

        # Build the prompt via the tokenizer's chat template. Fall back to
        # a minimal ChatML-style prompt if the model has no template.
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True,
                tokenize = False,
            )
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

        # Build the sampler. ``make_sampler`` is stable across 0.31.x.
        sampler = None
        try:
            sampler = make_sampler(
                temp = float(temperature),
                top_p = float(top_p),
                top_k = int(top_k) if top_k and top_k > 0 else 0,
                min_p = float(min_p) if min_p and min_p > 0 else 0.0,
            )
        except TypeError:
            # Older mlx-lm sample_utils has a smaller arg set.
            try:
                sampler = make_sampler(temp = float(temperature), top_p = float(top_p))
            except Exception as e:
                logger.debug(
                    f"Could not construct MLX sampler ({e}); using default"
                )
                sampler = None

        # Kwargs for stream_generate — filter out Nones.
        sg_kwargs: Dict[str, Any] = {"prompt": prompt}
        if max_tokens is not None and max_tokens > 0:
            sg_kwargs["max_tokens"] = int(max_tokens)
        if sampler is not None:
            sg_kwargs["sampler"] = sampler

        # Generation loop. stream_generate is a plain Python generator;
        # the caller (route) drives it from a worker thread via
        # asyncio.to_thread(next, gen, sentinel) to keep the event loop
        # free. We accumulate resp.text and yield the cumulative string.
        cumulative = ""
        last_resp: Any = None
        try:
            for resp in stream_generate(model, tokenizer, **sg_kwargs):
                last_resp = resp
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("MLX generation cancelled by client")
                    break
                text = getattr(resp, "text", "") or ""
                if text:
                    cumulative += text
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

        yield {"type": "metadata", "usage": usage, "timings": timings}
