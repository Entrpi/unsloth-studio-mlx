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
        raise NotImplementedError("load_model is implemented in step 3")

    def unload_model(self) -> bool:
        raise NotImplementedError("unload_model is implemented in step 3")

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
        raise NotImplementedError("generate_chat_completion is implemented in step 5")
