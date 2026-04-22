# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Apple-Silicon MLX-VLM (vision-language) backend for Unsloth Studio.

Peers :class:`core.inference.mlx_lm.MlxLmBackend` — **does not extend it**,
because ``mlx-vlm`` returns ``(model, processor)`` not
``(model, tokenizer)`` and the prompt-rendering / generate call shape
diverges enough that a shared parent would be more confusing than
duplicative. The route branches on ``is_mlx_vlm`` vs ``is_mlx`` /
``is_gguf`` to dispatch here.

Phase 9 scope:
- Load via ``mlx_vlm.load(path)`` returning ``(model, processor)``.
- Accept base64 images through ``generate_chat_completion(..., image_b64=...)``
  and write them to a per-generation temp file; ``mlx_vlm.generate`` /
  ``mlx_vlm.stream_generate`` accept either a path or a PIL Image.
- Use ``mlx_vlm.prompt_utils.apply_chat_template(processor, config,
  prompt, num_images=N)`` to render the model's expected chat template
  with image placeholders in the right spot.
- Reuse Chunk A's sampler helper indirectly — mlx-vlm's ``generate``
  accepts ``temperature``, ``top_p``, ``min_p``, ``top_k``,
  ``repetition_penalty``, ``max_tokens`` as plain kwargs on the public
  entrypoints, so we forward them as-is and let upstream do the
  make_sampler / make_logits_processors wiring.
- Tool calling: detection is the same chat-template probe the base MLX
  backend uses; when supported, tools flow through as a serialized
  system-prompt augmentation because VLM templates don't universally
  accept a ``tools`` kwarg in ``apply_chat_template``.

Non-goals (deferred):
- Video input (even though Qwen3VL ships a video processor — out of
  scope for Chunk D).
- Audio input on a VLM model.
- MLX-VLM LoRA adapter loading (Phase 6 does not extend to VLM).
- Draft-model speculative decoding on VLM.

The module is importable on any platform: ``mlx_vlm`` is lazy-imported
inside ``load_model``. Instantiating ``MlxVlmBackend`` does not import
``mlx_vlm``.
"""

from __future__ import annotations

import base64
import concurrent.futures
import gc
import json
import os
import platform
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from loggers import get_logger

logger = get_logger(__name__)

# Sentinel for a metadata event at the end of the generator stream.
MetadataEvent = Dict[str, Any]

# Reuse the MLX-LM variant regex — same naming conventions apply.
from core.inference.mlx_lm import _extract_mlx_variant  # noqa: E402


class MlxVlmBackend:
    """In-process MLX-VLM backend. One instance per Studio process.

    Lifecycle:
        1. ``load_model(local_path, model_identifier)`` — calls
           ``mlx_vlm.load`` and stashes the (model, processor, config).
        2. ``generate_chat_completion(...)`` — yields cumulative text
           strings followed by a final metadata dict. Matches
           :meth:`MlxLmBackend.generate_chat_completion` contract so the
           route layer's stream plumbing is shared.
        3. ``unload_model()`` — drops refs, ``gc.collect()``,
           ``mx.metal.clear_cache()`` if available.

    Not thread-safe against concurrent ``load_model`` / ``unload_model``
    — guarded by an internal lock.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        # The raw config dict (from config.json) required by
        # ``mlx_vlm.prompt_utils.apply_chat_template``. ``mlx_vlm.load``
        # also attaches it as ``model.config`` but we keep a separate
        # reference so we can inspect ``model_type`` / ``image_token_id``
        # / etc. without poking into the Module.
        self._config: Optional[Dict[str, Any]] = None
        self._model_identifier: Optional[str] = None
        self._local_path: Optional[str] = None
        self._context_length: Optional[int] = None
        self._hf_variant: Optional[str] = None
        # Tool-calling support flag — introspected at load time off the
        # processor's tokenizer chat_template, same rule MlxLmBackend uses.
        self._supports_tools: bool = False
        # Reasoning / <think> support flag — Qwen2.5-VL, Qwen3-VL and
        # Molmo variants all advertise <think>/<thinking> in their chat
        # templates. The base MLX path's ``chat_template_kwargs`` story
        # does not fully apply here because VLM apply_chat_template is
        # called off ``prompt_utils``, not the tokenizer directly.
        self._supports_reasoning: bool = False
        self._reasoning_always_on: bool = False
        self._reasoning_default: bool = True
        self._chat_template: Optional[str] = None
        # Simple load-progress surface; VLM loads are typically < 10 s
        # for a 4-bit 4B so we don't mirror the base backend's
        # download-bytes tracking.
        self._load_phase: Optional[str] = None
        self._lock = threading.Lock()

    # ── Platform gate ────────────────────────────────────────────────
    @staticmethod
    def _platform_ok() -> bool:
        """Return True on macOS with Apple Silicon.

        ``mlx_vlm`` depends on ``mlx`` which is Darwin/arm64 only.
        """
        if platform.system() != "Darwin":
            return False
        machine = platform.machine().lower()
        return machine in ("arm64", "aarch64")

    # ── Properties ────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def is_active(self) -> bool:
        return self.is_loaded

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return True

    @property
    def is_lora(self) -> bool:
        # Phase 6 LoRA for VLM is out of scope; peers must still honor
        # this attribute for the generic status-wiring layer.
        return False

    @property
    def adapter_path(self) -> Optional[str]:
        return None

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    @property
    def context_length(self) -> Optional[int]:
        return self._context_length

    @property
    def max_context_length(self) -> Optional[int]:
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
        return self._reasoning_default

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    @property
    def cache_type_kv(self) -> Optional[str]:
        # KV quantization on VLM: upstream ``mlx_vlm.generate`` forwards
        # ``kv_bits`` / ``kv_group_size`` via **kwargs to the underlying
        # cache path, but the surface is less tested than ``mlx-lm`` and
        # the UI already hides the KV dropdown for vision models. Return
        # None for now — revisit in a later chunk if a user asks.
        return None

    @property
    def speculative_type(self) -> Optional[str]:
        return None

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        return self._config

    # ── Load / unload ─────────────────────────────────────────────
    def load_model(
        self,
        local_path: str,
        model_identifier: str,
        hf_token: Optional[str] = None,
        n_ctx: Optional[int] = None,
    ) -> bool:
        """Load an MLX-VLM checkpoint.

        Args:
            local_path: Directory containing ``config.json``,
                ``preprocessor_config.json`` (and optionally
                ``processor_config.json``), and the MLX weight shards.
            model_identifier: Public-facing id the UI and /models/list
                surface.
            hf_token: Accepted for symmetry; not used for local loads.
            n_ctx: Optional cap on effective context length.

        Returns:
            True on success. False on a failed load (logs the exception).
        """
        if not self._platform_ok():
            raise RuntimeError(
                "mlx_vlm is not available on this platform "
                "(requires macOS on Apple Silicon)"
            )

        with self._lock:
            if self.is_loaded:
                logger.warning(
                    "MlxVlmBackend.load_model called while a model is "
                    "already loaded; unloading first"
                )
                self._unload_locked()

            path = Path(local_path)
            if not path.is_dir():
                raise RuntimeError(
                    f"MLX-VLM model path is not a directory: {local_path}"
                )

            self._load_phase = "loading"

            try:
                from mlx_vlm import load as _mlx_vlm_load  # type: ignore
            except ImportError as e:
                self._load_phase = None
                raise RuntimeError(
                    f"mlx_vlm is not installed in this Python env: {e}"
                ) from e

            t0 = time.time()
            try:
                model, processor = _mlx_vlm_load(str(path))
            except Exception as e:
                self._load_phase = None
                logger.error(f"mlx_vlm.load failed for {local_path}: {e}")
                return False
            load_s = time.time() - t0

            # Read raw config dict from disk for apply_chat_template and
            # for context-length inspection. ``mlx_vlm.load`` attaches it
            # to ``model.config`` but we want it as a plain dict.
            cfg: Optional[Dict[str, Any]] = None
            native_ctx: Optional[int] = None
            try:
                with open(path / "config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except (OSError, ValueError):
                cfg = None
            if isinstance(cfg, dict):
                # Prefer top-level max_position_embeddings; fall back to
                # text_config.max_position_embeddings (Qwen3.5-VL style).
                for k in ("max_position_embeddings",):
                    if isinstance(cfg.get(k), int) and cfg[k] > 0:
                        native_ctx = cfg[k]
                        break
                if native_ctx is None:
                    tc = cfg.get("text_config")
                    if isinstance(tc, dict):
                        v = tc.get("max_position_embeddings")
                        if isinstance(v, int) and v > 0:
                            native_ctx = v

            effective_ctx = native_ctx
            if n_ctx is not None and n_ctx > 0:
                effective_ctx = (
                    min(native_ctx, n_ctx) if native_ctx else n_ctx
                )

            self._model = model
            self._processor = processor
            self._config = cfg
            self._model_identifier = model_identifier
            self._local_path = str(path)
            self._context_length = effective_ctx
            self._hf_variant = _extract_mlx_variant(
                model_identifier
            ) or _extract_mlx_variant(str(path))
            self._load_phase = "loaded"

            # Introspect chat template for tool + reasoning support.
            try:
                tok = getattr(processor, "tokenizer", None)
                tpl = getattr(tok, "chat_template", None) if tok else None
                self._chat_template = tpl if isinstance(tpl, str) else None
            except Exception:
                self._chat_template = None

            self._supports_tools = _detect_tools_from_template(self._chat_template)
            (
                self._supports_reasoning,
                self._reasoning_always_on,
                self._reasoning_default,
            ) = _detect_reasoning_from_template(self._chat_template)

            logger.info(
                f"MLX-VLM model loaded in {load_s:.2f}s: "
                f"identifier={model_identifier} path={path} "
                f"context_length={effective_ctx} "
                f"reasoning={self._supports_reasoning} "
                f"tools={self._supports_tools}"
            )
            return True

    def _unload_locked(self) -> bool:
        if self._model is None and self._processor is None:
            return False
        self._model = None
        self._processor = None
        self._config = None
        self._model_identifier = None
        self._local_path = None
        self._context_length = None
        self._hf_variant = None
        self._supports_tools = False
        self._supports_reasoning = False
        self._reasoning_always_on = False
        self._reasoning_default = True
        self._chat_template = None
        self._load_phase = None
        gc.collect()
        try:
            import mlx.core as _mx  # type: ignore

            if hasattr(_mx, "metal") and hasattr(_mx.metal, "clear_cache"):
                _mx.metal.clear_cache()
        except Exception:
            pass
        return True

    def unload_model(self) -> bool:
        with self._lock:
            return self._unload_locked()

    # ── Prompt rendering ──────────────────────────────────────────
    def _render_prompt(
        self,
        messages: List[Dict[str, Any]],
        *,
        num_images: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Render a VLM prompt via ``mlx_vlm.prompt_utils.apply_chat_template``.

        The upstream signature is::

            apply_chat_template(processor, config, prompt, ...,
                                num_images=0, num_audios=0, **kwargs)

        where ``prompt`` may be a bare string (single-turn ask) OR a
        list of OpenAI-format messages. We always pass the full messages
        list so multi-turn conversations render correctly.

        Tool-schema injection: ``apply_chat_template`` in ``mlx-vlm``
        does NOT accept a ``tools=`` kwarg reliably across models, so
        when tools are supplied we add a synthetic system message with
        a compact JSON-schema block rather than routing the tools
        kwarg through the template engine. This mirrors what the GGUF
        backend's no-template-tools fallback does.
        """
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore

        # Tool-injection: prepend a system message with the schema.
        effective_messages = messages
        if tools:
            schema_lines = ["You have access to the following tools:"]
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name", "<unnamed>")
                desc = fn.get("description", "")
                schema_lines.append(f"- {name}: {desc}")
            schema_lines.append(
                "To call a tool, respond with a JSON object of the form "
                "{\"tool_calls\":[{\"name\":...,\"arguments\":{...}}]}."
            )
            tool_system = "\n".join(schema_lines)
            # Merge with an existing system message if present.
            effective_messages = list(messages)
            if effective_messages and effective_messages[0].get("role") == "system":
                old = effective_messages[0].get("content", "")
                if isinstance(old, str) and old:
                    effective_messages[0] = {
                        "role": "system",
                        "content": f"{old}\n\n{tool_system}",
                    }
                else:
                    effective_messages[0] = {
                        "role": "system",
                        "content": tool_system,
                    }
            else:
                effective_messages = [
                    {"role": "system", "content": tool_system}
                ] + effective_messages

        kwargs: Dict[str, Any] = {
            "num_images": int(num_images),
            "num_audios": 0,
            "add_generation_prompt": True,
        }
        # Forward enable_thinking only when the model advertises it. Not
        # every VLM template will honor the kwarg, so wrap in try/except.
        if self._supports_reasoning and enable_thinking is not None:
            kwargs["chat_template_kwargs"] = {
                "enable_thinking": bool(enable_thinking)
            }

        try:
            return apply_chat_template(
                self._processor, self._config, effective_messages, **kwargs
            )
        except TypeError:
            # Older mlx-vlm may not accept chat_template_kwargs — retry
            # without it.
            kwargs.pop("chat_template_kwargs", None)
            return apply_chat_template(
                self._processor, self._config, effective_messages, **kwargs
            )
        except Exception as e:
            logger.warning(
                f"mlx_vlm apply_chat_template failed ({e}); falling back "
                f"to a minimal ChatML-style render"
            )
            parts: List[str] = []
            for m in effective_messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    # Content-parts: collapse to text-only here; the
                    # image is still passed through the generate call.
                    text = "".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                    content = text
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    # ── Image handling ────────────────────────────────────────────
    @staticmethod
    def _decode_image_b64_to_path(image_b64: str, tmpdir: str) -> str:
        """Decode a base64 image to a file on disk; return the path.

        mlx-vlm accepts either a path or a PIL Image as ``image=``. We
        write to a temp file because passing a path is the least
        version-sensitive of the two and keeps us out of PIL format
        drama (RGBA / P / CMYK) for now.

        Data-URL prefix is stripped if present. Format is auto-detected
        by Pillow when mlx-vlm loads it.
        """
        # Strip data URL prefix if present.
        s = image_b64.strip()
        if s.startswith("data:"):
            _, _, s = s.partition(",")
        raw = base64.b64decode(s)
        from io import BytesIO
        from PIL import Image as _Image

        # Normalize to RGB PNG for consistency.
        img = _Image.open(BytesIO(raw)).convert("RGB")
        fd, outpath = tempfile.mkstemp(suffix=".png", dir=tmpdir, prefix="mlxvlm_")
        os.close(fd)
        img.save(outpath, format="PNG")
        return outpath

    # ── Generate ──────────────────────────────────────────────────
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
        """Stream a VLM chat completion.

        Contract mirrors :meth:`MlxLmBackend.generate_chat_completion`:

        - Yields cumulative text strings (not deltas).
        - After the text stream ends, yields exactly one
          ``{"type": "metadata", "usage": {...}, "timings": {...},
          "finish_reason": ...}`` dict.

        Image input:
        - ``image_b64`` may be omitted — in which case this is a
          text-only turn on a vision-capable model (valid, Qwen3.5-VL
          responds fine to no-image questions).
        - Base64 is decoded and written to a temp file; ``mlx_vlm``
          reads it. The temp file is removed after generation.
        - Only one image per turn is supported in Phase 9.
        """
        if not self.is_loaded:
            raise RuntimeError("MLX-VLM model is not loaded")

        try:
            from mlx_vlm import stream_generate  # type: ignore
        except ImportError as e:
            raise RuntimeError(f"mlx_vlm is not installed: {e}") from e

        num_images = 1 if image_b64 else 0
        prompt = self._render_prompt(
            messages,
            num_images=num_images,
            tools=None,
            enable_thinking=enable_thinking,
        )

        tmpdir_ctx: Optional[tempfile.TemporaryDirectory] = None
        image_path: Optional[str] = None
        try:
            if image_b64:
                tmpdir_ctx = tempfile.TemporaryDirectory(prefix="mlxvlm-")
                try:
                    image_path = self._decode_image_b64_to_path(
                        image_b64, tmpdir_ctx.name
                    )
                except Exception as e:
                    raise ValueError(f"Failed to decode image_b64: {e}") from e

            sg_kwargs: Dict[str, Any] = {
                "prompt": prompt,
            }
            if image_path is not None:
                sg_kwargs["image"] = image_path
            if max_tokens is not None and max_tokens > 0:
                sg_kwargs["max_tokens"] = int(max_tokens)
            # mlx-vlm ``stream_generate`` forwards unknown kwargs to the
            # sampler; pass the common sampling knobs through.
            if temperature and temperature > 0:
                sg_kwargs["temperature"] = float(temperature)
            else:
                sg_kwargs["temperature"] = 0.0
            if top_p and top_p > 0:
                sg_kwargs["top_p"] = float(top_p)
            if top_k and top_k > 0:
                sg_kwargs["top_k"] = int(top_k)
            if min_p and min_p > 0:
                sg_kwargs["min_p"] = float(min_p)
            if repetition_penalty is not None and float(repetition_penalty) > 1.0:
                sg_kwargs["repetition_penalty"] = float(repetition_penalty)

            # Normalize stop strings same as MlxLmBackend.
            stop_strings: List[str] = []
            if stop:
                if isinstance(stop, str):
                    stop_strings = [stop]
                else:
                    stop_strings = [s for s in stop if isinstance(s, str) and s]
            max_stop_len = max((len(s) for s in stop_strings), default=0)

            cumulative = ""
            last_resp: Any = None
            finish_reason = "stop"

            try:
                for resp in stream_generate(
                    self._model, self._processor, **sg_kwargs
                ):
                    last_resp = resp
                    if cancel_event is not None and cancel_event.is_set():
                        finish_reason = "cancelled"
                        break
                    text = getattr(resp, "text", None)
                    if text is None and isinstance(resp, str):
                        # Older mlx-vlm yielded plain strings.
                        text = resp
                    if not text:
                        continue
                    cumulative += text

                    if stop_strings:
                        scan_start = max(
                            0, len(cumulative) - (max_stop_len + len(text))
                        )
                        hay = cumulative[scan_start:]
                        earliest_rel: Optional[int] = None
                        for s in stop_strings:
                            idx = hay.find(s)
                            if idx != -1 and (
                                earliest_rel is None or idx < earliest_rel
                            ):
                                earliest_rel = idx
                        if earliest_rel is not None:
                            cut = scan_start + earliest_rel
                            cumulative = cumulative[:cut]
                            yield cumulative
                            break

                    yield cumulative
            except Exception as e:
                logger.error(f"mlx_vlm.stream_generate raised: {e}")
                raise

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
                    "predicted_per_second": getattr(
                        last_resp, "generation_tps", None
                    ),
                }

            yield {
                "type": "metadata",
                "usage": usage,
                "timings": timings,
                "finish_reason": finish_reason,
            }
        finally:
            if tmpdir_ctx is not None:
                try:
                    tmpdir_ctx.cleanup()
                except Exception:
                    pass

    # ── Agentic tool-call loop ────────────────────────────────────
    def generate_chat_completion_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Any] = None,
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
        max_tool_iterations: int = 10,
        auto_heal_tool_calls: bool = True,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
    ) -> Generator[Union[Dict[str, Any], str], None, None]:
        """VLM peer of :meth:`MlxLmBackend.generate_chat_completion_with_tools`.

        Runs the same agentic tool-use loop the GGUF and MLX-LM
        backends run, with one addition: ``image_b64`` is accepted on
        the FIRST turn (tool-calling with an image attached). Later
        iterations run text-only — mlx-vlm's ``stream_generate`` accepts
        ``image=None`` and the tool-call history is passed via the chat
        template.

        Yielded events mirror the MLX-LM method exactly so the route
        layer's ``_mlx_vlm_agentic_stream`` can reuse the same SSE-frame
        shapes:

        - ``{"type": "content", "text": cumulative}``
        - ``{"type": "status", "text": "Calling tool X ..."}``
        - ``{"type": "tool_start", "tool_name", "tool_call_id",
          "arguments"}``
        - ``{"type": "tool_end", "tool_name", "tool_call_id",
          "result"}``
        - ``{"type": "metadata", "usage": {...}, "timings": {...}}``

        Semantics:

        - ``tool_choice="none"`` short-circuits — one plain generation
          turn runs without tools.
        - The tool schema is injected via :meth:`_render_prompt`'s
          synthetic-system-message path (the VLM ``apply_chat_template``
          doesn't reliably accept a ``tools=`` kwarg across models).
        - After each turn the cumulative text goes through
          :func:`core.inference._tool_call_parser.parse_tool_calls_from_text`
          — no calls → emit final content and metadata, done. Calls →
          execute serially via :func:`core.inference.tools.execute_tool`
          with the same per-call hard-timeout wrapper MLX-LM uses.
        - Cancellation is checked at every iteration boundary and
          every 0.5 s during a tool execution.
        """
        if not self.is_loaded:
            raise RuntimeError("MLX-VLM model is not loaded")
        if not self.supports_tools:
            raise RuntimeError(
                "Loaded MLX-VLM model does not advertise tool-calling "
                "support (chat template does not mention tools / "
                "tool_calls). Reload a tool-capable VLM checkpoint "
                "(e.g. mlx-community Gemma-4 / Qwen3-VL)."
            )

        from core.inference._tool_call_parser import (
            TOOL_XML_SIGNALS,
            parse_tool_calls_from_text,
            strip_tool_markup,
        )
        from core.inference.tools import execute_tool

        tool_choice_norm = self._normalize_tool_choice(tool_choice)
        conversation = [dict(m) for m in messages]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        accumulated_predicted_ms = 0.0
        accumulated_predicted_n = 0

        if tool_choice_norm == "none":
            # Plain single-turn; image still honoured.
            yield from self._run_plain_vlm_tool_turn(
                conversation = conversation,
                tools = tools,
                tool_choice = tool_choice_norm,
                image_b64 = image_b64,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = max_tokens,
                repetition_penalty = repetition_penalty,
                repetition_context_size = repetition_context_size,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                cancel_event = cancel_event,
                enable_thinking = enable_thinking,
            )
            return

        for iteration in range(max_tool_iterations):
            if cancel_event is not None and cancel_event.is_set():
                return

            # Only the FIRST iteration uses the user-provided image.
            # Subsequent iterations are text-only (the image has already
            # been consumed by the encoder, and passing it again would
            # re-encode + bloat context).
            iter_image = image_b64 if iteration == 0 else None

            turn_text = ""
            turn_usage: Dict[str, Any] = {}
            turn_timings: Dict[str, Any] = {}
            for event in self._stream_vlm_assistant_turn(
                conversation = conversation,
                tools = tools,
                tool_choice = tool_choice_norm,
                image_b64 = iter_image,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = max_tokens,
                repetition_penalty = repetition_penalty,
                repetition_context_size = repetition_context_size,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                cancel_event = cancel_event,
                enable_thinking = enable_thinking,
            ):
                if isinstance(event, dict):
                    if event.get("type") == "metadata":
                        turn_usage = event.get("usage", {}) or {}
                        turn_timings = event.get("timings", {}) or {}
                    continue
                turn_text = event
                # Same two-stage clean as MlxLmBackend: strip closed
                # tool-call blocks then hold back anything from the
                # earliest unclosed signal onwards so partial markup
                # doesn't leak to the SSE wire.
                cleaned = (
                    strip_tool_markup(turn_text)
                    if auto_heal_tool_calls
                    else turn_text
                )
                if auto_heal_tool_calls:
                    earliest_signal = -1
                    for sig in TOOL_XML_SIGNALS:
                        idx = cleaned.find(sig)
                        if idx >= 0 and (
                            earliest_signal < 0 or idx < earliest_signal
                        ):
                            earliest_signal = idx
                    if earliest_signal >= 0:
                        cleaned = cleaned[:earliest_signal]
                yield {"type": "content", "text": cleaned}

            if cancel_event is not None and cancel_event.is_set():
                return

            total_prompt_tokens = turn_usage.get(
                "prompt_tokens", total_prompt_tokens
            )
            total_completion_tokens += int(
                turn_usage.get("completion_tokens", 0) or 0
            )
            pm = (
                turn_timings.get("predicted_ms")
                if isinstance(turn_timings, dict)
                else None
            )
            pn = (
                turn_timings.get("predicted_n")
                if isinstance(turn_timings, dict)
                else None
            )
            if isinstance(pm, (int, float)):
                accumulated_predicted_ms += float(pm)
            if isinstance(pn, (int, float)):
                accumulated_predicted_n += int(pn)

            tool_calls = (
                parse_tool_calls_from_text(turn_text)
                if auto_heal_tool_calls
                else []
            )

            if not tool_calls:
                final_text = (
                    strip_tool_markup(turn_text, final = True)
                    if auto_heal_tool_calls
                    else turn_text
                )
                yield {"type": "content", "text": final_text}
                yield {"type": "status", "text": ""}
                yield self._build_vlm_metadata_event(
                    prompt_tokens = total_prompt_tokens,
                    completion_tokens = total_completion_tokens,
                    predicted_ms = accumulated_predicted_ms,
                    predicted_n = accumulated_predicted_n,
                    base_timings = turn_timings,
                )
                return

            # Record the assistant's turn with its tool_calls so
            # apply_chat_template in the next iteration has history.
            assistant_content = (
                strip_tool_markup(turn_text, final = True)
                if auto_heal_tool_calls
                else turn_text
            )
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
            conversation.append(assistant_msg)

            for tc in tool_calls:
                if cancel_event is not None and cancel_event.is_set():
                    return
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                raw_args = func.get("arguments", "")
                if isinstance(raw_args, str):
                    try:
                        arguments = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        arguments = {"raw": raw_args}
                else:
                    arguments = raw_args

                status = self._vlm_tool_status_text(tool_name, arguments)
                yield {"type": "status", "text": status}
                yield {
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "tool_call_id": tc.get("id", ""),
                    "arguments": arguments,
                }

                # Hard-timeout wrapper — same shape as MlxLmBackend uses.
                try:
                    effective_timeout = (
                        None if tool_call_timeout >= 9999
                        else tool_call_timeout
                    )
                    _TOOL_TIMEOUT_CAPS = {
                        "web_search": 30,
                        "fetch_url": 30,
                    }
                    cap = _TOOL_TIMEOUT_CAPS.get(tool_name)
                    if cap is not None and (
                        effective_timeout is None
                        or effective_timeout > cap
                    ):
                        effective_timeout = cap
                    tool_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers = 1,
                        thread_name_prefix = "mlx-vlm-tool-exec",
                    )
                    try:
                        tool_future = tool_executor.submit(
                            execute_tool,
                            tool_name,
                            arguments,
                            cancel_event = cancel_event,
                            timeout = effective_timeout,
                            session_id = session_id,
                        )
                        deadline = (
                            time.monotonic() + effective_timeout
                            if effective_timeout is not None
                            else None
                        )
                        while True:
                            remaining = (
                                deadline - time.monotonic()
                                if deadline is not None
                                else None
                            )
                            if remaining is not None and remaining <= 0:
                                raise concurrent.futures.TimeoutError()
                            wait_for = (
                                min(0.5, remaining)
                                if remaining is not None
                                else 0.5
                            )
                            try:
                                result = tool_future.result(timeout = wait_for)
                                break
                            except concurrent.futures.TimeoutError:
                                if (
                                    cancel_event is not None
                                    and cancel_event.is_set()
                                ):
                                    return
                                continue
                    finally:
                        tool_executor.shutdown(wait = False)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "VLM tool '%s' exceeded %ss timeout; abandoning "
                        "background thread.",
                        tool_name,
                        effective_timeout,
                    )
                    result = (
                        f"Error executing {tool_name}: timed out after "
                        f"{effective_timeout}s."
                    )
                except Exception as exc:
                    result = f"Error executing {tool_name}: {exc}"

                yield {
                    "type": "tool_end",
                    "tool_name": tool_name,
                    "tool_call_id": tc.get("id", ""),
                    "result": result,
                }

                tool_msg: Dict[str, Any] = {
                    "role": "tool",
                    "name": tool_name,
                    "content": (
                        result if isinstance(result, str) else str(result)
                    ),
                }
                tc_id = tc.get("id")
                if tc_id:
                    tool_msg["tool_call_id"] = tc_id
                conversation.append(tool_msg)

            yield {"type": "status", "text": ""}

        # ── Tool iteration cap reached ────────────────────────────
        passthrough_mode = max_tool_iterations == 0
        if max_tool_iterations > 0:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have used all available tool calls. Based on "
                        "everything you found so far, provide your final "
                        "answer now. Do not call any more tools."
                    ),
                }
            )
        yield {"type": "status", "text": ""}
        final_text_cum = ""
        final_usage: Dict[str, Any] = {}
        final_timings: Dict[str, Any] = {}
        for event in self._stream_vlm_assistant_turn(
            conversation = conversation,
            tools = tools if passthrough_mode else None,
            tool_choice = tool_choice_norm if passthrough_mode else "none",
            # Cap-reached final turn is always text-only — the image was
            # consumed on iteration 0.
            image_b64 = None,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            max_tokens = max_tokens,
            repetition_penalty = repetition_penalty,
            repetition_context_size = repetition_context_size,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            logit_bias = logit_bias,
            stop = stop,
            cancel_event = cancel_event,
            enable_thinking = enable_thinking,
        ):
            if isinstance(event, dict):
                if event.get("type") == "metadata":
                    final_usage = event.get("usage", {}) or {}
                    final_timings = event.get("timings", {}) or {}
                continue
            final_text_cum = event
            yield {"type": "content", "text": final_text_cum}

        total_prompt_tokens = final_usage.get(
            "prompt_tokens", total_prompt_tokens
        )
        total_completion_tokens += int(
            final_usage.get("completion_tokens", 0) or 0
        )
        yield self._build_vlm_metadata_event(
            prompt_tokens = total_prompt_tokens,
            completion_tokens = total_completion_tokens,
            predicted_ms = accumulated_predicted_ms,
            predicted_n = accumulated_predicted_n,
            base_timings = final_timings,
        )

    # ── Helpers for the VLM tool loop ─────────────────────────────

    @staticmethod
    def _normalize_tool_choice(tool_choice: Any) -> Optional[str]:
        """Collapse OpenAI ``tool_choice`` to ``"auto"`` / ``"required"`` /
        ``"none"`` / None. Mirrors :meth:`MlxLmBackend._normalize_tool_choice`.
        """
        if tool_choice is None:
            return "auto"
        if isinstance(tool_choice, str):
            low = tool_choice.strip().lower()
            if low in ("auto", "required", "none"):
                return low
            return "auto"
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                return "required"
            return "auto"
        return "auto"

    def _stream_vlm_assistant_turn(
        self,
        *,
        conversation: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        image_b64: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: Optional[int],
        repetition_penalty: float,
        repetition_context_size: Optional[int],
        presence_penalty: float,
        frequency_penalty: float,
        logit_bias: Optional[Dict[int, float]],
        stop: Optional[List[str]],
        cancel_event: Optional[threading.Event],
        enable_thinking: Optional[bool],
    ) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """Stream one assistant turn with the current conversation.

        Mirrors :meth:`generate_chat_completion` but passes ``tools``
        into the prompt builder and accepts a per-turn image. The body
        is a near-clone — the two generators can't share code because
        the prompt must be rebuilt per turn (the conversation grows
        between iterations).
        """
        try:
            from mlx_vlm import stream_generate  # type: ignore
        except ImportError as e:
            raise RuntimeError(f"mlx_vlm is not installed: {e}") from e

        render_tools = tools if (tools and tool_choice != "none") else None
        num_images = 1 if image_b64 else 0
        prompt = self._render_prompt(
            conversation,
            num_images = num_images,
            tools = render_tools,
            enable_thinking = enable_thinking,
        )

        tmpdir_ctx: Optional[tempfile.TemporaryDirectory] = None
        image_path: Optional[str] = None
        try:
            if image_b64:
                tmpdir_ctx = tempfile.TemporaryDirectory(prefix = "mlxvlm-")
                try:
                    image_path = self._decode_image_b64_to_path(
                        image_b64, tmpdir_ctx.name
                    )
                except Exception as e:
                    raise ValueError(f"Failed to decode image_b64: {e}") from e

            sg_kwargs: Dict[str, Any] = {"prompt": prompt}
            if image_path is not None:
                sg_kwargs["image"] = image_path
            if max_tokens is not None and max_tokens > 0:
                sg_kwargs["max_tokens"] = int(max_tokens)
            if temperature and temperature > 0:
                sg_kwargs["temperature"] = float(temperature)
            else:
                sg_kwargs["temperature"] = 0.0
            if top_p and top_p > 0:
                sg_kwargs["top_p"] = float(top_p)
            if top_k and top_k > 0:
                sg_kwargs["top_k"] = int(top_k)
            if min_p and min_p > 0:
                sg_kwargs["min_p"] = float(min_p)
            if (
                repetition_penalty is not None
                and float(repetition_penalty) > 1.0
            ):
                sg_kwargs["repetition_penalty"] = float(repetition_penalty)

            stop_strings: List[str] = []
            if stop:
                if isinstance(stop, str):
                    stop_strings = [stop]
                else:
                    stop_strings = [
                        s for s in stop if isinstance(s, str) and s
                    ]
            max_stop_len = max((len(s) for s in stop_strings), default = 0)

            cumulative = ""
            last_resp: Any = None
            finish_reason = "stop"

            try:
                for resp in stream_generate(
                    self._model, self._processor, **sg_kwargs
                ):
                    last_resp = resp
                    if cancel_event is not None and cancel_event.is_set():
                        finish_reason = "cancelled"
                        break
                    text = getattr(resp, "text", None)
                    if text is None and isinstance(resp, str):
                        text = resp
                    if not text:
                        continue
                    cumulative += text
                    if stop_strings:
                        scan_start = max(
                            0,
                            len(cumulative) - (max_stop_len + len(text)),
                        )
                        hay = cumulative[scan_start:]
                        earliest_rel: Optional[int] = None
                        for s in stop_strings:
                            idx = hay.find(s)
                            if idx != -1 and (
                                earliest_rel is None
                                or idx < earliest_rel
                            ):
                                earliest_rel = idx
                        if earliest_rel is not None:
                            cut = scan_start + earliest_rel
                            cumulative = cumulative[:cut]
                            yield cumulative
                            break
                    yield cumulative
            except Exception as e:
                logger.error(f"mlx_vlm stream_generate (tool turn) raised: {e}")
                raise

            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            timings: Dict[str, Any] = {
                "prompt_per_second": None,
                "predicted_per_second": None,
            }
            if last_resp is not None:
                pt = int(getattr(last_resp, "prompt_tokens", 0) or 0)
                gt = int(getattr(last_resp, "generation_tokens", 0) or 0)
                usage = {
                    "prompt_tokens": pt,
                    "completion_tokens": gt,
                    "total_tokens": pt + gt,
                }
                timings = {
                    "prompt_per_second": getattr(
                        last_resp, "prompt_tps", None
                    ),
                    "predicted_per_second": getattr(
                        last_resp, "generation_tps", None
                    ),
                }
            yield {
                "type": "metadata",
                "usage": usage,
                "timings": timings,
                "finish_reason": finish_reason,
            }
        finally:
            if tmpdir_ctx is not None:
                try:
                    tmpdir_ctx.cleanup()
                except Exception:
                    pass

    def _run_plain_vlm_tool_turn(
        self, **kwargs: Any
    ) -> Generator[Union[Dict[str, Any], str], None, None]:
        """``tool_choice="none"`` path: run one turn, relay events."""
        conversation = kwargs.pop("conversation")
        for event in self._stream_vlm_assistant_turn(
            conversation = conversation, **kwargs
        ):
            if isinstance(event, dict):
                if event.get("type") == "metadata":
                    yield {
                        "type": "metadata",
                        "usage": event.get("usage", {}),
                        "timings": event.get("timings", {}),
                    }
                continue
            yield {"type": "content", "text": event}
        yield {"type": "status", "text": ""}

    @staticmethod
    def _vlm_tool_status_text(
        tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Build UI status text for a tool invocation.

        Mirrors :meth:`MlxLmBackend._tool_status_text` so the frontend
        badges look identical regardless of backend.
        """
        if tool_name == "web_search":
            url = (arguments.get("url") or "").strip()
            if url:
                return f"Reading: {url[:80]}"
            return f"Searching: {arguments.get('query', '')[:80]}"
        if tool_name == "python":
            preview = (arguments.get("code") or "").strip().split("\n")[0][:60]
            return (
                f"Running Python: {preview}" if preview
                else "Running Python..."
            )
        if tool_name == "terminal":
            cmd = (arguments.get("command") or "")[:60]
            return f"Running: {cmd}" if cmd else "Running command..."
        return f"Calling: {tool_name}"

    @staticmethod
    def _build_vlm_metadata_event(
        *,
        prompt_tokens: int,
        completion_tokens: int,
        predicted_ms: float,
        predicted_n: int,
        base_timings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Final metadata event — same shape as MLX-LM so the SSE driver
        can be shared.
        """
        timings = dict(base_timings) if isinstance(base_timings, dict) else {}
        if predicted_ms > 0.0:
            timings["predicted_ms"] = predicted_ms
        if predicted_n > 0:
            timings["predicted_n"] = predicted_n
        return {
            "type": "metadata",
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            },
            "timings": timings,
        }

    # ── Progress surface (matches MlxLmBackend.load_progress) ─────
    def load_progress(self) -> Dict[str, Any]:
        """Return a minimal progress dict for the status endpoint.

        VLM loads are fast (< 10 s on a 4B 4-bit) so we don't track
        bytes — just report the phase for UI consistency.
        """
        return {
            "phase": self._load_phase,
            "bytes_loaded": 0,
            "bytes_total": 0,
            "fraction": 1.0 if self._load_phase == "loaded" else 0.0,
        }


# ── Template-based detection helpers (pure functions) ────────────
def _detect_tools_from_template(template: Optional[str]) -> bool:
    """True iff *template* renders tool_calls / tools blocks.

    Mirrors :meth:`MlxLmBackend._detect_tools` — same heuristic: look
    for ``tool_calls`` OR ``tools`` in the template body.
    """
    if not template or not isinstance(template, str):
        return False
    tpl_lc = template.lower()
    return "tool_calls" in tpl_lc or "tools" in tpl_lc


_REASONING_MARKERS = (
    "<think>",
    "<thinking>",
    "enable_thinking",
    "reasoning_content",
)


def _detect_reasoning_from_template(
    template: Optional[str],
) -> Tuple[bool, bool, bool]:
    """Return (supports_reasoning, always_on, default_on).

    - ``supports_reasoning``: template mentions any of the markers.
    - ``always_on``: template emits ``<think>`` unconditionally (no
      enable_thinking gate). Approximated by "contains <think> but not
      enable_thinking".
    - ``default_on``: True when the model reasons by default.

    Matches :meth:`MlxLmBackend._detect_reasoning` (same heuristic).
    """
    if not template or not isinstance(template, str):
        return (False, False, True)
    tpl_lc = template.lower()
    has_marker = any(m in tpl_lc for m in _REASONING_MARKERS)
    has_gate = "enable_thinking" in tpl_lc
    always_on = has_marker and not has_gate and ("<think>" in tpl_lc)
    return (has_marker, always_on, True)
