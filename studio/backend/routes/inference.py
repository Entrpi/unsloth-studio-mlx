# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference API routes for model loading and text generation.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from fastapi import (
    APIRouter,
    Depends,
    File as FastAPIFile,
    Form as FastAPIForm,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse, JSONResponse, Response
from typing import Optional
import json
import httpx
import structlog
from loggers import get_logger
import asyncio
import threading


import re as _re

# Model size extraction (shared with core/inference/llama_cpp.py)
from utils.models import extract_model_size_b as _extract_model_size_b


def _friendly_error(exc: Exception) -> str:
    """Extract a user-friendly message from known llama-server errors."""
    # httpx transport-layer failures reaching the managed llama-server —
    # raised by the async pass-through helpers that talk to llama-server
    # directly. Treat any RequestError subclass (ConnectError, ReadError,
    # RemoteProtocolError, WriteError, PoolTimeout, ...) as "the upstream
    # subprocess is unreachable", which for Studio always means the
    # llama-server subprocess crashed or is still coming up.
    if isinstance(exc, httpx.RequestError):
        return "Lost connection to the model server. It may have crashed -- try reloading the model."
    msg = str(exc)
    m = _re.search(
        r"request \((\d+) tokens?\) exceeds the available context size \((\d+) tokens?\)",
        msg,
    )
    if m:
        return (
            f"Message too long: {m.group(1)} tokens exceeds the {m.group(2)}-token "
            f"context window. Try increasing the Context Length in Model settings, "
            f"or shorten the conversation."
        )
    if "Lost connection to llama-server" in msg:
        return "Lost connection to the model server. It may have crashed -- try reloading the model."
    return "An internal error occurred"


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults

from models.inference import (
    LoadRequest,
    UnloadRequest,
    GenerateRequest,
    LoadResponse,
    LoadProgressResponse,
    UnloadResponse,
    InferenceStatusResponse,
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletion,
    ChatMessage,
    ChunkChoice,
    ChoiceDelta,
    ToolCallDelta,
    ToolCallFunctionDelta,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
    ValidateModelRequest,
    ValidateModelResponse,
    TextContentPart,
    ImageContentPart,
    ImageUrl,
    ResponsesRequest,
    ResponsesInputMessage,
    ResponsesInputTextPart,
    ResponsesInputImagePart,
    ResponsesOutputTextContent,
    ResponsesOutputMessage,
    ResponsesUsage,
    ResponsesResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicResponseTextBlock,
    AnthropicResponseToolUseBlock,
    AnthropicUsage,
)
from core.inference.anthropic_compat import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    anthropic_tool_choice_to_openai,
    AnthropicStreamEmitter,
    AnthropicPassthroughEmitter,
)
from auth.authentication import get_current_subject

import io
import wave
import base64
import numpy as np
from datetime import date as _date

router = APIRouter()

# Appended to tool-use nudge to discourage plan-without-action
_TOOL_ACTION_NUDGE = (
    " IMPORTANT: Always call tools directly -- never write code yourself."
    " Never describe what you plan to do -- just call the tool immediately."
    " For any code request, call the python tool. For any factual question, call web_search."
    " Do NOT output code blocks -- use the python tool instead."
)

# Regex for stripping leaked tool-call XML from assistant messages/stream
_TOOL_XML_RE = _re.compile(
    r"<tool_call>.*?</tool_call>|<function=\w+>.*?</function>",
    _re.DOTALL,
)
logger = get_logger(__name__)


# GGUF inference backend (llama-server)
_llama_cpp_backend = LlamaCppBackend()


def get_llama_cpp_backend() -> LlamaCppBackend:
    return _llama_cpp_backend


# MLX inference backend (Apple Silicon). Lazy-constructed so non-Darwin
# CI does not pay the import cost and does not fail when mlx_lm is
# unavailable.
_mlx_lm_backend: Optional["MlxLmBackend"] = None  # noqa: F821 (forward ref)


def get_mlx_lm_backend():
    """Return the process-wide ``MlxLmBackend`` singleton.

    Instantiation is deferred to the first call so Linux / Windows
    imports of this module do not need ``mlx_lm`` available. The class
    itself does not import ``mlx_lm`` at construction time; the actual
    ``mlx_lm.load`` call is deferred further to ``load_model``.
    """
    global _mlx_lm_backend
    if _mlx_lm_backend is None:
        from core.inference.mlx_lm import MlxLmBackend

        _mlx_lm_backend = MlxLmBackend()
    return _mlx_lm_backend


# Phase 9 (Chunk D) — MLX vision-language backend. Same lazy-singleton
# pattern as ``get_mlx_lm_backend`` — ``mlx-vlm`` imports transitively
# pull in ``mlx``, ``transformers``, and PIL and are Darwin-only.
_mlx_vlm_backend: Optional["MlxVlmBackend"] = None  # noqa: F821


def get_mlx_vlm_backend():
    """Return the process-wide ``MlxVlmBackend`` singleton."""
    global _mlx_vlm_backend
    if _mlx_vlm_backend is None:
        from core.inference.mlx_vlm import MlxVlmBackend

        _mlx_vlm_backend = MlxVlmBackend()
    return _mlx_vlm_backend


# Phase 10 (Chunk D) — MLX audio backend (TTS / ASR / omni). Same
# lazy-singleton pattern; ``mlx-audio`` pulls in numba / librosa /
# soundfile which are heavy imports.
_mlx_audio_backend: Optional["MlxAudioBackend"] = None  # noqa: F821


def get_mlx_audio_backend():
    """Return the process-wide ``MlxAudioBackend`` singleton."""
    global _mlx_audio_backend
    if _mlx_audio_backend is None:
        from core.inference.mlx_audio import MlxAudioBackend

        _mlx_audio_backend = MlxAudioBackend()
    return _mlx_audio_backend


def _unload_all_mlx_peers(
    *,
    keep: Optional[str] = None,
) -> None:
    """Unload all MLX / GGUF peer backends except the one named in *keep*.

    Chunk D peer-unload helper. Registered backends:

    - ``"mlx"``     — :func:`get_mlx_lm_backend`
    - ``"mlx_vlm"`` — :func:`get_mlx_vlm_backend`
    - ``"mlx_audio"`` — :func:`get_mlx_audio_backend`
    - ``"gguf"``    — :func:`get_llama_cpp_backend`
    - ``"unsloth"`` — :func:`get_inference_backend`

    Per the roadmap (Section 4.6), the peer-unload loop is pulled into
    a helper here rather than continuing to inline every branch for
    every new backend. This runs synchronously — callers from async
    context should still wrap in ``asyncio.to_thread`` when the peer
    unload is known to be slow (GGUF llama-server teardown is ~1 s
    worst case).
    """
    if keep != "mlx":
        b = get_mlx_lm_backend()
        if b.is_loaded:
            logger.info("Unloading MLX text model (peer-unload)")
            b.unload_model()
    if keep != "mlx_vlm":
        b = get_mlx_vlm_backend()
        if b.is_loaded:
            logger.info("Unloading MLX-VLM model (peer-unload)")
            b.unload_model()
    if keep != "mlx_audio":
        b = get_mlx_audio_backend()
        if b.is_loaded:
            logger.info("Unloading MLX-Audio model (peer-unload)")
            b.unload_model()
    if keep != "gguf":
        b = get_llama_cpp_backend()
        if b.is_loaded:
            logger.info("Unloading GGUF model (peer-unload)")
            b.unload_model()
    if keep != "unsloth":
        b = get_inference_backend()
        if b.active_model_name:
            logger.info(
                f"Unloading Unsloth model '{b.active_model_name}' (peer-unload)"
            )
            b.unload_model(b.active_model_name)


@router.post("/load", response_model = LoadResponse)
async def load_model(
    request: LoadRequest,
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Load a model for inference.

    The model_path should be a clean identifier from GET /models/list.
    Returns inference configuration parameters (temperature, top_p, top_k, min_p)
    from the model's YAML config, falling back to default.yaml for missing values.

    GGUF models are loaded via llama-server (llama.cpp) instead of Unsloth.
    """
    try:
        # Version switching is handled automatically by the subprocess-based
        # inference backend — no need for ensure_transformers_version() here.

        # ── Already-loaded check: skip reload if the exact model is active ──
        backend = get_inference_backend()
        llama_backend = get_llama_cpp_backend()
        mlx_backend = get_mlx_lm_backend()

        # MLX short-circuit (must precede GGUF/Unsloth checks because the
        # MLX backend's model_identifier is the local dir name, which may
        # collide with a user-visible Unsloth path).
        if (
            mlx_backend.is_loaded
            and mlx_backend.model_identifier
            and mlx_backend.model_identifier.lower() == request.model_path.lower()
            and not request.gguf_variant
        ):
            logger.info(
                f"Model already loaded (MLX): {request.model_path}, skipping reload"
            )
            inference_config = load_inference_config(mlx_backend.model_identifier)
            return LoadResponse(
                status = "already_loaded",
                model = mlx_backend.model_identifier,
                display_name = mlx_backend.model_identifier,
                is_vision = False,
                is_lora = False,
                is_gguf = False,
                is_mlx = True,
                # Phase 6: surface lora-active state on the already-loaded branch.
                is_mlx_lora = mlx_backend.is_lora,
                is_audio = False,
                inference = inference_config,
                requires_trust_remote_code = False,
                context_length = mlx_backend.context_length,
                max_context_length = mlx_backend.max_context_length,
                native_context_length = mlx_backend.native_context_length,
                # Phase 4: surface reasoning introspection from the backend.
                supports_reasoning = mlx_backend.supports_reasoning,
                reasoning_always_on = mlx_backend.reasoning_always_on,
                # Phase 5: surface tool-calling support on the
                # "already loaded" short-circuit too.
                supports_tools = mlx_backend.supports_tools,
                # Phase 8: surface the effective KV-cache dtype. On the
                # "already loaded" short-circuit the backend state is
                # authoritative — don't echo the request field.
                cache_type_kv = mlx_backend.cache_type_kv,
                chat_template = mlx_backend.chat_template,
                # Phase 7: surface speculative-active state even on the
                # already-loaded path.
                speculative_type = mlx_backend.speculative_type,
                # Chunk D: collapsed backend enum.
                backend_kind = "mlx+lora" if mlx_backend.is_lora else "mlx",
            )

        if request.gguf_variant:
            if (
                llama_backend.is_loaded
                and llama_backend.hf_variant
                and llama_backend.hf_variant.lower() == request.gguf_variant.lower()
                and llama_backend.model_identifier
                and llama_backend.model_identifier.lower() == request.model_path.lower()
            ):
                logger.info(
                    f"Model already loaded (GGUF): {request.model_path} variant={request.gguf_variant}, skipping reload"
                )
                inference_config = load_inference_config(llama_backend.model_identifier)
                from utils.models import is_audio_input_type

                _gguf_audio = (
                    llama_backend._audio_type
                    if hasattr(llama_backend, "_audio_type")
                    else None
                )
                _gguf_is_audio = getattr(llama_backend, "_is_audio", False)
                return LoadResponse(
                    status = "already_loaded",
                    model = llama_backend.model_identifier,
                    display_name = llama_backend.model_identifier,
                    is_vision = llama_backend._is_vision,
                    is_lora = False,
                    is_gguf = True,
                    is_audio = _gguf_is_audio,
                    audio_type = _gguf_audio,
                    has_audio_input = is_audio_input_type(_gguf_audio)
                    if _gguf_audio
                    else False,
                    inference = inference_config,
                    requires_trust_remote_code = bool(
                        inference_config.get("trust_remote_code", False)
                    ),
                    context_length = llama_backend.context_length,
                    max_context_length = llama_backend.max_context_length,
                    native_context_length = llama_backend.native_context_length,
                    supports_reasoning = llama_backend.supports_reasoning,
                    reasoning_always_on = llama_backend.reasoning_always_on,
                    chat_template = llama_backend.chat_template,
                    speculative_type = llama_backend.speculative_type,
                    backend_kind = "gguf",
                )
        else:
            if (
                backend.active_model_name
                and backend.active_model_name.lower() == request.model_path.lower()
            ):
                logger.info(
                    f"Model already loaded (Unsloth): {request.model_path}, skipping reload"
                )
                inference_config = load_inference_config(backend.active_model_name)
                _model_info = backend.models.get(backend.active_model_name, {})
                _chat_template = None
                try:
                    _tpl_info = _model_info.get("chat_template_info", {})
                    _chat_template = _tpl_info.get("template")
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve chat template for {backend.active_model_name}: {e}"
                    )
                return LoadResponse(
                    status = "already_loaded",
                    model = backend.active_model_name,
                    display_name = backend.active_model_name,
                    is_vision = _model_info.get("is_vision", False),
                    is_lora = _model_info.get("is_lora", False),
                    is_gguf = False,
                    is_audio = _model_info.get("is_audio", False),
                    audio_type = _model_info.get("audio_type"),
                    has_audio_input = _model_info.get("has_audio_input", False),
                    inference = inference_config,
                    requires_trust_remote_code = bool(
                        inference_config.get("trust_remote_code", False)
                    ),
                    chat_template = _chat_template,
                    backend_kind = "unsloth",
                )

        # Create config using clean factory method
        # is_lora is auto-detected from adapter_config.json on disk/HF
        config = ModelConfig.from_identifier(
            model_id = request.model_path,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {request.model_path}",
            )

        # Normalize gpu_ids: empty list means auto-selection, same as None
        effective_gpu_ids = request.gpu_ids if request.gpu_ids else None

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            if effective_gpu_ids is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = "gpu_ids is not supported for GGUF models yet.",
                )

            llama_backend = get_llama_cpp_backend()

            # Chunk D: peer-unload via the shared helper. Unloads Unsloth,
            # MLX text, MLX-VLM, and MLX-Audio so GGUF gets the full
            # unified-memory budget.
            await asyncio.to_thread(
                _unload_all_mlx_peers, keep = "gguf"
            )

            # Route to HF mode or local mode based on config
            # Run in a thread so the event loop stays free for progress
            # polling and other requests during the (potentially long)
            # GGUF download + llama-server startup.
            _n_parallel = getattr(fastapi_request.app.state, "llama_parallel_slots", 1)

            if config.gguf_hf_repo:
                # HF mode: download via huggingface_hub then start llama-server
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    hf_repo = config.gguf_hf_repo,
                    hf_variant = config.gguf_variant,
                    hf_token = request.hf_token,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                    speculative_type = request.speculative_type,
                    n_parallel = _n_parallel,
                )
            else:
                # Local mode: llama-server loads via -m <path>
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    gguf_path = config.gguf_file,
                    mmproj_path = config.gguf_mmproj_file,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                    speculative_type = request.speculative_type,
                    n_parallel = _n_parallel,
                )

            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load GGUF model: {config.display_name}",
                )

            logger.info(f"Loaded GGUF model via llama-server: {config.identifier}")

            # Detect TTS audio by probing the loaded model's vocabulary
            from utils.models import is_audio_input_type

            _gguf_audio = llama_backend.detect_audio_type()
            _gguf_is_audio = _gguf_audio in ("snac", "bicodec", "dac")
            llama_backend._is_audio = _gguf_is_audio
            llama_backend._audio_type = _gguf_audio
            if _gguf_is_audio:
                logger.info(f"GGUF model detected as audio: audio_type={_gguf_audio}")
                await asyncio.to_thread(llama_backend.init_audio_codec, _gguf_audio)

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = config.identifier,
                display_name = config.display_name,
                is_vision = config.is_vision,
                is_lora = False,
                is_gguf = True,
                is_audio = _gguf_is_audio,
                audio_type = _gguf_audio,
                has_audio_input = is_audio_input_type(_gguf_audio),
                inference = inference_config,
                requires_trust_remote_code = bool(
                    inference_config.get("trust_remote_code", False)
                ),
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_tools = llama_backend.supports_tools,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template = llama_backend.chat_template,
                speculative_type = llama_backend.speculative_type,
                backend_kind = "gguf",
            )

        # ── MLX-VLM path: load via mlx_vlm (Apple Silicon only) ────
        # Phase 9 (Chunk D). Runs BEFORE the ``config.is_mlx`` branch
        # because a VLM checkpoint can also carry the ``quantization``
        # block that would otherwise route it through ``MlxLmBackend``.
        if config.is_mlx_vlm:
            if effective_gpu_ids is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = "gpu_ids is not supported for MLX-VLM models.",
                )
            mlx_vlm_backend = get_mlx_vlm_backend()

            # Unload every other peer — VLM consumes unified memory
            # alongside the visual encoder.
            await asyncio.to_thread(
                _unload_all_mlx_peers, keep = "mlx_vlm"
            )

            success = await asyncio.to_thread(
                mlx_vlm_backend.load_model,
                local_path = config.mlx_vlm_path or config.path,
                model_identifier = config.identifier,
                hf_token = request.hf_token,
                n_ctx = request.max_seq_length,
            )
            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load MLX-VLM model: {config.display_name}",
                )

            logger.info(f"Loaded MLX-VLM model via mlx_vlm: {config.identifier}")
            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = config.identifier,
                display_name = config.display_name,
                is_vision = True,
                is_lora = False,
                is_gguf = False,
                is_mlx = False,
                is_mlx_vlm = True,
                is_mlx_lora = False,
                is_audio = False,
                audio_type = None,
                has_audio_input = False,
                inference = inference_config,
                requires_trust_remote_code = bool(
                    inference_config.get("trust_remote_code", False)
                ),
                context_length = mlx_vlm_backend.context_length,
                max_context_length = mlx_vlm_backend.max_context_length,
                native_context_length = mlx_vlm_backend.native_context_length,
                supports_reasoning = mlx_vlm_backend.supports_reasoning,
                reasoning_always_on = mlx_vlm_backend.reasoning_always_on,
                supports_tools = mlx_vlm_backend.supports_tools,
                cache_type_kv = mlx_vlm_backend.cache_type_kv,
                chat_template = mlx_vlm_backend.chat_template,
                speculative_type = mlx_vlm_backend.speculative_type,
                backend_kind = "mlx+vlm",
            )

        # ── MLX-Audio path: load via mlx_audio (Apple Silicon only) ─
        # Phase 10 (Chunk D). Runs BEFORE the ``config.is_mlx`` branch.
        if config.is_mlx_audio:
            if effective_gpu_ids is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = "gpu_ids is not supported for MLX-Audio models.",
                )
            mlx_audio_backend = get_mlx_audio_backend()

            await asyncio.to_thread(
                _unload_all_mlx_peers, keep = "mlx_audio"
            )

            success = await asyncio.to_thread(
                mlx_audio_backend.load_model,
                local_path = config.mlx_audio_path or config.path,
                model_identifier = config.identifier,
                hf_token = request.hf_token,
            )
            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load MLX-Audio model: {config.display_name}",
                )

            logger.info(f"Loaded MLX-Audio model via mlx_audio: {config.identifier}")
            inference_config = load_inference_config(config.identifier)

            _audio_caps = mlx_audio_backend.detect_audio_type()

            return LoadResponse(
                status = "loaded",
                model = config.identifier,
                display_name = config.display_name,
                is_vision = False,
                is_lora = False,
                is_gguf = False,
                is_mlx = False,
                is_mlx_vlm = False,
                is_mlx_audio = True,
                is_mlx_lora = False,
                is_audio = True,
                audio_type = _audio_caps,
                has_audio_input = mlx_audio_backend.has_audio_input,
                inference = inference_config,
                requires_trust_remote_code = False,
                context_length = mlx_audio_backend.context_length,
                max_context_length = mlx_audio_backend.context_length,
                native_context_length = mlx_audio_backend.context_length,
                supports_reasoning = False,
                reasoning_always_on = False,
                supports_tools = False,
                cache_type_kv = None,
                chat_template = None,
                speculative_type = None,
                backend_kind = "mlx+audio",
            )

        # ── MLX path: load via mlx_lm (Apple Silicon only) ─────────
        if config.is_mlx:
            if effective_gpu_ids is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = "gpu_ids is not supported for MLX models.",
                )

            mlx_backend = get_mlx_lm_backend()

            # Chunk D: peer-unload via the shared helper. Unloads GGUF,
            # Unsloth, MLX-VLM, and MLX-Audio — whichever happens to be
            # loaded — in one pass.
            await asyncio.to_thread(
                _unload_all_mlx_peers, keep = "mlx"
            )

            # Phase 6 — an explicit request.adapter_path wins over the
            # ModelConfig-derived one (e.g. when the user points at a
            # base model and supplies the adapter separately).
            _mlx_adapter_path = request.adapter_path or config.mlx_adapter_path
            success = await asyncio.to_thread(
                mlx_backend.load_model,
                local_path = config.mlx_path or config.path,
                model_identifier = config.identifier,
                hf_token = request.hf_token,
                n_ctx = request.max_seq_length,
                # Phase 8: forward the UI's KV-dtype label. ``None`` or
                # ``"f16"`` / ``"bf16"`` → unquantized (unchanged behaviour).
                cache_type_kv = request.cache_type_kv,
                # Phase 6: forward the LoRA adapter path when provided.
                adapter_path = _mlx_adapter_path,
                # Phase 7: forward the draft model path for speculative.
                draft_model_path = request.draft_model_path,
                # Chunk E (E3): forward optional override for number of draft
                # tokens speculated per step (None → backend default of 3).
                num_draft_tokens = request.num_draft_tokens,
            )
            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load MLX model: {config.display_name}",
                )

            logger.info(f"Loaded MLX model via mlx_lm: {config.identifier}")

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = config.identifier,
                display_name = config.display_name,
                is_vision = False,
                is_lora = False,
                is_gguf = False,
                is_mlx = True,
                # Phase 6: True when an adapter was layered in.
                is_mlx_lora = mlx_backend.is_lora,
                is_audio = False,
                audio_type = None,
                has_audio_input = False,
                inference = inference_config,
                requires_trust_remote_code = bool(
                    inference_config.get("trust_remote_code", False)
                ),
                context_length = mlx_backend.context_length,
                max_context_length = mlx_backend.max_context_length,
                native_context_length = mlx_backend.native_context_length,
                # Phase 4: surface reasoning introspection from the backend.
                supports_reasoning = mlx_backend.supports_reasoning,
                reasoning_always_on = mlx_backend.reasoning_always_on,
                # Phase 5: surface tool-calling support so the UI can
                # gate the tools panel on an MLX-loaded model just like
                # it already does for GGUF.
                supports_tools = mlx_backend.supports_tools,
                # Phase 8: surface the effective KV-cache dtype so the UI
                # can reflect what the backend actually applied (e.g. a
                # q5_1 request rounds down to q4_0).
                cache_type_kv = mlx_backend.cache_type_kv,
                chat_template = mlx_backend.chat_template,
                # Phase 7: "mlx-draft-model" when a draft was loaded, else None.
                speculative_type = mlx_backend.speculative_type,
                # Chunk D: collapsed backend enum.
                backend_kind = "mlx+lora" if mlx_backend.is_lora else "mlx",
            )

        # ── Standard path: load via Unsloth/transformers ──────────
        backend = get_inference_backend()

        # Chunk D: peer-unload via the shared helper. Unloads GGUF,
        # MLX text, MLX-VLM, and MLX-Audio so the Unsloth/transformers
        # load gets the full unified-memory budget.
        await asyncio.to_thread(
            _unload_all_mlx_peers, keep = "unsloth"
        )

        # Shut down any export subprocess to free VRAM
        try:
            from core.export import get_export_backend

            exp_backend = get_export_backend()
            if exp_backend.current_checkpoint:
                logger.info(
                    "Shutting down export subprocess to free GPU memory for inference"
                )
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as e:
            logger.warning("Could not shut down export subprocess: %s", e)

        # Auto-detect quantization for LoRA adapters from adapter_config.json
        # The training pipeline patches this file with "unsloth_training_method"
        # which is 'qlora' or 'lora'. Only LoRA (16-bit) needs load_in_4bit=False.
        load_in_4bit = request.load_in_4bit
        if config.is_lora and config.path:
            import json
            from pathlib import Path

            adapter_cfg_path = Path(config.path) / "adapter_config.json"
            if adapter_cfg_path.exists():
                try:
                    with open(adapter_cfg_path) as f:
                        adapter_cfg = json.load(f)
                    training_method = adapter_cfg.get("unsloth_training_method")
                    if training_method == "lora" and load_in_4bit:
                        logger.info(
                            f"adapter_config.json says unsloth_training_method='lora' — "
                            f"setting load_in_4bit=False to match 16-bit training"
                        )
                        load_in_4bit = False
                    elif training_method == "qlora" and not load_in_4bit:
                        logger.info(
                            f"adapter_config.json says unsloth_training_method='qlora' — "
                            f"setting load_in_4bit=True to match QLoRA training"
                        )
                        load_in_4bit = True
                    elif training_method:
                        logger.info(
                            f"Training method: {training_method}, load_in_4bit={load_in_4bit}"
                        )
                    else:
                        # No unsloth_training_method — fallback to base model name
                        if (
                            config.base_model
                            and "-bnb-4bit" not in config.base_model.lower()
                            and load_in_4bit
                        ):
                            logger.info(
                                f"No unsloth_training_method in adapter_config.json. "
                                f"Base model '{config.base_model}' has no -bnb-4bit suffix — "
                                f"setting load_in_4bit=False"
                            )
                            load_in_4bit = False
                except Exception as e:
                    logger.warning(f"Could not read adapter_config.json: {e}")

        # Load the model in a thread so the event loop stays free
        # for download progress polling and other requests.
        success = await asyncio.to_thread(
            backend.load_model,
            config = config,
            max_seq_length = request.max_seq_length,
            load_in_4bit = load_in_4bit,
            hf_token = request.hf_token,
            trust_remote_code = request.trust_remote_code,
            gpu_ids = effective_gpu_ids,
        )

        if not success:
            # Check if YAML says this model needs trust_remote_code
            if not request.trust_remote_code:
                model_defaults = load_model_defaults(config.identifier)
                yaml_trust = model_defaults.get("inference", {}).get(
                    "trust_remote_code", False
                )
                if yaml_trust:
                    raise HTTPException(
                        status_code = 400,
                        detail = (
                            f"Model '{config.display_name}' requires trust_remote_code to be enabled. "
                            f"Please enable 'Trust remote code' in Chat Settings and try again."
                        ),
                    )
            raise HTTPException(
                status_code = 500, detail = f"Failed to load model: {config.display_name}"
            )

        logger.info(f"Loaded model: {config.identifier}")

        # Load inference configuration parameters
        inference_config = load_inference_config(config.identifier)

        # Get chat template from tokenizer
        _chat_template = None
        try:
            _model_info = backend.models.get(config.identifier, {})
            _tpl_info = _model_info.get("chat_template_info", {})
            _chat_template = _tpl_info.get("template")
        except Exception:
            pass

        return LoadResponse(
            status = "loaded",
            model = config.identifier,
            display_name = config.display_name,
            is_vision = config.is_vision,
            is_lora = config.is_lora,
            is_gguf = False,
            is_audio = config.is_audio,
            audio_type = config.audio_type,
            has_audio_input = config.has_audio_input,
            inference = inference_config,
            requires_trust_remote_code = bool(
                inference_config.get("trust_remote_code", False)
            ),
            chat_template = _chat_template,
            backend_kind = "unsloth",
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Rejected inference GPU selection: %s", e)
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info = True)
        msg = str(e)
        # Surface a friendlier message for models that Unsloth cannot load
        not_supported_hints = [
            "No config file found",
            "not yet supported",
            "is not supported",
            "does not support",
        ]
        if any(h.lower() in msg.lower() for h in not_supported_hints):
            msg = f"This model is not supported yet. Try a different model. (Original error: {msg})"
        raise HTTPException(status_code = 500, detail = f"Failed to load model: {msg}")


@router.post("/validate", response_model = ValidateModelResponse)
async def validate_model(
    request: ValidateModelRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Lightweight validation endpoint for model identifiers.

    This checks that ModelConfig.from_identifier() can resolve the given
    model_path, but it does NOT actually load model weights into GPU memory.
    """
    try:
        config = ModelConfig.from_identifier(
            model_id = request.model_path,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {request.model_path}",
            )

        return ValidateModelResponse(
            valid = True,
            message = "Model identifier is valid.",
            identifier = config.identifier,
            display_name = getattr(config, "display_name", config.identifier),
            is_gguf = getattr(config, "is_gguf", False),
            is_lora = getattr(config, "is_lora", False),
            is_vision = getattr(config, "is_vision", False),
            requires_trust_remote_code = bool(
                load_inference_config(config.identifier).get("trust_remote_code", False)
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error validating model identifier '{request.model_path}': {e}",
            exc_info = True,
        )
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid model: {str(e)}",
        )


@router.post("/unload", response_model = UnloadResponse)
async def unload_model(
    request: UnloadRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Unload a model from memory.
    Routes to the correct backend (llama-server for GGUF, Unsloth otherwise).
    """
    try:
        # Check if the GGUF backend has this model loaded or is loading it
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_active and (
            llama_backend.model_identifier == request.model_path
            or not llama_backend.is_loaded
        ):
            llama_backend.unload_model()
            logger.info(f"Unloaded GGUF model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Check if the MLX backend has this model loaded
        mlx_backend = get_mlx_lm_backend()
        if mlx_backend.is_loaded and (
            mlx_backend.model_identifier == request.model_path
        ):
            await asyncio.to_thread(mlx_backend.unload_model)
            logger.info(f"Unloaded MLX model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Chunk D: check the MLX-VLM peer.
        mlx_vlm_backend = get_mlx_vlm_backend()
        if mlx_vlm_backend.is_loaded and (
            mlx_vlm_backend.model_identifier == request.model_path
        ):
            await asyncio.to_thread(mlx_vlm_backend.unload_model)
            logger.info(f"Unloaded MLX-VLM model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Chunk D: check the MLX-Audio peer.
        mlx_audio_backend = get_mlx_audio_backend()
        if mlx_audio_backend.is_loaded and (
            mlx_audio_backend.model_identifier == request.model_path
        ):
            await asyncio.to_thread(mlx_audio_backend.unload_model)
            logger.info(f"Unloaded MLX-Audio model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Otherwise, unload from Unsloth backend
        backend = get_inference_backend()
        backend.unload_model(request.model_path)
        logger.info(f"Unloaded model: {request.model_path}")
        return UnloadResponse(status = "unloaded", model = request.model_path)

    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to unload model: {str(e)}")


@router.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.

    For vision models, provide image_base64 with the base64-encoded image.
    """
    backend = get_inference_backend()

    if not backend.active_model_name:
        raise HTTPException(
            status_code = 400, detail = "No model loaded. Call POST /inference/load first."
        )

    # Decode image if provided (for vision models)
    image = None
    if request.image_base64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            # Check if current model supports vision
            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code = 400, detail = f"Failed to decode image: {str(e)}"
            )

    async def stream():
        try:
            for chunk in backend.generate_chat_response(
                messages = request.messages,
                system_prompt = request.system_prompt,
                image = image,
                temperature = request.temperature,
                top_p = request.top_p,
                top_k = request.top_k,
                max_new_tokens = request.max_new_tokens,
                repetition_penalty = request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during generation: {e}", exc_info = True)
            yield f"data: {json.dumps({'error': _friendly_error(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/status", response_model = InferenceStatusResponse)
async def get_status(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get current inference backend status.
    Reports whichever backend (Unsloth or llama-server) is currently active.
    """
    try:
        llama_backend = get_llama_cpp_backend()
        mlx_backend = get_mlx_lm_backend()
        mlx_vlm_backend = get_mlx_vlm_backend()
        mlx_audio_backend = get_mlx_audio_backend()

        # Chunk D — MLX-VLM peer has precedence over plain MLX so a
        # vision chat reports ``is_mlx_vlm=True`` even if someone had a
        # text MLX model loaded earlier (shouldn't happen because
        # peer-unload is mandatory, but defensive).
        if mlx_vlm_backend.is_loaded:
            _vlm_id = mlx_vlm_backend.model_identifier
            _inf_cfg = load_inference_config(_vlm_id) if _vlm_id else None
            return InferenceStatusResponse(
                active_model = _vlm_id,
                is_vision = True,
                is_gguf = False,
                is_mlx = False,
                is_mlx_vlm = True,
                is_audio = False,
                audio_type = None,
                has_audio_input = False,
                loading = [],
                loaded = [_vlm_id] if _vlm_id else [],
                inference = _inf_cfg,
                requires_trust_remote_code = bool(
                    (_inf_cfg or {}).get("trust_remote_code", False)
                ),
                supports_reasoning = mlx_vlm_backend.supports_reasoning,
                reasoning_always_on = mlx_vlm_backend.reasoning_always_on,
                supports_tools = mlx_vlm_backend.supports_tools,
                context_length = mlx_vlm_backend.context_length,
                max_context_length = mlx_vlm_backend.max_context_length,
                native_context_length = mlx_vlm_backend.native_context_length,
                speculative_type = None,
                backend_kind = "mlx+vlm",
            )

        # Chunk D — MLX-Audio peer status.
        if mlx_audio_backend.is_loaded:
            _aid = mlx_audio_backend.model_identifier
            _inf_cfg = load_inference_config(_aid) if _aid else None
            return InferenceStatusResponse(
                active_model = _aid,
                is_vision = False,
                is_gguf = False,
                is_mlx = False,
                is_mlx_audio = True,
                is_audio = True,
                audio_type = mlx_audio_backend.detect_audio_type(),
                has_audio_input = mlx_audio_backend.has_audio_input,
                loading = [],
                loaded = [_aid] if _aid else [],
                inference = _inf_cfg,
                requires_trust_remote_code = False,
                supports_reasoning = False,
                reasoning_always_on = False,
                supports_tools = False,
                context_length = mlx_audio_backend.context_length,
                max_context_length = mlx_audio_backend.context_length,
                native_context_length = mlx_audio_backend.context_length,
                speculative_type = None,
                backend_kind = "mlx+audio",
            )

        # If an MLX model is loaded, report that first (it is mutually
        # exclusive with the GGUF backend but we check it first because
        # users may switch backends and we want the newest one to win).
        if mlx_backend.is_loaded:
            _mlx_model_id = mlx_backend.model_identifier
            _inference_cfg = (
                load_inference_config(_mlx_model_id) if _mlx_model_id else None
            )
            return InferenceStatusResponse(
                active_model = _mlx_model_id,
                is_vision = False,
                is_gguf = False,
                is_mlx = True,
                is_audio = False,
                audio_type = None,
                has_audio_input = False,
                loading = [],
                loaded = [_mlx_model_id] if _mlx_model_id else [],
                inference = _inference_cfg,
                requires_trust_remote_code = bool(
                    (_inference_cfg or {}).get("trust_remote_code", False)
                ),
                # Phase 4: surface reasoning introspection from the backend.
                supports_reasoning = mlx_backend.supports_reasoning,
                reasoning_always_on = mlx_backend.reasoning_always_on,
                # Phase 5: surface tool-calling support in /status so
                # the frontend can enable the tools panel when an MLX
                # tool-capable model is active.
                supports_tools = mlx_backend.supports_tools,
                context_length = mlx_backend.context_length,
                max_context_length = mlx_backend.max_context_length,
                native_context_length = mlx_backend.native_context_length,
                # Phase 7: surface "mlx-draft-model" in status so the UI
                # can reflect the active speculative mode.
                speculative_type = mlx_backend.speculative_type,
                # Chunk D: collapsed backend enum.
                backend_kind = "mlx+lora" if mlx_backend.is_lora else "mlx",
            )

        # If a GGUF model is loaded via llama-server, report that
        if llama_backend.is_loaded:
            _model_id = llama_backend.model_identifier
            _inference_cfg = load_inference_config(_model_id) if _model_id else None
            return InferenceStatusResponse(
                active_model = _model_id,
                is_vision = llama_backend.is_vision,
                is_gguf = True,
                gguf_variant = llama_backend.hf_variant,
                is_audio = getattr(llama_backend, "_is_audio", False),
                audio_type = getattr(llama_backend, "_audio_type", None),
                loading = [],
                loaded = [_model_id],
                inference = _inference_cfg,
                requires_trust_remote_code = bool(
                    (_inference_cfg or {}).get("trust_remote_code", False)
                ),
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_tools = llama_backend.supports_tools,
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                speculative_type = llama_backend.speculative_type,
                backend_kind = "gguf",
            )

        # Otherwise, report Unsloth backend status
        backend = get_inference_backend()

        is_vision = False
        is_audio = False
        audio_type = None
        has_audio_input = False
        if backend.active_model_name:
            model_info = backend.models.get(backend.active_model_name, {})
            is_vision = model_info.get("is_vision", False)
            is_audio = model_info.get("is_audio", False)
            audio_type = model_info.get("audio_type")
            has_audio_input = model_info.get("has_audio_input", False)

        # gpt-oss safetensors models support reasoning via harmony channels
        supports_reasoning = False
        if backend.active_model_name and hasattr(backend, "_is_gpt_oss_model"):
            supports_reasoning = backend._is_gpt_oss_model()
        inference_config = (
            load_inference_config(backend.active_model_name)
            if backend.active_model_name
            else None
        )

        return InferenceStatusResponse(
            active_model = backend.active_model_name,
            is_vision = is_vision,
            is_gguf = False,
            is_audio = is_audio,
            audio_type = audio_type,
            has_audio_input = has_audio_input,
            loading = list(getattr(backend, "loading_models", set())),
            loaded = list(backend.models.keys()),
            inference = inference_config,
            requires_trust_remote_code = bool(
                (inference_config or {}).get("trust_remote_code", False)
            ),
            supports_reasoning = supports_reasoning,
            backend_kind = "unsloth" if backend.active_model_name else None,
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to get status: {str(e)}")


@router.get("/load-progress", response_model = LoadProgressResponse)
async def get_load_progress(
    current_subject: str = Depends(get_current_subject),
):
    """
    Return the active GGUF load's mmap/upload progress.

    During the warmup window after a GGUF download -- when llama-server
    is paging ~tens-to-hundreds of GB of shards into the page cache
    before pushing layers to VRAM -- ``/api/inference/status`` only
    shows a generic spinner. This endpoint exposes sampled progress so
    the UI can render a real bar plus rate/ETA during that window.

    Returns an empty payload (``phase=null, bytes=0``) when no load is
    in flight. The frontend should stop polling once ``phase`` becomes
    ``ready``.
    """
    try:
        # Phase 3 — MLX uses the same endpoint. When the MLX backend has
        # a load in flight (phase != None), prefer it over GGUF. This
        # is safe because only one peer can be mid-load at a time: the
        # route unloads the other peer before starting a new load.
        mlx_backend = get_mlx_lm_backend()
        mlx_progress = mlx_backend.load_progress()
        if mlx_progress is not None and mlx_progress.get("phase") is not None:
            # Chunk E (E4): the schema now has a warnings field — surface
            # any memory-headroom advisories the backend collected during
            # load alongside the phase/bytes data.
            filtered = {
                k: v
                for k, v in mlx_progress.items()
                if k in ("phase", "bytes_loaded", "bytes_total", "fraction", "warnings")
            }
            return LoadProgressResponse(**filtered)

        # Chunk D — check the MLX-VLM / MLX-Audio peers.
        for peer in (get_mlx_vlm_backend(), get_mlx_audio_backend()):
            if hasattr(peer, "load_progress"):
                prog = peer.load_progress()
                if prog and prog.get("phase") is not None:
                    filtered = {
                        k: v
                        for k, v in prog.items()
                        if k in ("phase", "bytes_loaded", "bytes_total", "fraction", "warnings")
                    }
                    return LoadProgressResponse(**filtered)

        llama_backend = get_llama_cpp_backend()
        progress = llama_backend.load_progress()
        if progress is None:
            return LoadProgressResponse()
        # GGUF backend does not emit warnings today; if/when it does the
        # LoadProgressResponse default_factory tolerates absence and the
        # schema accepts additional keys via the filtered passthrough below.
        return LoadProgressResponse(**progress)
    except Exception as e:
        logger.warning(f"Error sampling load progress: {e}")
        return LoadProgressResponse()


# =====================================================================
# Audio (TTS) Generation  (/audio/generate)
# =====================================================================


@router.post("/audio/generate")
async def generate_audio(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate audio (TTS) from the latest user message.
    Returns a JSON response with base64-encoded WAV audio.
    Works with both GGUF (llama-server) and Unsloth/transformers backends.
    """
    import base64

    # Extract text from the last user message
    _, chat_messages, _ = _extract_content_parts(payload.messages)
    if not chat_messages:
        raise HTTPException(status_code = 400, detail = "No messages provided.")
    last_user_msg = next(
        (m for m in reversed(chat_messages) if m["role"] == "user"), None
    )
    if not last_user_msg:
        raise HTTPException(status_code = 400, detail = "No user message found.")
    text = last_user_msg["content"]

    # Pick backend — all return (wav_bytes, sample_rate)
    llama_backend = get_llama_cpp_backend()
    # Chunk D (Phase 10): MLX audio peer.
    mlx_audio_backend = get_mlx_audio_backend()
    if mlx_audio_backend.is_loaded:
        model_name = mlx_audio_backend.model_identifier
        gen = lambda: mlx_audio_backend.generate_tts(
            text = text,
            max_new_tokens = payload.max_tokens or 2048,
            temperature = payload.temperature or 0.7,
            top_k = payload.top_k or 50,
        )
    elif llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        model_name = llama_backend.model_identifier
        gen = lambda: llama_backend.generate_audio_response(
            text = text,
            audio_type = llama_backend._audio_type,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = payload.max_tokens or 2048,
            repetition_penalty = payload.repetition_penalty,
        )
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(status_code = 400, detail = "No model loaded.")
        model_info = backend.models.get(backend.active_model_name, {})
        if not model_info.get("is_audio"):
            raise HTTPException(
                status_code = 400, detail = "Active model is not an audio model."
            )
        model_name = backend.active_model_name
        gen = lambda: backend.generate_audio_response(
            text = text,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = payload.max_tokens or 2048,
            repetition_penalty = payload.repetition_penalty,
            use_adapter = payload.use_adapter,
        )

    try:
        wav_bytes, sample_rate = await asyncio.get_event_loop().run_in_executor(
            None, gen
        )
    except Exception as e:
        logger.error(f"Audio generation error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return JSONResponse(
        content = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.audio",
            "model": model_name,
            "audio": {"data": audio_b64, "format": "wav", "sample_rate": sample_rate},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f'[Generated audio from: "{text[:100]}"]',
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )


# =====================================================================
# OpenAI-Compatible TTS + ASR  (Phase 10 / Chunk D)
# =====================================================================
# OpenAI's public API exposes two audio endpoints:
#   POST /v1/audio/speech          — body: {model, input, voice, response_format}
#                                     returns: raw audio bytes (default mp3).
#   POST /v1/audio/transcriptions  — multipart: file (audio), model, prompt...
#                                     returns: {"text": "..."} (default json).
# We implement both against the MLX audio backend. When no MLX audio
# model is loaded we fall through to the GGUF / Unsloth audio paths
# that already exist in Studio via ``/audio/generate``. The new
# endpoints live on ``/audio/speech`` and ``/audio/transcriptions``
# so they're reachable at ``/v1/audio/speech`` /
# ``/v1/audio/transcriptions`` (OpenAI shape) AND
# ``/api/inference/audio/speech`` / ``.../transcriptions`` (Studio
# internal).


from pydantic import BaseModel as _OpenAIBaseModel, Field as _OpenAIField


class _OpenAITtsRequest(_OpenAIBaseModel):
    """Subset of OpenAI's /v1/audio/speech request body.

    Unknown fields are accepted silently via ``model_config = ConfigDict(
    extra='allow')`` so future OpenAI-shape fields (voice, speed…) land
    without a 422.
    """

    model: Optional[str] = _OpenAIField(None, description = "Model name (ignored; uses the active MLX audio checkpoint)")
    input: str = _OpenAIField(..., description = "Text to synthesize")
    voice: Optional[str] = _OpenAIField(None, description = "Voice identifier (currently ignored — LFM2.5-Audio uses a single default voice)")
    response_format: Optional[str] = _OpenAIField(
        "wav",
        description = "Output format: 'wav' | 'mp3'. MP3 falls through to WAV when mp3 encoding is unavailable.",
    )
    speed: Optional[float] = _OpenAIField(None, description = "Speech rate (currently ignored)")


@router.post("/audio/speech")
async def openai_audio_speech(
    payload: _OpenAITtsRequest,
    current_subject: str = Depends(get_current_subject),
):
    """OpenAI-compatible TTS endpoint. Returns raw audio bytes.

    Dispatch order:
    1. MLX audio backend (Phase 10 / Chunk D) when loaded.
    2. GGUF audio backend when an audio-capable GGUF model is loaded.
    3. Unsloth TTS fallback.

    The return body is raw audio (not JSON), matching OpenAI's shape.
    """
    mlx_audio_backend = get_mlx_audio_backend()
    llama_backend = get_llama_cpp_backend()

    if mlx_audio_backend.is_loaded:
        loop = asyncio.get_event_loop()
        try:
            wav_bytes, _sr = await loop.run_in_executor(
                None,
                mlx_audio_backend.generate_tts,
                payload.input,
            )
        except Exception as e:
            logger.error(f"MLX-Audio TTS error: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))
        return Response(content = wav_bytes, media_type = "audio/wav")

    if llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        # Defer to the GGUF-shaped helper by synthesizing a
        # ChatCompletionRequest-like object.
        loop = asyncio.get_event_loop()
        try:
            wav_bytes, _sr = await loop.run_in_executor(
                None,
                lambda: llama_backend.generate_audio_response(
                    text = payload.input,
                    audio_type = llama_backend._audio_type,
                ),
            )
        except Exception as e:
            logger.error(f"GGUF TTS error: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))
        return Response(content = wav_bytes, media_type = "audio/wav")

    # Unsloth / transformers TTS fallback.
    backend = get_inference_backend()
    if backend.active_model_name and backend.models.get(
        backend.active_model_name, {}
    ).get("is_audio"):
        try:
            wav_bytes, _sr = backend.generate_audio_response(text = payload.input)
        except Exception as e:
            logger.error(f"Unsloth TTS error: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))
        return Response(content = wav_bytes, media_type = "audio/wav")

    raise HTTPException(
        status_code = 400,
        detail = "No audio-capable model is loaded. Load an MLX-Audio / GGUF-audio / Unsloth-audio checkpoint first.",
    )


@router.post("/audio/transcriptions")
async def openai_audio_transcriptions(
    file: "UploadFile" = FastAPIFile(...),
    model: Optional[str] = FastAPIForm(None),
    prompt: Optional[str] = FastAPIForm(None),
    language: Optional[str] = FastAPIForm(None),
    response_format: Optional[str] = FastAPIForm("json"),
    temperature: Optional[float] = FastAPIForm(None),
    current_subject: str = Depends(get_current_subject),
):
    """OpenAI-compatible ASR endpoint. Multipart file upload.

    Dispatch order:
    1. MLX audio backend (Phase 10 / Chunk D) when loaded and it
       advertises ``has_audio_input``. Best-effort — LFM2.5-Audio is a
       voice assistant, not a dedicated ASR (see PROBE_RESULTS.md).
    2. GGUF audio backend when loaded.
    3. Unsloth Whisper / audio-input fallback.

    Returns JSON ``{"text": "..."}`` by default.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code = 400, detail = "No audio data in upload.")

    mlx_audio_backend = get_mlx_audio_backend()

    if mlx_audio_backend.is_loaded and mlx_audio_backend.has_audio_input:
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                lambda: mlx_audio_backend.transcribe(
                    audio_bytes,
                    prompt = prompt or "Please transcribe the audio.",
                    temperature = temperature or 0.0,
                ),
            )
        except Exception as e:
            logger.error(f"MLX-Audio transcribe error: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))
        return JSONResponse(content = {"text": text})

    # Fallback — Unsloth whisper-style audio input path.
    backend = get_inference_backend()
    if backend.active_model_name:
        model_info = backend.models.get(backend.active_model_name, {})
        if model_info.get("audio_type") == "whisper" or model_info.get(
            "has_audio_input"
        ):
            # Reuse the /v1/audio/transcriptions Unsloth path via the
            # chat-completions audio_base64 branch — it decodes,
            # transcribes, and returns text.
            arr = _decode_audio_base64(base64.b64encode(audio_bytes).decode("ascii"))
            text = ""
            for chunk in backend.generate_whisper_response(audio_array = arr):
                text += chunk or ""
            return JSONResponse(content = {"text": text})

    raise HTTPException(
        status_code = 400,
        detail = "No ASR-capable model is loaded. Load an MLX-Audio, GGUF-audio, or Whisper checkpoint first.",
    )


# =====================================================================
# OpenAI-Compatible Chat Completions  (/chat/completions)
# =====================================================================


def _decode_audio_base64(b64: str) -> np.ndarray:
    """Decode base64 audio (any format) → float32 numpy array at 16kHz."""
    import torch
    import torchaudio
    import tempfile
    import os
    from utils.paths import ensure_dir, tmp_root

    raw = base64.b64decode(b64)
    # torchaudio.load needs a file path or file-like object with format hint
    # Write to a temp file so torchaudio can auto-detect the format
    with tempfile.NamedTemporaryFile(
        suffix = ".audio",
        delete = False,
        dir = str(ensure_dir(tmp_root())),
    ) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        waveform, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def _extract_content_parts(
    messages: list,
    *,
    preserve_tool_history: bool = False,
) -> tuple[str, list[dict], "Optional[str]"]:
    """
    Parse OpenAI-format messages into components the inference backend expects.

    Handles both plain-string ``content`` and multimodal content-part arrays
    (``[{type: "text", ...}, {type: "image_url", ...}]``).

    Args:
        messages: List of ChatMessage instances.
        preserve_tool_history: When True, keep ``tool_calls`` /
            ``tool_call_id`` / ``name`` / ``reasoning_content`` fields on
            the messages the backend sees. Used by the tool-calling path
            so the backend can feed a complete conversation (including a
            prior assistant's tool_calls and the subsequent tool
            results) back into ``apply_chat_template``. Default False
            preserves Phase-1 behaviour for non-tool chat paths.

    Returns:
        system_prompt:  The system message text (empty string if none provided).
        chat_messages:  Non-system messages with content flattened to strings.
        image_base64:   Base64 data of the *first* image found, or ``None``.
    """
    system_prompt = ""
    chat_messages: list[dict] = []
    first_image_b64: Optional[str] = None

    for msg in messages:
        # ── System messages → extract as system_prompt ────────
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content
            elif isinstance(msg.content, list):
                # Unlikely but handle: join text parts
                system_prompt = "\n".join(
                    p.text for p in msg.content if p.type == "text"
                )
            continue

        # ── Tool-role messages (tool results) ─────────────────
        if msg.role == "tool":
            if not preserve_tool_history:
                # Legacy chat paths never expect a role="tool" message;
                # drop it to avoid confusing the non-tool template.
                continue
            entry = {
                "role": "tool",
                "content": msg.content if isinstance(msg.content, str) else "",
            }
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            chat_messages.append(entry)
            continue

        # ── User / assistant messages ─────────────────────────
        entry: dict = {"role": msg.role}

        if isinstance(msg.content, str):
            entry["content"] = msg.content
        elif isinstance(msg.content, list):
            text_parts: list[str] = []
            for part in msg.content:
                if part.type == "text":
                    text_parts.append(part.text)
                elif part.type == "image_url" and first_image_b64 is None:
                    url = part.image_url.url
                    if url.startswith("data:"):
                        first_image_b64 = url.split(",", 1)[1] if "," in url else None
                    else:
                        logger.warning(
                            f"Remote image URLs not yet supported: {url[:80]}..."
                        )
            entry["content"] = "\n".join(text_parts) if text_parts else ""
        else:
            entry["content"] = msg.content

        # Assistant tool-calling metadata: only surface when the caller
        # opted in, so the non-tool chat paths keep their compact
        # {"role","content"} shape and don't risk feeding an unknown
        # field into a chat template.
        if preserve_tool_history and msg.role == "assistant":
            if msg.tool_calls:
                # Normalize ToolCall instances back to dicts so the
                # tokenizer's apply_chat_template sees plain JSON-like
                # structures (Jinja can't render Pydantic models).
                norm: list[dict] = []
                for tc in msg.tool_calls:
                    if hasattr(tc, "model_dump"):
                        norm.append(tc.model_dump(exclude_none = True))
                    elif isinstance(tc, dict):
                        norm.append(tc)
                entry["tool_calls"] = norm
            if msg.reasoning_content:
                entry["reasoning_content"] = msg.reasoning_content

        chat_messages.append(entry)

    return system_prompt, chat_messages, first_image_b64


@router.post("/chat/completions")
async def openai_chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports multimodal messages: ``content`` may be a plain string or a
    list of content parts (``text`` / ``image_url``).

    Streaming (default):  returns SSE chunks matching OpenAI's format.
    Non-streaming:        returns a single ChatCompletion JSON object.

    Automatically routes to the correct backend:
    - GGUF models → llama-server via LlamaCppBackend
    - Other models → Unsloth/transformers via InferenceBackend
    """
    llama_backend = get_llama_cpp_backend()
    mlx_backend = get_mlx_lm_backend()
    # Chunk D peers.
    vlm_backend = get_mlx_vlm_backend()
    audio_backend = get_mlx_audio_backend()

    using_gguf = llama_backend.is_loaded
    using_mlx = mlx_backend.is_loaded
    using_vlm = vlm_backend.is_loaded
    using_mlx_audio = audio_backend.is_loaded

    # ── Determine which backend is active ─────────────────────
    if using_gguf:
        model_name = llama_backend.model_identifier or payload.model
        if getattr(llama_backend, "_is_audio", False):
            return await generate_audio(payload, request)
    elif using_vlm:
        model_name = vlm_backend.model_identifier or payload.model
    elif using_mlx_audio:
        model_name = audio_backend.model_identifier or payload.model
        # Audio-in path: when the caller supplies ``audio_base64`` the
        # MLX audio backend runs a best-effort transcription. We return
        # an OpenAI-shape chat completion with the transcribed text
        # so chat clients get a recognizable response.
        if payload.audio_base64:
            loop = asyncio.get_event_loop()
            raw = base64.b64decode(payload.audio_base64)
            try:
                text = await loop.run_in_executor(
                    None,
                    lambda: audio_backend.transcribe(
                        raw,
                        prompt = None,
                        temperature = payload.temperature or 0.0,
                    ),
                )
            except Exception as e:
                logger.error(f"MLX-Audio transcribe error: {e}", exc_info = True)
                raise HTTPException(status_code = 500, detail = str(e))
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            response = ChatCompletion(
                id = completion_id,
                created = int(time.time()),
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(content = text),
                        finish_reason = "stop",
                    )
                ],
            )
            return JSONResponse(
                content = response.model_dump(exclude_none = True)
            )
        # TTS path: route to /audio/generate for the OpenAI-style
        # JSON response.
        return await generate_audio(payload, request)
    elif using_mlx:
        model_name = mlx_backend.model_identifier or payload.model
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(
                status_code = 400,
                detail = "No model loaded. Call POST /inference/load first.",
            )
        model_name = backend.active_model_name or payload.model

        # ── Audio TTS path: auto-route to audio generation ────
        # (Whisper is ASR not TTS — handled below in audio input path)
        model_info = backend.models.get(backend.active_model_name, {})
        if model_info.get("is_audio") and model_info.get("audio_type") != "whisper":
            return await generate_audio(payload, request)

        # ── Whisper without audio: return clear error ──
        if model_info.get("audio_type") == "whisper" and not payload.audio_base64:
            raise HTTPException(
                status_code = 400,
                detail = "Whisper models require audio input. Please upload an audio file.",
            )

        # ── Audio INPUT path: decode WAV and route to audio input generation ──
        if payload.audio_base64 and model_info.get("has_audio_input"):
            audio_array = _decode_audio_base64(payload.audio_base64)
            system_prompt, chat_messages, _ = _extract_content_parts(payload.messages)
            cancel_event = threading.Event()
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            def audio_input_generate():
                if model_info.get("audio_type") == "whisper":
                    return backend.generate_whisper_response(
                        audio_array = audio_array,
                        cancel_event = cancel_event,
                    )
                return backend.generate_audio_input_response(
                    messages = chat_messages,
                    system_prompt = system_prompt,
                    audio_array = audio_array,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_new_tokens = payload.max_tokens or 2048,
                    repetition_penalty = payload.repetition_penalty,
                    cancel_event = cancel_event,
                )

            if payload.stream:

                async def audio_input_stream():
                    try:
                        first_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(role = "assistant"),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                        for chunk_text in audio_input_generate():
                            if await request.is_disconnected():
                                cancel_event.set()
                                return
                            if chunk_text:
                                chunk = ChatCompletionChunk(
                                    id = completion_id,
                                    created = created,
                                    model = model_name,
                                    choices = [
                                        ChunkChoice(
                                            delta = ChoiceDelta(content = chunk_text),
                                            finish_reason = None,
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                        final_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(delta = ChoiceDelta(), finish_reason = "stop")
                            ],
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        cancel_event.set()
                        raise
                    except Exception as e:
                        logger.error(
                            f"Error during audio input streaming: {e}", exc_info = True
                        )
                        yield f"data: {json.dumps({'error': {'message': _friendly_error(e), 'type': 'server_error'}})}\n\n"

                return StreamingResponse(
                    audio_input_stream(),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                full_text = "".join(audio_input_generate())
                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())

    # ── Standard OpenAI function-calling pass-through (GGUF only) ────
    # When a client (opencode / Claude Code via OpenAI compat / Cursor /
    # Continue / ...) sends standard OpenAI `tools` without Studio's
    # `enable_tools` shorthand, forward the request to llama-server
    # verbatim so structured `tool_calls` flow back to the client. This
    # branch runs BEFORE `_extract_content_parts` because that helper is
    # unaware of `role="tool"` messages and assistant messages that only
    # carry `tool_calls` (content=None) — both of which are valid in
    # multi-turn client-side tool loops.
    _has_tool_messages = any(m.role == "tool" or m.tool_calls for m in payload.messages)
    if (
        using_gguf
        and llama_backend.supports_tools
        and not payload.enable_tools
        and ((payload.tools and len(payload.tools) > 0) or _has_tool_messages)
    ):
        # Preserve the vision guard that would otherwise run in the
        # non-passthrough path below: text-only tool-capable GGUFs
        # should return a clear 400 here rather than forwarding the
        # image to llama-server and surfacing an opaque upstream error.
        if not llama_backend.is_vision and (
            payload.image_base64
            or any(
                isinstance(m.content, list)
                and any(isinstance(p, ImageContentPart) for p in m.content)
                for m in payload.messages
            )
        ):
            raise HTTPException(
                status_code = 400,
                detail = "Image provided but current GGUF model does not support vision.",
            )

        cancel_event = threading.Event()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if payload.stream:
            return await _openai_passthrough_stream(
                request,
                cancel_event,
                llama_backend,
                payload,
                model_name,
                completion_id,
            )
        return await _openai_passthrough_non_streaming(
            llama_backend,
            payload,
            model_name,
        )

    # ── Parse messages (handles multimodal content parts) ─────
    system_prompt, chat_messages, extracted_image_b64 = _extract_content_parts(
        payload.messages
    )

    if not chat_messages:
        raise HTTPException(
            status_code = 400,
            detail = "At least one non-system message is required.",
        )

    # ── GGUF path: proxy to llama-server /v1/chat/completions ──
    if using_gguf:
        # Reject images if this GGUF model doesn't support vision
        image_b64 = extracted_image_b64 or payload.image_base64
        if image_b64 and not llama_backend.is_vision:
            raise HTTPException(
                status_code = 400,
                detail = "Image provided but current GGUF model does not support vision.",
            )

        # Convert image to PNG for llama-server (stb_image has limited format support)
        if image_b64:
            try:
                import base64 as _b64
                from io import BytesIO as _BytesIO
                from PIL import Image as _Image

                raw = _b64.b64decode(image_b64)
                # Normalize to RGB so PNG encoding succeeds regardless of
                # source mode (RGBA, P, L, CMYK, I, F, ...). Previously
                # we only converted RGBA, which left CMYK/I/F to raise at
                # img.save(PNG).
                img = _Image.open(_BytesIO(raw)).convert("RGB")
                buf = _BytesIO()
                img.save(buf, format = "PNG")
                image_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                raise HTTPException(
                    status_code = 400, detail = f"Failed to process image: {e}"
                )

        # Build message list with system prompt prepended
        gguf_messages = []
        if system_prompt:
            gguf_messages.append({"role": "system", "content": system_prompt})
        gguf_messages.extend(chat_messages)

        cancel_event = threading.Event()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # ── Tool-calling path (agentic loop) ──────────────────
        use_tools = (
            payload.enable_tools and llama_backend.supports_tools and not image_b64
        )

        if use_tools:
            from core.inference.tools import ALL_TOOLS

            if payload.enabled_tools is not None:
                tools_to_use = [
                    t
                    for t in ALL_TOOLS
                    if t["function"]["name"] in payload.enabled_tools
                ]
            else:
                tools_to_use = ALL_TOOLS

            # ── Tool-use system prompt nudge ──────────────────────
            _tool_names = {t["function"]["name"] for t in tools_to_use}
            _has_web = "web_search" in _tool_names
            _has_code = "python" in _tool_names or "terminal" in _tool_names

            _date_line = f"The current date is {_date.today().isoformat()}."

            # Small models (<9B) struggle with multi-step search plans,
            # so simplify the web tips to avoid plan-then-stall behavior.
            _model_size_b = _extract_model_size_b(model_name)
            _is_small_model = _model_size_b is not None and _model_size_b < 9

            if _is_small_model:
                _web_tips = "Do not repeat the same search query."
            else:
                _web_tips = (
                    "When you search and find a relevant URL in the results, "
                    "fetch its full content by calling web_search with the url parameter. "
                    "Do not repeat the same search query. If a search returns "
                    "no useful results, try rephrasing or fetching a result URL directly."
                )
            _code_tips = (
                "Use code execution for math, calculations, data processing, "
                "or to parse and analyze information from tool results."
            )

            if _has_web and _has_code:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "tools rather than answering from memory. "
                    + _web_tips
                    + " "
                    + _code_tips
                )
            elif _has_code:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "code execution rather than answering from memory. " + _code_tips
                )
            elif _has_web:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "web search for up-to-date or uncertain factual "
                    "information rather than answering from memory. " + _web_tips
                )
            else:
                _nudge = ""

            if _nudge:
                _nudge += _TOOL_ACTION_NUDGE
                # Append nudge to system prompt (preserve user's prompt)
                if system_prompt:
                    system_prompt = system_prompt.rstrip() + "\n\n" + _nudge
                else:
                    system_prompt = _nudge
                # Rebuild gguf_messages with updated system prompt
                gguf_messages = []
                if system_prompt:
                    gguf_messages.append({"role": "system", "content": system_prompt})
                gguf_messages.extend(chat_messages)

            # ── Strip stale tool-call XML from conversation history ─
            for _msg in gguf_messages:
                if _msg.get("role") == "assistant" and isinstance(
                    _msg.get("content"), str
                ):
                    _msg["content"] = _TOOL_XML_RE.sub("", _msg["content"]).strip()

            def gguf_generate_with_tools():
                return llama_backend.generate_chat_completion_with_tools(
                    messages = gguf_messages,
                    tools = tools_to_use,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_tokens = payload.max_tokens,
                    repetition_penalty = payload.repetition_penalty,
                    presence_penalty = payload.presence_penalty,
                    cancel_event = cancel_event,
                    enable_thinking = payload.enable_thinking,
                    auto_heal_tool_calls = payload.auto_heal_tool_calls
                    if payload.auto_heal_tool_calls is not None
                    else True,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 25,
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                )

            _tool_sentinel = object()

            async def gguf_tool_stream():
                try:
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Iterate the synchronous generator in a thread so
                    # the event loop stays free for disconnect detection.
                    gen = gguf_generate_with_tools()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return

                        event = await asyncio.to_thread(next, gen, _tool_sentinel)
                        if event is _tool_sentinel:
                            break

                        if event["type"] == "status":
                            # Empty status marks an iteration boundary
                            # in the GGUF tool loop (e.g. after a
                            # re-prompt).  Reset the cumulative cursor
                            # so the next assistant turn streams cleanly.
                            if not event["text"]:
                                prev_text = ""
                            # Emit tool status as a custom SSE event
                            # (including empty ones to clear UI badges)
                            status_data = json.dumps(
                                {
                                    "type": "tool_status",
                                    "content": event["text"],
                                }
                            )
                            yield f"data: {status_data}\n\n"
                            continue

                        if event["type"] in ("tool_start", "tool_end"):
                            if event["type"] == "tool_start":
                                prev_text = ""
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        if event["type"] == "metadata":
                            _stream_usage = event.get("usage")
                            _stream_timings = event.get("timings")
                            continue

                        # "content" type -- cumulative text
                        # Sanitize the full cumulative then diff against
                        # the last sanitized snapshot so cross-chunk XML
                        # tags are handled correctly.
                        raw_cumulative = event.get("text", "")
                        clean_cumulative = _TOOL_XML_RE.sub("", raw_cumulative)
                        new_text = clean_cumulative[len(prev_text) :]
                        prev_text = clean_cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    # Usage chunk (OpenAI-standard: choices=[], usage populated)
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get("total_tokens", 0),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(f"Error during GGUF tool streaming: {e}\n{tb}")
                    error_chunk = {
                        "error": {
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_tool_stream(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Standard GGUF path (no tools) ─────────────────────

        def gguf_generate():
            return llama_backend.generate_chat_completion(
                messages = gguf_messages,
                image_b64 = image_b64,
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = payload.max_tokens,
                repetition_penalty = payload.repetition_penalty,
                presence_penalty = payload.presence_penalty,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
            )

        _gguf_sentinel = object()

        if payload.stream:

            async def gguf_stream_chunks():
                try:
                    # First chunk: role
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Iterate the synchronous generator in a thread so
                    # the event loop stays free for disconnect detection.
                    gen = gguf_generate()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        cumulative = await asyncio.to_thread(next, gen, _gguf_sentinel)
                        if cumulative is _gguf_sentinel:
                            break
                        # Capture server metadata for final usage chunk
                        if isinstance(cumulative, dict):
                            if cumulative.get("type") == "metadata":
                                _stream_usage = cumulative.get("usage")
                                _stream_timings = cumulative.get("timings")
                            else:
                                logger.warning(
                                    "gguf_stream_chunks: unexpected dict event: %s",
                                    {
                                        k: v
                                        for k, v in cumulative.items()
                                        if k != "timings"
                                    },
                                )
                            continue
                        new_text = cumulative[len(prev_text) :]
                        prev_text = cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Final chunk
                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    # Usage chunk (OpenAI-standard: choices=[], usage populated)
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get("total_tokens", 0),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info = True)
                    error_chunk = {
                        "error": {
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_stream_chunks(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                full_text = ""
                for token in gguf_generate():
                    if isinstance(token, dict):
                        continue  # skip metadata dict in non-streaming path
                    full_text = token

                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())

            except Exception as e:
                logger.error(f"Error during GGUF completion: {e}", exc_info = True)
                raise HTTPException(status_code = 500, detail = str(e))

    # ── MLX-VLM path: stream via mlx_vlm.stream_generate ──────
    # Phase 9 (Chunk D). Image-bearing multimodal chats route here.
    if using_vlm:
        image_b64 = extracted_image_b64 or payload.image_base64

        # Build message list with system prompt prepended. VLM models
        # don't support Studio's built-in tool-agentic loop yet (Phase 9
        # non-goal); tool_calls passthrough from clients is accepted
        # but the content/parsing is done client-side.
        vlm_messages: List[Dict[str, Any]] = []
        if system_prompt:
            vlm_messages.append({"role": "system", "content": system_prompt})
        vlm_messages.extend(chat_messages)

        cancel_event = threading.Event()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        _vlm_stop: Optional[list[str]] = None
        if payload.stop is not None:
            if isinstance(payload.stop, str):
                _vlm_stop = [payload.stop]
            elif isinstance(payload.stop, list):
                _vlm_stop = [s for s in payload.stop if isinstance(s, str) and s]

        def vlm_generate():
            return vlm_backend.generate_chat_completion(
                messages = vlm_messages,
                image_b64 = image_b64,
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = payload.max_tokens,
                repetition_penalty = payload.repetition_penalty,
                presence_penalty = payload.presence_penalty,
                stop = _vlm_stop,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
            )

        _vlm_sentinel = object()

        if payload.stream:

            async def vlm_stream_chunks():
                try:
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    gen = vlm_generate()
                    prev_text = ""
                    _vlm_usage = None
                    _vlm_timings = None
                    _vlm_finish = "stop"
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            break
                        chunk = await asyncio.to_thread(
                            next, gen, _vlm_sentinel
                        )
                        if chunk is _vlm_sentinel:
                            break
                        if isinstance(chunk, dict) and chunk.get("type") == "metadata":
                            _vlm_usage = chunk.get("usage")
                            _vlm_timings = chunk.get("timings")
                            _vlm_finish = chunk.get("finish_reason") or "stop"
                            continue
                        if isinstance(chunk, str):
                            delta = chunk[len(prev_text):]
                            prev_text = chunk
                            if delta:
                                out = ChatCompletionChunk(
                                    id = completion_id,
                                    created = created,
                                    model = model_name,
                                    choices = [
                                        ChunkChoice(
                                            delta = ChoiceDelta(content = delta),
                                            finish_reason = None,
                                        )
                                    ],
                                )
                                yield f"data: {out.model_dump_json(exclude_none = True)}\n\n"
                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = _vlm_finish,
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(
                        f"Error during MLX-VLM streaming: {e}", exc_info = True
                    )
                    yield f"data: {json.dumps({'error': {'message': _friendly_error(e), 'type': 'server_error'}})}\n\n"

            return StreamingResponse(
                vlm_stream_chunks(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming: accumulate then return one JSON body.
            full_text = ""
            usage: Optional[dict] = None
            _finish = "stop"
            try:
                for chunk in vlm_generate():
                    if isinstance(chunk, dict) and chunk.get("type") == "metadata":
                        usage = chunk.get("usage")
                        _finish = chunk.get("finish_reason") or "stop"
                        continue
                    if isinstance(chunk, str):
                        full_text = chunk
            except Exception as e:
                logger.error(f"MLX-VLM non-stream error: {e}", exc_info = True)
                raise HTTPException(status_code = 500, detail = str(e))
            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(content = full_text),
                        finish_reason = _finish,
                    )
                ],
                usage = usage,
            )
            return JSONResponse(content = response.model_dump(exclude_none = True))

    # ── MLX path: stream via mlx_lm.stream_generate ───────────
    if using_mlx:
        # Reject images: MLX text backend has no vision support. The
        # vision-bearing chats are handled by the ``using_vlm`` branch
        # above — this is the text-only MLX path.
        image_b64 = extracted_image_b64 or payload.image_base64
        if image_b64:
            raise HTTPException(
                status_code = 400,
                detail = "MLX text backend does not support image inputs. Load a VLM checkpoint instead.",
            )

        # Tool-calling: three possible paths, identical to the GGUF branch.
        #   1. ``enable_tools=true`` → agentic loop with Studio's built-in
        #      tools (web_search / python / terminal). Requires the model
        #      to advertise ``supports_tools``.
        #   2. ``tools=[...]`` client pass-through → parse calls out of
        #      the model output and emit OpenAI ``tool_calls`` deltas so
        #      external clients (opencode / Claude Code / Cursor) can
        #      execute the tools themselves.
        #   3. Neither → plain chat (the Chunk A/B streaming path).
        _mlx_has_tool_messages = any(
            m.role == "tool" or m.tool_calls for m in payload.messages
        )
        _mlx_wants_client_tools = (
            mlx_backend.supports_tools
            and not payload.enable_tools
            and (
                (payload.tools and len(payload.tools) > 0)
                or _mlx_has_tool_messages
            )
        )

        if payload.enable_tools and not mlx_backend.supports_tools:
            raise HTTPException(
                status_code = 400,
                detail = (
                    "Loaded MLX model does not advertise tool-calling "
                    "support. Load a tool-capable model "
                    "(e.g. mlx-community Qwen3 / Bonsai / Hermes)."
                ),
            )
        if payload.tools and not mlx_backend.supports_tools:
            raise HTTPException(
                status_code = 400,
                detail = (
                    "Client-side tools were requested but the loaded MLX "
                    "model does not advertise tool-calling support."
                ),
            )

        # Build message list with system prompt prepended. When tools
        # are in play we re-extract with ``preserve_tool_history=True``
        # so ``role="tool"`` results and assistant ``tool_calls`` reach
        # the backend for apply_chat_template. The regular chat path
        # keeps Phase-1's lean message shape.
        _mlx_using_tools = payload.enable_tools or _mlx_wants_client_tools
        if _mlx_using_tools:
            _tool_system, _tool_msgs, _ = _extract_content_parts(
                payload.messages, preserve_tool_history = True
            )
            mlx_messages = []
            if _tool_system:
                mlx_messages.append({"role": "system", "content": _tool_system})
            mlx_messages.extend(_tool_msgs)
        else:
            mlx_messages = []
            if system_prompt:
                mlx_messages.append({"role": "system", "content": system_prompt})
            mlx_messages.extend(chat_messages)

        cancel_event = threading.Event()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Normalize OpenAI's ``stop`` field: str | list[str] | None.
        _mlx_stop: Optional[list[str]] = None
        if payload.stop is not None:
            if isinstance(payload.stop, str):
                _mlx_stop = [payload.stop]
            elif isinstance(payload.stop, list):
                _mlx_stop = [s for s in payload.stop if isinstance(s, str) and s]

        # ── MLX server-side tool agentic loop (enable_tools=true) ──
        if payload.enable_tools:
            from core.inference.tools import ALL_TOOLS

            if payload.enabled_tools is not None:
                mlx_tools = [
                    t
                    for t in ALL_TOOLS
                    if t["function"]["name"] in payload.enabled_tools
                ]
            else:
                mlx_tools = ALL_TOOLS

            def mlx_generate_with_tools():
                return mlx_backend.generate_chat_completion_with_tools(
                    messages = mlx_messages,
                    tools = mlx_tools,
                    tool_choice = payload.tool_choice,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_tokens = payload.max_tokens,
                    repetition_penalty = payload.repetition_penalty,
                    presence_penalty = payload.presence_penalty,
                    stop = _mlx_stop,
                    cancel_event = cancel_event,
                    enable_thinking = payload.enable_thinking,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 10,
                    auto_heal_tool_calls = (
                        payload.auto_heal_tool_calls
                        if payload.auto_heal_tool_calls is not None
                        else True
                    ),
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                )

            if payload.stream:
                return StreamingResponse(
                    _mlx_agentic_stream(
                        request = request,
                        cancel_event = cancel_event,
                        run_gen = mlx_generate_with_tools,
                        completion_id = completion_id,
                        created = created,
                        model_name = model_name,
                    ),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            return await _mlx_agentic_non_streaming(
                run_gen = mlx_generate_with_tools,
                completion_id = completion_id,
                created = created,
                model_name = model_name,
            )

        # ── MLX client-side tools pass-through (standard OpenAI) ──
        if _mlx_wants_client_tools:
            if payload.stream:
                return StreamingResponse(
                    _mlx_openai_passthrough_stream(
                        request = request,
                        cancel_event = cancel_event,
                        mlx_backend = mlx_backend,
                        payload = payload,
                        messages = mlx_messages,
                        stop = _mlx_stop,
                        completion_id = completion_id,
                        created = created,
                        model_name = model_name,
                    ),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            return await _mlx_openai_passthrough_non_streaming(
                mlx_backend = mlx_backend,
                payload = payload,
                messages = mlx_messages,
                stop = _mlx_stop,
                completion_id = completion_id,
                created = created,
                model_name = model_name,
            )

        def mlx_generate():
            return mlx_backend.generate_chat_completion(
                messages = mlx_messages,
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = payload.max_tokens,
                repetition_penalty = payload.repetition_penalty,
                presence_penalty = payload.presence_penalty,
                stop = _mlx_stop,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
            )

        _mlx_sentinel = object()

        if payload.stream:

            async def mlx_stream_chunks():
                try:
                    # First chunk: role
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    gen = mlx_generate()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        cumulative = await asyncio.to_thread(
                            next, gen, _mlx_sentinel
                        )
                        if cumulative is _mlx_sentinel:
                            break
                        if isinstance(cumulative, dict):
                            if cumulative.get("type") == "metadata":
                                _stream_usage = cumulative.get("usage")
                                _stream_timings = cumulative.get("timings")
                            else:
                                logger.warning(
                                    "mlx_stream_chunks: unexpected dict event: %s",
                                    cumulative,
                                )
                            continue
                        new_text = cumulative[len(prev_text):]
                        prev_text = cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get(
                                "prompt_tokens", 0
                            ),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get(
                                "total_tokens", 0
                            ),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(
                        f"Error during MLX streaming: {e}", exc_info = True
                    )
                    error_chunk = {
                        "error": {
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                mlx_stream_chunks(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                full_text = ""
                for token in mlx_generate():
                    if isinstance(token, dict):
                        continue
                    full_text = token
                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())
            except Exception as e:
                logger.error(f"Error during MLX completion: {e}", exc_info = True)
                raise HTTPException(status_code = 500, detail = str(e))

    # ── Standard Unsloth path ─────────────────────────────────

    # Decode image (from content parts OR legacy field)
    image_b64 = extracted_image_b64 or payload.image_base64
    image = None

    if image_b64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code = 400, detail = f"Failed to decode image: {e}")

    # Shared generation kwargs
    gen_kwargs = dict(
        messages = chat_messages,
        system_prompt = system_prompt,
        image = image,
        temperature = payload.temperature,
        top_p = payload.top_p,
        top_k = payload.top_k,
        min_p = payload.min_p,
        max_new_tokens = payload.max_tokens or 2048,
        repetition_penalty = payload.repetition_penalty,
    )

    # Choose generation path (adapter-controlled or standard)
    cancel_event = threading.Event()

    if payload.use_adapter is not None:

        def generate():
            return backend.generate_with_adapter_control(
                use_adapter = payload.use_adapter,
                cancel_event = cancel_event,
                **gen_kwargs,
            )
    else:

        def generate():
            return backend.generate_chat_response(
                cancel_event = cancel_event, **gen_kwargs
            )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming response ────────────────────────────────────────
    if payload.stream:

        async def stream_chunks():
            try:
                first_chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(role = "assistant"),
                            finish_reason = None,
                        )
                    ],
                )
                yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                prev_text = ""
                # Run sync generator in thread pool to avoid blocking
                # the event loop. Critical for compare mode: two SSE
                # requests arrive concurrently but the orchestrator
                # serializes them via _gen_lock. Without run_in_executor
                # the second request's blocking lock acquisition would
                # freeze the entire event loop, stalling both streams.
                _DONE = object()  # sentinel for generator exhaustion
                loop = asyncio.get_event_loop()
                gen = generate()
                while True:
                    # next(gen, _DONE) returns _DONE instead of raising
                    # StopIteration — StopIteration cannot propagate
                    # through asyncio futures (Python limitation).
                    cumulative = await loop.run_in_executor(None, next, gen, _DONE)
                    if cumulative is _DONE:
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
                        return
                    new_text = cumulative[len(prev_text) :]
                    prev_text = cumulative
                    if not new_text:
                        continue
                    chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(content = new_text),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                final_chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(),
                            finish_reason = "stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                raise
            except Exception as e:
                backend.reset_generation_state()
                logger.error(f"Error during OpenAI streaming: {e}", exc_info = True)
                error_chunk = {
                    "error": {
                        "message": _friendly_error(e),
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            stream_chunks(),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming response ────────────────────────────────────
    else:
        try:
            full_text = ""
            for token in generate():
                full_text = token

            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(content = full_text),
                        finish_reason = "stop",
                    )
                ],
            )
            return JSONResponse(content = response.model_dump())

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during OpenAI completion: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))


# =====================================================================
# Sandbox file serving  (/sandbox/{session_id}/{filename})
# =====================================================================

_SANDBOX_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@router.get("/sandbox/{session_id}/{filename}")
async def serve_sandbox_file(
    session_id: str,
    filename: str,
    request: Request,
    token: Optional[str] = None,
):
    """
    Serve image files created by Python tool execution.

    Accepts auth via Authorization header OR ?token= query param
    (needed because <img src> cannot send custom headers).
    """
    from fastapi.responses import FileResponse

    # ── Authentication (header or query param) ──────────────────
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        jwt_token = auth_header[7:]
    elif token:
        jwt_token = token
    else:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing authentication token",
        )
    from fastapi.security import HTTPAuthorizationCredentials

    creds = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = jwt_token)
    await get_current_subject(creds)

    # ── Filename sanitization ───────────────────────────────────
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code = 404, detail = "Not found")

    # ── Extension allowlist ─────────────────────────────────────
    ext = os.path.splitext(safe_filename)[1].lower()
    media_type = _SANDBOX_MEDIA_TYPES.get(ext)
    if not media_type:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "File type not allowed",
        )

    # ── Path containment check ──────────────────────────────────
    home = os.path.expanduser("~")
    sandbox_root = os.path.realpath(os.path.join(home, "studio_sandbox"))
    safe_session = os.path.basename(session_id.replace("..", ""))
    if not safe_session:
        raise HTTPException(status_code = 404, detail = "Not found")

    file_path = os.path.realpath(
        os.path.join(sandbox_root, safe_session, safe_filename)
    )
    if not file_path.startswith(sandbox_root + os.sep):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Access denied",
        )

    if not os.path.isfile(file_path):
        raise HTTPException(status_code = 404, detail = "Not found")

    return FileResponse(
        path = file_path,
        media_type = media_type,
        headers = {
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# =====================================================================
# OpenAI-Compatible Models Listing  (/models → /v1/models)
# =====================================================================


@router.get("/models")
async def openai_list_models(
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible model listing endpoint.

    Returns the currently loaded model in the format expected by
    OpenAI-compatible clients (``GET /v1/models``).
    """
    models = []

    # Check GGUF backend
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded:
        models.append(
            {
                "id": llama_backend.model_identifier,
                "object": "model",
                "owned_by": "local",
            }
        )

    # Check Unsloth backend
    backend = get_inference_backend()
    if backend.active_model_name:
        models.append(
            {
                "id": backend.active_model_name,
                "object": "model",
                "owned_by": "local",
            }
        )

    return {"object": "list", "data": models}


# =====================================================================
# OpenAI-Compatible Completions Proxy  (/completions → /v1/completions)
# =====================================================================


@router.post("/completions")
async def openai_completions(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible text completions endpoint (non-chat).

    Transparently proxies to the running llama-server's ``/v1/completions``.
    Only available when a GGUF model is loaded.
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    body = await request.json()
    target_url = f"{llama_backend.base_url}/v1/completions"
    is_stream = body.get("stream", False)

    if is_stream:

        async def _stream():
            # Manual httpx client/response lifecycle AND explicit
            # aiter_bytes() iterator close — see _anthropic_passthrough_stream
            # for the full rationale. Saving `bytes_iter = resp.aiter_bytes()`
            # and `await bytes_iter.aclose()` in the finally block is the
            # part that matters for avoiding the Python 3.13 + httpcore
            # 1.0.x "Exception ignored in: <async_generator>" / anyio
            # cancel-scope trace: an anonymous async for leaves the
            # iterator unclosed, so Python's asyncgen GC finalizer runs
            # cleanup on a later pass in a different asyncio task.
            client = httpx.AsyncClient(timeout = 600)
            resp = None
            bytes_iter = None
            try:
                req = client.build_request("POST", target_url, json = body)
                resp = await client.send(req, stream = True)
                bytes_iter = resp.aiter_bytes()
                async for chunk in bytes_iter:
                    yield chunk
            except Exception as e:
                logger.error("openai_completions stream error: %s", e)
            finally:
                if bytes_iter is not None:
                    try:
                        await bytes_iter.aclose()
                    except Exception:
                        pass
                if resp is not None:
                    try:
                        await resp.aclose()
                    except Exception:
                        pass
                try:
                    await client.aclose()
                except Exception:
                    pass

        return StreamingResponse(_stream(), media_type = "text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(target_url, json = body, timeout = 600)
        return Response(
            content = resp.content,
            status_code = resp.status_code,
            media_type = "application/json",
        )


# =====================================================================
# OpenAI-Compatible Embeddings Proxy  (/embeddings → /v1/embeddings)
# =====================================================================


@router.post("/embeddings")
async def openai_embeddings(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible embeddings endpoint.

    Transparently proxies to the running llama-server's ``/v1/embeddings``.
    Only available when a GGUF model is loaded.
    Note: the loaded model must support pooling; otherwise llama-server
    will return an error (expected).
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    body = await request.json()
    target_url = f"{llama_backend.base_url}/v1/embeddings"

    async with httpx.AsyncClient() as client:
        resp = await client.post(target_url, json = body, timeout = 600)
    return Response(
        content = resp.content,
        status_code = resp.status_code,
        media_type = "application/json",
    )


# =====================================================================
# OpenAI Responses API  (/responses → /v1/responses)
# =====================================================================


def _normalise_responses_input(payload: ResponsesRequest) -> list[ChatMessage]:
    """Convert a ResponsesRequest into a list of ChatMessage for the completions backend."""
    messages: list[ChatMessage] = []

    # System / developer instructions
    if payload.instructions:
        messages.append(ChatMessage(role = "system", content = payload.instructions))

    # Simple string input
    if isinstance(payload.input, str):
        if payload.input:
            messages.append(ChatMessage(role = "user", content = payload.input))
        return messages

    # List of ResponsesInputMessage
    for msg in payload.input:
        role = "system" if msg.role == "developer" else msg.role

        if isinstance(msg.content, str):
            messages.append(ChatMessage(role = role, content = msg.content))
        else:
            # Convert Responses content parts -> Chat content parts
            parts = []
            for part in msg.content:
                if isinstance(part, ResponsesInputTextPart):
                    parts.append(TextContentPart(type = "text", text = part.text))
                elif isinstance(part, ResponsesInputImagePart):
                    parts.append(
                        ImageContentPart(
                            type = "image_url",
                            image_url = ImageUrl(url = part.image_url, detail = part.detail),
                        )
                    )
            messages.append(ChatMessage(role = role, content = parts if parts else ""))

    return messages


def _build_chat_request(
    payload: ResponsesRequest, messages: list[ChatMessage], stream: bool
) -> ChatCompletionRequest:
    """Build a ChatCompletionRequest from a ResponsesRequest."""
    chat_kwargs = dict(
        model = payload.model,
        messages = messages,
        stream = stream,
    )
    if payload.temperature is not None:
        chat_kwargs["temperature"] = payload.temperature
    if payload.top_p is not None:
        chat_kwargs["top_p"] = payload.top_p
    if payload.max_output_tokens is not None:
        chat_kwargs["max_tokens"] = payload.max_output_tokens
    return ChatCompletionRequest(**chat_kwargs)


async def _responses_non_streaming(
    payload: ResponsesRequest,
    messages: list[ChatMessage],
    request: Request,
) -> JSONResponse:
    """Handle a non-streaming Responses API call."""
    chat_req = _build_chat_request(payload, messages, stream = False)
    result = await openai_chat_completions(chat_req, request)

    # openai_chat_completions returns a JSONResponse for non-streaming
    if isinstance(result, JSONResponse):
        body = json.loads(result.body.decode())
    elif isinstance(result, Response):
        body = json.loads(result.body.decode())
    else:
        body = result

    # Extract content and usage from the Chat Completions response
    choices = body.get("choices", [])
    text = ""
    if choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "") or ""

    usage_data = body.get("usage", {})
    input_tokens = usage_data.get("prompt_tokens", 0)
    output_tokens = usage_data.get("completion_tokens", 0)

    resp_id = f"resp_{uuid.uuid4().hex[:12]}"
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"

    response = ResponsesResponse(
        id = resp_id,
        created_at = int(time.time()),
        status = "completed",
        model = body.get("model", payload.model),
        output = [
            ResponsesOutputMessage(
                id = msg_id,
                status = "completed",
                role = "assistant",
                content = [
                    ResponsesOutputTextContent(text = text),
                ],
            ),
        ],
        usage = ResponsesUsage(
            input_tokens = input_tokens,
            output_tokens = output_tokens,
            total_tokens = input_tokens + output_tokens,
        ),
        temperature = payload.temperature,
        top_p = payload.top_p,
        max_output_tokens = payload.max_output_tokens,
        instructions = payload.instructions,
    )
    return JSONResponse(content = response.model_dump())


async def _responses_stream(
    payload: ResponsesRequest,
    messages: list[ChatMessage],
    request: Request,
):
    """Handle a streaming Responses API call, emitting named SSE events."""
    resp_id = f"resp_{uuid.uuid4().hex[:12]}"
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    item_id = f"item_{uuid.uuid4().hex[:12]}"
    created_at = int(time.time())

    chat_req = _build_chat_request(payload, messages, stream = True)
    result = await openai_chat_completions(chat_req, request)

    async def event_generator():
        full_text = ""
        input_tokens = 0
        output_tokens = 0

        # ── Preamble events ──
        yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'status': 'in_progress', 'model': payload.model, 'output': [], 'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}}})}\n\n"

        # output_item.added
        output_item = {
            "type": "message",
            "id": msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': output_item})}\n\n"

        # content_part.added
        content_part = {"type": "output_text", "text": "", "annotations": []}
        yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': content_part})}\n\n"

        # ── Stream delta events from the inner chat completions stream ──
        if isinstance(result, StreamingResponse):
            async for raw_chunk in result.body_iterator:
                if isinstance(raw_chunk, bytes):
                    raw_chunk = raw_chunk.decode("utf-8", errors = "replace")

                for line in raw_chunk.split("\n"):
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk_data.get("choices", [])
                    if not choices:
                        # Check for usage in final chunk
                        usage = chunk_data.get("usage")
                        if usage:
                            input_tokens = usage.get("prompt_tokens", input_tokens)
                            output_tokens = usage.get(
                                "completion_tokens", output_tokens
                            )
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        full_text += content
                        delta_event = {
                            "type": "response.output_text.delta",
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": content,
                        }
                        yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"

                    # Check for usage in chunk
                    usage = chunk_data.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)

        # ── Closing events ──
        # output_text.done
        yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': full_text})}\n\n"

        # content_part.done
        yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': full_text, 'annotations': []}})}\n\n"

        # output_item.done
        yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'type': 'message', 'id': msg_id, 'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_text, 'annotations': []}]}})}\n\n"

        # response.completed
        total_tokens = input_tokens + output_tokens
        completed_response = {
            "type": "response.completed",
            "response": {
                "id": resp_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "model": payload.model,
                "output": [
                    {
                        "type": "message",
                        "id": msg_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": full_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            },
        }
        yield f"event: response.completed\ndata: {json.dumps(completed_response)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/responses")
async def openai_responses(
    payload: ResponsesRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI Responses API endpoint.

    Accepts the Responses-format request, converts it to a
    ChatCompletionRequest internally, and returns a response
    matching the OpenAI Responses API schema (output array,
    input_tokens/output_tokens, named SSE events for streaming).
    """
    messages = _normalise_responses_input(payload)
    if not messages:
        raise HTTPException(status_code = 400, detail = "No input provided.")

    if payload.stream:
        return await _responses_stream(payload, messages, request)
    return await _responses_non_streaming(payload, messages, request)


# =====================================================================
# Anthropic-Compatible Messages API  (/messages → /v1/messages)
# =====================================================================


@router.post("/messages")
async def anthropic_messages(
    payload: AnthropicMessagesRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Anthropic-compatible Messages API endpoint.

    Translates Anthropic message format to internal OpenAI format, runs
    through the existing agentic tool loop when tools are provided, and
    returns responses in Anthropic Messages API format (streaming SSE or
    non-streaming JSON).
    """
    llama_backend = get_llama_cpp_backend()
    mlx_backend = get_mlx_lm_backend()

    # Prefer MLX when it's the active backend; fall back to GGUF;
    # otherwise 503. Mirrors the /v1/chat/completions routing logic.
    using_mlx = mlx_backend.is_loaded
    using_gguf = llama_backend.is_loaded

    if not using_mlx and not using_gguf:
        raise HTTPException(
            status_code = 503,
            detail = "No model loaded. Load a GGUF or MLX model first.",
        )

    if using_mlx:
        model_name = (
            getattr(mlx_backend, "model_identifier", None) or payload.model
        )
    else:
        model_name = (
            getattr(llama_backend, "model_identifier", None) or payload.model
        )
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # ── Translate Anthropic → OpenAI ──────────────────────────
    openai_messages = anthropic_messages_to_openai(
        [m.model_dump() for m in payload.messages],
        payload.system,
    )

    temperature = payload.temperature if payload.temperature is not None else 0.6
    top_p = payload.top_p if payload.top_p is not None else 0.95
    top_k = payload.top_k if payload.top_k is not None else 20
    min_p = payload.min_p if payload.min_p is not None else 0.01
    repetition_penalty = (
        payload.repetition_penalty if payload.repetition_penalty is not None else 1.0
    )
    presence_penalty = (
        payload.presence_penalty if payload.presence_penalty is not None else 0.0
    )
    stop = payload.stop_sequences or None

    # Translate Anthropic tool_choice to OpenAI format for forwarding to
    # llama-server. Falls back to "auto" when unset or unrecognized, which
    # matches the prior hardcoded behavior.
    openai_tool_choice = anthropic_tool_choice_to_openai(payload.tool_choice)
    if openai_tool_choice is None:
        openai_tool_choice = "auto"

    cancel_event = threading.Event()

    # ── Tool routing ──────────────────────────────────────────
    # Three paths, evaluated per the active backend:
    # 1. enable_tools=true → server-side execution of built-in tools (Unsloth shorthand)
    # 2. tools=[...] only  → client-side pass-through (standard Anthropic behavior)
    # 3. neither           → plain chat
    active_backend = mlx_backend if using_mlx else llama_backend
    server_tools = bool(payload.enable_tools and active_backend.supports_tools)
    client_tools = bool(
        not server_tools
        and payload.tools
        and len(payload.tools) > 0
        and active_backend.supports_tools
    )

    # ── MLX branch for tools ──────────────────────────────────
    # MLX has no HTTP endpoint to proxy so we drive the same
    # OpenAI-format pass-through helpers the /v1/chat/completions
    # route uses and convert the emitted SSE to Anthropic's wire
    # format via AnthropicPassthroughEmitter — exactly the
    # translation llama-server's SSE goes through for GGUF clients.
    if using_mlx and (server_tools or client_tools):
        # Convert Anthropic tools to OpenAI format for the backend's
        # apply_chat_template. Studio's built-in tools (web_search /
        # python / terminal) use the OpenAI schema shape unchanged.
        if server_tools:
            from core.inference.tools import ALL_TOOLS as _ALL_TOOLS

            if payload.enabled_tools is not None:
                openai_tools = [
                    t
                    for t in _ALL_TOOLS
                    if t["function"]["name"] in payload.enabled_tools
                ]
            else:
                openai_tools = _ALL_TOOLS
        else:
            openai_tools = anthropic_tools_to_openai(payload.tools)

        # Build an MLX message list that preserves prior tool history
        # (assistant tool_calls, role=tool results) so the model sees
        # the full conversation context.
        mlx_conv_messages = list(openai_messages)

        # Build a synthetic ChatCompletionRequest-compatible payload so
        # we can reuse the OpenAI pass-through helper. Only the fields
        # the helper reads need to be populated.
        _ns = type("NS", (), {})()
        _ns.tools = openai_tools
        _ns.tool_choice = openai_tool_choice
        _ns.temperature = temperature
        _ns.top_p = top_p
        _ns.top_k = top_k
        _ns.min_p = min_p
        _ns.max_tokens = payload.max_tokens
        _ns.presence_penalty = presence_penalty
        _ns.repetition_penalty = repetition_penalty
        _ns.enable_thinking = None
        _ns.stop = stop
        _ns.enable_tools = bool(server_tools)
        _ns.enabled_tools = payload.enabled_tools
        _ns.auto_heal_tool_calls = True
        _ns.max_tool_calls_per_message = 10
        _ns.tool_call_timeout = 300
        _ns.session_id = payload.session_id

        # Generate OpenAI-format frames from MLX, translate to Anthropic
        # as we go. For server_tools (agentic), the route just needs to
        # bridge the agentic events into Anthropic SSE — reuse
        # AnthropicStreamEmitter with the MLX generator.
        if server_tools:

            def _mlx_anthropic_run_gen():
                return mlx_backend.generate_chat_completion_with_tools(
                    messages = mlx_conv_messages,
                    tools = openai_tools,
                    tool_choice = openai_tool_choice,
                    temperature = temperature,
                    top_p = top_p,
                    top_k = top_k,
                    min_p = min_p,
                    max_tokens = payload.max_tokens,
                    repetition_penalty = repetition_penalty,
                    presence_penalty = presence_penalty,
                    stop = stop,
                    cancel_event = cancel_event,
                    enable_thinking = None,
                    max_tool_iterations = 10,
                    auto_heal_tool_calls = True,
                    tool_call_timeout = 300,
                    session_id = payload.session_id,
                )

            if payload.stream:
                return await _anthropic_tool_stream(
                    request,
                    cancel_event,
                    _mlx_anthropic_run_gen,
                    message_id,
                    model_name,
                )
            return await _anthropic_tool_non_streaming(
                _mlx_anthropic_run_gen,
                message_id,
                model_name,
            )

        # client_tools path: drive the MLX passthrough helper and
        # translate its OpenAI-SSE output through the Anthropic
        # passthrough emitter. We capture the emitted data strings
        # and feed each chunk dict into the emitter.
        async def _mlx_to_anthropic_stream():
            emitter = AnthropicPassthroughEmitter()
            for line in emitter.start(message_id, model_name):
                yield line

            try:
                async for raw in _mlx_openai_passthrough_stream(
                    request = request,
                    cancel_event = cancel_event,
                    mlx_backend = mlx_backend,
                    payload = _ns,
                    messages = mlx_conv_messages,
                    stop = stop,
                    completion_id = message_id,
                    created = int(time.time()),
                    model_name = model_name,
                ):
                    # raw is an SSE "data: {...}\n\n" string.
                    line_body = raw.strip()
                    if not line_body.startswith("data: "):
                        continue
                    payload_body = line_body[6:]
                    if payload_body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_body)
                    except json.JSONDecodeError:
                        continue
                    for ev in emitter.feed_chunk(chunk):
                        yield ev
            except Exception as e:
                logger.error(
                    "anthropic_messages MLX passthrough stream error: %s", e,
                    exc_info = True,
                )

            for ev in emitter.finish():
                yield ev

        if payload.stream:
            return StreamingResponse(
                _mlx_to_anthropic_stream(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming MLX client-side tools: run the passthrough
        # non-streaming helper, then convert the OpenAI JSON response
        # into Anthropic Messages format via the same inline logic
        # that _anthropic_passthrough_non_streaming uses for GGUF.
        resp = await _mlx_openai_passthrough_non_streaming(
            mlx_backend = mlx_backend,
            payload = _ns,
            messages = mlx_conv_messages,
            stop = stop,
            completion_id = message_id,
            created = int(time.time()),
            model_name = model_name,
        )
        data = json.loads(resp.body.decode())
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")

        content_blocks = []
        text = message.get("content") or ""
        if text:
            text = _TOOL_XML_RE.sub("", text).strip()
            if text:
                content_blocks.append(AnthropicResponseTextBlock(text = text))

        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            content_blocks.append(
                AnthropicResponseToolUseBlock(
                    id = tc.get("id", ""),
                    name = fn.get("name", ""),
                    input = args,
                )
            )

        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        usage = data.get("usage") or {}
        resp_obj = AnthropicMessagesResponse(
            id = message_id,
            model = model_name,
            content = content_blocks,
            stop_reason = stop_reason,
            usage = AnthropicUsage(
                input_tokens = usage.get("prompt_tokens", 0),
                output_tokens = usage.get("completion_tokens", 0),
            ),
        )
        return JSONResponse(content = resp_obj.model_dump())

    # When MLX is loaded without tools, fall through to the plain path
    # below, but the plain-path helpers assume llama_backend — so branch:
    if using_mlx:

        def _mlx_plain_run_gen():
            return mlx_backend.generate_chat_completion(
                messages = openai_messages,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = payload.max_tokens,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                stop = stop,
                cancel_event = cancel_event,
                enable_thinking = None,
            )

        if payload.stream:
            return await _anthropic_plain_stream(
                request,
                cancel_event,
                _mlx_plain_run_gen,
                message_id,
                model_name,
            )
        return await _anthropic_plain_non_streaming(
            _mlx_plain_run_gen, message_id, model_name
        )

    # ── Client-side pass-through path ─────────────────────────
    if client_tools:
        openai_tools = anthropic_tools_to_openai(payload.tools)

        if payload.stream:
            return await _anthropic_passthrough_stream(
                request,
                cancel_event,
                llama_backend,
                openai_messages,
                openai_tools,
                temperature,
                top_p,
                top_k,
                payload.max_tokens,
                message_id,
                model_name,
                stop = stop,
                min_p = min_p,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                tool_choice = openai_tool_choice,
            )
        return await _anthropic_passthrough_non_streaming(
            llama_backend,
            openai_messages,
            openai_tools,
            temperature,
            top_p,
            top_k,
            payload.max_tokens,
            message_id,
            model_name,
            stop = stop,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            tool_choice = openai_tool_choice,
        )

    if server_tools:
        from core.inference.tools import ALL_TOOLS

        if payload.enabled_tools is not None:
            openai_tools = [
                t for t in ALL_TOOLS if t["function"]["name"] in payload.enabled_tools
            ]
        else:
            openai_tools = ALL_TOOLS

        # Build tool-use system prompt nudge (same logic as /chat/completions)
        _tool_names = {t["function"]["name"] for t in openai_tools}
        _has_web = "web_search" in _tool_names
        _has_code = "python" in _tool_names or "terminal" in _tool_names

        _date_line = f"The current date is {_date.today().isoformat()}."
        _model_size_b = _extract_model_size_b(model_name)
        _is_small_model = _model_size_b is not None and _model_size_b < 9

        if _is_small_model:
            _web_tips = "Do not repeat the same search query."
        else:
            _web_tips = (
                "When you search and find a relevant URL in the results, "
                "fetch its full content by calling web_search with the url parameter. "
                "Do not repeat the same search query. If a search returns "
                "no useful results, try rephrasing or fetching a result URL directly."
            )
        _code_tips = (
            "Use code execution for math, calculations, data processing, "
            "or to parse and analyze information from tool results."
        )

        if _has_web and _has_code:
            _nudge = (
                _date_line + " "
                "You have access to tools. When appropriate, prefer using "
                "tools rather than answering from memory. "
                + _web_tips
                + " "
                + _code_tips
            )
        elif _has_code:
            _nudge = (
                _date_line + " "
                "You have access to tools. When appropriate, prefer using "
                "code execution rather than answering from memory. " + _code_tips
            )
        elif _has_web:
            _nudge = (
                _date_line + " "
                "You have access to tools. When appropriate, prefer using "
                "web search for up-to-date or uncertain factual "
                "information rather than answering from memory. " + _web_tips
            )
        else:
            _nudge = ""

        if _nudge:
            _nudge += _TOOL_ACTION_NUDGE
            # Inject into system prompt
            if openai_messages and openai_messages[0].get("role") == "system":
                openai_messages[0]["content"] = (
                    openai_messages[0]["content"].rstrip() + "\n\n" + _nudge
                )
            else:
                openai_messages.insert(0, {"role": "system", "content": _nudge})

        # Strip stale tool-call XML from conversation
        for _msg in openai_messages:
            if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                _msg["content"] = _TOOL_XML_RE.sub("", _msg["content"]).strip()

        def _run_tool_gen():
            return llama_backend.generate_chat_completion_with_tools(
                messages = openai_messages,
                tools = openai_tools,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                max_tokens = payload.max_tokens,
                stop = stop,
                cancel_event = cancel_event,
                max_tool_iterations = 25,
                auto_heal_tool_calls = True,
                tool_call_timeout = 300,
                session_id = payload.session_id,
            )

        if payload.stream:
            return await _anthropic_tool_stream(
                request,
                cancel_event,
                _run_tool_gen,
                message_id,
                model_name,
            )
        return await _anthropic_tool_non_streaming(
            _run_tool_gen,
            message_id,
            model_name,
        )

    # ── No-tool path ──────────────────────────────────────────
    def _run_plain_gen():
        return llama_backend.generate_chat_completion(
            messages = openai_messages,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            max_tokens = payload.max_tokens,
            stop = stop,
            cancel_event = cancel_event,
        )

    if payload.stream:
        return await _anthropic_plain_stream(
            request,
            cancel_event,
            _run_plain_gen,
            message_id,
            model_name,
        )
    return await _anthropic_plain_non_streaming(
        _run_plain_gen,
        message_id,
        model_name,
    )


async def _anthropic_tool_stream(
    request,
    cancel_event,
    run_gen,
    message_id,
    model_name,
):
    """Streaming response for the tool-calling path."""
    _sentinel = object()

    async def _stream():
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name):
            yield line

        gen = run_gen()
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                event = await asyncio.to_thread(next, gen, _sentinel)
                if event is _sentinel:
                    break
                # Strip leaked tool-call XML from content events
                if event.get("type") == "content":
                    event = dict(event)
                    event["text"] = _TOOL_XML_RE.sub("", event["text"])
                for line in emitter.feed(event):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)

        for line in emitter.finish("end_turn"):
            yield line

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_plain_stream(
    request,
    cancel_event,
    run_gen,
    message_id,
    model_name,
):
    """Streaming response for the no-tool path."""
    _sentinel = object()

    async def _stream():
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name):
            yield line

        gen = run_gen()
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                cumulative = await asyncio.to_thread(next, gen, _sentinel)
                if cumulative is _sentinel:
                    break
                if isinstance(cumulative, dict):
                    if cumulative.get("type") == "metadata":
                        for line in emitter.feed(cumulative):
                            yield line
                    continue
                # Plain generator yields cumulative text strings
                for line in emitter.feed({"type": "content", "text": cumulative}):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)

        for line in emitter.finish("end_turn"):
            yield line

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_tool_non_streaming(run_gen, message_id, model_name):
    """Non-streaming response for the tool-calling path.

    Builds ``content_blocks`` in generation order (text → tool_use → text →
    tool_use → ...), mirroring the streaming emitter's behavior. Deltas
    within a single synthesis turn are merged into the trailing text block;
    tool_use blocks interrupt the text sequence and open a new text block on
    the next content event.

    ``prev_text`` is reset on ``tool_end`` because
    ``generate_chat_completion_with_tools`` yields cumulative content *per
    turn* — the first content event of turn N+1 must diff against an empty
    baseline, not against turn N's final length.
    """
    content_blocks: list = []
    usage = {}
    prev_text = ""

    for event in run_gen():
        etype = event.get("type", "")
        if etype == "content":
            # Strip leaked tool-call XML
            clean = _TOOL_XML_RE.sub("", event["text"])
            new = clean[len(prev_text) :]
            prev_text = clean
            if new:
                if content_blocks and isinstance(
                    content_blocks[-1], AnthropicResponseTextBlock
                ):
                    content_blocks[-1].text += new
                else:
                    content_blocks.append(AnthropicResponseTextBlock(text = new))
        elif etype == "tool_start":
            content_blocks.append(
                AnthropicResponseToolUseBlock(
                    id = event["tool_call_id"],
                    name = event["tool_name"],
                    input = event.get("arguments", {}),
                )
            )
        elif etype == "tool_end":
            prev_text = ""
        elif etype == "metadata":
            usage = event.get("usage", {})

    resp = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = "end_turn",
        usage = AnthropicUsage(
            input_tokens = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
        ),
    )
    return JSONResponse(content = resp.model_dump())


async def _anthropic_plain_non_streaming(run_gen, message_id, model_name):
    """Non-streaming response for the no-tool path."""
    text_parts = []
    usage = {}
    prev_text = ""

    for cumulative in run_gen():
        if isinstance(cumulative, dict):
            if cumulative.get("type") == "metadata":
                usage = cumulative.get("usage", {})
            continue
        new = cumulative[len(prev_text) :]
        prev_text = cumulative
        if new:
            text_parts.append(new)

    full_text = "".join(text_parts)
    content_blocks = []
    if full_text:
        content_blocks.append(AnthropicResponseTextBlock(text = full_text))

    resp = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = "end_turn",
        usage = AnthropicUsage(
            input_tokens = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
        ),
    )
    return JSONResponse(content = resp.model_dump())


# =====================================================================
# Client-side tool pass-through (Anthropic-native tools field)
# =====================================================================


def _build_passthrough_payload(
    openai_messages,
    openai_tools,
    temperature,
    top_p,
    top_k,
    max_tokens,
    stream,
    stop = None,
    min_p = None,
    repetition_penalty = None,
    presence_penalty = None,
    tool_choice = "auto",
):
    body = {
        "messages": openai_messages,
        "tools": openai_tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stop:
        body["stop"] = stop
    if min_p is not None:
        body["min_p"] = min_p
    if repetition_penalty is not None:
        # llama-server's field is "repeat_penalty", not "repetition_penalty"
        body["repeat_penalty"] = repetition_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    return body


async def _anthropic_passthrough_stream(
    request,
    cancel_event,
    llama_backend,
    openai_messages,
    openai_tools,
    temperature,
    top_p,
    top_k,
    max_tokens,
    message_id,
    model_name,
    stop = None,
    min_p = None,
    repetition_penalty = None,
    presence_penalty = None,
    tool_choice = "auto",
):
    """Streaming client-side pass-through: forward tools to llama-server and
    translate its streaming response to Anthropic SSE without executing anything."""
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_passthrough_payload(
        openai_messages,
        openai_tools,
        temperature,
        top_p,
        top_k,
        max_tokens,
        True,
        stop = stop,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        tool_choice = tool_choice,
    )

    async def _stream():
        emitter = AnthropicPassthroughEmitter()
        for line in emitter.start(message_id, model_name):
            yield line

        # Manage the httpx client, response, AND the aiter_lines() async
        # generator MANUALLY — no `async with`, no anonymous iterator.
        #
        # On Python 3.13 + httpcore 1.0.x, `async for raw_line in
        # resp.aiter_lines():` creates an anonymous async generator. When
        # the loop exits via `break` (or the generator is orphaned when a
        # client disconnects mid-stream), Python's `async for` protocol
        # does NOT auto-close the iterator the way a sync `for` loop
        # would. The iterator remains reachable only from the current
        # coroutine frame; once `_stream()` returns, the frame is GC'd
        # and the iterator becomes unreachable. Python's asyncgen
        # finalizer hook then runs its aclose() on a LATER GC pass in a
        # DIFFERENT asyncio task, where httpcore's
        # `HTTP11ConnectionByteStream.aclose()` enters
        # `anyio.CancelScope.__exit__` with a mismatched task and prints
        # `RuntimeError: Attempted to exit cancel scope in a different
        # task` / `RuntimeError: async generator ignored GeneratorExit`
        # as "Exception ignored in:" unraisable warnings.
        #
        # The fix: save `resp.aiter_lines()` as `lines_iter`, and in the
        # finally block explicitly `await lines_iter.aclose()` BEFORE
        # `resp.aclose()` / `client.aclose()`. This closes the iterator
        # inside our own task's event loop, so the internal httpcore
        # byte-stream is cleaned up before Python's asyncgen finalizer
        # has anything orphaned to finalize. Each aclose is wrapped in
        # `try: ... except Exception: pass` so anyio cleanup noise from
        # nested aclose paths can't bubble out.
        client = httpx.AsyncClient(timeout = 600)
        resp = None
        lines_iter = None
        try:
            req = client.build_request("POST", target_url, json = body)
            resp = await client.send(req, stream = True)

            lines_iter = resp.aiter_lines()
            async for raw_line in lines_iter:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for line in emitter.feed_chunk(chunk):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages passthrough stream error: %s", e)
        finally:
            if lines_iter is not None:
                try:
                    await lines_iter.aclose()
                except Exception:
                    pass
            if resp is not None:
                try:
                    await resp.aclose()
                except Exception:
                    pass
            try:
                await client.aclose()
            except Exception:
                pass

        for line in emitter.finish():
            yield line

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_passthrough_non_streaming(
    llama_backend,
    openai_messages,
    openai_tools,
    temperature,
    top_p,
    top_k,
    max_tokens,
    message_id,
    model_name,
    stop = None,
    min_p = None,
    repetition_penalty = None,
    presence_penalty = None,
    tool_choice = "auto",
):
    """Non-streaming client-side pass-through."""
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_passthrough_payload(
        openai_messages,
        openai_tools,
        temperature,
        top_p,
        top_k,
        max_tokens,
        False,
        stop = stop,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        tool_choice = tool_choice,
    )

    async with httpx.AsyncClient() as client:
        resp = await client.post(target_url, json = body, timeout = 600)

    if resp.status_code != 200:
        raise HTTPException(
            status_code = resp.status_code,
            detail = f"llama-server error: {resp.text[:500]}",
        )

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    content_blocks = []
    text = message.get("content") or ""
    if text:
        text = _TOOL_XML_RE.sub("", text).strip()
        if text:
            content_blocks.append(AnthropicResponseTextBlock(text = text))

    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            AnthropicResponseToolUseBlock(
                id = tc.get("id", ""),
                name = fn.get("name", ""),
                input = args,
            )
        )

    if tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage = data.get("usage") or {}
    resp_obj = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = stop_reason,
        usage = AnthropicUsage(
            input_tokens = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
        ),
    )
    return JSONResponse(content = resp_obj.model_dump())


# =====================================================================
# Client-side tool pass-through (OpenAI-native /v1/chat/completions)
# =====================================================================


def _openai_messages_for_passthrough(payload) -> list[dict]:
    """Build OpenAI-format message dicts for the /v1/chat/completions
    passthrough path.

    Messages from ``payload.messages`` are dumped through Pydantic (dropping
    unset optional fields) so they are already in standard OpenAI format
    — including ``role="tool"`` tool-result messages and assistant messages
    that carry structured ``tool_calls``. Content-parts images already in
    the message list are left untouched.

    When a client uses Studio's legacy ``image_base64`` top-level field, the
    image is re-encoded to PNG (llama-server's stb_image has limited format
    support) and spliced into the last user message as an OpenAI
    ``image_url`` content part so vision + function-calling requests work
    transparently.
    """
    messages = [m.model_dump(exclude_none = True) for m in payload.messages]

    if not payload.image_base64:
        return messages

    try:
        import base64 as _b64
        from io import BytesIO as _BytesIO
        from PIL import Image as _Image

        raw = _b64.b64decode(payload.image_base64)
        img = _Image.open(_BytesIO(raw)).convert("RGB")
        buf = _BytesIO()
        img.save(buf, format = "PNG")
        png_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"Failed to process image: {e}",
        )

    data_url = f"data:image/png;base64,{png_b64}"
    image_part = {"type": "image_url", "image_url": {"url": data_url}}

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        existing = msg.get("content")
        if isinstance(existing, str):
            msg["content"] = [{"type": "text", "text": existing}, image_part]
        elif isinstance(existing, list):
            existing.append(image_part)
        else:
            msg["content"] = [image_part]
        break
    else:
        messages.append({"role": "user", "content": [image_part]})

    return messages


def _build_openai_passthrough_body(payload) -> dict:
    """Assemble the llama-server request body from a ChatCompletionRequest.

    Only explicitly-known OpenAI / llama-server fields are forwarded so that
    Studio-specific extensions (``enable_tools``, ``enabled_tools``,
    ``session_id``, ...) never leak to the backend.
    """
    messages = _openai_messages_for_passthrough(payload)
    tool_choice = payload.tool_choice if payload.tool_choice is not None else "auto"
    return _build_passthrough_payload(
        messages,
        payload.tools,
        payload.temperature,
        payload.top_p,
        payload.top_k,
        payload.max_tokens,
        payload.stream,
        stop = payload.stop,
        min_p = payload.min_p,
        repetition_penalty = payload.repetition_penalty,
        presence_penalty = payload.presence_penalty,
        tool_choice = tool_choice,
    )


async def _openai_passthrough_stream(
    request,
    cancel_event,
    llama_backend,
    payload,
    model_name,
    completion_id,
):
    """Streaming client-side pass-through for /v1/chat/completions.

    Forwards the client's OpenAI function-calling request to llama-server and
    relays the SSE stream back verbatim. This preserves llama-server's
    native response ``id``, ``finish_reason`` (including ``"tool_calls"``),
    ``delta.tool_calls``, and the trailing ``usage`` chunk so the client
    observes a standard OpenAI response.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_openai_passthrough_body(payload)

    # Dispatch the upstream request BEFORE returning StreamingResponse so
    # transport errors and non-200 upstream statuses surface as real HTTP
    # errors to the client. OpenAI SDKs rely on status codes to raise
    # ``APIError``/``BadRequestError``/...; burying the failure inside a
    # 200 SSE ``error`` frame silently breaks their error handling.
    client = httpx.AsyncClient(timeout = 600)
    resp = None
    try:
        req = client.build_request("POST", target_url, json = body)
        resp = await client.send(req, stream = True)
    except httpx.RequestError as e:
        # llama-server subprocess crashed / still starting / unreachable.
        logger.error("openai passthrough stream: upstream unreachable: %s", e)
        if resp is not None:
            try:
                await resp.aclose()
            except Exception:
                pass
        try:
            await client.aclose()
        except Exception:
            pass
        raise HTTPException(
            status_code = 502,
            detail = _friendly_error(e),
        )

    if resp.status_code != 200:
        err_bytes = await resp.aread()
        err_text = err_bytes.decode("utf-8", errors = "replace")
        logger.error(
            "openai passthrough upstream error: status=%s body=%s",
            resp.status_code,
            err_text[:500],
        )
        upstream_status = resp.status_code
        try:
            await resp.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass
        raise HTTPException(
            status_code = upstream_status,
            detail = f"llama-server error: {err_text[:500]}",
        )

    async def _stream():
        # Same httpx lifecycle pattern as _anthropic_passthrough_stream:
        # avoid `async with` on the client/response AND explicitly save
        # resp.aiter_lines() so we can close it ourselves in the finally
        # block. See the long comment there for the full rationale on
        # why the anonymous `async for raw_line in resp.aiter_lines():`
        # pattern leaks an unclosed async generator that Python's
        # asyncgen GC hook then finalizes in a different asyncio task,
        # producing "Exception ignored in:" / "async generator ignored
        # GeneratorExit" / anyio cancel-scope traces on Python 3.13 +
        # httpcore 1.0.x.
        lines_iter = None
        try:
            lines_iter = resp.aiter_lines()
            async for raw_line in lines_iter:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue
                # Relay the llama-server SSE chunk verbatim so the client
                # sees its native `id`, `finish_reason`, `delta.tool_calls`,
                # and final `usage` unchanged.
                yield raw_line + "\n\n"
                if raw_line[6:].strip() == "[DONE]":
                    break
        except Exception as e:
            # Mid-stream failures still have to be reported inside the SSE
            # body because the 200 response headers have already been
            # committed by the time the first chunk flushes.
            logger.error("openai passthrough stream error: %s", e)
            err = {
                "error": {
                    "message": _friendly_error(e),
                    "type": "server_error",
                },
            }
            yield f"data: {json.dumps(err)}\n\n"
        finally:
            if lines_iter is not None:
                try:
                    await lines_iter.aclose()
                except Exception:
                    pass
            try:
                await resp.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _openai_passthrough_non_streaming(
    llama_backend,
    payload,
    model_name,
):
    """Non-streaming client-side pass-through for /v1/chat/completions.

    Returns llama-server's JSON response verbatim (via JSONResponse) so the
    client sees the native response ``id``, ``finish_reason`` (including
    ``"tool_calls"``), structured ``tool_calls``, and accurate ``usage``
    token counts.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_openai_passthrough_body(payload)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(target_url, json = body, timeout = 600)
    except httpx.RequestError as e:
        # llama-server subprocess crashed / still starting / unreachable.
        # Surface the same friendly message the sync chat path emits so
        # operators don't see a bare 500 with no diagnostic.
        logger.error("openai passthrough non-streaming: upstream unreachable: %s", e)
        raise HTTPException(
            status_code = 502,
            detail = _friendly_error(e),
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code = resp.status_code,
            detail = f"llama-server error: {resp.text[:500]}",
        )

    # Pass the upstream body through as raw bytes — skips a redundant
    # parse+re-serialize round-trip and keeps the response truly
    # verbatim (matches the docstring). Status is guaranteed 200 by
    # the check above.
    return Response(content = resp.content, media_type = "application/json")


# =====================================================================
# MLX-LM tool-calling helpers (Phase 5 / Chunk C)
# =====================================================================
#
# The GGUF backend proxies to llama-server, which speaks OpenAI's
# function-calling wire format out of the box; the helpers above simply
# relay llama-server's SSE verbatim. MLX is in-process and only hands
# us cumulative text plus agentic-loop events (tool_start / tool_end),
# so we have to *synthesise* the OpenAI SSE frames ourselves. The two
# public entry points here are:
#
#   • _mlx_agentic_stream / _mlx_agentic_non_streaming — used when the
#     request carries Studio's `enable_tools=true` shorthand. Studio
#     executes the built-in tools server-side; we emit a final
#     assistant text plus a `finish_reason: "stop"` completion.
#
#   • _mlx_openai_passthrough_stream / _mlx_openai_passthrough_non_streaming
#     — used when the request carries standard OpenAI `tools=[...]`.
#     The model's tool-call XML is parsed into structured
#     `delta.tool_calls`, streamed to the client, and terminated with
#     `finish_reason: "tool_calls"` so the external client (opencode /
#     Claude Code / Cursor) can execute the tools itself.


async def _mlx_agentic_stream(
    *,
    request,
    cancel_event,
    run_gen,
    completion_id: str,
    created: int,
    model_name: str,
):
    """Stream the output of ``generate_chat_completion_with_tools``
    as OpenAI SSE frames.

    Studio's `enable_tools=true` path runs the tools server-side, so
    from the OpenAI client's perspective the response is a normal
    streaming completion — content deltas plus a
    `finish_reason: "stop"` at the end. The custom ``tool_status`` /
    ``tool_start`` / ``tool_end`` events are surfaced as Studio-specific
    SSE events (mirrors the existing GGUF tool loop route shape so the
    frontend doesn't need to branch).
    """
    _sentinel = object()

    # First chunk: role.
    first_chunk = ChatCompletionChunk(
        id = completion_id,
        created = created,
        model = model_name,
        choices = [
            ChunkChoice(
                delta = ChoiceDelta(role = "assistant"),
                finish_reason = None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

    gen = run_gen()
    prev_text = ""
    _stream_usage = None
    _stream_timings = None

    try:
        while True:
            if await request.is_disconnected():
                cancel_event.set()
                return
            event = await asyncio.to_thread(next, gen, _sentinel)
            if event is _sentinel:
                break

            etype = event.get("type") if isinstance(event, dict) else None

            if etype == "status":
                # Empty status = tool-iteration boundary; reset cursor so
                # the next assistant turn streams cleanly.
                if not event.get("text"):
                    prev_text = ""
                yield f"data: {json.dumps({'type': 'tool_status', 'content': event.get('text', '')})}\n\n"
                continue

            if etype in ("tool_start", "tool_end"):
                if etype == "tool_start":
                    prev_text = ""
                yield f"data: {json.dumps(event)}\n\n"
                continue

            if etype == "metadata":
                _stream_usage = event.get("usage")
                _stream_timings = event.get("timings")
                continue

            if etype == "content":
                raw_cumulative = event.get("text", "")
                new_text = raw_cumulative[len(prev_text) :]
                prev_text = raw_cumulative
                if not new_text:
                    continue
                chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(content = new_text),
                            finish_reason = None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

        # Final chunk with finish_reason=stop.
        final_chunk = ChatCompletionChunk(
            id = completion_id,
            created = created,
            model = model_name,
            choices = [
                ChunkChoice(delta = ChoiceDelta(), finish_reason = "stop"),
            ],
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"

        if _stream_usage or _stream_timings:
            usage_obj = CompletionUsage(
                prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                completion_tokens = (_stream_usage or {}).get("completion_tokens", 0),
                total_tokens = (_stream_usage or {}).get("total_tokens", 0),
            )
            usage_chunk = ChatCompletionChunk(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [],
                usage = usage_obj,
                timings = _stream_timings,
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        cancel_event.set()
        raise
    except Exception as e:
        logger.error("MLX agentic stream error: %s", e, exc_info = True)
        err = {"error": {"message": _friendly_error(e), "type": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"


async def _mlx_agentic_non_streaming(
    *,
    run_gen,
    completion_id: str,
    created: int,
    model_name: str,
):
    """Drive ``generate_chat_completion_with_tools`` to completion and
    return a single ChatCompletion JSON object.
    """
    final_text = ""
    prev_text = ""
    usage: Dict[str, Any] = {}
    for event in run_gen():
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "content":
            prev_text = event.get("text", "")
            final_text = prev_text
        elif etype == "metadata":
            usage = event.get("usage", {}) or {}

    response = ChatCompletion(
        id = completion_id,
        created = created,
        model = model_name,
        choices = [
            CompletionChoice(
                message = CompletionMessage(content = final_text),
                finish_reason = "stop",
            )
        ],
        usage = CompletionUsage(
            prompt_tokens = usage.get("prompt_tokens", 0),
            completion_tokens = usage.get("completion_tokens", 0),
            total_tokens = usage.get("total_tokens", 0),
        ),
    )
    return JSONResponse(content = response.model_dump())


async def _mlx_openai_passthrough_stream(
    *,
    request,
    cancel_event,
    mlx_backend,
    payload,
    messages: list[dict],
    stop: Optional[list[str]],
    completion_id: str,
    created: int,
    model_name: str,
):
    """Client-side tools pass-through: stream the model output, hold
    back tokens that might be forming a tool-call XML block, parse at
    end of turn, and emit structured ``delta.tool_calls`` per OpenAI's
    wire format.

    The buffering strategy:

    - While cumulative text starts with (a prefix of) ``<tool_call>``
      or ``<function=``, we hold back emission. This avoids leaking
      half-formed XML into the client's content stream before we know
      whether it's a real tool call or a false positive.
    - When the buffer grows past :data:`_MLX_PASSTHROUGH_BUFFER_MAX` and
      still matches a prefix, we flush it as plain text — the model is
      probably echoing the markup literally.
    - When the whole turn ends we parse the cumulative text one last
      time; any calls found get streamed out as OpenAI deltas with the
      tool call arguments chunked character-by-character (matching
      llama-server's behaviour and what the official Python SDK
      assembles transparently for client code).
    """
    from core.inference._tool_call_parser import (
        TOOL_XML_SIGNALS,
        parse_tool_calls_from_text,
        strip_tool_markup,
    )

    _MLX_BUFFER_MAX = 64
    _sentinel = object()

    # First chunk: role.
    first_chunk = ChatCompletionChunk(
        id = completion_id,
        created = created,
        model = model_name,
        choices = [
            ChunkChoice(
                delta = ChoiceDelta(role = "assistant"),
                finish_reason = None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

    def _make_gen():
        return mlx_backend.generate_chat_completion(
            messages = messages,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_tokens = payload.max_tokens,
            repetition_penalty = payload.repetition_penalty,
            presence_penalty = payload.presence_penalty,
            stop = stop,
            cancel_event = cancel_event,
            enable_thinking = payload.enable_thinking,
        )

    # For client-side tools we also need the tool schema in the prompt,
    # so call the render path with tools. The easiest way is to go
    # through generate_chat_completion_with_tools with ``tool_choice="none"``
    # semantics for the agentic portion — but we want the parser to
    # still see the tool call. Instead, pre-render with tools and feed
    # generate_chat_completion with the rendered prompt injection by
    # relying on the tokenizer's template. We already have the model
    # loaded and ``supports_tools``, so delegate to the tool-loop
    # method with a custom implementation: run exactly ONE turn, never
    # execute tools, expose the text.
    #
    # Simpler: run generate_chat_completion_with_tools with
    # ``max_tool_iterations=0`` so the tool-detect + execute path is
    # bypassed and only the raw model output comes through. When
    # max_tool_iterations=0 the loop body is skipped entirely, which
    # goes straight to the final "no more tools" fallback — that
    # fallback runs one plain turn. Our path-through constraint is that
    # tools were already passed to the prompt builder on that final
    # turn via the tools argument, so the model is still prompted with
    # the schema. Good.

    def _make_passthrough_gen():
        return mlx_backend.generate_chat_completion_with_tools(
            messages = messages,
            tools = payload.tools or [],
            tool_choice = payload.tool_choice or "auto",
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_tokens = payload.max_tokens,
            repetition_penalty = payload.repetition_penalty,
            presence_penalty = payload.presence_penalty,
            stop = stop,
            cancel_event = cancel_event,
            enable_thinking = payload.enable_thinking,
            # Do NOT execute tools; we're proxying them to the client.
            # max_tool_iterations=0 still emits the single "final turn"
            # with the full tool schema in the rendered prompt.
            max_tool_iterations = 0,
            auto_heal_tool_calls = True,
            tool_call_timeout = 300,
            session_id = None,
        )

    gen = _make_passthrough_gen()
    cumulative = ""
    prev_content_emitted = 0
    content_buffer = ""
    # State machine: "buffering" (might be entering a tool call),
    # "streaming" (flushing content unchanged), "draining" (inside a
    # tool call, hold back emission until end-of-turn).
    state = "buffering"
    usage: Dict[str, Any] = {}

    try:
        while True:
            if await request.is_disconnected():
                cancel_event.set()
                return
            event = await asyncio.to_thread(next, gen, _sentinel)
            if event is _sentinel:
                break

            if isinstance(event, dict):
                etype = event.get("type")
                if etype == "metadata":
                    usage = event.get("usage", {}) or {}
                continue

            if not isinstance(event, str):
                continue

            cumulative = event

            if state == "draining":
                # Hold back — tool-call detected, assemble at end.
                continue

            if state == "streaming":
                new_text = cumulative[prev_content_emitted:]
                if not new_text:
                    continue
                prev_content_emitted = len(cumulative)
                chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(content = new_text),
                            finish_reason = None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"
                continue

            # state == "buffering" — decide what to do with cumulative.
            stripped = cumulative.lstrip()
            is_prefix = False
            is_match = False
            for sig in TOOL_XML_SIGNALS:
                if stripped.startswith(sig):
                    is_match = True
                    break
                if sig.startswith(stripped):
                    is_prefix = True
                    break
            if is_match:
                state = "draining"
            elif is_prefix and len(stripped) < _MLX_BUFFER_MAX:
                # Keep buffering — small chance this is still a tool call.
                continue
            else:
                # Plain content — flush everything accumulated.
                state = "streaming"
                flush_text = cumulative[prev_content_emitted:]
                prev_content_emitted = len(cumulative)
                if flush_text:
                    chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(content = flush_text),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

        # ── End of turn: parse cumulative for tool calls ──
        tool_calls = parse_tool_calls_from_text(cumulative)

        if tool_calls:
            # Build structured tool_calls deltas. Emit one delta per
            # call that carries the id + name, then a second delta per
            # call that streams the arguments in a single chunk. This
            # matches the most common shape OpenAI clients expect —
            # providers sometimes split arguments across multiple
            # deltas but a single chunk is a valid superset.
            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args)
                    except (TypeError, ValueError):
                        args = "{}"

                header = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(
                                tool_calls = [
                                    ToolCallDelta(
                                        index = i,
                                        id = tc.get("id", f"call_{i}"),
                                        type = "function",
                                        function = ToolCallFunctionDelta(
                                            name = name,
                                            arguments = "",
                                        ),
                                    )
                                ]
                            ),
                            finish_reason = None,
                        )
                    ],
                )
                yield f"data: {header.model_dump_json(exclude_none = True)}\n\n"

                if args:
                    args_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(
                                    tool_calls = [
                                        ToolCallDelta(
                                            index = i,
                                            function = ToolCallFunctionDelta(
                                                arguments = args,
                                            ),
                                        )
                                    ]
                                ),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {args_chunk.model_dump_json(exclude_none = True)}\n\n"

            # Final chunk: finish_reason=tool_calls.
            final_chunk = ChatCompletionChunk(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    ChunkChoice(delta = ChoiceDelta(), finish_reason = "tool_calls")
                ],
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
        else:
            # No tool calls — if we were draining (false-positive XML
            # prefix that didn't resolve to a real call), flush the
            # cleaned cumulative as content now.
            if state == "draining":
                cleaned = strip_tool_markup(cumulative, final = True)
                if cleaned:
                    chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(content = cleaned),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"
            # Final chunk: finish_reason=stop.
            final_chunk = ChatCompletionChunk(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    ChunkChoice(delta = ChoiceDelta(), finish_reason = "stop")
                ],
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"

        if usage:
            usage_obj = CompletionUsage(
                prompt_tokens = usage.get("prompt_tokens", 0),
                completion_tokens = usage.get("completion_tokens", 0),
                total_tokens = usage.get("total_tokens", 0),
            )
            usage_chunk = ChatCompletionChunk(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [],
                usage = usage_obj,
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        cancel_event.set()
        raise
    except Exception as e:
        logger.error("MLX passthrough stream error: %s", e, exc_info = True)
        err = {"error": {"message": _friendly_error(e), "type": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"


async def _mlx_openai_passthrough_non_streaming(
    *,
    mlx_backend,
    payload,
    messages: list[dict],
    stop: Optional[list[str]],
    completion_id: str,
    created: int,
    model_name: str,
):
    """Client-side tools pass-through (non-streaming).

    Runs the same one-shot generation as the streaming path, collects
    the cumulative text, parses it for tool calls, and returns a
    single ChatCompletion JSON body. When tool calls are found they
    are attached to ``choices[0].message.tool_calls`` and
    ``finish_reason`` is set to ``"tool_calls"``; otherwise it's a
    standard content-only response.
    """
    from core.inference._tool_call_parser import (
        parse_tool_calls_from_text,
        strip_tool_markup,
    )

    cumulative = ""
    usage: Dict[str, Any] = {}
    for event in mlx_backend.generate_chat_completion_with_tools(
        messages = messages,
        tools = payload.tools or [],
        tool_choice = payload.tool_choice or "auto",
        temperature = payload.temperature,
        top_p = payload.top_p,
        top_k = payload.top_k,
        min_p = payload.min_p,
        max_tokens = payload.max_tokens,
        repetition_penalty = payload.repetition_penalty,
        presence_penalty = payload.presence_penalty,
        stop = stop,
        cancel_event = None,
        enable_thinking = payload.enable_thinking,
        max_tool_iterations = 0,
        auto_heal_tool_calls = True,
        tool_call_timeout = 300,
    ):
        if isinstance(event, dict):
            if event.get("type") == "metadata":
                usage = event.get("usage", {}) or {}
            continue
        cumulative = event

    tool_calls = parse_tool_calls_from_text(cumulative)

    if tool_calls:
        # Build non-streaming CompletionMessage with tool_calls.
        message_body = {
            "role": "assistant",
            "content": strip_tool_markup(cumulative, final = True) or None,
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"
    else:
        message_body = {"role": "assistant", "content": cumulative}
        finish_reason = "stop"

    payload_out = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message_body,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }
    return JSONResponse(content = payload_out)
