# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 9 (Chunk D) — Darwin-gated integration test.

Loads the Qwen3.5-4B-MLX-4bit VLM checkpoint (if present locally),
sends a red-square test image, and asserts the output contains
``"red"`` or ``"square"``. Also tests the peer-unload contract: loading
an MLX text model after a VLM should unload the VLM.

Skipped when:
- Platform is not macOS arm64.
- The model dir doesn't exist locally.
- The mlx_vlm package isn't importable.
"""

from __future__ import annotations

import base64
import io
import os
import platform
from pathlib import Path

import pytest

VLM_PATH_CANDIDATES = [
    "/Users/ent/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit",
    os.environ.get("UNSLOTH_E2E_MLX_VLM_PATH", ""),
]


def _pick_vlm_path() -> str | None:
    for p in VLM_PATH_CANDIDATES:
        if p and Path(p).is_dir() and (Path(p) / "config.json").is_file():
            return p
    return None


def _make_red_square_png_b64(size: int = 256) -> str:
    """Return a base64-encoded PNG of a red square on a white bg."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    m = size // 4
    draw.rectangle([m, m, size - m, size - m], fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Darwin"
        or platform.machine().lower() not in ("arm64", "aarch64"),
        reason = "MLX-VLM requires macOS on Apple Silicon",
    ),
]


@pytest.fixture(scope = "module")
def vlm_path() -> str:
    p = _pick_vlm_path()
    if p is None:
        pytest.skip(
            "Qwen3.5-4B-MLX-4bit not found at any expected local path; "
            "skipping VLM integration test"
        )
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        pytest.skip("mlx_vlm not installed in this env")
    return p


@pytest.fixture(scope = "module")
def loaded_vlm_backend(vlm_path):
    """Load Qwen3.5-4B-VLM once per module; unload at teardown."""
    from core.inference.mlx_vlm import MlxVlmBackend

    b = MlxVlmBackend()
    ok = b.load_model(local_path = vlm_path, model_identifier = vlm_path)
    assert ok, "MlxVlmBackend.load_model returned False"
    try:
        yield b
    finally:
        b.unload_model()


def test_vlm_load_surfaces_properties(loaded_vlm_backend):
    b = loaded_vlm_backend
    assert b.is_loaded is True
    assert b.is_vision is True
    assert b.context_length is not None and b.context_length > 0
    assert b.model_identifier is not None


def test_vlm_describes_red_square(loaded_vlm_backend):
    b = loaded_vlm_backend
    img_b64 = _make_red_square_png_b64()

    cumulative = ""
    metadata = None
    for chunk in b.generate_chat_completion(
        messages = [
            {"role": "user", "content": "Describe this image in one short sentence."}
        ],
        image_b64 = img_b64,
        max_tokens = 96,
        temperature = 0.0,
        top_p = 1.0,
        top_k = 0,
        min_p = 0.0,
    ):
        if isinstance(chunk, dict) and chunk.get("type") == "metadata":
            metadata = chunk
            break
        if isinstance(chunk, str):
            cumulative = chunk

    assert cumulative, "VLM produced no text output"
    lower = cumulative.lower()
    assert (
        "red" in lower or "square" in lower
    ), f"VLM output does not contain 'red' or 'square': {cumulative!r}"
    assert metadata is not None
    assert metadata["usage"]["completion_tokens"] > 0


def test_vlm_text_only_question_also_works(loaded_vlm_backend):
    """A VLM should answer a text-only question too (no image_b64)."""
    b = loaded_vlm_backend
    cumulative = ""
    for chunk in b.generate_chat_completion(
        messages = [{"role": "user", "content": "Say the single word: hello"}],
        image_b64 = None,
        max_tokens = 16,
        temperature = 0.0,
        top_p = 1.0,
        top_k = 0,
        min_p = 0.0,
    ):
        if isinstance(chunk, dict):
            break
        if isinstance(chunk, str):
            cumulative = chunk
    assert cumulative, "VLM text-only call produced no output"


def test_peer_unload_vlm_then_mlx_text(vlm_path, tmp_path):
    """Loading an MLX text model after a VLM must unload the VLM.

    We fake the text-load via the route helper (``_unload_all_mlx_peers``)
    which is the entry point the real /inference/load uses. This
    avoids needing a second real checkpoint on disk.
    """
    from core.inference.mlx_vlm import MlxVlmBackend
    from routes.inference import (
        _unload_all_mlx_peers,
        get_mlx_vlm_backend,
        get_mlx_lm_backend,
    )

    vlm = get_mlx_vlm_backend()
    ok = vlm.load_model(local_path = vlm_path, model_identifier = vlm_path)
    assert ok

    assert vlm.is_loaded is True
    # Simulate "user loads an MLX text model" peer-unload.
    _unload_all_mlx_peers(keep = "mlx")
    assert vlm.is_loaded is False, "VLM peer was not unloaded"
