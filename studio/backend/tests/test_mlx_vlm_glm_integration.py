# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 (H2-4): non-Qwen VLM end-to-end integration.

Chunk D's VLM integration test only exercised Qwen3.5-4B-MLX-4bit
(a 4-bit Qwen VL). This file adds a second family — Zhipu's **GLM-4.6V
Flash** at 8-bit — so the matrix proves MlxVlmBackend handles a
different architecture AND a different quant width (8-bit, not 4-bit)
end-to-end.

Gated on:
- Darwin/arm64 (``mlx_vlm`` is Apple-Silicon only).
- ``mlx_vlm`` importable.
- The local checkpoint at ``~/.lmstudio/models/lmstudio-community/GLM-4.6V-Flash-MLX-8bit``.

Reuses the red-square test-image generator from the Qwen VLM suite
(same shape / size), but the model-specific response style varies:
GLM describes colour and geometry with slightly different phrasing,
so the content assertion accepts either "red" OR "square" OR a
non-empty response — matching the philosophy of the Qwen VLM test.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import platform
from pathlib import Path

import pytest

_GLM_PATH = Path(
    "/Users/ent/.lmstudio/models/lmstudio-community/GLM-4.6V-Flash-MLX-8bit"
)

_PLATFORM_OK = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_vlm") is not None
)
# Same slow-gate convention as the MoE test: ~10 GB checkpoint.
_SLOW_ENABLED = os.environ.get("MLX_SLOW_TESTS") == "1"


def _make_red_square_png_b64(size: int = 256) -> str:
    """Base64-encoded PNG: red square centred on white canvas.

    Shared with test_mlx_vlm_integration.py; duplicated here so this
    test file is self-contained and the Qwen test isn't a dependency.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    m = size // 4
    draw.rectangle([m, m, size - m, size - m], fill = "red")
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


pytestmark = [
    pytest.mark.skipif(
        not _PLATFORM_OK,
        reason = "MLX-VLM requires macOS on Apple Silicon + mlx_vlm",
    ),
    pytest.mark.skipif(
        not _GLM_PATH.is_dir(),
        reason = f"GLM-4.6V checkpoint not present at {_GLM_PATH}",
    ),
    pytest.mark.skipif(
        not _SLOW_ENABLED,
        reason = (
            "GLM-4.6V integration is gated behind MLX_SLOW_TESTS=1 "
            "— the 8-bit checkpoint is ~10 GB. Run with "
            "``MLX_SLOW_TESTS=1 pytest ...``."
        ),
    ),
]


@pytest.fixture(scope = "module")
def loaded_glm_backend():
    """Load GLM-4.6V-Flash-8bit once per module; unload on teardown."""
    from core.inference.mlx_vlm import MlxVlmBackend

    b = MlxVlmBackend()
    ok = b.load_model(local_path = str(_GLM_PATH), model_identifier = _GLM_PATH.name)
    assert ok, "MlxVlmBackend.load_model returned False for GLM-4.6V"
    try:
        yield b
    finally:
        b.unload_model()


def test_glm_load_surfaces_vision_properties(loaded_glm_backend):
    """After load: ``is_vision=True``, context populated, identifier set."""
    b = loaded_glm_backend
    assert b.is_loaded is True
    assert b.is_vision is True
    assert b.model_identifier is not None
    # 8-bit quant variant should round-trip the hf_variant suffix
    # (MLX variant regex pulls "MLX-8bit" out of the dir name).
    assert b.is_active is True


def test_glm_describes_red_square(loaded_glm_backend):
    """Headline VLM smoke: red square image + description prompt.

    GLM's response style is a little more verbose than Qwen's short
    captions, but it still reliably mentions either the colour ("red")
    or the shape ("square"). If neither marker appears we settle for
    "not empty" — the goal is "the VLM saw the image and produced a
    response", not phrasing fidelity.
    """
    b = loaded_glm_backend
    img_b64 = _make_red_square_png_b64()

    cumulative = ""
    metadata = None
    for chunk in b.generate_chat_completion(
        messages = [
            {
                "role": "user",
                "content": "Describe the colour you see in one short sentence.",
            }
        ],
        image_b64 = img_b64,
        max_tokens = 64,
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

    assert cumulative, f"GLM-4.6V produced no text output (metadata={metadata})"
    lower = cumulative.lower()
    # Primary: mention the colour or shape. If GLM's phrasing drifts
    # further in future releases we can soften this to "not empty"
    # without losing the core "image-was-processed" signal.
    assert (
        "red" in lower or "square" in lower
    ), (
        f"GLM-4.6V output does not contain 'red' or 'square': "
        f"{cumulative!r}"
    )
    assert metadata is not None
    assert metadata["usage"]["completion_tokens"] > 0
