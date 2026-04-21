# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 9 (Chunk D) — unit tests for ``MlxVlmBackend`` without loading
``mlx-vlm``. Exercises pure-Python helpers (tool/reasoning detection)
and property defaults on an unloaded instance.
"""

from __future__ import annotations

import pytest

from core.inference.mlx_vlm import (
    MlxVlmBackend,
    _detect_reasoning_from_template,
    _detect_tools_from_template,
)


def test_unloaded_backend_reports_not_loaded():
    b = MlxVlmBackend()
    assert b.is_loaded is False
    assert b.is_active is False
    assert b.model_identifier is None
    assert b.context_length is None
    assert b.supports_tools is False
    assert b.supports_reasoning is False
    assert b.is_vision is True
    assert b.is_lora is False


def test_is_vision_is_always_true():
    """VLM backend always advertises is_vision=True so the UI gates
    the image picker correctly regardless of load state."""
    b = MlxVlmBackend()
    assert b.is_vision is True
    # After (hypothetical) unload the flag stays True too.
    b._unload_locked()
    assert b.is_vision is True


def test_detect_tools_from_template_positive():
    tpl = "{% if tool_calls %}{{ tool_calls }}{% endif %}"
    assert _detect_tools_from_template(tpl) is True

    tpl2 = "{% for tool in tools %}{{ tool.name }}{% endfor %}"
    assert _detect_tools_from_template(tpl2) is True


def test_detect_tools_from_template_negative():
    assert _detect_tools_from_template("") is False
    assert _detect_tools_from_template(None) is False
    assert _detect_tools_from_template("hello world") is False


def test_detect_reasoning_supports_thinking():
    tpl = "{% if enable_thinking %}<think></think>{% endif %}"
    supports, always_on, default_on = _detect_reasoning_from_template(tpl)
    assert supports is True
    assert always_on is False  # gate is present
    assert default_on is True


def test_detect_reasoning_always_on():
    tpl = "<think>reasoning</think>user message"
    supports, always_on, default_on = _detect_reasoning_from_template(tpl)
    assert supports is True
    assert always_on is True  # <think> without an enable_thinking gate
    assert default_on is True


def test_detect_reasoning_absent():
    tpl = "{{ messages[0].content }}"
    supports, always_on, default_on = _detect_reasoning_from_template(tpl)
    assert supports is False
    assert always_on is False


def test_decode_image_b64_handles_data_url(tmp_path):
    """The image helper must strip ``data:image/png;base64,`` prefixes."""
    from PIL import Image
    import base64
    import io

    # Make a 4×4 red image.
    img = Image.new("RGB", (4, 4), "red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")

    # With and without a data-URL prefix.
    path1 = MlxVlmBackend._decode_image_b64_to_path(b64, str(tmp_path))
    path2 = MlxVlmBackend._decode_image_b64_to_path(
        "data:image/png;base64," + b64, str(tmp_path)
    )
    # Both must produce readable PNGs at 4×4.
    for p in (path1, path2):
        im = Image.open(p)
        assert im.size == (4, 4)
        assert im.mode == "RGB"


def test_load_progress_shape_on_unloaded_backend():
    b = MlxVlmBackend()
    p = b.load_progress()
    assert "phase" in p and "fraction" in p
    assert p["phase"] is None
    assert p["fraction"] == 0.0
