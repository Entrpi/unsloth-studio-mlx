# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for MLX-VLM (Phase 9) and MLX-Audio (Phase 10) detection.

Exercise :func:`_detect_mlx_vlm_model` / :func:`_detect_mlx_audio_model`
and :meth:`ModelConfig.from_identifier` routing so a Qwen3.5-VL-style
directory classifies as ``is_mlx_vlm`` and an LFM2.5-Audio-style
directory classifies as ``is_mlx_audio``, without importing
``mlx_vlm`` or ``mlx_audio``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.models.model_config import (
    ModelConfig,
    _detect_mlx_audio_model,
    _detect_mlx_model,
    _detect_mlx_vlm_model,
)


def _write_json(p: Path, payload) -> None:
    p.parent.mkdir(parents = True, exist_ok = True)
    p.write_text(json.dumps(payload))


# ── VLM detection ─────────────────────────────────────────────
def test_detect_vlm_via_preprocessor_config(tmp_path: Path) -> None:
    """A dir shipping preprocessor_config.json + config.json is VLM."""
    _write_json(
        tmp_path / "config.json",
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"},
    )
    _write_json(tmp_path / "preprocessor_config.json", {"image_processor_type": "Qwen2VLImageProcessor"})
    assert _detect_mlx_vlm_model(tmp_path) is True


def test_detect_vlm_via_processor_config(tmp_path: Path) -> None:
    """A dir shipping processor_config.json is also VLM."""
    _write_json(
        tmp_path / "config.json",
        {"architectures": ["Llava"], "model_type": "llava"},
    )
    _write_json(tmp_path / "processor_config.json", {"processor_class": "LlavaProcessor"})
    assert _detect_mlx_vlm_model(tmp_path) is True


def test_detect_vlm_via_architecture_allowlist(tmp_path: Path) -> None:
    """A dir with ConditionalGeneration arch + known model_type is VLM."""
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
        },
    )
    # No preprocessor_config.json — fallback path.
    assert _detect_mlx_vlm_model(tmp_path) is True


def test_detect_vlm_rejects_unknown_model_type(tmp_path: Path) -> None:
    """Unknown model_type without preprocessor_config.json is not VLM."""
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["SomethingForConditionalGeneration"],
            "model_type": "made_up_model",
        },
    )
    assert _detect_mlx_vlm_model(tmp_path) is False


def test_detect_vlm_rejects_plain_text_mlx(tmp_path: Path) -> None:
    """A plain MLX text checkpoint (quantization block only) is not VLM."""
    _write_json(
        tmp_path / "config.json",
        {"quantization": {"bits": 4, "group_size": 64}, "model_type": "qwen3"},
    )
    assert _detect_mlx_vlm_model(tmp_path) is False


def test_detect_vlm_missing_config(tmp_path: Path) -> None:
    """No config.json → not VLM."""
    assert _detect_mlx_vlm_model(tmp_path) is False


def test_detect_vlm_empty_dir(tmp_path: Path) -> None:
    """Empty dir → not VLM."""
    (tmp_path / "subdir").mkdir()
    assert _detect_mlx_vlm_model(tmp_path / "subdir") is False


# ── Audio detection ───────────────────────────────────────────
def test_detect_audio_via_architecture(tmp_path: Path) -> None:
    """A dir whose first arch ends AudioForConditionalGeneration is audio."""
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Lfm2AudioForConditionalGeneration"],
            "model_type": "lfm_audio",
        },
    )
    assert _detect_mlx_audio_model(tmp_path) is True


def test_detect_audio_via_model_type_allowlist(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"architectures": ["SomethingElse"], "model_type": "lfm_audio"},
    )
    assert _detect_mlx_audio_model(tmp_path) is True


def test_detect_audio_rejects_plain_text(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
    )
    assert _detect_mlx_audio_model(tmp_path) is False


# ── ModelConfig.from_identifier routing ───────────────────────
def test_from_identifier_routes_vlm(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "quantization": {"bits": 4, "group_size": 64},
            "max_position_embeddings": 32768,
        },
    )
    _write_json(tmp_path / "preprocessor_config.json", {"image_processor_type": "Qwen2VLImageProcessor"})

    cfg = ModelConfig.from_identifier(str(tmp_path))
    assert cfg is not None
    assert cfg.is_mlx_vlm is True
    assert cfg.is_mlx is False  # VLM wins over base MLX
    assert cfg.is_vision is True
    assert cfg.mlx_vlm_path == str(tmp_path)
    assert cfg.native_context_length == 32768


def test_from_identifier_routes_audio(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Lfm2AudioForConditionalGeneration"],
            "model_type": "lfm_audio",
        },
    )
    cfg = ModelConfig.from_identifier(str(tmp_path))
    assert cfg is not None
    assert cfg.is_mlx_audio is True
    assert cfg.is_mlx is False
    assert cfg.is_mlx_vlm is False
    assert cfg.is_audio is True
    assert cfg.mlx_audio_path == str(tmp_path)


def test_from_identifier_routes_plain_mlx_when_no_vlm_signal(tmp_path: Path) -> None:
    """A quant-only checkpoint still routes as ``is_mlx`` (base)."""
    _write_json(
        tmp_path / "config.json",
        {
            "quantization": {"bits": 2, "group_size": 64},
            "max_position_embeddings": 4096,
        },
    )
    cfg = ModelConfig.from_identifier(str(tmp_path))
    assert cfg is not None
    assert cfg.is_mlx is True
    assert cfg.is_mlx_vlm is False
    assert cfg.is_mlx_audio is False
