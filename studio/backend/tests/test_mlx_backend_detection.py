# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for MLX checkpoint detection in ``ModelConfig``.

These tests do not import ``mlx_lm``; they exercise only the on-disk
detection heuristic in ``utils/models/model_config.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.models.model_config import (
    ModelConfig,
    _detect_mlx_adapter,
    _detect_mlx_model,
    _read_mlx_max_position_embeddings,
)


def _write_json(p: Path, payload) -> None:
    p.parent.mkdir(parents = True, exist_ok = True)
    p.write_text(json.dumps(payload))


def test_detect_mlx_with_quantization_block(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {
            "quantization": {"group_size": 128, "bits": 2},
            "max_position_embeddings": 8192,
        },
    )
    assert _detect_mlx_model(tmp_path) is True

    cfg = ModelConfig.from_identifier(str(tmp_path))
    assert cfg is not None
    assert cfg.is_mlx is True
    assert cfg.is_gguf is False
    assert cfg.mlx_path == str(tmp_path)
    assert cfg.native_context_length == 8192


def test_detect_mlx_missing_group_size(tmp_path: Path) -> None:
    _write_json(tmp_path / "config.json", {"quantization": {"bits": 2}})
    assert _detect_mlx_model(tmp_path) is False

    cfg = ModelConfig.from_identifier(str(tmp_path))
    # Without the MLX shape the fallthrough goes into the transformers path
    # which probes HF. We only assert the MLX flag is not set; cfg may be
    # None if the transformers branch can't resolve the name as a repo.
    if cfg is not None:
        assert cfg.is_mlx is False


def test_detect_mlx_with_gguf_present(tmp_path: Path) -> None:
    """GGUF wins when both formats coexist in the same directory."""
    _write_json(
        tmp_path / "config.json",
        {"quantization": {"group_size": 128, "bits": 2}},
    )
    (tmp_path / "model.gguf").write_bytes(b"GGUF\x00\x00\x00\x00")

    cfg = ModelConfig.from_identifier(str(tmp_path))
    assert cfg is not None
    assert cfg.is_gguf is True
    assert cfg.is_mlx is False


def test_detect_mlx_with_bnb_config(tmp_path: Path) -> None:
    """bnb-4bit checkpoints use ``quantization_config`` — no collision."""
    _write_json(
        tmp_path / "config.json",
        {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
            }
        },
    )
    assert _detect_mlx_model(tmp_path) is False


def test_detect_mlx_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{ not valid json")
    # Must not raise
    assert _detect_mlx_model(tmp_path) is False


def test_detect_mlx_missing_config(tmp_path: Path) -> None:
    assert _detect_mlx_model(tmp_path) is False


def test_detect_mlx_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("not a dir")
    assert _detect_mlx_model(f) is False


def test_read_max_position_embeddings_happy(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config.json",
        {"max_position_embeddings": 4096},
    )
    assert _read_mlx_max_position_embeddings(tmp_path) == 4096


def test_read_max_position_embeddings_missing(tmp_path: Path) -> None:
    _write_json(tmp_path / "config.json", {})
    assert _read_mlx_max_position_embeddings(tmp_path) is None


def test_read_max_position_embeddings_malformed(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{bad")
    assert _read_mlx_max_position_embeddings(tmp_path) is None


# ── Phase 6 — MLX LoRA adapter detection ─────────────────────────────


def test_detect_mlx_adapter_happy_path(tmp_path: Path) -> None:
    """A dir with both adapters.safetensors + adapter_config.json is an
    MLX LoRA adapter."""
    (tmp_path / "adapters.safetensors").write_bytes(b"\x00\x00")
    _write_json(tmp_path / "adapter_config.json", {"peft_type": "LORA"})
    assert _detect_mlx_adapter(tmp_path) is True


def test_detect_mlx_adapter_missing_weights(tmp_path: Path) -> None:
    """Without the weights file the detector must return False."""
    _write_json(tmp_path / "adapter_config.json", {"peft_type": "LORA"})
    assert _detect_mlx_adapter(tmp_path) is False


def test_detect_mlx_adapter_missing_config(tmp_path: Path) -> None:
    """Without the adapter_config.json the detector must return False."""
    (tmp_path / "adapters.safetensors").write_bytes(b"\x00\x00")
    assert _detect_mlx_adapter(tmp_path) is False


def test_detect_mlx_adapter_hf_peft_is_not_mlx(tmp_path: Path) -> None:
    """HuggingFace PEFT adapters write adapter_model.safetensors (singular,
    different stem) — not adapters.safetensors (plural). The MLX adapter
    detector must NOT misclassify an HF PEFT adapter as MLX."""
    (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00\x00")
    _write_json(tmp_path / "adapter_config.json", {"peft_type": "LORA"})
    assert _detect_mlx_adapter(tmp_path) is False


def test_detect_mlx_adapter_base_mlx_is_not_adapter(tmp_path: Path) -> None:
    """A base MLX model dir has config.json + a quantization block but no
    adapters.safetensors. It must not register as an adapter."""
    _write_json(
        tmp_path / "config.json",
        {"quantization": {"bits": 2, "group_size": 128}},
    )
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x00")
    assert _detect_mlx_adapter(tmp_path) is False
    # The base detector still fires:
    assert _detect_mlx_model(tmp_path) is True


def test_detect_mlx_adapter_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "notdir"
    f.write_text("not a dir")
    assert _detect_mlx_adapter(f) is False


# ── Phase 3 — remote HF MLX detection ───────────────────────────────


def test_remote_hf_mlx_detected_via_probed_config(tmp_path: Path) -> None:
    """When the remote HF probe returns an MLX-shaped config.json, the
    resulting ModelConfig must carry ``is_mlx=True`` / ``mlx_path=None``
    (download deferred to the backend)."""
    from unittest import mock

    # Stage a fake config.json on disk that the patched hf_hub_download
    # returns the path to. Mirrors the real HF cache shape.
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "quantization": {"bits": 4, "group_size": 64},
                "max_position_embeddings": 32768,
            }
        )
    )

    def _fake_hf_hub_download(repo_id, filename, *args, **kwargs):
        assert filename == "config.json"
        return str(cfg)

    # Also stub detect_gguf_model_remote so we don't touch the network
    # in the GGUF branch.
    with mock.patch(
        "utils.models.model_config.detect_gguf_model_remote",
        return_value = None,
    ):
        with mock.patch(
            "huggingface_hub.hf_hub_download", _fake_hf_hub_download
        ):
            model_cfg = ModelConfig.from_identifier(
                "mlx-community/Fake-Model-4bit"
            )

    assert model_cfg is not None
    assert model_cfg.is_mlx is True
    assert model_cfg.mlx_path is None  # download deferred
    assert model_cfg.path == "mlx-community/Fake-Model-4bit"
    assert model_cfg.native_context_length == 32768
    assert model_cfg.is_local is False


def test_remote_hf_non_mlx_falls_through(tmp_path: Path) -> None:
    """A remote repo whose config.json lacks the MLX quantization block
    must NOT be misclassified as MLX."""
    from unittest import mock

    cfg = tmp_path / "config.json"
    # Standard transformers config — no quantization block.
    cfg.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 4096,
            }
        )
    )

    def _fake_hf_hub_download(repo_id, filename, *args, **kwargs):
        return str(cfg)

    with mock.patch(
        "utils.models.model_config.detect_gguf_model_remote",
        return_value = None,
    ):
        with mock.patch(
            "huggingface_hub.hf_hub_download", _fake_hf_hub_download
        ):
            # The function will proceed into the transformers / Unsloth
            # fallback, which may or may not return something depending
            # on environment. We just assert MLX is not claimed.
            try:
                model_cfg = ModelConfig.from_identifier(
                    "some/plain-transformers-repo"
                )
            except Exception:
                model_cfg = None
    if model_cfg is not None:
        assert model_cfg.is_mlx is False
