# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 (H2-3): MoE end-to-end integration.

Prior MLX chunks exercised dense decoders (Bonsai 8B / 1.7B, Qwen3.5-4B,
Hermes-3, Llama-3.2, Ministral-3). This file closes the MoE-routing
cell in the parity matrix by loading MoE models — starting with
**Qwen3.5-35B-A3B-4bit** (35B-total / 4B-active) and extended on the
Gemma-4 matrix closure pass to **gemma-4-26b-a4b-4bit** — and streaming
a handful of tokens through each.

Goal: prove MoE routing actually works on hardware across distinct
architectures (Qwen3 vs Gemma-4), not that the models generate
high-quality prose. Each row is deliberately tiny (one short prompt,
≤16 new tokens) so wall time is bounded even when the 4-bit shards
cold-load from disk.

Cost caveats:
- Model weights are ~14-17 GB on disk — once resident, they occupy
  unified memory. The M5 / 32 GB machine this chunk targets can hold
  them alongside the process's own overhead but not alongside a peer
  MLX model. Each row unloads explicitly.
- Cold load time depends on page-cache state; at worst ~60 s from
  cold spinning-disk pages, ~6-10 s with warm cache.

Gated on:
- Darwin/arm64 + ``mlx_lm`` importable (platform).
- The local checkpoint at the canonical ``~/.lmstudio`` path (per row).
- ``MLX_SLOW_TESTS=1`` env var — prevents default CI runs from
  paying the 14-17 GB per-row load cost.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path

import pytest

_QWEN_MOE_PATH = Path(
    "/Users/ent/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
)
# Chunk H-2 (Gemma-4 matrix closure): second MoE architecture. Gemma-4
# 26B-a4b is a sparse mixture-of-experts in the Gemma-4 family;
# including it proves MoE routing isn't Qwen-specific. ~14 GB on disk.
_GEMMA_MOE_PATH = Path(
    "/Users/ent/.lmstudio/models/mlx-community/gemma-4-26b-a4b-4bit"
)

_PLATFORM_OK = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_lm") is not None
)
_SLOW_ENABLED = os.environ.get("MLX_SLOW_TESTS") == "1"

_MOE_ROWS = [
    pytest.param(
        "qwen3.5-35b-a3b",
        _QWEN_MOE_PATH,
        id = "qwen3.5-35b-a3b",
    ),
    pytest.param(
        "gemma-4-26b-a4b",
        _GEMMA_MOE_PATH,
        id = "gemma-4-26b-a4b",
    ),
]


@pytest.mark.skipif(
    not _PLATFORM_OK,
    reason = "MLX requires macOS on Apple Silicon + mlx_lm",
)
@pytest.mark.skipif(
    not _SLOW_ENABLED,
    reason = (
        "MoE end-to-end is gated behind MLX_SLOW_TESTS=1 — these tests "
        "load 14-17 GB checkpoints which are too expensive for the "
        "default suite. Run with ``MLX_SLOW_TESTS=1 pytest ...``."
    ),
)
@pytest.mark.parametrize("family,model_dir", _MOE_ROWS)
def test_moe_load_chat_unload(family: str, model_dir: Path):
    """Load an MoE checkpoint, stream a short completion, unload.

    Contract:
    - ``is_loaded`` goes False → True after load.
    - At least one cumulative text chunk is emitted.
    - A terminal metadata event lands with ``completion_tokens >= 1``.
    - ``unload_model`` returns True and the backend resets.

    Output correctness is NOT asserted — the goal is "MoE routing
    produced tokens end-to-end", not "the model is well-behaved on
    a novel prompt." A probabilistic model doing 16 tokens of greedy
    decoding on a trivial prompt is the cheapest signal we can ship
    for "gate routing is wired through mlx_lm correctly".

    Parametrised across every locally-cached MoE in ``_MOE_ROWS``;
    a missing directory skips only that row rather than failing.
    Each row unloads its weights before returning so a subsequent row
    starts from cold state (memory discipline on 32 GB unified).
    """
    if not model_dir.is_dir():
        pytest.skip(
            f"{family}: MoE checkpoint not present at {model_dir} — "
            f"see docs/chunk-h2-matrix/downloads.md for the fetch command"
        )

    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    # Pre-flight: ensure no stale state from prior tests. The
    # auto-peer-unload in the route isn't exercised here because we
    # hit the backend directly.
    assert backend.is_loaded is False

    ok = backend.load_model(
        local_path = str(model_dir),
        model_identifier = model_dir.name,
    )
    assert ok is True, f"{family}: MlxLmBackend.load_model returned False"
    assert backend.is_loaded is True
    assert backend.model_identifier == model_dir.name

    try:
        messages = [
            {"role": "user", "content": "Count: 1 2"},
        ]
        cumulative_last = ""
        saw_metadata = False
        for event in backend.generate_chat_completion(
            messages = messages,
            max_tokens = 16,
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            min_p = 0.0,
        ):
            if isinstance(event, dict):
                saw_metadata = True
                assert event.get("type") == "metadata"
                usage = event.get("usage", {})
                assert usage.get("completion_tokens", 0) >= 1, (
                    f"{family}: no completion tokens emitted — "
                    f"metadata={event}"
                )
            else:
                assert isinstance(event, str)
                assert event.startswith(cumulative_last) or cumulative_last == ""
                cumulative_last = event

        assert saw_metadata, f"{family}: generate never emitted metadata"
        assert len(cumulative_last) > 0, f"{family}: empty cumulative stream"
    finally:
        assert backend.unload_model()

    # Post-unload: fully cold.
    assert backend.is_loaded is False
    assert backend.model_identifier is None
