# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 (H2-3): MoE end-to-end integration.

Prior MLX chunks exercised dense decoders (Bonsai 8B / 1.7B, Qwen3.5-4B,
Hermes-3, Llama-3.2, Ministral-3). This file closes the MoE-routing
cell in the parity matrix by loading **Qwen3.5-35B-A3B-4bit** — a
35B-total / 4B-active mixture-of-experts model — and streaming a
handful of tokens through it.

Goal: prove MoE routing actually works on hardware, not that the
model generates high-quality prose. The test is deliberately tiny
(one short prompt, ≤16 new tokens) so wall time is bounded even
when the 4-bit shards cold-load from disk.

Cost caveats:
- The model weights are ~17 GB on disk — once resident, they occupy
  unified memory. The M5 / 32 GB machine this chunk targets can hold
  them alongside the process's own overhead but not alongside a peer
  MLX model. The test unloads explicitly.
- Cold load time depends on page-cache state; at worst ~60 s from
  cold spinning-disk pages, ~6-10 s with warm cache.

Gated on:
- Darwin/arm64 + ``mlx_lm`` importable (platform).
- The local checkpoint at the canonical ``~/.lmstudio`` path.
- ``MLX_SLOW_TESTS=1`` env var — prevents default CI runs from
  paying the ~17 GB load cost.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path

import pytest

_MOE_PATH = Path(
    "/Users/ent/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit"
)

_PLATFORM_OK = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_lm") is not None
)
_SLOW_ENABLED = os.environ.get("MLX_SLOW_TESTS") == "1"


@pytest.mark.skipif(
    not _PLATFORM_OK,
    reason = "MLX requires macOS on Apple Silicon + mlx_lm",
)
@pytest.mark.skipif(
    not _SLOW_ENABLED,
    reason = (
        "MoE end-to-end is gated behind MLX_SLOW_TESTS=1 — this test "
        "loads a ~17 GB checkpoint which is too expensive for the "
        "default suite. Run with ``MLX_SLOW_TESTS=1 pytest ...``."
    ),
)
@pytest.mark.skipif(
    not _MOE_PATH.is_dir(),
    reason = f"MoE checkpoint not present at {_MOE_PATH}",
)
def test_qwen35_moe_load_chat_unload():
    """Load Qwen3.5-35B-A3B-4bit, stream a short completion, unload.

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
    """
    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    # Pre-flight: ensure no stale state from prior tests. The
    # auto-peer-unload in the route isn't exercised here because we
    # hit the backend directly.
    assert backend.is_loaded is False

    ok = backend.load_model(
        local_path = str(_MOE_PATH),
        model_identifier = _MOE_PATH.name,
    )
    assert ok is True, "MlxLmBackend.load_model returned False for MoE"
    assert backend.is_loaded is True
    assert backend.model_identifier == _MOE_PATH.name

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
                    f"MoE: no completion tokens emitted — metadata={event}"
                )
            else:
                assert isinstance(event, str)
                assert event.startswith(cumulative_last) or cumulative_last == ""
                cumulative_last = event

        assert saw_metadata, "MoE: generate never emitted metadata"
        assert len(cumulative_last) > 0, "MoE: empty cumulative stream"
    finally:
        assert backend.unload_model()

    # Post-unload: fully cold.
    assert backend.is_loaded is False
    assert backend.model_identifier is None
