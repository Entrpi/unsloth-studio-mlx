# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 (H2-2): per-family text-chat smoke tests.

Before this chunk, the MLX parity matrix only had real-hardware chat
coverage against Qwen3-derived bases (Bonsai 8B / 1.7B, Qwen3.5-4B
text, Qwen3.5-4B VLM). This file extends smoke-level chat coverage
to three non-Qwen families that were template-probed but never
actually loaded + streamed tokens against:

- Hermes-3 Llama-3.2 3B (bf16) — Nous Research fine-tune.
- Ministral-3 3B Instruct (4-bit) — Mistral 2512.
- Llama-3.2 3B Instruct (4-bit) — Meta baseline.
- Gemma-4 E4B Instruct (4-bit) — Google Gemma-4 dense. Added after
  the initial H-2 landing to close the last Gemma-4 cell that only
  had a tokenizer probe against 31B and zero weights-loaded coverage.

For each: load → 1-token-chat → assert cumulative text contract +
metadata event → unload. These are gate tests: they prove the MLX
backend actually drives these bases on hardware, not just that their
configs parse or their templates render.

Gated on each model directory existing locally; skipped cleanly on
CI without the cache. Hardware gate: Darwin/arm64 + ``mlx_lm``
importable — inherited from the other MLX integration suites.
"""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import pytest

_LMSTUDIO_ROOT = Path("/Users/ent/.lmstudio/models")

_PLATFORM_OK = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_lm") is not None
)

# Three new candidates. Each is a (family_id, local_path) pair. The
# parametrised test skips cleanly when any one path is missing rather
# than failing — some CI machines won't have the full download set.
_FAMILIES = [
    pytest.param(
        "hermes-3-3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Hermes-3-Llama-3.2-3B-bf16",
        id = "hermes-3-3b",
    ),
    pytest.param(
        "ministral-3-3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Ministral-3-3B-Instruct-2512-4bit",
        id = "ministral-3-3b",
    ),
    pytest.param(
        "llama-3.2-3b",
        _LMSTUDIO_ROOT / "mlx-community" / "Llama-3.2-3B-Instruct-4bit",
        id = "llama-3.2-3b",
    ),
    # Chunk H-2 (Gemma-4 matrix closure): Gemma-4 E4B dense. The 31B
    # sibling had template-probe coverage but never loaded weights;
    # E4B at ~4.9 GB is small enough to ship in the default fast suite
    # so the "Gemma-4 actually drives on hardware" claim is real rather
    # than inferred. Uses the same Gemma-4 chat template family as 31B.
    pytest.param(
        "gemma-4-e4b",
        _LMSTUDIO_ROOT / "mlx-community" / "gemma-4-e4b-it-4bit",
        id = "gemma-4-e4b",
    ),
]


@pytest.mark.skipif(not _PLATFORM_OK, reason = "mlx_lm not available")
@pytest.mark.parametrize("family,model_dir", _FAMILIES)
def test_load_chat_unload_family(family: str, model_dir: Path):
    """End-to-end smoke: load, stream ≥1 cumulative token chunk,
    terminate with a metadata event, unload, confirm state resets.

    Parametrised per family so a missing local checkpoint skips only
    that one row, not the whole file.
    """
    if not model_dir.is_dir():
        pytest.skip(
            f"{family}: model dir not present at {model_dir} — "
            f"see docs/chunk-h2-matrix/downloads.md for the fetch command"
        )

    from core.inference.mlx_lm import MlxLmBackend

    backend = MlxLmBackend()
    assert backend.is_loaded is False

    ok = backend.load_model(
        local_path = str(model_dir),
        model_identifier = model_dir.name,
    )
    assert ok is True, f"{family}: MlxLmBackend.load_model returned False"
    assert backend.is_loaded is True
    assert backend.model_identifier == model_dir.name
    # Every family must advertise a context length from config.
    # Chunk H-2 (B2): Ministral-3 is ``Mistral3ForConditionalGeneration``
    # with ``max_position_embeddings`` nested inside ``text_config``;
    # the MLX backend's config reader now descends into that nested
    # block so the surfaced property is a real integer (32768) rather
    # than None. Qwen / Llama / Hermes populate it from the top-level
    # key directly.
    assert backend.context_length is not None, (
        f"{family}: context_length is None — the backend's config "
        f"reader failed to extract max_position_embeddings from "
        f"config.json (top-level or text_config fallback)"
    )
    assert backend.context_length > 0

    try:
        messages = [
            {
                "role": "user",
                "content": "Reply with exactly one word: hi",
            }
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
                # ≥1 completion token is the core "it generated"
                # assertion. The output is probabilistic so we don't
                # pin the text value.
                assert usage.get("completion_tokens", 0) >= 1, (
                    f"{family}: no completion tokens in metadata {event}"
                )
                assert usage.get("prompt_tokens", 0) >= 1
            else:
                assert isinstance(event, str)
                # Cumulative text contract: each yield is a prefix of
                # the next yield (monotonic append-only stream).
                assert event.startswith(cumulative_last) or cumulative_last == ""
                cumulative_last = event

        assert saw_metadata, (
            f"{family}: generate_chat_completion never emitted a "
            f"terminal metadata event"
        )
        assert len(cumulative_last) > 0, (
            f"{family}: cumulative text stream was empty"
        )
    finally:
        assert backend.unload_model()

    # Post-unload: backend resets to cold-start. This is the contract
    # the route's peer-unload logic depends on so a user swapping
    # between families doesn't leak state across loads.
    assert backend.is_loaded is False
    assert backend.model_identifier is None
