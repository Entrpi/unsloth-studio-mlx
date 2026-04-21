# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 (H2-5): real mlx-whisper end-to-end.

Chunk E-9 added ``MlxAudioBackend.transcribe_with_whisper`` but only
mock-tested it (``test_mlx_audio_backend_unit.py`` monkeypatches
``mlx_whisper.transcribe`` wholesale). This file loads the dedicated
4-bit Whisper-small MLX checkpoint and actually transcribes a short
audio clip through it.

Fixture choice: we bundle a ~32 KB pre-recorded WAV of the phrase
"Hello world." under ``tests/fixtures/audio/hello_world_16khz.wav``
(generated locally via macOS ``say``). The LFM2.5-Audio TTS → Whisper
round-trip was probed first but produced unreliable transcriptions
("Bye." / "E.B." / "I'll call you back!" on successive runs for the
prompt "Hello world") — per PROBE_RESULTS.md, LFM2.5-Audio's TTS is
not intelligibility-focused. The ``say``-sourced fixture is fully
deterministic (bit-exact across runs on the same macOS version) and
unambiguous — Whisper-small transcribes it to " Hello world." with
no jitter.

Gated on:
- Darwin/arm64 + ``mlx_whisper`` importable.
- The local Whisper-small checkpoint at
  ``~/.lmstudio/models/mlx-community/whisper-small-mlx-4bit``.
- The bundled WAV fixture present in the tree.
"""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import pytest

_WHISPER_PATH = Path(
    "/Users/ent/.lmstudio/models/mlx-community/whisper-small-mlx-4bit"
)
_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "audio" / "hello_world_16khz.wav"
)

_PLATFORM_OK = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_whisper") is not None
)


@pytest.mark.skipif(
    not _PLATFORM_OK,
    reason = "mlx_whisper requires macOS on Apple Silicon",
)
@pytest.mark.skipif(
    not _WHISPER_PATH.is_dir(),
    reason = f"whisper-small-mlx-4bit not present at {_WHISPER_PATH}",
)
@pytest.mark.skipif(
    not _FIXTURE_PATH.is_file(),
    reason = f"audio fixture not present at {_FIXTURE_PATH}",
)
def test_transcribe_with_whisper_real_model_recovers_hello_world():
    """End-to-end: MlxAudioBackend.transcribe_with_whisper against the
    4-bit Whisper-small checkpoint returns a string containing "hello"
    or "world" (case-insensitive).

    The fixture was generated with macOS ``say --data-format=LEI16@16000``
    so the exact PCM bytes are deterministic across runs on the same OS
    build. Whisper-small transcribes it to ``" Hello world."`` reliably
    — the assertion stays loose to tolerate minor decoder drift across
    Whisper versions without losing the "ASR actually ran" signal.
    """
    from core.inference.mlx_audio import MlxAudioBackend

    audio_bytes = _FIXTURE_PATH.read_bytes()
    assert len(audio_bytes) > 1000, (
        f"Fixture at {_FIXTURE_PATH} is suspiciously small "
        f"({len(audio_bytes)} bytes) — regenerate via "
        f"``say --data-format=LEI16@16000 -o ... 'Hello world.'``"
    )

    # The backend doesn't need to be "loaded" — transcribe_with_whisper
    # is a side channel: it loads mlx-whisper lazily from the model
    # path (or HF repo id) passed in via ``model_hint``. This matches
    # the E9 design: LFM2.5-Audio and mlx-whisper are independent
    # residents, a caller can hit one without materialising the other.
    b = MlxAudioBackend()
    text = b.transcribe_with_whisper(
        audio_bytes,
        model_hint = str(_WHISPER_PATH),
    )
    assert isinstance(text, str)
    lower = text.lower().strip()
    assert "hello" in lower or "world" in lower, (
        f"Whisper-small did not recover 'hello' or 'world' from the "
        f"fixture — got {text!r}"
    )


@pytest.mark.skipif(
    not _PLATFORM_OK,
    reason = "mlx_whisper requires macOS on Apple Silicon",
)
@pytest.mark.skipif(
    not _WHISPER_PATH.is_dir(),
    reason = f"whisper-small-mlx-4bit not present at {_WHISPER_PATH}",
)
@pytest.mark.skipif(
    not _FIXTURE_PATH.is_file(),
    reason = f"audio fixture not present at {_FIXTURE_PATH}",
)
def test_transcribe_with_whisper_returns_stripped_text():
    """Contract: the returned string is stripped of leading/trailing
    whitespace. Whisper emits ``" Hello world."`` with a leading space;
    the backend's ``return text.strip()`` pass must remove it.
    """
    from core.inference.mlx_audio import MlxAudioBackend

    audio_bytes = _FIXTURE_PATH.read_bytes()
    b = MlxAudioBackend()
    text = b.transcribe_with_whisper(
        audio_bytes,
        model_hint = str(_WHISPER_PATH),
    )
    # Strip contract — no leading whitespace even if Whisper emits one.
    assert text == text.strip()
