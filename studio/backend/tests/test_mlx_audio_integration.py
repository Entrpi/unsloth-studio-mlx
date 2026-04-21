# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 10 (Chunk D) — Darwin-gated integration test for ``MlxAudioBackend``.

Loads LFM2.5-Audio-1.5B-bf16 (if present locally), runs a TTS
round-trip, and asserts the WAV has non-trivial duration + energy.
Also exercises the best-effort ASR path.

Skipped when:
- Platform is not macOS arm64.
- The model dir doesn't exist locally.
- The mlx_audio package isn't importable.
"""

from __future__ import annotations

import io
import os
import platform
import wave
from pathlib import Path

import numpy as np
import pytest

AUDIO_PATH_CANDIDATES = [
    "/Users/ent/.lmstudio/models/mlx-community/LFM2.5-Audio-1.5B-bf16",
    os.environ.get("UNSLOTH_E2E_MLX_AUDIO_PATH", ""),
]


def _pick_audio_path() -> str | None:
    for p in AUDIO_PATH_CANDIDATES:
        if p and Path(p).is_dir() and (Path(p) / "config.json").is_file():
            return p
    return None


pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Darwin"
        or platform.machine().lower() not in ("arm64", "aarch64"),
        reason = "MLX-Audio requires macOS on Apple Silicon",
    ),
]


@pytest.fixture(scope = "module")
def audio_path() -> str:
    p = _pick_audio_path()
    if p is None:
        pytest.skip("LFM2.5-Audio-1.5B-bf16 not found locally")
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        pytest.skip("mlx_audio not installed in this env")
    return p


@pytest.fixture(scope = "module")
def loaded_audio_backend(audio_path):
    from core.inference.mlx_audio import MlxAudioBackend

    b = MlxAudioBackend()
    ok = b.load_model(local_path = audio_path, model_identifier = audio_path)
    assert ok, "MlxAudioBackend.load_model returned False"
    try:
        yield b
    finally:
        b.unload_model()


def test_audio_load_properties(loaded_audio_backend):
    b = loaded_audio_backend
    assert b.is_loaded is True
    assert b.sample_rate == 24000
    assert b.is_audio is True
    # LFM2.5 is omni per detection.
    assert b.detect_audio_type() in ("omni", "tts")


def test_tts_roundtrip(loaded_audio_backend):
    """Generate TTS for 'Say hello world' and validate the WAV."""
    b = loaded_audio_backend
    wav_bytes, sr = b.generate_tts(
        "Say hello world.",
        max_new_tokens = 1024,
        temperature = 0.7,
    )
    assert sr == 24000
    assert isinstance(wav_bytes, (bytes, bytearray))
    assert len(wav_bytes) > 1024  # a few hundred ms of audio minimum

    with io.BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        frames = wav.readframes(wav.getnframes())
    data = np.frombuffer(frames, dtype = np.int16).astype(np.float32) / 32767.0
    duration = len(data) / sr
    energy = float(np.mean(np.abs(data)))
    assert duration > 0.25, f"TTS duration too short: {duration:.2f}s"
    assert energy > 0.001, f"TTS energy too low (silent?): {energy:.4f}"


def test_transcribe_returns_string(loaded_audio_backend):
    """Best-effort transcribe contract: returns a string (possibly empty).

    We don't assert the transcription contains the original phrase
    because per PROBE_RESULTS.md, LFM2.5-Audio's self-ASR on its own
    short TTS output is unreliable. What we validate is that the call
    doesn't raise, returns a str, and exercises the audio-in path.
    """
    b = loaded_audio_backend
    # Generate a 1-ish second clip then feed it back.
    wav_bytes, _sr = b.generate_tts(
        "Hello world, this is a test.", max_new_tokens = 2048
    )
    text = b.transcribe(wav_bytes, prompt = "Repeat what you hear.", max_new_tokens = 64)
    assert isinstance(text, str)
    # No content assertion — see PROBE_RESULTS.md.
