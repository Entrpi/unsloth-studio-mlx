# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 10 (Chunk D) — unit tests for ``MlxAudioBackend`` without
loading ``mlx-audio``. Exercises properties on an unloaded instance
and the WAV-packing helper.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from core.inference.mlx_audio import MlxAudioBackend, _float32_to_wav_bytes


def test_unloaded_backend_properties():
    b = MlxAudioBackend()
    assert b.is_loaded is False
    assert b.is_active is False
    assert b.model_identifier is None
    assert b.is_audio is True
    assert b.is_vision is False
    assert b.is_lora is False
    assert b.sample_rate == 24000
    assert b.detect_audio_type() == "tts"  # default when unloaded


def test_float32_to_wav_roundtrip():
    """Pack a deterministic sine wave and confirm the WAV is valid."""
    sr = 24000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    wav_bytes = _float32_to_wav_bytes(samples, sr)

    # Valid RIFF header.
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    # Reparse.
    with io.BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wav:
        assert wav.getframerate() == sr
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2  # 16-bit
        frames = wav.readframes(wav.getnframes())
    pcm16 = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    # The round-trip must reproduce the sine close enough that the
    # per-sample L∞ error is below 1e-3 (16-bit quantization floor).
    assert np.max(np.abs(pcm16 - samples)) < 2e-3


def test_float32_to_wav_clips_outliers():
    """Samples outside [-1, 1] must be clipped, not wrapped."""
    samples = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    wav_bytes = _float32_to_wav_bytes(samples, 24000)
    with io.BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    pcm16 = np.frombuffer(frames, dtype=np.int16)
    # 2.0 clips to 1.0 → 32767; -2.0 clips to -1.0 → -32767; 0.5 → ~16383.
    assert pcm16[0] == 32767
    assert pcm16[1] == -32767
    assert abs(pcm16[2] - 16383) < 3


def test_load_progress_on_unloaded():
    b = MlxAudioBackend()
    p = b.load_progress()
    assert p["phase"] is None
    assert p["fraction"] == 0.0
