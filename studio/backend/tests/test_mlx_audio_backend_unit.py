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


def test_transcribe_with_whisper_raises_when_package_missing(monkeypatch):
    """E9: whisper transcription surfaces a clear error when mlx-whisper
    isn't installed. The route handler translates this to HTTP 503.
    """
    import sys

    # Force the lazy import inside the method to fail.
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    b = MlxAudioBackend()
    with pytest.raises(RuntimeError, match = "mlx-whisper is not installed"):
        b.transcribe_with_whisper(b"RIFF" + b"\x00" * 100)


def test_transcribe_with_whisper_dispatches_to_mlx_whisper(monkeypatch, tmp_path):
    """E9: backend method writes bytes to a tempfile and forwards to
    mlx_whisper.transcribe, returning the 'text' field. No backend
    state (model/processor) is needed.
    """
    import sys
    import types

    captured: dict = {}

    def fake_transcribe(audio, *, path_or_hf_repo, **_kwargs):
        captured["audio_path"] = audio
        captured["repo"] = path_or_hf_repo
        return {"text": "  hello world  ", "segments": [], "language": "en"}

    fake_module = types.ModuleType("mlx_whisper")
    fake_module.transcribe = fake_transcribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)

    b = MlxAudioBackend()
    # Default model-hint resolution uses MLX_WHISPER_MODEL env var then
    # falls back to mlx-community/whisper-tiny.
    monkeypatch.delenv("MLX_WHISPER_MODEL", raising = False)
    text = b.transcribe_with_whisper(b"RIFF\x00\x00\x00\x24WAVEfmt ")
    assert text == "hello world"  # stripped
    assert captured["repo"] == "mlx-community/whisper-tiny"

    # Explicit model_hint overrides the default.
    text = b.transcribe_with_whisper(b"ID3\x03", model_hint = "mlx-community/whisper-small")
    assert captured["repo"] == "mlx-community/whisper-small"

    # MLX_WHISPER_MODEL env var takes precedence over the fallback.
    monkeypatch.setenv("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3")
    b.transcribe_with_whisper(b"RIFF\x00")
    assert captured["repo"] == "mlx-community/whisper-large-v3"


def test_transcribe_with_whisper_handles_non_dict_result(monkeypatch):
    """E9: degraded return (e.g. future API change) surfaces as empty string."""
    import sys
    import types

    fake_module = types.ModuleType("mlx_whisper")
    fake_module.transcribe = lambda *a, **k: "legacy-string-return"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)

    b = MlxAudioBackend()
    assert b.transcribe_with_whisper(b"RIFF\x00") == ""
