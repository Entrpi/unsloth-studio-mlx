# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Apple-Silicon MLX-Audio backend for Unsloth Studio.

Peers :class:`core.inference.mlx_lm.MlxLmBackend` and
:class:`core.inference.mlx_vlm.MlxVlmBackend` — **does not extend
either**, because ``mlx-audio``'s LFM2AudioModel has a non-standard
load signature (``LFM2AudioModel.from_pretrained`` instead of
``mlx_audio.load``) and the generate surface is a
``(token, modality)`` generator rather than the ``mlx-lm``
``GenerationResult`` stream.

Phase 10 scope:
- Load LFM2.5-Audio checkpoints via
  ``LFM2AudioModel.from_pretrained`` + ``LFM2AudioProcessor.from_pretrained``.
- TTS path: ``generate_tts(text) -> bytes`` (WAV at model sample_rate).
- ASR path: ``transcribe(audio_bytes) -> str`` (best-effort — see the
  probe report; LFM2.5-Audio's self-transcription of its own TTS is
  unreliable).
- Omni / speech-to-speech is **deferred** — the chat composer would
  need bigger changes than Phase 10 can afford in one chunk.

Non-goals (deferred):
- Streaming audio output.
- Speech-to-speech conversational turns.
- Phoneme-level alignment, SSML, voice cloning.
- LoRA adapter loading on audio models.

The module is importable on any platform: ``mlx_audio`` is lazy-imported
inside ``load_model``. Instantiating ``MlxAudioBackend`` does not
import ``mlx_audio``.
"""

from __future__ import annotations

import gc
import io
import json
import platform
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

AudioCapability = Literal["tts", "asr", "omni"]


class MlxAudioBackend:
    """In-process MLX-Audio backend (TTS + ASR + omni).

    Lifecycle:
        1. ``load_model(local_path, model_identifier)`` — calls
           ``LFM2AudioModel.from_pretrained`` and
           ``LFM2AudioProcessor.from_pretrained``. Both expect a
           local directory path.
        2. ``generate_tts(text)`` — returns WAV bytes at 24 kHz.
        3. ``transcribe(audio_bytes)`` — returns the model's text
           response to the supplied audio. Best-effort on LFM2.5-Audio
           (see ``PROBE_RESULTS.md``).
        4. ``unload_model()`` — drops refs, ``gc.collect()``,
           ``mx.metal.clear_cache()`` if available.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._model_identifier: Optional[str] = None
        self._local_path: Optional[str] = None
        self._sample_rate: int = 24000
        self._load_phase: Optional[str] = None
        # Detected capabilities based on architectures / model_type.
        self._capability: AudioCapability = "tts"
        # True when the model accepts audio-in (voice assistant / ASR / omni).
        self._has_audio_input: bool = False
        self._lock = threading.Lock()

    # ── Platform gate ────────────────────────────────────────────
    @staticmethod
    def _platform_ok() -> bool:
        if platform.system() != "Darwin":
            return False
        return platform.machine().lower() in ("arm64", "aarch64")

    # ── Properties ───────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def is_active(self) -> bool:
        return self.is_loaded

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return False

    @property
    def is_audio(self) -> bool:
        return True

    @property
    def has_audio_input(self) -> bool:
        return self._has_audio_input

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def context_length(self) -> Optional[int]:
        # LFM2.5-Audio's underlying LFM2 backbone advertises 128k in the
        # nested ``lfm`` config block. We don't cap on load.
        return None

    @property
    def is_lora(self) -> bool:
        return False

    @property
    def cache_type_kv(self) -> Optional[str]:
        return None

    @property
    def speculative_type(self) -> Optional[str]:
        return None

    @property
    def chat_template(self) -> Optional[str]:
        # Audio models don't have a standard user-facing chat template
        # we surface — the UI doesn't render one for TTS.
        return None

    def detect_audio_type(self) -> AudioCapability:
        """Return the model's audio capability classification.

        - ``"tts"``: text-in, audio-out only.
        - ``"asr"``: audio-in, text-out only.
        - ``"omni"``: both. This is the LFM2.5-Audio case.
        """
        return self._capability

    # ── Load / unload ─────────────────────────────────────────────
    def load_model(
        self,
        local_path: str,
        model_identifier: str,
        hf_token: Optional[str] = None,
    ) -> bool:
        """Load an MLX audio checkpoint (TTS / ASR / omni).

        Args:
            local_path: Directory containing ``config.json`` and the
                MLX weight shards. For LFM2.5-Audio this is the
                ``mlx-community/LFM2.5-Audio-1.5B-bf16`` snapshot dir.
            model_identifier: Public-facing id the UI surfaces.
            hf_token: Accepted for symmetry; LFM2.5-Audio is public.

        Returns:
            True on success. False on a failed load (logs the
            exception).
        """
        if not self._platform_ok():
            raise RuntimeError(
                "mlx_audio is not available on this platform "
                "(requires macOS on Apple Silicon)"
            )

        with self._lock:
            if self.is_loaded:
                logger.warning(
                    "MlxAudioBackend.load_model called while a model is "
                    "already loaded; unloading first"
                )
                self._unload_locked()

            path = Path(local_path)
            if not path.is_dir():
                raise RuntimeError(
                    f"MLX-Audio model path is not a directory: {local_path}"
                )

            self._load_phase = "loading"

            # Lazy import — heavy numba / librosa chain.
            try:
                from mlx_audio.sts.models.lfm_audio import (  # type: ignore
                    LFM2AudioModel,
                    LFM2AudioProcessor,
                )
            except ImportError as e:
                self._load_phase = None
                raise RuntimeError(
                    f"mlx_audio is not installed in this Python env: {e}"
                ) from e

            t0 = time.time()
            try:
                model = LFM2AudioModel.from_pretrained(str(path))
                processor = LFM2AudioProcessor.from_pretrained(str(path))
            except Exception as e:
                self._load_phase = None
                logger.error(f"LFM2AudioModel.from_pretrained failed: {e}")
                return False
            load_s = time.time() - t0

            # Read ``model_type`` / ``architectures`` to classify the
            # capability. LFM2.5-Audio is omni (both directions).
            cap: AudioCapability = "omni"
            has_input = True
            try:
                with open(path / "config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                archs = cfg.get("architectures") or []
                first = archs[0] if archs else ""
                model_type = (cfg.get("model_type") or "").lower()
                # LFM2.5-Audio: Lfm2AudioForConditionalGeneration — omni.
                if "lfm" in first.lower() or model_type.startswith("lfm"):
                    cap = "omni"
                    has_input = True
                elif first.endswith("ForConditionalGeneration"):
                    cap = "omni"
                    has_input = True
                else:
                    # Unknown; default to TTS-only (safest — the route's
                    # /v1/audio/transcriptions branch will surface a
                    # clean 400).
                    cap = "tts"
                    has_input = False
            except Exception:
                pass

            self._model = model
            self._processor = processor
            self._model_identifier = model_identifier
            self._local_path = str(path)
            self._sample_rate = int(getattr(model, "sample_rate", 24000))
            self._capability = cap
            self._has_audio_input = has_input
            self._load_phase = "loaded"

            logger.info(
                f"MLX-Audio model loaded in {load_s:.2f}s: "
                f"identifier={model_identifier} path={path} "
                f"sample_rate={self._sample_rate} capability={cap}"
            )
            return True

    def _unload_locked(self) -> bool:
        if self._model is None and self._processor is None:
            return False
        self._model = None
        self._processor = None
        self._model_identifier = None
        self._local_path = None
        self._sample_rate = 24000
        self._capability = "tts"
        self._has_audio_input = False
        self._load_phase = None
        gc.collect()
        try:
            import mlx.core as _mx  # type: ignore

            if hasattr(_mx, "metal") and hasattr(_mx.metal, "clear_cache"):
                _mx.metal.clear_cache()
        except Exception:
            pass
        return True

    def unload_model(self) -> bool:
        with self._lock:
            return self._unload_locked()

    # ── TTS ───────────────────────────────────────────────────────
    def generate_tts(
        self,
        text: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_k: int = 4,
    ) -> Tuple[bytes, int]:
        """Generate speech from *text*. Returns ``(wav_bytes, sample_rate)``.

        Uses the model's ``generate_from_chat_state(mode="interleaved")``
        path per the LFM2.5-Audio HF card and collects ``AUDIO_OUT``
        frames into a (codebooks, T) array which is decoded through
        the processor's Mimi detokenizer.

        Args:
            text: Text to synthesize. Long texts will be truncated when
                ``max_new_tokens`` is exhausted.
            max_new_tokens: Generator cap. ~12 audio frames per second
                of speech at 24 kHz, so 2048 yields up to ~170 s.
            temperature, top_k: Text-head sampling parameters.
            audio_temperature, audio_top_k: Audio-codebook sampling
                parameters (LFM2.5-Audio uses separate samplers for the
                text and audio heads).

        Returns:
            ``(wav_bytes, sample_rate)``. ``wav_bytes`` is a complete
            little-endian 16-bit PCM WAV file with a single mono
            channel.
        """
        if not self.is_loaded:
            raise RuntimeError("MLX-Audio model is not loaded")

        try:
            from mlx_audio.sts.models.lfm_audio import ChatState  # type: ignore
            from mlx_audio.sts.models.lfm_audio.model import (  # type: ignore
                LFMModality,
            )
        except ImportError as e:
            raise RuntimeError(f"mlx_audio is not installed: {e}") from e

        import mlx.core as mx  # type: ignore
        import numpy as np

        chat = ChatState(self._processor)
        chat.new_turn("user")
        chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")

        audio_codes: List[Any] = []
        try:
            for token, modality in self._model.generate_from_chat_state(
                chat,
                mode = "interleaved",
                max_new_tokens = int(max_new_tokens),
                temperature = float(temperature),
                top_k = int(top_k),
                audio_temperature = float(audio_temperature),
                audio_top_k = int(audio_top_k),
            ):
                if modality == LFMModality.AUDIO_OUT:
                    audio_codes.append(token)
        except Exception as e:
            logger.error(f"LFM2AudioModel.generate_from_chat_state raised: {e}")
            raise

        if not audio_codes:
            raise RuntimeError(
                "TTS produced no audio frames — the model returned a "
                "text-only response. Try a different prompt."
            )

        # Each yielded token has shape (8,) for 8-codebook Mimi.
        # Stack along time axis → (codebooks, T).
        stacked = mx.stack(audio_codes, axis = -1)
        try:
            waveform = self._processor.decode_audio(stacked)
        except Exception:
            # Probed fallback: some bf16 checkpoints skip the
            # ``audio_detokenizer/config.json`` sub-dir. Use the
            # processor's Mimi directly.
            waveform = self._processor.mimi.decode(stacked[None])

        # Normalize to a 1-D float32 array at self._sample_rate.
        arr = np.asarray(waveform).squeeze()
        if arr.ndim > 1:
            arr = arr[0]
        arr = np.asarray(arr, dtype = np.float32)

        wav_bytes = _float32_to_wav_bytes(arr, self._sample_rate)
        return wav_bytes, self._sample_rate

    # ── ASR / audio understanding ─────────────────────────────────
    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Transcribe / respond to *audio_bytes* (a WAV/MP3 blob).

        Best-effort on LFM2.5-Audio (the model is a voice assistant,
        not a dedicated ASR — see ``PROBE_RESULTS.md`` for the probe
        outcome). When *prompt* is None we feed the audio alone; the
        model treats it as an S2S turn and usually echoes or responds.
        Passing a prompt like ``"Please transcribe the audio"`` often
        improves results but never to the level of a dedicated ASR.

        Returns:
            The model's text output as a single string (empty string
            if the model generated no text tokens).
        """
        if not self.is_loaded:
            raise RuntimeError("MLX-Audio model is not loaded")
        if not self._has_audio_input:
            raise RuntimeError(
                "Loaded MLX-Audio model does not support audio input "
                "(TTS-only)."
            )

        try:
            from mlx_audio.sts.models.lfm_audio import ChatState  # type: ignore
            from mlx_audio.sts.models.lfm_audio.model import (  # type: ignore
                LFMModality,
            )
        except ImportError as e:
            raise RuntimeError(f"mlx_audio is not installed: {e}") from e

        import soundfile as sf  # type: ignore
        import numpy as np
        import mlx.core as mx  # type: ignore

        # Decode the input WAV/MP3 bytes into a float array.
        with io.BytesIO(audio_bytes) as bio:
            data, in_sr = sf.read(bio, dtype = "float32", always_2d = False)
        if data.ndim > 1:
            data = data.mean(axis = 1)  # to mono
        audio_arr = mx.array(data.astype(np.float32))

        chat = ChatState(self._processor)
        chat.new_turn("user")
        if prompt:
            chat.add_text(prompt)
        chat.add_audio(audio_arr, sample_rate = int(in_sr))
        chat.end_turn()
        chat.new_turn("assistant")

        text_tokens: List[Any] = []
        try:
            for token, modality in self._model.generate_from_chat_state(
                chat,
                mode = "text",
                max_new_tokens = int(max_new_tokens),
                temperature = float(temperature),
            ):
                if modality == LFMModality.TEXT:
                    text_tokens.append(token)
        except Exception as e:
            logger.error(f"mlx_audio transcribe raised: {e}")
            raise

        if not text_tokens:
            return ""

        toks = mx.concatenate([t.reshape(-1) for t in text_tokens])
        text = self._processor.decode_text(toks)
        # Strip the common ``<|im_end|>`` suffix the model appends on
        # turn close.
        for end in ("<|im_end|>", "</s>"):
            if text.endswith(end):
                text = text[: -len(end)]
        return text.strip()

    # ── Progress surface ─────────────────────────────────────────
    def load_progress(self) -> Dict[str, Any]:
        return {
            "phase": self._load_phase,
            "bytes_loaded": 0,
            "bytes_total": 0,
            "fraction": 1.0 if self._load_phase == "loaded" else 0.0,
        }

    # ── Chunk E (E9): mlx-whisper as a parallel ASR path ─────────
    #
    # LFM2.5-Audio's self-transcription is conversational rather than
    # verbatim (see PROBE_RESULTS.md). For real ASR, route to
    # ``mlx-whisper`` — a dedicated Whisper port on MLX. This sits
    # alongside the LFM2.5 ASR path; it doesn't replace it. Callers
    # opt in via the ``X-ASR-Backend: whisper`` header or ``backend``
    # body field on /v1/audio/transcriptions.
    #
    # The whisper model is loaded lazily on first call and kept
    # process-lifetime (mlx-whisper doesn't expose a lifecycle
    # surface). It does not affect LFM2.5-Audio memory residency.
    def transcribe_with_whisper(
        self,
        audio_bytes: bytes,
        *,
        model_hint: str = "auto",
    ) -> str:
        """Transcribe *audio_bytes* using ``mlx-whisper`` — a dedicated
        verbatim-ASR path independent of the loaded LFM2.5-Audio model.

        Args:
            audio_bytes: Raw WAV/MP3 bytes. Decoded via soundfile.
            model_hint: Either ``"auto"`` (honour ``MLX_WHISPER_MODEL``
                env var, default to ``mlx-community/whisper-tiny``) or
                an explicit HF repo id / local path passed to
                ``mlx_whisper.transcribe(path_or_hf_repo=...)``.

        Returns:
            Transcribed text. Empty string on failure to decode.

        Raises:
            RuntimeError: if ``mlx-whisper`` is not installed or the
                model cannot be loaded.
        """
        try:
            import mlx_whisper  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"mlx-whisper is not installed: {e}. Install via "
                f"`pip install mlx-whisper` (macOS / Apple Silicon only)."
            ) from e

        if model_hint == "auto":
            import os as _os

            model_repo = _os.environ.get(
                "MLX_WHISPER_MODEL", "mlx-community/whisper-tiny"
            )
        else:
            model_repo = model_hint

        import tempfile
        import os

        # mlx_whisper.transcribe accepts path, ndarray, or mx.array. We
        # already accept bytes on the API surface; write to a tempfile
        # so whisper's own audio loader handles format sniffing
        # (WAV/MP3/FLAC) — cheaper than re-decoding via soundfile.
        with tempfile.NamedTemporaryFile(
            suffix = ".audio", delete = False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = mlx_whisper.transcribe(
                tmp_path,
                path_or_hf_repo = model_repo,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # ``mlx_whisper.transcribe`` returns a dict with keys
        # ``"text"``, ``"segments"``, ``"language"``. We only need the
        # full concatenated text.
        if not isinstance(result, dict):
            return ""
        text = result.get("text", "")
        return text.strip() if isinstance(text, str) else ""


# ── WAV helpers ──────────────────────────────────────────────────
def _float32_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """Pack a 1-D float32 array of samples in [-1, 1] into a WAV blob.

    We use the stdlib ``wave`` module so the output is a standards-
    compliant RIFF WAV even when Studio runs in sandboxes without
    ``soundfile``. Samples outside [-1, 1] are clipped before the
    16-bit cast so any TTS over-shoot doesn't wrap around.
    """
    import numpy as np

    arr = np.asarray(samples, dtype = np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()
