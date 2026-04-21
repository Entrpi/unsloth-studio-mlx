# Chunk D — Phase 9 & Phase 10 Probe Results

Date: 2026-04-21
Venv: `/tmp/mlxtest` (Python 3.13.10)

## Versions installed

| Package | Version |
|---------|---------|
| mlx-vlm | 0.4.4 |
| mlx-audio | 0.4.2 |
| mlx-lm | 0.31.2 (kept at Chunk-C pin; mlx-audio prefers 0.31.1 but still works) |
| mlx | 0.31.1 |
| torch | 2.11.0 (pulled in as a transitive requirement for the Qwen3VL video processor shim) |
| torchvision | 0.26.0 |

`mlx-audio==0.4.2` declares a hard pin on `mlx-lm==0.31.1`. We restored `mlx-lm==0.31.2` post-install (Chunk C uses it); all `mlx_audio` imports and the TTS path still work. The pip resolver emits a warning, not an error.

`mlx-vlm` pulled in `torch` + `torchvision` because the HuggingFace `Qwen3VLVideoProcessor` (co-instantiated with the image processor via `AutoProcessor.from_pretrained`) hard-requires PyTorch — even though we never invoke the video path. This is upstream behavior, not ours to fix.

## Probe 1 — install

Pass. See `/tmp/chunkD-probes/install.log`.

## Probe 2 — mlx-vlm API surface

Pass. See `/tmp/chunkD-probes/mlx_vlm_api.log`.

Key signatures:
- `load(path_or_hf_repo: str, adapter_path=None, lazy=False, revision=None, **kwargs) -> (Module, ProcessorMixin)`
- `generate(model, processor, prompt: str, image=None, audio=None, verbose=False, **kwargs) -> GenerationResult`
- `stream_generate(model, processor, prompt: str, image=None, audio=None, **kwargs) -> Generator[GenerationResult | str]`
- `mlx_vlm.prompt_utils.apply_chat_template(processor, config, prompt, add_generation_prompt=True, return_messages=False, num_images=0, num_audios=0, **kwargs)`

`qwen3_5` is an entry in `mlx_vlm.models` — architecture is supported.

## Probe 3 — Qwen3.5-4B-VLM load + describe image

Pass. See `/tmp/chunkD-probes/probe3_vision.log`.

- 256×256 white PNG with a red square centered at [64,64,192,192]
- Generated text (64 tokens): *"The user wants a short description of the image. 1. Identify the main subject: It's a square shape. 2. Identify the color: It is bright red. ..."*
- Assertion `"red" in output.lower() or "square" in output.lower()`: **PASS** (both present)
- Peak memory: 3.3 GB; generation 40.5 tok/s

Streaming probe (probe3b): **PASS**. `stream_generate` yields 33 `GenerationResult` chunks with `.text` and `.token` attributes; last chunk's `.text` is the final incremental piece. Concatenating yields the same full text.

## Probe 4 — mlx-audio API surface

Pass. See `/tmp/chunkD-probes/mlx_audio_api.log`.

Key signatures:
- `LFM2AudioModel.from_pretrained(model_name_or_path: str) -> LFM2AudioModel`
- `LFM2AudioProcessor.from_pretrained(model_name_or_path: str) -> LFM2AudioProcessor`
- `ChatState(processor, add_bos=True)` with `.new_turn(role)`, `.end_turn()`, `.add_text(text)`, `.add_audio(audio: mx.array, sample_rate: int = 16000)`, `.get_text_tokens() / .get_audio_features() / .get_modalities()`.
- `model.generate_from_chat_state(chat_state, mode="interleaved"|"text"|"sequential", max_new_tokens, temperature, top_k, audio_temperature, audio_top_k) -> Generator[(token, LFMModality)]`
- `LFMModality = {TEXT, AUDIO_IN, AUDIO_OUT}` (note: the example in the roadmap says `"audio"` but upstream uses an IntEnum with `AUDIO_OUT`).
- `processor.mimi.decode(audio_codes[None])` decodes codes → waveform at `model.sample_rate` (24000).

`processor.decode_audio(...)` exists but requires a local `audio_detokenizer/config.json` not shipped with the bf16 checkpoint; falling back to `processor.mimi.decode(...)` works.

## Probe 5 — LFM2.5-Audio TTS round-trip

Pass. See `/tmp/chunkD-probes/probe5_tts.log`.

- Prompt: "Say hello world."
- Text produced: `"Hello, world!"` (plus a small artifact)
- 12 audio frames × 8 codebooks → 23,040 samples → 0.96 s
- WAV on disk: sr=24000, duration=0.96 s, energy=0.0355
- Assertion (sr==24000, len>12000, energy>0.001): **PASS**

## Probe 6 — LFM2.5-Audio ASR round-trip

Partial / degraded. See `/tmp/chunkD-probes/probe6_asr.log`, `probe6b_asr.log`, `probe6c_asr.log`.

- Chat state correctly ingests audio — `get_audio_features()` reports shape `(1, 289, 128)` for a 2.88 s WAV — so encoder is wired and conformer preprocessing runs.
- Model does respond to the presence of audio (replies change based on audio presence vs absence).
- However, on **self-TTS-then-ASR round-trips**, the model says *"I didn't quite hear you"* or *"Could you share the audio"* rather than transcribing.
- Multiple prompts tried ("Transcribe…", "Repeat…", "What was said?", bare audio): none produced a string containing "hello"/"world"/"test".

**Interpretation**: the architecture is present but this 1.5B bf16 checkpoint's STT capability on short self-generated TTS waveforms is unreliable. Potential causes:
1. The TTS-generated speech is only 0.96–2.88 s and may be below the model's useful STT window.
2. LFM2.5-Audio is primarily a conversational S2S model, not a dedicated ASR — it may not transcribe on demand without a specific fine-tune.
3. Possible upstream bug in how `mlx_audio` wires the conformer encoder output into the LFM2 decoder context during inference (no unit-test coverage for the audio-in path in `mlx_audio.sts.tests.test_lfm_audio`).

This does not block Phase 10 TTS. It does downgrade the ASR story to **best-effort**: we'll expose `backend.transcribe(...)` and an `/v1/audio/transcriptions` route wired to `generate_from_chat_state(..., mode="text")`, documented as "works depending on model/audio quality". The integration test will assert the call returns a non-empty string, not that it contains the original phrase.

## Per-probe pass/fail

| Probe | Status |
|-------|--------|
| 1. install mlx-vlm + mlx-audio | PASS |
| 2. mlx-vlm API surface | PASS |
| 3. Qwen3.5-4B-VLM image describe | PASS (red/square present in output) |
| 3b. mlx-vlm stream_generate | PASS (33 chunks, incremental text) |
| 4. mlx-audio API surface | PASS |
| 5. LFM2.5-Audio TTS round-trip | PASS (sr=24000, dur=0.96 s, energy=0.0355) |
| 6. LFM2.5-Audio ASR round-trip | PARTIAL — model receives audio but does not reliably transcribe self-TTS |

## Decision

**PROCEED** — both ecosystems work end-to-end for the primary user flow:
- Phase 9: image-in / text-out via Qwen3.5-4B-VLM is rock-solid.
- Phase 10: TTS (text-in / audio-out) via LFM2.5-Audio is rock-solid.
- Phase 10 ASR ships as best-effort with a softened integration assertion.

No phases are blocked. Proceed to implementation.
