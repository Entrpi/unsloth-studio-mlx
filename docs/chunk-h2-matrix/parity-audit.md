# Chunk H-2 — GGUF / MLX parity audit

Systematic backend-surface comparison across the four Studio inference
backends. Produced during the Chunk H-2 VLM tool-calling closure.

Backends under audit:

- **LlamaCpp** — `studio/backend/core/inference/llama_cpp.py` (GGUF,
  Chunks A/B).
- **MlxLm** — `studio/backend/core/inference/mlx_lm.py` (MLX text,
  Chunk C/F).
- **MlxVlm** — `studio/backend/core/inference/mlx_vlm.py` (MLX vision,
  Chunk D).
- **MlxAudio** — `studio/backend/core/inference/mlx_audio.py` (MLX
  audio, Chunk D).

Legend: `✓` present, `—` not applicable, `✗` gap.

## Surface matrix

| Dimension | LlamaCpp | MlxLm | MlxVlm | MlxAudio |
|---|---|---|---|---|
| `is_loaded` / `is_active` | ✓ | ✓ | ✓ | ✓ |
| `model_identifier` | ✓ | ✓ | ✓ | ✓ |
| `is_vision` | ✓ (`_is_vision` on load) | ✓ (False) | ✓ (always True) | ✓ (False) |
| `is_audio` | — | — | — | ✓ |
| `has_audio_input` | — | — | — | ✓ |
| `is_lora` / `adapter_path` | ✗ (no LoRA path) | ✓ | ✓ (hard-wired False — Phase 6 LoRA deferred) | ✓ (False) |
| `hf_variant` | ✓ | ✓ | ✓ | ✗ (not surfaced) |
| `context_length` / `max_context_length` / `native_context_length` | ✓ | ✓ | ✓ (all three = same value) | ✓ (context_length only, returns None) |
| `chat_template` | ✓ | ✓ | ✓ | ✓ (None by design — TTS/ASR has no template) |
| `supports_reasoning` / `reasoning_always_on` / `reasoning_default` | ✓ | ✓ | ✓ | ✗ (not surfaced) |
| `supports_tools` | ✓ | ✓ | ✓ (detection works — but no generator!) | ✗ (not applicable) |
| `cache_type_kv` | ✓ | ✓ (`q8_0`/`q4_0`) | ✓ (returns None — VLM KV quant surface deferred) | ✓ (None) |
| `speculative_type` / `draft_model_path` | ✓ (`ngram-simple`/`mod`) | ✓ (`mlx-draft-model`) | ✓ (None) | ✓ (None) |
| `load_progress` (phase / bytes / fraction / warnings) | ✓ | ✓ (+warnings) | ✓ (minimal — no byte counters, always fraction=1.0 when loaded) | ✓ (minimal) |
| HF remote download with tqdm mirror | ✓ (`_download_gguf`) | ✓ (`_download_mlx`, snapshot_download + custom tqdm subclass) | ✗ (route layer pre-downloads; no backend-level fetch) | ✗ (route layer pre-downloads) |
| RAM-pressure preflight warnings | ✓ (GPU VRAM fit + mmproj split) | ✓ (`_draft_mem_preflight`, RAM checks) | ✗ | ✗ |
| Context auto-fit to available memory | ✓ (`_fit_context_to_vram`) | ✗ (passed `n_ctx` or native) | ✗ | — |
| `n_ctx` override on load | ✓ | ✓ | ✓ | ✗ |
| `enable_thinking` passthrough into prompt builder | ✓ | ✓ | ✓ (via `chat_template_kwargs` + TypeError fallback) | — |
| `generate_chat_completion` streaming | ✓ | ✓ | ✓ | — (TTS/ASR use `generate_tts` / `transcribe`) |
| `generate_chat_completion_with_tools` agentic loop | ✓ | ✓ | **✗ MISSING — this chunk closes it** | — (not applicable) |
| Tool-loop sub-features (timeout wrapper, cancel-event polling, content hold-back, status text, metadata sum) | ✓ | ✓ | **✗ — closed this chunk** | — |
| Client-side `tools=[...]` passthrough (OpenAI) | ✓ (route `using_gguf` branch) | ✓ (`_mlx_openai_passthrough_stream`) | ✗ (flagged) | — |
| Anthropic `/v1/messages` tool-call support | ✓ (route handler) | ✓ (`test_mlx_anthropic_passthrough.py` coverage) | ✗ (flagged) | — |
| Tool-use nudge injection in route layer | ✓ | ✓ (Chunk H-2 `45aea900`) | ✗ (this chunk wires it into the new VLM agentic path) | — |
| `detect_audio_type` | ✓ (GPT-OSS style) | ✓ (returns None) | — (VLM isn't a TTS model) | ✓ |
| `generate_audio_response` / `init_audio_codec` | ✓ (GPT-OSS) | — | — | — (separate `generate_tts` / `transcribe` / `transcribe_with_whisper`) |
| Streamed transcription (ASR) | ✗ | — | — | ✓ (`transcribe`, `transcribe_with_whisper`) |
| Image-b64 input → PIL decode → backend | ✓ (route level) | — | ✓ (`_decode_image_b64_to_path`) | — |
| `base_url` / subprocess boundary | ✓ (llama-server) | — (in-process) | — (in-process) | — (in-process) |
| `_platform_ok` gate | — (cross-platform) | ✓ (Darwin arm64) | ✓ (Darwin arm64) | ✓ (Darwin arm64/aarch64) |
| `reset_generation_state` | — (llama-server stateless) | ✗ (not needed — no hidden state) | ✗ | — |

## Confirmed closed

These dimensions are already at full parity or the gap is genuinely
"not applicable":

- **`is_active` / `is_loaded` / `model_identifier` / `chat_template`** —
  identical semantics across all four.
- **Reasoning detection (`supports_reasoning`, `always_on`, `default`)** —
  MlxLm and MlxVlm share `_detect_reasoning_from_template` with the
  same `<think>` / `enable_thinking` heuristic. GGUF's equivalent lives
  in its own `_read_gguf_metadata`. Audio correctly returns nothing.
- **Speculative decoding** — GGUF uses llama.cpp's built-in n-gram
  speculator (`ngram-simple` / `ngram-mod`), MlxLm loads a separate
  `draft_model` via `mlx_lm`. VLM and Audio both return None; neither
  upstream library supports a parallel speculator and neither is
  user-visible in the UI dropdown (confirmed by `speculative_type`
  surface check).
- **Quantized KV cache surface** — GGUF and MlxLm both surface
  `q8_0` / `q4_0` and honor the UI dropdown. VLM intentionally returns
  None — `mlx_vlm` forwards `kv_bits` via `**kwargs` but the UI hides
  the KV dropdown for vision models; flagged as P2 if a user requests.
  Audio returns None and the surface is not wired.
- **Audio input** — only MlxAudio has `has_audio_input`; GGUF's
  GPT-OSS audio path is distinct (`init_audio_codec` + audio-decoder).
- **LoRA on audio models** — N/A; no one ships audio LoRA adapters
  for LFM2.5-Audio today.

## Closed this chunk

**VLM tool-calling (Option A) — RESOLVED.**
See `blockers.md::B5` for the full closure trail.

Landed in commits:
- `MlxVlmBackend.generate_chat_completion_with_tools` agentic loop
  with image-b64 passthrough, shared-parser reuse, hold-back, and
  timeout wrapper.
- Route `using_vlm:` branch gains a `payload.enable_tools` sub-branch
  that dispatches to `_mlx_vlm_agentic_stream`.
- `_render_prompt` now tries passing `tools=` through
  `apply_chat_template` first so Gemma-4's native
  `<|tool>declaration:NAME{...}<tool|>` dialect is emitted — falls
  back to the old system-prompt injection for templates that reject
  the kwarg.
- 12 unit tests in `test_mlx_vlm_tool_loop_unit.py` covering
  guards, tool-choice, happy path, Gemma dialect, image-b64 first-
  iteration-only contract, content hold-back, timeout wrapper,
  cancel-event exit, max-iterations cap, metadata sum.
- 1 real-model integration test in
  `test_mlx_vlm_gemma_tool_calling.py` — empirically verified:
  Gemma-4 E4B VLM emits `tool_name='get_weather'`,
  `arguments={'city': 'Paris'}` through the VLM backend.

## Flagged for follow-up

### P0 — user-blocking gaps

None beyond the VLM tool-calling gap being closed this chunk.

### P1 — capability gaps (do-next candidates)

- **VLM client-side tools passthrough** (OpenAI `tools=[...]` without
  `enable_tools`). The route has `_mlx_openai_passthrough_stream` for
  MlxLm but no peer for MlxVlm. External clients (opencode / Claude
  Code / Cursor) that drive a VLM checkpoint with client-side tools
  silently fall through to plain chat. Effort: medium — mostly a
  port of `_mlx_openai_passthrough_stream` that also honors
  `image_b64`; the buffering state machine translates cleanly because
  `generate_chat_completion_with_tools` now exists on MlxVlm.
- **VLM Anthropic `/v1/messages` tool-calling**. Same story on the
  Anthropic route — the existing `test_mlx_anthropic_passthrough.py`
  covers text-only. Effort: small — reuses the new VLM tool-call
  backend once the OpenAI passthrough is ported.
- **MlxAudio `hf_variant`**. Audio backend doesn't surface the
  `4bit`/`bf16` variant tag the UI uses for dropdown grouping. Users
  with multiple LFM2.5 variants see them collapsed into the same row.
  Effort: trivial — one property + one extraction call in `load_model`.
- **VLM `load_progress` byte counters**. MlxVlm returns
  `bytes_loaded=0`, `bytes_total=0`, `fraction=1.0` on load. The UI
  shows a spinner that never advances. Audio has the same issue. Not
  user-visible on a 4–8 GB VLM (loads in < 10 s) but it's noticeable
  on 31B / MoE. Effort: medium — port `_download_mlx` + the psutil
  RSS sampler from MlxLm.
- **VLM HF remote load**. `load_model` requires a local directory
  path; there's no path for "user pastes an HF repo id, backend
  downloads it". Route layer currently pre-downloads via the
  download manager. Effort: medium — port `_download_mlx` as above
  or reuse the download manager more cleanly.
- **GGUF LoRA support**. `LlamaCppBackend` has no LoRA attach path.
  MlxLm supports `is_lora` / `adapter_path`. llama.cpp itself accepts
  `--lora-adapter` so the underlying surface exists; just not plumbed
  through Studio. Effort: medium.

### P2 — nice-to-have

- **VLM quantized KV cache UI dropdown**. Backend returns None; UI
  hides the dropdown. If a user explicitly asks, add `kv_bits` / 
  `kv_group_size` pass-through similar to MlxLm. Effort: small.
- **VLM speculative-draft path**. `mlx_vlm` doesn't expose a draft
  model today. If upstream adds one, the `speculative_type` surface
  is already there — just needs a loader call and property wiring.
- **MlxAudio `supports_reasoning`**. Audio models don't reason, but
  the UI logic checks the attribute; relying on `AttributeError`
  today works but a stable `False` would be cleaner. Effort: trivial.
- **MlxAudio `n_ctx` override on load**. LFM2.5-Audio has a 128k
  context window; not user-tunable via the Studio UI today. Effort:
  trivial.
- **MlxLm `reset_generation_state`**. MLX backends hold no hidden
  state between turns (each `stream_generate` starts fresh), so the
  method from the Unsloth `backend.reset_generation_state()` surface
  is a no-op. Documented as N/A.

---

This audit was produced on 2026-04-22. The single P0 gap (VLM
tool-calling) lands in the same chunk. P1/P2 items become their own
chunks.
