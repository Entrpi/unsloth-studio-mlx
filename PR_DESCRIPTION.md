# MLX-LM backend for Unsloth Studio (Phase 1)

Adds an in-process `MlxLmBackend` as a peer to the existing `LlamaCppBackend`
so that on Apple Silicon, users can load MLX checkpoints — specifically
`prism-ml/Ternary-Bonsai-8B-mlx-2bit` — and chat with them through Studio's
existing OpenAI-compatible `/v1/chat/completions` route.

Detection keys off a `"quantization": {"bits", "group_size"}` block in
`config.json` (empirically verified for the target model and confirmed to
have no collision with `quantization_config`-shaped bnb / GPTQ configs).
The backend exposes the same public surface as the GGUF path, so the route
branches on `is_mlx` / `is_gguf` without touching the Unsloth / transformers
path.

## Scope

**In scope (Phase 1):**

- Detect MLX checkpoints by `config.json["quantization"]["bits"]` presence.
- Load the target MLX model via `mlx_lm.load()`.
- Stream chat completions via `mlx_lm.stream_generate()` yielding OpenAI
  `chat.completion.chunk` SSE frames.
- Unload/reload cleanly; peer backends are unloaded automatically on load.
- Only activate on macOS (`platform.system() == "Darwin"` and `mlx_lm`
  importable). Linux / Windows CI imports the module cleanly but cannot
  load.
- Propagate minimal metadata to the UI (`is_mlx`, `context_length`,
  `native_context_length`).

**Out of scope (defer):**

- Tool calling (the backend returns `supports_tools = False`; the route
  rejects tool requests with 400).
- Vision / multimodal (`is_vision = False`; the route rejects image
  inputs with 400).
- Audio (TTS/ASR).
- Reasoning / `<think>` tag wrapping.
- Speculative decoding, quantized KV cache, LoRA adapters, training.
- Remote HF-hub MLX download — Phase 1 loads only local paths.
- GPU memory estimation / context autofit — MLX uses unified memory; we
  expose only `native_context_length` from config.

## Files

| File | Change |
|---|---|
| `studio/backend/core/inference/mlx_lm.py` | **new** — `MlxLmBackend` |
| `studio/backend/utils/models/model_config.py` | `_detect_mlx_model`, `is_mlx` / `mlx_path` / `native_context_length` fields |
| `studio/backend/routes/inference.py` | `get_mlx_lm_backend`, MLX branches in `load`, `unload`, `chat/completions`, `/status` |
| `studio/backend/models/inference.py` | `is_mlx` on `LoadResponse`, `ValidateModelResponse`, `InferenceStatusResponse` |
| `studio/backend/models/models.py` | `is_mlx` on `ModelDetails` |
| `studio/backend/routes/models.py` | MLX passthrough in `list_models`; delete-cached guard |
| `studio/backend/requirements/studio.txt` | `mlx-lm>=0.31.2,<0.32 ; sys_platform == "darwin"` |
| `studio/frontend/src/features/chat/types/api.ts` | `is_mlx?` on four interfaces |
| `studio/frontend/src/features/chat/types/runtime.ts` | `isMlx?` on `ChatModelSummary` |
| `studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts` | "MLX" tag push; MLX-aware context-length derivation |
| `studio/backend/tests/test_mlx_backend_detection.py` | **new** — 10 unit tests |
| `studio/backend/tests/test_mlx_backend_unit.py` | **new** — 8 unit tests |
| `studio/backend/tests/test_mlx_backend_lifecycle.py` | **new** — 2 macOS-gated integration tests |

## Streaming contract

- `MlxLmBackend.generate_chat_completion(...)` **yields cumulative text**
  strings, then a final `{"type": "metadata", "usage": {...}, "timings":
  {...}}` dict — identical to the GGUF backend's contract, so the route
  can reuse its `new_text = cumulative[len(prev_text):]` diff unchanged.
- First SSE frame is the role chunk. Per-token frames carry the delta.
  The final frame carries `finish_reason="stop"`. An optional usage
  chunk follows. Terminator is `data: [DONE]`.
- The route drives the backend's synchronous generator via
  `asyncio.to_thread(next, gen, sentinel)` so the event loop stays free
  for disconnect detection. `cancel_event` is checked between tokens.

## MLX `stream_generate` response shape (verified)

Probed `mlx-lm==0.31.2` against `Ternary-Bonsai-8B-mlx-2bit`:

```
dir(resp) ≈ [finish_reason, from_draft, generation_tokens, generation_tps,
             logprobs, peak_memory, prompt_tokens, prompt_tps, text, token]
```

So the metadata dict maps `resp.prompt_tokens` → `usage.prompt_tokens`,
`resp.generation_tokens` → `usage.completion_tokens`, and the two `_tps`
fields → `timings.prompt_per_second` / `timings.predicted_per_second`.

## Performance (Apple M5, 32 GB unified)

Numbers from `/tmp/mlx_load_test.py` running `mlx-lm==0.31.2` against
`Ternary-Bonsai-8B-mlx-2bit`:

| Metric | Value |
|---|---|
| Cold load (`mlx_lm.load`) | ~1.5–2 s |
| Resident memory after load | ~2.9 GB |
| Prefill | **~711 tok/s** |
| Decode | **~47 tok/s** |

47 tok/s decode puts cancel latency at ~21 ms per token, which matches or
beats the GGUF cancel-during-streaming latency.

## Manual smoke-test checklist

1. `pip install 'mlx-lm>=0.31.2,<0.32'` on the macOS dev box.
2. Start Studio: `unsloth studio run`.
3. In the UI, paste `/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit`
   into the model picker. Verify the **"MLX"** tag shows after load.
4. Send "What is 2+2? Answer in one word." — verify tokens stream at
   ~30+ tok/s and the response ends cleanly.
5. Click **Stop** mid-generation — verify it stops within 1 s and no
   zombie Python threads are left (Activity Monitor).
6. Load a GGUF model next — verify MLX unloads first (memory drops in
   Activity Monitor) and GGUF loads cleanly.
7. Re-load the MLX model — verify GGUF unloads first.
8. `GET /api/inference/status` returns `is_mlx=true`; `GET /api/models/list`
   returns the MLX entry with `is_mlx=true`.
9. `POST /v1/chat/completions` with `{"image_base64": "...some img..."}`
   returns 400 ("MLX backend does not support image inputs in Phase 1").
10. `POST /v1/chat/completions` with `{"tools": [...]}` returns 400
    ("MLX backend does not support tool calling in Phase 1").

## Risks

| Risk | Mitigation |
|---|---|
| `stream_generate` blocks the event loop | Route wraps each `next()` in `asyncio.to_thread`. |
| `mx.metal.clear_cache()` API drift | `hasattr`-guarded in `_unload_locked`. |
| Tokenizer missing `chat_template` | Backend falls back to a minimal ChatML prompt with a warning. |
| `mlx_lm` minor-version API drift | Pinned `>=0.31.2,<0.32`; bump after smoke-testing 0.32. |
| 16 GB RAM boxes swap when loading 8B | Logged warning planned — out of scope for Phase 1. |

## Follow-ups (Phase 2+)

- Collapse `is_gguf` / `is_mlx` / Unsloth flags into a single `backend_kind`
  enum once a fourth backend lands.
- Remote-HF MLX loading by teaching `from_identifier` to pull `config.json`
  from HF Hub.
- MLX vision via `mlx-vlm` when the UI shows a vision MLX model.
