# MLX-LM backend for Unsloth Studio (Phase 1 + Chunk A)

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

## Follow-ups (Phase 3, 5–10)

- Remote-HF MLX loading by teaching `from_identifier` to pull `config.json`
  from HF Hub — Phase 3 of the parity roadmap.
- Tool calling (XML parser extraction + in-process OpenAI passthrough) —
  Phase 5.
- LoRA adapter loading, speculative decoding, quantized KV, vision
  (`mlx-vlm`), audio (`mlx-audio` + `mlx-whisper`).
- Collapse `is_gguf` / `is_mlx` / Unsloth flags into a single `backend_kind`
  enum once a fourth backend lands.

---

# Chunk A (Phases 2 + 4)

Adds sampling fidelity and reasoning/`<think>` support on top of Phase 1.

## Phase 2 — Sampling fidelity

Brings MLX chat quality to parity with the GGUF backend for a given set of
sampling settings.

- `_build_mlx_sampler_and_processors` (module-private helper) centralizes
  the `mlx_lm.sample_utils` surface: `make_sampler(temp, top_p, top_k, min_p)`
  + `make_logits_processors(repetition_penalty, presence_penalty,
  frequency_penalty, logit_bias, *context_size)`. Empty processor list is
  the fast path — defaults never pay per-token overhead. Processor kwargs
  are only forwarded when non-default (`repetition_penalty > 1.0`,
  `presence/frequency != 0`, non-empty `logit_bias`).
- `temperature <= 0` → greedy / argmax (upstream `make_sampler(temp=0)`).
- Defensive fallback: if a kwarg is rejected by a different `mlx-lm` patch
  release, drop it and retry. Mirrors Phase 1's pattern.
- **Backend-side stop-string enforcement.** `mlx-lm` 0.31.2 has no native
  `stop` kwarg (verified against upstream). After each token tick we
  scan the tail of the cumulative decoded text — window size is
  `max_stop_len + len(latest_delta)` so cross-boundary matches are caught
  without rescanning the full buffer each tick. On match we truncate
  cumulative, yield it once, and terminate with `finish_reason="stop"`
  in the metadata event.
- Route: `payload.stop` is normalized (str → [str], empty strings
  filtered) and forwarded into the MLX generator.

## Phase 4 — Reasoning / `<think>` support

Detection at load time by inspecting `tokenizer.chat_template`:

1. Template contains literal `enable_thinking` → `supports_reasoning=True`,
   `reasoning_always_on=False`. Qwen3.5/3.6 `<9B` → `reasoning_default=False`,
   else `True`. Ported from `llama_cpp.py:1519-1535`.
2. Template contains both `<think>` and `</think>` (but not
   `enable_thinking`) → `supports_reasoning=True`,
   `reasoning_always_on=True`, `reasoning_default=True`. Template
   hardcodes the tags, so the toggle has no effect and the UI hides it.
3. Otherwise → no reasoning support.

`generate_chat_completion` threads `enable_thinking` into
`tokenizer.apply_chat_template` via `chat_template_kwargs={"enable_thinking":
bool}` **only when** the backend advertises reasoning support AND the
caller explicitly set the flag. Non-reasoning models never see the kwarg
(protects templates that reject unknown keys). Literal `<think>...</think>`
tags in the stream are passed through unmodified — the existing
frontend parser handles them directly.

Route / `LoadResponse` / `InferenceStatusResponse` now surface
`supports_reasoning`, `reasoning_always_on`, and `chat_template` from
`MlxLmBackend`. The frontend's existing reasoning-toggle UI in
`shared-composer.tsx` and `use-chat-model-runtime.ts` is already
backend-agnostic (keys off `supportsReasoning` in the runtime store), so
the thinking panel lights up for MLX reasoning models automatically — no
frontend code changes required.

## Verified against Ternary-Bonsai-8B-mlx-2bit

Probed the Bonsai 8B MLX 2-bit checkpoint at load time. Its chat template
contains literal `<think>\n\n</think>\n\n` in the assistant prefix (a
Qwen3-derived template that seeds empty thinking — effectively always-on
reasoning), so Phase 4 detects it correctly: `supports_reasoning=True`,
`reasoning_always_on=True`, `chat_template` length ≈ 4063 chars.

## Files (Chunk A additions)

| File | Change |
|---|---|
| `studio/backend/core/inference/mlx_lm.py` | +`_build_mlx_sampler_and_processors`, `_detect_reasoning`, stop-string loop, reasoning-state fields, `chat_template` / `supports_reasoning` / `reasoning_always_on` / `reasoning_default` properties, `enable_thinking` threading into `apply_chat_template` |
| `studio/backend/routes/inference.py` | MLX `LoadResponse` / `InferenceStatusResponse` surface the new reasoning flags; MLX chat branch normalizes + forwards `payload.stop` |
| `studio/backend/tests/test_mlx_backend_unit.py` | +18 unit tests (sampler helper, stop strings, reasoning detection, enable_thinking threading) |
| `studio/backend/tests/test_mlx_backend_lifecycle.py` | +5 integration tests (stop string end-to-end, repetition penalty changes output, reasoning flags populated after load, enable_thinking changes prompt, <think> tags pass through) |
| `PR_DESCRIPTION.md` | this section |

Frontend: **no changes**. The existing reasoning/thinking UI in
`shared-composer.tsx` already reads `supportsReasoning` / `reasoningAlwaysOn`
from the chat runtime store, and `use-chat-model-runtime.ts` already
populates those fields from any `LoadResponse` (GGUF or MLX).
`chat-adapter.ts` already forwards `enable_thinking` when
`supportsReasoning` is true.

## Chunk A smoke-test checklist

Build on the Phase 1 checklist above.

11. Load Bonsai MLX. `GET /api/inference/status` returns
    `supports_reasoning=true`, `reasoning_always_on=true`, and a
    non-empty `chat_template` string.
12. `POST /v1/chat/completions` with `{"stop": ["END"]}`:

    ```bash
    curl -sN http://127.0.0.1:8000/v1/chat/completions \
      -H 'content-type: application/json' \
      -d '{
        "model": "Ternary-Bonsai-8B-mlx-2bit",
        "messages": [{"role": "user", "content": "Reply exactly: ok END more"}],
        "stream": true,
        "stop": ["END"]
      }'
    ```

    Assert the concatenated `delta.content` does NOT contain `END`.

13. `POST /v1/chat/completions` with `{"repetition_penalty": 1.3,
    "temperature": 0}` twice (once at 1.0, once at 1.3) — greedy, identical
    prompt, outputs differ.

14. `POST /v1/chat/completions` with `{"enable_thinking": true}` against
    a reasoning model:

    ```bash
    curl -sN http://127.0.0.1:8000/v1/chat/completions \
      -H 'content-type: application/json' \
      -d '{
        "model": "Ternary-Bonsai-8B-mlx-2bit",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "enable_thinking": true,
        "stream": true
      }'
    ```

    Assert the server emits a 200 (the flag is forwarded) and the UI's
    thinking toggle becomes visible/active after load.

## Follow-ups (remaining phases)

- Phase 3: remote HF-hub download + load_progress UI.
- Phase 5: tool calling (XML parser extraction, in-process OpenAI/Anthropic
  passthrough, agentic loop).
- Phases 6–10 as per `/tmp/mlx-parity-roadmap.md`.
