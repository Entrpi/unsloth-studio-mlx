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

## Follow-ups (remaining phases after Chunk B)

- Phase 5: tool calling (XML parser extraction, in-process OpenAI/Anthropic
  passthrough, agentic loop).
- Phases 9–10 as per `/tmp/mlx-parity-roadmap.md` (vision, audio).

---

# Chunk B (Phases 3 + 6 + 7 + 8)

Four orthogonal phases on top of Chunk A. Built in order 8 → 6 → 7 → 3
to minimize merge churn (Phase 3 touches route/config/frontend, so it
wraps the finalized load signature rather than re-doing Phases 6/7/8
arguments afterwards).

## Phase 8 — Quantized KV cache

Adds KV-cache quantization pass-through for MLX. Mirrors the GGUF UI's
`cache_type_kv` dropdown (`f16`/`bf16`/`q8_0`/`q5_1`/`q4_1`/`q4_0`).

- `_cache_type_kv_to_mlx(value) -> (kv_bits, kv_group_size)` in
  `mlx_lm.py` maps the UI dropdown strings to mlx-lm's `(int, int)`.
  `f16`/`bf16` → `(None, 64)` meaning "don't pass the kwargs". `q5_1`
  logs a warning and rounds **down** to 4-bit (mlx-lm has no 5-bit KV
  path; rounding down preserves the user's "smaller cache" intent).
- `MlxLmBackend.cache_type_kv` property returns `None` / `"q8_0"` /
  `"q4_0"` / `f"q{bits}_0"` depending on the loaded state.
- `load_model(cache_type_kv=...)` stores `_kv_bits` / `_kv_group_size` /
  `_quantized_kv_start` on the backend; `generate_chat_completion` adds
  the kwargs to `stream_generate` **only when `kv_bits is not None`**.
  This preserves bit-for-bit behavior of Chunk A when quantization is
  off (empty-kwargs fast path).
- Route: `LoadRequest.cache_type_kv` → `load_model(cache_type_kv=...)`;
  `LoadResponse.cache_type_kv` surfaces the effective value so the UI
  learns about the `q5_1 → q4_0` rounding.
- Frontend: added `activeIsMlx` to the chat runtime store; the main
  `isGguf` gate in `chat-settings-sheet.tsx` becomes `isGgufOrMlx` so
  KV cache dtype, speculative, and context-length controls now show
  for MLX too. No visual refinement yet; the existing dropdown is
  re-used.
- `stream_generate` probe on 0.31.2: explicit kwarg `draft_model`;
  `kv_bits`/`kv_group_size`/`quantized_kv_start` forwarded via
  `**kwargs` into `generate_step` (and `speculative_generate_step`),
  which accept them as explicit parameters.

## Phase 6 — LoRA adapter loading

- `_detect_mlx_adapter(path)` in `model_config.py`: True when the
  directory has both `adapters.safetensors` **and**
  `adapter_config.json`. Keying on `adapters.safetensors` (plural)
  specifically — HuggingFace PEFT writes `adapter_model.safetensors`
  (singular) so there's no collision with PEFT adapters. Base MLX
  models have neither so there's no collision there either.
- `ModelConfig` gains `is_mlx_lora: bool` + `mlx_adapter_path:
  Optional[str]`.
- `MlxLmBackend.load_model(adapter_path=...)` threads the path through
  to `mlx_lm.load(..., adapter_path=...)`. `is_lora` property returns
  True iff an adapter was loaded.
- `LoadRequest.adapter_path` new schema field; route forwards.
  `LoadResponse.is_mlx_lora` new schema field surfaces the backend's
  `is_lora` state.
- **Opportunistic `backend_kind` enum — deferred.** Adding a new enum
  alongside the boolean flags (`is_gguf`, `is_mlx`, `is_mlx_lora`,
  `is_lora`) without frontend consumers would just add a field to
  sign off on. Better to wait until Phase 9 (vision) lands a fourth
  distinct backend and forces the collapse.
- Integration testing: no local MLX LoRA adapters available. Covered
  by 10 unit tests: detection (5), backend `is_lora` property (3),
  and `mlx_lm.load` kwarg-threading with a mock (2). The integration
  path will light up once a user drops an MLX LoRA into the model
  picker in a real Studio session.

## Phase 7 — Speculative decoding

Pairs the Bonsai 8B base with the Bonsai 1.7B MLX 2-bit as draft —
the headline Chunk B demo.

- `MlxLmBackend.load_model(draft_model_path=...)`: when non-None, loads
  a second model via `mlx_lm.load` and stores `_draft_model` +
  `_draft_tokenizer` alongside the base. Single lock guards both.
- `_draft_mem_preflight`: before loading, sum the draft's
  `*.safetensors` sizes; refuse with `RuntimeError("draft model would
  exceed 75% of available memory...")` if combined footprint
  `(total - available) + draft_bytes` crosses 75% of total RAM. No-op
  when `psutil` isn't importable (graceful degradation).
- `stream_generate(draft_model=..., num_draft_tokens=3)` — `draft_model`
  is an explicit kwarg on 0.31.2; `num_draft_tokens` forwards via
  `**kwargs` into `speculative_generate_step`.
- `speculative_type` property returns `"mlx-draft-model"` (distinct
  from GGUF's `"ngram-simple"` / `"ngram-mod"` labels) when a draft
  is loaded.
- `unload_model` drops the draft alongside the base.
- Vocab-size mismatch between base and draft logs a warning at load
  time. mlx-lm requires the same tokenizer; mismatched vocab sizes
  will typically fail at token-verification time.
- Route: `LoadRequest.draft_model_path` → `load_model(...)`;
  `LoadResponse.speculative_type` surfaces the active mode.
- Frontend: added `draftModelPath` / `loadedDraftModelPath` to the
  runtime store; the settings sheet's Speculative Decoding dropdown
  maps MLX "On" → `mlx-draft-model` and shows a path input
  (`placeholder="/absolute/path/to/mlx-draft-dir"`) only when both
  MLX is active and the dropdown is "On".
- Integration test (`test_load_with_draft_and_stream_tokens`): loads
  Bonsai 8B + Bonsai 1.7B, asserts `speculative_type==
  "mlx-draft-model"`, streams 32 tokens, asserts non-empty output.
  Skipped automatically when RAM headroom < ~25% (the 75% preflight
  cap), since a 32 GB M5 under baseline load often can't fit both.
  The preflight itself is exercised by dedicated unit tests.

## Phase 3 — Remote-HF pulls, load_progress, memory warnings

- `_extract_mlx_variant(name)` in `mlx_lm.py`: regex-parses the trailing
  quant suffix (`-4bit`, `-mlx-2bit`, `-fp16`, …). Exposed as
  `MlxLmBackend.hf_variant` after load.
- `_download_mlx(repo, hf_token)`: wraps
  `huggingface_hub.snapshot_download` with:
  - `allow_patterns = ["*.safetensors",
    "*.safetensors.index.json", "config.json", "tokenizer*",
    "special_tokens_map.json", "*.json", "*.jinja",
    "chat_template*", "generation_config.json"]` — tight enough to
    skip READMEs and eval tensors, loose enough to catch whatever
    tokenizer layout the repo uses.
  - A custom `tqdm_class` subclass that writes per-shard totals into
    `_download_bytes_total` on init and per-update bumps into
    `_download_bytes_loaded`. Strictly additive; if the tqdm API
    drifts, counters go stale but the download still succeeds.
- `ModelConfig.from_identifier` remote-HF branch now probes
  `config.json` via `hf_hub_download`; when the MLX `quantization
  {"bits","group_size"}` block is present, returns an `is_mlx=True`
  ModelConfig with `mlx_path=None` (download deferred to the
  backend) and `native_context_length` from `max_position_embeddings`.
- `load_model`:
  - Accepts non-existent `local_path` that looks like a repo id
    (has a `/`, not a filesystem path) — routes through
    `_download_mlx`.
  - Sets `_load_phase` to `"downloading"` → `"loading"` → `"loaded"`
    across the load.
  - Computes `_weights_bytes_total` from local safetensors after
    download. Compares to `psutil.virtual_memory().total`; when
    `total < 1.5 × weight bytes`, appends a RAM-pressure warning to
    `_load_warnings` (surfaced via `load_progress().warnings`).
- `load_progress()`:
  - `"downloading"` → counters from the HF tqdm subclass.
  - `"loading"` → samples `psutil.Process().memory_info().rss`
    against `_weights_bytes_total` for a best-effort bar.
  - `"loaded"` → fraction=1.0 with final byte counts.
  - `None` → no load in flight.
  - Additive `warnings: List[str]` field when `_load_warnings` has
    entries.
- Route: `/api/inference/load-progress` checks the MLX backend first;
  when an MLX load is in flight it delegates. The MLX-only
  `warnings` field is filtered out before handing to the existing
  `LoadProgressResponse` (no frontend schema break).
- Reset semantics: `unload_model` clears all progress + warning
  state so a subsequent `load_progress()` returns `None`.

## Chunk B smoke-test checklist

Build on Chunk A's checklist. All curl commands assume the default
local port `http://127.0.0.1:8000`.

### Phase 8 — Quantized KV cache

```bash
# Load Bonsai with 8-bit quantized KV:
curl -sN http://127.0.0.1:8000/api/inference/load \
  -H 'content-type: application/json' \
  -d '{
    "model_path": "/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit",
    "cache_type_kv": "q8_0",
    "max_seq_length": 0
  }'
# Expect: LoadResponse with cache_type_kv=="q8_0".

# q5_1 rounds down to q4_0 with a warning in logs:
curl -sN http://127.0.0.1:8000/api/inference/load \
  -H 'content-type: application/json' \
  -d '{
    "model_path": "/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit",
    "cache_type_kv": "q5_1"
  }'
# Expect: LoadResponse with cache_type_kv=="q4_0".
```

### Phase 6 — LoRA adapter loading

```bash
# (Requires an MLX LoRA adapter at /path/to/adapter — not shipped in
# this repo.) The backend routes adapter_path through to
# mlx_lm.load(..., adapter_path=...).
curl -sN http://127.0.0.1:8000/api/inference/load \
  -H 'content-type: application/json' \
  -d '{
    "model_path": "/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit",
    "adapter_path": "/path/to/mlx-lora-adapter"
  }'
# Expect: LoadResponse with is_mlx=true AND is_mlx_lora=true.
```

### Phase 7 — Speculative decoding (Bonsai 8B + 1.7B)

```bash
curl -sN http://127.0.0.1:8000/api/inference/load \
  -H 'content-type: application/json' \
  -d '{
    "model_path": "/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit",
    "draft_model_path": "/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit"
  }'
# Expect: LoadResponse with speculative_type=="mlx-draft-model".

# Stream a chat completion — throughput should be higher than Bonsai 8B alone:
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Ternary-Bonsai-8B-mlx-2bit",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}],
    "stream": true,
    "max_tokens": 64
  }'
```

### Phase 3 — Remote HF MLX + load_progress

```bash
# Kick off a remote download (any public mlx-community repo works):
curl -sN http://127.0.0.1:8000/api/inference/load \
  -H 'content-type: application/json' \
  -d '{
    "model_path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  }' &
LOAD_PID=$!

# Poll progress in a second shell:
while kill -0 $LOAD_PID 2>/dev/null; do
  curl -s http://127.0.0.1:8000/api/inference/load-progress
  echo
  sleep 1
done
# Expect: a sequence of {phase="downloading", bytes_loaded, bytes_total, fraction},
# then {phase="loading"}, then {phase="loaded"}.

# Final load response carries hf_variant (exposed via backend property,
# not yet routed to LoadResponse — follow-up).
```

---

## Chunk C (Phase 5) — Tool calling

Wires MLX into the full tool-calling stack so external clients
(opencode, Claude Code via OpenAI-compat, Cursor, Continue) and
Studio's own agentic chat mode can drive MLX-backed models on the
same terms they already drive GGUF-backed models.

### New surface

| Layer | New entry point |
|---|---|
| Shared parser | `studio/backend/core/inference/_tool_call_parser.py` — `parse_tool_calls_from_text(text, model_family="auto")` + `strip_tool_markup`. Lifted from the inline GGUF implementation. |
| MLX backend | `MlxLmBackend.generate_chat_completion_with_tools(...)` — event shape matches the GGUF backend (`status` / `content` / `tool_start` / `tool_end` / `metadata`). |
| MLX backend | `supports_tools` property now returns the chat-template probe result (True for Qwen3 / Bonsai / Hermes / Mistral-instruct dialects). |
| Schema | `FunctionCall`, `ToolCall`, `ToolCallDelta`, `ToolCallFunctionDelta` Pydantic models + `ChoiceDelta.tool_calls` + `ChunkChoice.finish_reason = "tool_calls"`. `ChatMessage.reasoning_content` added. |
| Route helper | `_extract_content_parts(..., preserve_tool_history=True)` keeps `role='tool'` / `assistant.tool_calls` / `reasoning_content` on the backend message list. |
| OpenAI route | `_mlx_agentic_stream` / `_mlx_agentic_non_streaming` (Studio `enable_tools=true`); `_mlx_openai_passthrough_stream` / `_mlx_openai_passthrough_non_streaming` (standard OpenAI `tools=[...]`). |
| Anthropic route | MLX-aware branching in `/v1/messages`. Reuses `AnthropicStreamEmitter` for agentic flow and `AnthropicPassthroughEmitter` for client-side pass-through. |

### Detection: what Bonsai actually emits

Tested against `/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-8B-mlx-2bit`:

- `apply_chat_template` accepts the `tools` kwarg (so the fallback
  prompt-injection path is rarely exercised in practice).
- Tool calls come out in the Qwen3 JSON dialect:
  ```
  <tool_call>
  {"name": "get_weather", "arguments": {"city": "Paris"}}
  </tool_call>
  ```
  The closing tag is present on most turns but the parser handles its
  absence (the GGUF backend's auto-heal also covers this).
- Tool results are rendered back into the template as a
  `<tool_response>...</tool_response>` block inside a user turn.

### Tests

| Suite | Count | Path |
|---|---:|---|
| Shared parser | 28 | `tests/test_tool_call_parser.py` |
| Schema round-trip + `_extract_content_parts(preserve_tool_history=…)` | 19 | `tests/test_mlx_tool_schemas.py` |
| MLX tool-loop units | 18 (inside a larger 82-test file) | `tests/test_mlx_backend_unit.py::Test*Tool*` |
| MLX OpenAI SSE shape | 6 | `tests/test_mlx_openai_passthrough.py` |
| MLX Anthropic SSE shape | 5 | `tests/test_mlx_anthropic_passthrough.py` |
| GGUF tool regression | 97 (existing, re-ran green) | `tests/test_openai_tool_passthrough.py` + `tests/test_anthropic_messages.py` |
| Bonsai real-model integration | 2 | `tests/test_mlx_backend_lifecycle.py::test_bonsai_*tool*` |

### Smoke-test curls

All three paths below reuse the existing Bonsai 8B MLX checkpoint.
Load it first with a plain local-path `POST /api/inference/load` (the
Phase 5 changes don't touch the load payload).

**1. OpenAI `/v1/chat/completions` with client-side `tools=[...]`:**

```bash
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Ternary-Bonsai-8B-mlx-2bit",
    "messages": [{"role":"user","content":"What is the weather in Paris? Use the get_weather function."}],
    "stream": true,
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Return current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
# Expect: delta.tool_calls fragments carrying id+name then arguments,
# followed by finish_reason="tool_calls" and data: [DONE].
```

**2. Anthropic `/v1/messages` with `tools=[...]`:**

```bash
curl -sN http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "Ternary-Bonsai-8B-mlx-2bit",
    "max_tokens": 256,
    "messages": [{"role":"user","content":"What is the weather in Paris? Use the get_weather function."}],
    "stream": true,
    "tools": [{
      "name": "get_weather",
      "description": "Return current weather for a city",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type":"string"}},
        "required": ["city"]
      }
    }]
  }'
# Expect: message_start / content_block_start with tool_use /
# content_block_delta with input_json_delta fragments / message_stop.
```

**3. Studio-internal `enable_tools=true` (built-in web_search / python / terminal):**

```bash
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Ternary-Bonsai-8B-mlx-2bit",
    "messages": [{"role":"user","content":"Use python to compute 2**100."}],
    "stream": true,
    "enable_tools": true,
    "enabled_tools": ["python"]
  }'
# Expect: Studio-specific tool_status / tool_start / tool_end custom
# SSE events interleaved with standard content deltas. Final
# finish_reason="stop" once the agentic loop concludes.
```

### Deviations from the roadmap

- **`backend_kind` enum**: left deferred as the roadmap allows (Phase 6
  opportunistic). All Phase 5 code paths branch on the existing boolean
  flags; no change.
- **Parser dialect hint**: the shared parser auto-detects both dialects
  by default. Callers can still force the JSON or XML dialect via
  `model_family="qwen"` / `"xml"` / etc. but no current caller does —
  Bonsai always takes the JSON path.
- **Parallel tool calls in a single turn**: supported in the parser
  (multiple `<tool_call>` blocks produce multiple `tool_calls`) but
  Bonsai usually emits one at a time. Not exercised in integration
  tests because the model doesn't tend to do it.
- **`_mlx_openai_passthrough_stream` buffering**: uses a 64-char
  buffer cap before flushing a speculative XML prefix as plain
  content. The GGUF side uses 32; the larger MLX value accommodates
  Bonsai's longer tool-call XML prefix that appears when the model
  starts thinking out loud.
- **Arguments streaming**: client-side pass-through emits arguments as
  one chunk per call rather than character-by-character. The OpenAI SDK
  accepts both shapes (it assembles via string concatenation
  internally) and single-chunk emission is noticeably simpler to get
  right.

### Known follow-ups for Chunk D

- Tool-call cancellation: the `cancel_event` is honoured at iteration
  boundaries but not mid-tool execution (inherits the GGUF behaviour).
- The MLX route's preserve-tool-history branch doesn't yet synthesise
  XML for assistant turns that only carry `tool_calls` with empty
  content; current behaviour ends up with an empty assistant
  string — rendering correctness depends on the chat template.
  Verified to work with Bonsai but not proven out for Hermes / Mistral
  templates that don't re-serialise `tool_calls` cleanly.
