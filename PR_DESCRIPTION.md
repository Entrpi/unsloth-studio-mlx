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

---

# Chunk D (Phases 9 + 10) — Vision + Audio

Adds two new MLX peer backends completing the parity roadmap:

- **`MlxVlmBackend`** (Phase 9, `core/inference/mlx_vlm.py`) — load
  `mlx-community` vision-language checkpoints via `mlx_vlm.load()` and
  stream multimodal chat with `mlx_vlm.stream_generate()`. Target and
  smoke model: `mlx-community/Qwen3.5-4B-MLX-4bit`.
- **`MlxAudioBackend`** (Phase 10, `core/inference/mlx_audio.py`) — TTS
  via `LFM2AudioModel.from_pretrained` + `generate_from_chat_state`
  (`mode="interleaved"`), ASR via best-effort `mode="text"` with audio
  injected through `ChatState.add_audio`. Target and smoke model:
  `mlx-community/LFM2.5-Audio-1.5B-bf16`.

Additive companion: a `BackendKind` enum
(`"gguf" | "mlx" | "mlx+lora" | "mlx+vlm" | "mlx+audio" | "unsloth"`)
lands on `LoadResponse` and `InferenceStatusResponse`. The boolean
flags (`is_gguf` / `is_mlx` / `is_mlx_vlm` / `is_mlx_audio`) are
preserved — nothing is deprecated yet.

## Probe-phase findings (pre-implementation)

Committed separately as `PROBE_RESULTS.md`. Highlights:

- `mlx-vlm==0.4.4` — works. `load()` returns `(model, processor)`;
  `stream_generate` yields `GenerationResult` objects with `.text`.
  Qwen3.5-4B-VLM describes a red-square test image with 100 %
  reliability (assertion: output contains `"red"` or `"square"`).
  Installing `mlx-vlm` transitively pulls in `torch` + `torchvision`
  because the HF `Qwen3VLVideoProcessor` is hard-required by
  `AutoProcessor.from_pretrained` even when video is never used.
- `mlx-audio==0.4.2` — TTS works reliably. LFM2.5-Audio-1.5B-bf16
  synthesizes "Say hello world" to a 0.96 s / 24 kHz mono WAV with
  non-trivial energy. ASR is **best-effort** — the model is a voice
  assistant, not a dedicated ASR, and does not reliably transcribe
  its own short TTS outputs. Documented as a softer assertion in
  the integration test.
- `mlx-audio` pins `mlx-lm==0.31.1` in its `install_requires`; we
  keep the Chunk-C pin `mlx-lm==0.31.2` and the resolver emits a
  warning rather than an error. All `mlx-audio` imports + TTS paths
  verified working with the mismatched pin.

Decision: **PROCEED** — both ecosystems are production-ready for the
primary flows; ASR degrades to best-effort with a softened test.

## Scope

**In scope (Chunk D):**

- `_detect_mlx_vlm_model` — dual-signal detector (preprocessor /
  processor config JSON files OR `architectures[*]` ending
  `ConditionalGeneration` + known-VLM `model_type` allow-list).
- `_detect_mlx_audio_model` — `architectures[*]` ending
  `AudioForConditionalGeneration` OR known audio-model_type allow-list.
- `MlxVlmBackend` with properties and public surface matching the
  other peer backends (so the route's dispatch stays trivial):
  `is_loaded`, `is_active`, `is_vision=True`, `model_identifier`,
  `context_length`, `supports_tools`, `supports_reasoning`,
  `load_model`, `unload_model`, `generate_chat_completion(..., image_b64=...)`.
- `MlxAudioBackend` with `generate_tts(text) -> (wav_bytes,
  sample_rate)` and `transcribe(audio_bytes, prompt=...) -> str`.
  Returns valid RIFF/WAVE using the stdlib `wave` module so the
  output is parseable everywhere.
- Route wiring:
  - `get_mlx_vlm_backend()` + `get_mlx_audio_backend()` lazy singletons.
  - `_unload_all_mlx_peers(keep=...)` helper — Section 4.6 of the
    roadmap's "peer-unload loop" (applied ~5 call sites across
    GGUF / Unsloth / MLX / MLX-VLM / MLX-Audio loads).
  - `/api/inference/load` branches for `is_mlx_vlm` and
    `is_mlx_audio` (before the base MLX branch because both detect
    on VLM/audio-specific signals).
  - `/api/inference/unload` and `/api/inference/status` surface the
    new peers with `backend_kind` enum values.
  - `/api/inference/load-progress` polls the VLM / audio peers too.
  - **`POST /v1/audio/speech`** (and `/api/inference/audio/speech`)
    — OpenAI-compatible TTS endpoint. Body: `{input, model, voice,
    response_format}`. Returns raw WAV bytes.
  - **`POST /v1/audio/transcriptions`** (and
    `/api/inference/audio/transcriptions`) — OpenAI-compatible ASR
    endpoint. Multipart upload. Returns JSON `{"text": "..."}`.
- Schemas: `is_mlx_vlm` / `is_mlx_audio` / `backend_kind` on
  `LoadResponse`, `ValidateModelResponse`, `InferenceStatusResponse`,
  `ModelDetails`.
- Frontend:
  - `BackendKind` TS enum + `is_mlx_vlm` / `is_mlx_audio` /
    `backend_kind` on the API interfaces.
  - `ChatModelSummary.isMlxVlm` / `isMlxAudio`.
  - `chat-runtime-store.ts` now tracks `activeIsMlxVlm` /
    `activeIsMlxAudio` / `activeBackendKind`; both are set from
    `LoadModelResponse` and `InferenceStatusResponse`.
  - Model-tag logic emits "MLX-VLM" and "MLX-Audio" labels.
  - A VLM load treats `isVision = true` so the image composer shows
    automatically — no UI changes required downstream of the flag.
  - `npm run typecheck` clean.
- Requirements: `mlx-vlm>=0.4.4,<0.5` and `mlx-audio>=0.4.2,<0.5`,
  both `sys_platform == "darwin"` gated.

**Out of scope (deferred):**

- Video input on VLM models (even though Qwen3VL ships a video
  processor config — video is off the Chunk D menu).
- Audio input on VLM models (Qwen3-Omni exists but we didn't probe it).
- VLM LoRA adapter loading (Phase 6 LoRA story is base-text only).
- Streaming audio output (roadmap-defined non-goal).
- Voice-to-voice conversational chat composer (too much UI work).
- A dedicated ASR model path — we rely on LFM2.5-Audio's built-in
  audio-in channel rather than shipping `mlx-whisper`. That's a
  reasonable follow-up if ASR quality becomes a priority.

## Deviations from the roadmap

- The roadmap suggested **extending** `MlxLmBackend` to carry audio
  behavior (mirroring `LlamaCppBackend`'s audio_type fork). We
  implemented a **peer class** instead because
  `LFM2AudioModel.from_pretrained` has a different load signature
  and the generate surface yields `(token, modality)` tuples rather
  than `GenerationResult`. The peer-class approach kept `MlxLmBackend`
  unchanged and made the route dispatch simpler.
- `BackendKind` is rolled out **additively in Phase 9+10** (not "in
  Phase 6 additively + deprecate in 9/10"). Booleans stay on every
  response; no deprecation warnings yet. A future chunk can flip
  the UI to read `backend_kind` exclusively and then trim the
  booleans.
- The peer-unload refactor (roadmap §4.6) was **applied**. A new
  `_unload_all_mlx_peers(keep=...)` helper replaces the ~5 inline
  chains of `if backend.is_loaded: backend.unload_model()`.

## Smoke-test curls

### Load + describe an image (Phase 9)

```bash
# Load Qwen3.5-4B-VLM
curl -s -X POST http://localhost:8085/api/inference/load \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $STUDIO_API_KEY' \
  -d '{"model_path":"/Users/ent/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit","max_seq_length":0,"load_in_4bit":false,"is_lora":false}'

# Send a multimodal chat with a base64 PNG
B64=$(python3 -c 'from PIL import Image,ImageDraw;import io,base64;img=Image.new("RGB",(256,256),"white");ImageDraw.Draw(img).rectangle([64,64,192,192],fill="red");b=io.BytesIO();img.save(b,format="PNG");print(base64.b64encode(b.getvalue()).decode())')
curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $STUDIO_API_KEY' \
  -d "{\"model\":\"qwen3.5-vlm\",\"stream\":false,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image in one short sentence.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}]}"
# Expected: response text mentions 'red' or 'square'.
```

### TTS via /v1/audio/speech (Phase 10)

```bash
# Load LFM2.5-Audio
curl -s -X POST http://localhost:8085/api/inference/load \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $STUDIO_API_KEY' \
  -d '{"model_path":"/Users/ent/.lmstudio/models/mlx-community/LFM2.5-Audio-1.5B-bf16","max_seq_length":0,"load_in_4bit":false,"is_lora":false}'

# OpenAI-shape TTS
curl -s -X POST http://localhost:8085/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $STUDIO_API_KEY' \
  -d '{"model":"lfm2-audio","input":"Hello world, this is Unsloth Studio on MLX.","response_format":"wav"}' \
  --output /tmp/studio_tts.wav
# Expected: /tmp/studio_tts.wav is a playable 24 kHz mono WAV.
```

### ASR via /v1/audio/transcriptions (Phase 10)

```bash
curl -s -X POST http://localhost:8085/v1/audio/transcriptions \
  -H 'Authorization: Bearer $STUDIO_API_KEY' \
  -F file=@/tmp/studio_tts.wav \
  -F model=lfm2-audio \
  -F response_format=json
# Expected: {"text": "..."}.
# Best-effort — LFM2.5-Audio may respond conversationally rather than
# verbatim transcribing. See PROBE_RESULTS.md for the limitation.
```

## Test results

- 201 tests in `tests/test_mlx*.py` + `tests/test_tool_call_parser.py`
  all pass on macOS arm64, including:
  - 22 new VLM detection / backend unit tests.
  - 4 new VLM Darwin-gated integration tests (Qwen3.5-4B-VLM
    end-to-end: load, describe red square, text-only fallback,
    peer-unload after loading another MLX model).
  - 4 new audio backend unit tests (WAV packing round-trip).
  - 3 new audio Darwin-gated integration tests (LFM2.5-Audio TTS
    round-trip, ASR call contract).
  - All pre-existing Chunks A–C tests unchanged.
- `npm run typecheck` in `studio/frontend` clean.

---

# Chunk E — Refinement pass

Quality / polish pass across Chunks A–D follow-ups. No new user-visible features; tightens correctness, ergonomics, robustness, and test coverage.

## Items landed

| # | Item | Status |
|---|---|---|
| E1 | Fix stale Phase-1 `test_non_gguf_load_responses_omit_field` (MLX now populates `native_context_length`) | landed |
| E2 | Pin `python-multipart` as explicit runtime dep (was ad-hoc install during Chunk C) | landed |
| E3 | Expose `num_draft_tokens` via `LoadRequest` (was hardcoded to 3) | landed |
| E4 | Surface backend `warnings: list[str]` on `LoadProgressResponse` + frontend | landed |
| E5 | Hard-refuse tokenizer mismatch on speculative decoding (was soft warning) | landed |
| E6 | Fix cross-test pollution via module-load stubs (httpx / import-order issues) | landed |
| E7 | Stream tool-call arguments across multiple SSE deltas (OpenAI spec compliance) | landed |
| E8 | Non-Bonsai tool-calling template coverage — new `test_mlx_tool_templates.py`, 2 xfail cases for known template gaps | landed (partial) |
| E9 | `mlx-whisper` as parallel ASR path (opt-in via `backend=whisper`) | landed |
| E10 | `BackendKind` boolean deprecation (optional) | deferred |

## E9 — Whisper ASR opt-in

`LFM2.5-Audio` ASR is conversational rather than verbatim (see PROBE_RESULTS.md). For real transcription, clients now opt in to `mlx-whisper` via the `backend` form field on `/v1/audio/transcriptions`.

```bash
curl -s -X POST http://localhost:8085/v1/audio/transcriptions \
  -H "Authorization: Bearer $STUDIO_API_KEY" \
  -F file=@/tmp/speech.wav \
  -F backend=whisper \
  -F response_format=json
# Expected: {"text": "verbatim transcription"}.
# Whisper model resolves via MLX_WHISPER_MODEL env var, falls back to
# mlx-community/whisper-tiny. Lazy-imported; the ASR path continues to
# work without mlx-whisper for the LFM2.5-Audio default.
```

Non-goals: replacing LFM2.5-Audio as the default ASR. Whisper is opt-in; the prior path is preserved.

## Test results after Chunk E

- 248 MLX tests pass + 2 xfail (E8's documented non-Bonsai template gaps).
- 0 regressions from Chunks A–D.
- `/tmp/mlxtest/bin/pytest tests/test_mlx_*.py tests/test_tool_call_parser.py tests/test_native_context_length.py` completes in ~3 min.

---

# Chunk F — Final cleanup

Closes the last two deferred follow-ups. After this chunk the roadmap work is a clean PR candidate.

## Items landed

| # | Item | Status |
|---|---|---|
| F1 | Resolve the 2 E8 xfails (Qwen3.5 template + args dict) + synthesise `<tool_call>` content for templates without a `tool_calls` iterator | landed |
| F2 | Mark `is_gguf` / `is_mlx` / `is_mlx_vlm` / `is_mlx_audio` / `is_mlx_lora` as `deprecated=True` on the Pydantic schemas; migrate frontend chat reads to `backend_kind` | landed |

## F1 — Template-aware tool-call routing + argument dict decode

`_extract_content_parts(preserve_tool_history=True)` now accepts an optional `chat_template` kwarg and routes per template idiom:

- **Camp (a) — templates that iterate `message.tool_calls`** (Qwen3.5, Bonsai, Gemma-4): ship `tool_calls` natively. Opportunistically `json.loads` the OpenAI JSON-string `arguments` into a dict. Qwen3.5's template specifically needs a mapping for `tool_call.arguments | items`; Bonsai / Gemma accept both forms so the upgrade is safe across the whole native-iteration set. This flips both Chunk-E xfails green.
- **Camp (b) — templates without `tool_calls` iteration** (Hermes / Ministral / some Llama-3.1 fine-tunes): synthesise `<tool_call>{...}</tool_call>` JSON-in-tags content into the assistant turn's content field. Dialect matches the shared parser (`core.inference._tool_call_parser.parse_tool_calls_from_text`) so round-tripping reconstructs the original `ToolCall` shape — a new unit test exercises this round-trip.

Heuristic: plain substring scan of the template for `message.tool_calls` / `message['tool_calls']` / `message["tool_calls"]`. Cheap, no Jinja parsing. False-positives keep the native path — the safer side.

## F2 — `BackendKind` boolean deprecation

`is_gguf` / `is_mlx` / `is_mlx_vlm` / `is_mlx_audio` / `is_mlx_lora` are now marked `deprecated=True` on every Pydantic schema that carries them:
- `LoadResponse`
- `InferenceStatusResponse`
- `ValidateModelResponse`
- `ModelDetails`

Pydantic v2 surfaces the deprecation as a `DeprecationWarning` on field reads AND a `{"deprecated": true}` flag in the generated JSON schema / OpenAPI docs. The booleans remain populated on every response — no external API consumer is broken by this change; removal is a future chunk.

`backend_kind` on `LoadResponse` / `InferenceStatusResponse` is now documented as the primary source of truth. It's `BackendKind | None` — `None` is returned only on cold-start (`/status` before any load); every successful load populates the enum deterministically.

### Frontend migration

Every `isGguf` / `isMlx` / `isMlxVlm` / `isMlxAudio` read in `studio/frontend/src/features/chat/**` now routes through `backend_kind` first and falls back to the deprecated boolean. Two small helpers in `use-chat-model-runtime.ts` centralise the routing:

```ts
resolveBackendKind(source)     // → BackendKind | null
kindHasNativeContextLength(kind) // → boolean  (GGUF / any MLX variant)
```

`ChatModelSummary` gains a new `backendKind?: BackendKind | null` field; the deprecated booleans are still written so in-flight consumers keep working.

JSDoc `@deprecated` comments were added to every legacy boolean on `types/api.ts`, `types/runtime.ts`, and the relevant fields on `chat-runtime-store.ts`.

## Test results after Chunk F

- **260 MLX tests pass, 0 xfail.** Delta from Chunk E: +8 tests from the new deprecation suite, +2 new F1 tests (content-synthesis round-trip + template-iteration heuristic), +2 xfails converted to passes.
- 28 tool-call parser tests continue to pass — GGUF tool-call path unchanged.
- `npm run typecheck` in `studio/frontend` clean.
- No regressions in the ~250 existing MLX tests from Chunks A–E.

## New deprecation-contract tests

`studio/backend/tests/test_mlx_backend_kind_deprecation.py`:
- Every legacy boolean field has `deprecated=True` in the generated JSON schema.
- `backend_kind` itself is NOT deprecated (it's the replacement).
- Reading a deprecated field on an instance raises `DeprecationWarning`; construction stays silent; reading `backend_kind` stays silent.
- Every `BackendKind` literal variant is accepted by Pydantic; invalid variants are rejected.

---

# Chunk G — Residual follow-ups

Closes the two items Chunk F's deliverable flagged as still open after the main deprecation landed. After this chunk the follow-up list is empty.

## Items landed

| # | Item | Status |
|---|---|---|
| G1 | Migrate `ModelOption.isGguf` to `backendKind` across `components/assistant-ui/model-selector/**` so the frontend deprecation is uniform outside `features/chat/**` | landed |
| G2 | Add `backend_kind` to `ValidateModelResponse` so the `/validate` endpoint is symmetric with `LoadResponse` / `InferenceStatusResponse` | landed |

## G1 — `ModelOption.backendKind`

`ModelOption` in `studio/frontend/src/components/assistant-ui/model-selector/types.ts` now carries an additive `backendKind?: BackendKind | null` alongside the deprecated `isGguf`. `BackendKind` is imported from `@/features/chat/types/api` — the same path already used for `GgufVariantDetail` in `pickers.tsx`, so no new layering.

The single `ModelOption.isGguf` read site (`pickers.tsx` line 493, in the `modelGgufIds` memo that feeds `isKnownGgufRepo`) now prefers `backendKind === "gguf"` and falls back to the deprecated boolean — the same fallback pattern F2 used in `chat-page.tsx`. `isGguf` on `ModelOption` carries a JSDoc `@deprecated` note mirroring the backend Pydantic field.

`chat-page.tsx` now populates `backendKind` on each `ModelOption` it builds from the store (reading through `ChatModelSummary.backendKind` which F2 introduced); `isGguf` stays populated for compat with any picker that hasn't migrated yet.

Per the chunk's hard rules, `LoraModelOption` / `HfModelResult` / `LocalModelInfo` / training-flow `ModelOption` reads in `pickers.tsx` (rows 1192, 1199, 1543…) were left alone — they construct `isGguf` locally from `.gguf` suffix heuristics and don't carry a backend summary. They're intentionally out of scope for G1.

## G2 — `ValidateModelResponse.backend_kind`

`ValidateModelResponse` in `studio/backend/models/inference.py` now carries `backend_kind: Optional[BackendKind] = Field(default=None, description=...)`. The field style matches `LoadResponse.backend_kind` verbatim — same `None`-on-cold-start semantics, same "primary source of truth / booleans deprecated" docstring.

The single `/validate` construction site in `routes/inference.py` now derives `backend_kind` from the resolved `ModelConfig`:

| `ModelConfig` flag | → `backend_kind` |
|---|---|
| `is_mlx_vlm` | `"mlx+vlm"` |
| `is_mlx_audio` | `"mlx+audio"` |
| `is_mlx_lora` (or `is_mlx && is_lora`) | `"mlx+lora"` |
| `is_mlx` | `"mlx"` |
| `is_gguf` | `"gguf"` |
| none of the above | `"unsloth"` |

The booleans `is_mlx` / `is_mlx_vlm` / `is_mlx_audio` are now also populated on `ValidateModelResponse` (they weren't before — only `is_gguf` was set pre-G2). This is additive and keeps the wire shape symmetric with `LoadResponse`.

## Test results after Chunk G

- **276 MLX tests pass, 0 xfail.** Delta from Chunk F: +16 new G2 tests in `test_mlx_validate_backend_kind.py`; no regressions in the 260 pre-existing tests.
- 28 tool-call parser tests continue to pass.
- `npm run typecheck` in `studio/frontend` clean.

## New G2 tests

`studio/backend/tests/test_mlx_validate_backend_kind.py` (16 tests):
- **Compat path** — constructing `ValidateModelResponse(is_gguf=True, …)` with no `backend_kind` still works; reading the deprecated boolean still emits a `DeprecationWarning` per the F2 contract.
- **Dump round-trip** — constructing with `backend_kind="gguf"` carries both the enum AND the deprecated boolean through `model_dump()`.
- **Schema contract** — `backend_kind` is NOT `deprecated` in the generated schema even though it sits alongside deprecated peers.
- **Enum variants** — every `BackendKind` literal is accepted; invalid strings are rejected.
- **Route-level** — POST `/inference/validate` against a stubbed `ModelConfig` returns the correct `backend_kind` for each of gguf / mlx / mlx+lora / mlx+vlm / mlx+audio / unsloth; legacy booleans mirror the config deterministically.

## Residual follow-ups after Chunk G

None. The deprecation is uniform across the Pydantic schemas that carry backend identity (`LoadResponse`, `InferenceStatusResponse`, `ValidateModelResponse`, `ModelDetails`) and across the frontend chat + model-selector surfaces. Removal of the legacy booleans remains deferred to a future chunk once telemetry confirms no external readers still hit them.

---

# Chunk H — Real LoRA fixture

Closes the last end-to-end gap in the Phase 6 matrix: until now `mlx_lm.load(..., adapter_path=...)` was only exercised through mocks. Chunk H ships a vendored real MLX LoRA adapter + integration test that actually fuses the adapter onto a base model at load time.

## Items landed

| # | Item | Status |
|---|---|---|
| H1 | Probe — confirm `mlx_lm.lora` trains against a 2-bit quantized MLX base (not explicitly covered by mlx-lm docs) | landed |
| H2 | Dataset fixture — 400 train / 80 valid `dair-ai/emotion` rows as JSONL under `tests/fixtures/lora_dataset/emotion/` | landed |
| H3 | Adapter fixture — 4.8 MB `adapters.safetensors` + 1 KB `adapter_config.json` under `tests/fixtures/lora_adapter/`, trained on H2 against Bonsai 1.7B 2-bit | landed |
| H4 | Integration test — `test_load_bonsai_with_real_lora_adapter` loads the fixture through the real `mlx_lm.load` path and streams tokens through the fused model | landed |
| H5 | Detection test — `test_detect_mlx_adapter_real_fixture` points `_detect_mlx_adapter` at the vendored fixture, proving the plural-`adapters` key survives round-tripping through the mlx-lm trainer | landed |

## Base choice

**Bonsai 1.7B 2-bit** (`/Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit`) — already the Phase 7 speculative-decoding draft, so the existing test suite knows how to locate it via `MLX_TEST_DRAFT_MODEL_PATH`. Probing first: a 10-iter smoke train dropped val loss 6.77 → 1.63 with ~0.6 GB peak mem, so 2-bit QLoRA works on mlx-lm 0.31.2 despite not being explicitly advertised. Full audit trail in `docs/chunk-h-lora/probe.md`.

## Training recipe

```
mlx_lm.lora \
  --train \
  --model /Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
  --data studio/backend/tests/fixtures/lora_dataset/emotion \
  --adapter-path studio/backend/tests/fixtures/lora_adapter \
  --iters 200 --batch-size 2 --num-layers 4 \
  --learning-rate 1e-4 --val-batches 5 \
  --steps-per-report 20 --steps-per-eval 50 \
  --fine-tune-type lora --seed 42
```

- Trainable params: 1.245 M / 1720 M (0.072%) — LoRA rank 8, scale 20, 4 layers.
- Loss trajectory (val): 5.522 → 2.663 (iter 50) → 2.756 (iter 100) → 2.711 (iter 200).
- Wall clock ~25 s on M5 32 GB, peak mem 0.99 GB.
- Output: `adapters.safetensors` (4.8 MB) + `adapter_config.json` (1 KB).

We are NOT chasing classification accuracy — the recipe is just enough to produce a legitimate adapter that exercises Phase 6 loading end-to-end. The fixture's value is **structural**, not semantic.

## Tests added

- `test_mlx_backend_lifecycle.py::test_load_bonsai_with_real_lora_adapter` (gated on `MLX_LORA_AVAILABLE` — requires the 1.7B draft base + fixture files). Loads Bonsai 1.7B 2-bit with `adapter_path=<fixture>`, asserts `is_lora` flips, `adapter_path` round-trips, `load_progress` terminates at `phase="loaded"`, streams 8 tokens through the fused model without error, and resets cleanly on unload.
- `test_mlx_backend_detection.py::test_detect_mlx_adapter_real_fixture` — points the pure-Python detector at the vendored fixture and asserts `True`. Skips cleanly in checkouts where the fixture is absent.

## Test results after Chunk H

- **278 MLX tests pass, 0 xfail.** Delta from Chunk G: +1 integration test (real adapter load/generate) +1 unit test (real fixture detection).
- 28 tool-call parser tests continue to pass.
- Full regression: `pytest tests/test_mlx_*.py tests/test_tool_call_parser.py tests/test_native_context_length.py` — 278 passed in 66 s.

## Fixture inventory

```
studio/backend/tests/fixtures/lora_dataset/emotion/
  train.jsonl                    60 KB  (400 rows)
  valid.jsonl                    11 KB  ( 80 rows)
studio/backend/tests/fixtures/lora_adapter/
  adapters.safetensors           4.8 MB
  adapter_config.json            1 KB
docs/chunk-h-lora/
  probe.md                       probe trail (2-bit decision gate)
  training.log                   full training output
```

Both `adapters.safetensors` and the training log are force-added past the root `.gitignore` patterns (`*.safetensors` / `*.log`) — this is intentional, small (<6 MB total), deterministic-ish test-fixture material.

## Regenerating the fixture

If the fixture needs refreshing (e.g. because mlx-lm's adapter shape changes):

```
# 1. Re-run the probe if the target base changed.
# See docs/chunk-h-lora/probe.md for the canonical smoke-training command.

# 2. Re-emit the JSONL dataset from HF.
cd /tmp/unsloth-mlx-chunkH-lora
/tmp/mlxtest/bin/python -c "
from datasets import load_dataset
import json, pathlib
ds = load_dataset('dair-ai/emotion', 'split')
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
outdir = pathlib.Path('studio/backend/tests/fixtures/lora_dataset/emotion')
outdir.mkdir(parents=True, exist_ok=True)
for split, split_out_name, n in [('train', 'train', 400), ('validation', 'valid', 80)]:
    rows = ds[split].select(range(n))
    with open(outdir / f'{split_out_name}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps({'text': f\"Classify the emotion: {r['text']}\nLabel: {LABELS[r['label']]}\"}) + '\n')
"

# 3. Re-train the adapter (destructive — overwrites the fixture).
/tmp/mlxtest/bin/mlx_lm.lora \
  --train \
  --model /Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
  --data studio/backend/tests/fixtures/lora_dataset/emotion \
  --adapter-path studio/backend/tests/fixtures/lora_adapter \
  --iters 200 --batch-size 2 --num-layers 4 \
  --learning-rate 1e-4 --val-batches 5 \
  --steps-per-report 20 --steps-per-eval 50 \
  --fine-tune-type lora --seed 42

# 4. Drop the per-checkpoint copies mlx-lm writes alongside the final.
rm -f studio/backend/tests/fixtures/lora_adapter/0000*_adapters.safetensors
```

## Residual follow-ups after Chunk H

None. Phase 6 (LoRA) now has both mock-level unit coverage (pre-existing) AND real-adapter end-to-end coverage (this chunk). Every row in the MLX parity matrix has at least one real-hardware test.
