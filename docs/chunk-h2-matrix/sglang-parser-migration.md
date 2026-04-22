# Decision: vendor SGLang's detector framework for MLX tool-call parsing

**Status**: deferred. Pick up after streaming Phases 2–3 land.

## Why this exists

Our MLX backend carries a hand-rolled 5-dialect parser
(`studio/backend/core/inference/_tool_call_parser.py`, ~881 LoC).
It has real gaps (Harmony/gpt-oss, Llama 3.1/3.2 `<|python_tag|>`,
Mistral Tekken, Command-R, DeepSeek-R1) and load-bearing workarounds
(stream hold-back via `TOOL_XML_SIGNALS.find` truncation, the
`_global_tool_counter` fix from commit `2e86a974` for duplicate
`toolCallId`s). SGLang's `python/sglang/srt/function_call/` ships a
well-tested `BaseFormatDetector` ABC plus ~20 per-family detectors
(Apache-2.0) that solve the same problems cleanly.

## Decision

**MLX-only migration** — Option 1 in the scoping research. Rejected:
whole-system unification (blocked on llama-server's `--jinja` dropping
raw content when it emits structured `tool_calls`, so SGLang has
nothing to parse on the GGUF path; unifying would mean giving up
llama.cpp's constrained-generation quality).

## Plan (4 staged commits)

- **Phase A** (~1-2 days): vendor `base_format_detector.py`,
  `core_types.py`, `utils.py`, `HermesDetector` into
  `studio/backend/core/inference/_sglang_detectors/` with SPDX
  Apache-2.0 headers. Keep `_tool_call_parser.py` as a facade so the
  40+ current call sites stay source-compatible. Feature-flag via
  `STUDIO_USE_SGLANG_PARSER=1`. Covers Qwen / Hermes / Bonsai.
- **Phase B**: add `Gemma4Detector` + `Glm4MoeDetector`. Keep
  `extract_channel_thought` as a helper wrapped around the detector
  (SGLang's Gemma4 doesn't handle `<|channel>thought` blocks).
- **Phase C**: port Claude-XML and loose-JSON as local
  `BaseFormatDetector` subclasses. SGLang has no equivalents; we own
  these dialects. Retire old dialect implementations.
- **Phase D** (optional — the one targeted unification win): vendor
  `GptOssDetector` + `HarmonyParser`. For GGUF, detect gpt-oss family
  at load time in `llama_cpp.py:1428` and launch without `--jinja` so
  both backends flow through the same Harmony parser. This is the
  only family llama.cpp doesn't handle natively; ~250 extra LoC.

## Key wins over current code

- Drops `_global_tool_counter`: SGLang uses integer `tool_index` +
  UUID synthesis, so duplicate IDs are impossible by construction.
- Drops `TOOL_XML_SIGNALS.find` hold-back loop: SGLang's
  `_buffer` + `_ends_with_partial_token` state machine handles it.
- Adds OpenAI-reference streaming semantics: name-first, then
  argument-diff chunks via `_find_common_prefix`, replacing our
  ad-hoc 8-char splitting at `routes/inference.py:5978`.
- Coverage of ~20 model families out of the box.

## Costs

- ~+1700 net LoC (vendored, upstream-maintained).
- ~60/73 parser tests pass unchanged; ~10 need assertion relaxation;
  ~3 may need rewriting on malformed-input edge cases.
- New runtime dep: `partial_json_parser` (pure-Python, MIT, ~3 KLOC).
- Minor dependency shim: duck-typed `Tool` / `ToolChoice` Pydantic
  models matching what SGLang's detectors read (~60 LoC).

## One semantic change to flag during rollout

SGLang's `HermesDetector` validates tool names against the `tools`
list and drops unknowns (`base_format_detector.py:79-82`). Ours
doesn't. Gated behind the vendored `SGLANG_FORWARD_UNKNOWN_TOOLS`
env-var knob in the shim.

## License

AGPL-3.0 (Studio) ← Apache-2.0 (SGLang) is cleanly one-way
compatible. Vendored files carry SPDX-License-Identifier headers +
a pinned-commit provenance comment.
`studio/backend/core/inference/_sglang_detectors/LICENSE` contains
the Apache-2.0 full text. No CLA required (no upstream contribution
back).

## When to revisit

Any of the following should reopen this bookmark:

- A user reports tool-calling failure on a family we don't cover
  (gpt-oss/Harmony is the most likely near-term trigger).
- Upstream SGLang ships a detector for a model we want to support
  faster than we can hand-roll the dialect.
- We accumulate a third load-bearing workaround on the current
  parser's streaming semantics.

See the scoping research in the conversation for full file:line
citations and the dialect-mapping table.
