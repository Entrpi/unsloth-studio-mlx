# Chunk H-2 — blockers and known-issue trail

Chunk H-2 expands real-model coverage across the MLX parity matrix.
Everything that landed is marked PASS in `PR_DESCRIPTION.md`; the few
things that didn't are tracked here.

## B1. Ministral-3 template rejects `content=None` on assistant-with-tool_calls

**Resolved in `bf2d4bd9` + `9deafdd4`.**

**Family.** `mlx-community/Ministral-3-3B-Instruct-2512-4bit`.

**Symptom.** `tokenizer.apply_chat_template(...)` raises
`TypeError: object of type 'NoneType' has no len()` when the assistant
message has `tool_calls` and `content=None` — the exact shape our
extractor (`_extract_content_parts(preserve_tool_history=True)`) emits
for "assistant issued a tool call and no accompanying prose".

**Root cause.** Ministral's chat template executes
`message['content'] | length > 0` unconditionally on the assistant
branch before checking `tool_calls`. Jinja's `length` filter surfaces
`len(None)` which explodes. The template expects `content=""` (empty
string) for the tool-call-only case.

**Fix landed.** `_extract_content_parts` now coerces `content=None`
to `content=""` when the assistant message has `tool_calls` populated
AND we're taking the native-iteration branch (template iterates
`message.tool_calls` directly). Templates that iterate `tool_calls`
don't care whether content is `""` or `None` — they render the tool
call off the list and ignore content for that turn — so the coercion
is purely defensive. The synthesis branch (templates that don't
iterate `tool_calls`, like Hermes) always carries a non-empty
`<tool_call>` JSON string in content and is unchanged.

A secondary issue emerged during verification: Ministral's template
also enforces strict user/assistant alternation over content-bearing
turns (tool results and assistant-only-tool-calls are exempt from the
count), and the existing 4-turn fixture (`user → assistant-tc → tool →
user`) tripped that alternation check independently of content-None.
The shared `_build_tool_call_history` fixture is now 3-turn (drops
the trailing `user="Thanks!"`), which represents the canonical OpenAI
tool-call round-trip shape — the next generation would produce the
final assistant response — and renders identically across every
other family (Qwen3.5-4B/35B, Gemma-4, Bonsai, Hermes-3, Llama-3.2).

**Downstream impact.** H2-2 (plain chat smoke) on Ministral is
unaffected — the smoke test doesn't send a `tool_calls`-shaped
assistant turn. The model loads, chats, and unloads cleanly; the only
failure mode is the specific tool-template round-trip.

## B2. `context_length` detection doesn't descend into `text_config`

**Resolved in `2946a07e`.**

**Family.** `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
(and any future VLM-architectured model loaded via the text path).

**Symptom.** `MlxLmBackend.context_length` returned `None` for
Ministral-3 even though the model loaded and generated cleanly.

**Root cause.** Ministral-3's `config.json` is shaped like a VLM —
top-level architecture is `Mistral3ForConditionalGeneration` with
vision / image-processor fields at the top level and
`max_position_embeddings`, `hidden_size`, etc. nested under a
`text_config` dict. Our config reader looked only at the top-level
key. `mlx_lm` consumes the nested config internally so generation
worked, but the Studio-level metadata surface was lossy.

**Fix landed.** Both config reader call sites —
`_read_mlx_max_position_embeddings` in `utils/models/model_config.py`
and the inline read in `core/inference/mlx_lm.py` — now fall back to
`config["text_config"]["max_position_embeddings"]` when the top-level
key is absent or None. Ministral-3 now surfaces
`context_length=262144` (its actual native context).

**Downstream impact.** The H2-2 smoke test previously exempted
Ministral from the `context_length is not None` check with a pointer
back to this doc; that exemption is removed. `mlx_vlm` had a similar
fallback for Qwen3.5-VL since Chunk D, so the VLM path was already
covered — this fix closes the gap on the text-path side.

## B3. Gemma-4 `<|tool_call>` idiom not recognised by the shared parser

**Resolved inline during the H-2 Gemma closure.**

**Family.** `mlx-community/gemma-4-e4b-it-4bit` (and its 31B / MoE
siblings — all share this template dialect).

**Symptom.** Gemma-4 renders assistant tool calls as
`<|tool_call>call:NAME{...}<tool_call|>` with string values delimited
by Gemma's `<|"|>...<|"|>` escaped-quote token pair and bare
(unquoted) object keys. This is distinct from the
`<tool_call>{...}</tool_call>` Qwen / Bonsai / Hermes JSON dialect
and the `<function=…><parameter=…>` Claude/Mistral XML dialect that
`core.inference._tool_call_parser.parse_tool_calls_from_text`
recognised pre-fix.

**Fix landed.** The shared parser gained a third dialect:

- New `_TC_GEMMA_START_RE`, `_TC_GEMMA_QUOTE`, `_TC_GEMMA_KEY_RE`
  constants.
- New `_parse_gemma_dialect(content)` helper that does balanced-brace
  extraction respecting Gemma's quote tokens, then normalises the
  extracted body to JSON in two steps (Gemma-quotes → `"`, bare-key
  quoting via regex) and runs `json.loads`.
- `parse_tool_calls_from_text` tries the new dialect last in `auto`
  mode; `model_family="gemma"` / `"gemma4"` forces it.
- `TOOL_CLOSED_PATS` / `TOOL_ALL_PATS` grew matching strip patterns
  for closed and unclosed Gemma blocks.
- `TOOL_XML_SIGNALS` now lists `"<|tool_call>"` so the speculative
  buffer holds Gemma streams mid-emission just like the other two
  dialects.
- 10 new unit tests in `TestGemmaDialect` cover the happy path,
  chain-of-thought prefix, multi-arg with mixed types, nested
  objects, unclosed calls, multiple calls, forced dialect hint,
  suppression under other dialect hints, strip-closed, strip-final,
  and the signal export.

**Downstream impact.** `tests/test_mlx_gemma_tool_calling.py`'s
aspirational assertion is no longer aspirational — the test now
asserts `tool_name == "get_weather"` and `arguments` contains
`"Paris"` as the primary contract. Gemma-4 users can now drive
real tool-call workflows on the MLX backend.

## B5. MLX-VLM advertised `supports_tools=True` but had no agentic loop

**Resolved in this chunk (see commits following the parity audit).**

**Family.** Any MLX-VLM checkpoint whose chat template mentions
`tool_calls` / `tools` — surfaced first by
`mlx-community/gemma-4-e4b-it-4bit` (a Gemma-4 VLM, routes as VLM via
the `is_mlx_vlm` predicate).

**Symptom.** Loading Gemma 4 E4B through the VLM backend and enabling
the tool-calling toggle in the UI produced plain chat responses — no
tool invocations, no `tool_start` / `tool_end` events. The route
layer's `if using_vlm:` branch only called
`generate_chat_completion`, and `MlxVlmBackend` had no
`generate_chat_completion_with_tools` method at all, despite
`_detect_tools_from_template(self._chat_template)` returning True.

**Root cause.** Phase 9 (Chunk D) explicitly deferred VLM
tool-calling; the backend grew a `supports_tools` property (line 188)
but not the agentic-loop method. The route layer's VLM branch
(around `routes/inference.py:2734`) has no `if payload.enable_tools:`
sub-branch — in contrast with the MLX-LM branch at line 2978.

**Fix landed.**

- Ported `generate_chat_completion_with_tools` to `MlxVlmBackend`,
  mirroring the MLX-LM implementation. Uses `mlx_vlm.stream_generate`
  instead of `mlx_lm.stream_generate`, and accepts an optional
  `image_b64` so tool-calling with an image input is supported (e.g.
  "analyse this chart and call `python` to compute stats").
- Reuses the shared parser (`TOOL_XML_SIGNALS`,
  `parse_tool_calls_from_text`, `strip_tool_markup`) so Gemma-4's
  `<|tool_call>` dialect, Qwen/Bonsai's `<tool_call>` JSON dialect,
  and Claude/Mistral's `<function=...>` XML dialect all parse
  correctly.
- Reuses the content hold-back logic (2026-04-22 MLX-LM fix) so
  partial `<tool_call>` markup doesn't leak to the SSE wire.
- Reuses the `concurrent.futures` tool-execution wrapper with the
  30 s per-invocation cap on `web_search` / `fetch_url` and a
  0.5 s cancel-event poll.
- Prompt-injects the tool schema as a synthetic system message
  (`_render_prompt(tools=...)` already did this) rather than
  threading `tools=` through `apply_chat_template` — the VLM
  template path doesn't reliably accept the kwarg across models,
  and the JSON tool_calls output from the model is still parsed by
  the shared parser regardless.
- Route `if using_vlm:` branch gained a parallel `if payload.enable_tools:`
  sub-branch that dispatches to a new `_mlx_vlm_agentic_stream`
  SSE driver (mirror of `_mlx_agentic_stream` with image passthrough).
- `_build_tool_use_nudge` is applied to the VLM system prompt the
  same way MLX-LM does it, so small tool-capable VLMs (Gemma 4 E4B)
  reliably call tools rather than replying "I can't do that".

**Downstream impact.** Gemma 4 E4B VLM users can now drive tool
workflows. The existing `test_mlx_gemma_tool_calling.py` contract
(`tool_name == "get_weather"` and `"Paris"` in arguments) is mirrored
for the VLM backend in
`tests/test_mlx_vlm_gemma_tool_calling.py`. Non-goals for this chunk:
client-side tools passthrough on VLM and Anthropic `/v1/messages` VLM
tool-calling (both flagged P1 in `parity-audit.md`).

## B4. `@pytest.mark.slow` not yet a convention in this suite

**Observation.** `pyproject.toml` doesn't register a `slow` mark, and
no other MLX test gates behind one today. The MoE / GLM-VLM tests
gate behind `MLX_SLOW_TESTS=1` instead (explicit env var), which is
local to this chunk. A future chunk could promote this to a
configured `slow` marker if the convention catches on.

---

Nothing else was encountered as a hard blocker. All other matrix
cells (H2-1 for Hermes/Llama, H2-2 for all three, H2-3 MoE, H2-4
GLM-VLM, H2-5 Whisper) closed to PASS.
