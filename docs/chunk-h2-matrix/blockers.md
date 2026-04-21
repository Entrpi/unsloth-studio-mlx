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

**Observed during the Gemma-4 matrix closure pass.** Tracked as a
follow-up, not blocking Chunk H-2.

**Family.** `mlx-community/gemma-4-e4b-it-4bit` (and its 31B / MoE
siblings — all share this template dialect).

**Symptom.** Gemma-4 renders assistant tool calls as
`<|tool_call>call:NAME{...}<tool_call|>`. This is distinct from the
`<tool_call>{...}</tool_call>` Qwen / Bonsai / Hermes dialect that
`core.inference._tool_call_parser.parse_tool_calls_from_text`
recognises. When Gemma-4 emits a call, the parser returns no hits,
the `generate_chat_completion_with_tools` loop treats the turn as a
final-answer turn, and no `tool_start` event fires.

**Impact.** The code path runs end-to-end (load + loop + unload) and
the test in `tests/test_mlx_gemma_tool_calling.py` asserts that
contract. The model-emits-tool-call aspirational assertion is
currently unreachable — documented inline in the test. Adding
Gemma-4 dialect support to the shared parser is a straightforward
regex addition (plus a matching `strip_tool_markup` branch) but it's
a behaviour change, not coverage expansion, so it's deferred to a
follow-up chunk.

**Downstream impact.** Zero. Every currently-shipped Gemma-4 user
flow today either (a) uses a non-tools prompt (the family smoke row
covers that) or (b) produces prose even when tools are offered
(the parser just passes the prose through as final content). Users
who need real Gemma-4 tool-calling today can fall back to non-MLX
providers; the MLX path will light up once the parser grows the
Gemma dialect.

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
