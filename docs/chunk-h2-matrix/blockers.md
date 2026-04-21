# Chunk H-2 — blockers and known-issue trail

Chunk H-2 expands real-model coverage across the MLX parity matrix.
Everything that landed is marked PASS in `PR_DESCRIPTION.md`; the few
things that didn't are tracked here.

## B1. Ministral-3 template rejects `content=None` on assistant-with-tool_calls

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

**Why we didn't fix this in H-2.** The extractor is intentionally
shape-preserving: `content=None` means "the model emitted no text
tokens", which is distinct from `content=""` ("the model emitted an
explicit empty string"). Coercing None → "" at the extractor level
would ripple into the Chunks F/G code paths that already do the right
thing for Qwen/Bonsai/Gemma/Hermes. The correct fix is a per-template
coercion hook (something like "if the template is known to choke on
None, normalise to ''") or an extractor-level flag. That belongs in a
future chunk — Chunk H-2 is coverage expansion, not behaviour change.

**Current status.** The H2-1 parametrised row for Ministral is marked
`pytest.mark.xfail(strict=False, raises=(TypeError, AssertionError))`
with an explicit reason pointer back to this doc.

**What would close this.** A small change to
`_extract_content_parts` that looks up `tokenizer.chat_template` (or
accepts an explicit coercion flag) and, when the template matches a
known-None-hostile family, emits `content=""` instead of `content=None`
on assistant turns that have tool_calls. Alternatively: detect the
TypeError pattern and retry. Either is ~30 lines + one new test. Out of
scope here.

**Downstream impact.** H2-2 (plain chat smoke) on Ministral is
unaffected — the smoke test doesn't send a `tool_calls`-shaped
assistant turn. The model loads, chats, and unloads cleanly; the only
failure mode is the specific tool-template round-trip.

## B2. `@pytest.mark.slow` not yet a convention in this suite

**Observation.** `pyproject.toml` doesn't register a `slow` mark, and
no other MLX test gates behind one today. The MoE / GLM-VLM tests
gate behind `MLX_SLOW_TESTS=1` instead (explicit env var), which is
local to this chunk. A future chunk could promote this to a
configured `slow` marker if the convention catches on.

---

Nothing else was encountered as a hard blocker. All other matrix
cells (H2-1 for Hermes/Llama, H2-2 for all three, H2-3 MoE, H2-4
GLM-VLM, H2-5 Whisper) closed to PASS.
