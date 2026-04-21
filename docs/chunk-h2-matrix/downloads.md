# Chunk H-2 — new local MLX models

All four models were fetched via `hf download` into the existing
`~/.lmstudio/models/<org>/<name>` layout. Downloads are deterministic
— any future re-sync can use the same commands below.

## Sizes

| Local path | Role | Size |
|---|---|---|
| `~/.lmstudio/models/mlx-community/Hermes-3-Llama-3.2-3B-bf16` | Text + tool-template (synthesis branch) | ~6.0 GB |
| `~/.lmstudio/models/mlx-community/Ministral-3-3B-Instruct-2512-4bit` | Text + tool-template (native tool_calls) | ~2.6 GB |
| `~/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit` | Text + tool-template (Llama3 JSON dialect) | ~1.7 GB |
| `~/.lmstudio/models/mlx-community/whisper-small-mlx-4bit` | ASR (mlx-whisper) | ~187 MB |

Already local and reused by the matrix expansion:

- `~/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit` (~17 GB) — MoE.
- `~/.lmstudio/models/lmstudio-community/GLM-4.6V-Flash-MLX-8bit` (~10 GB) — non-Qwen VLM.

## Fetch commands

Run these in parallel (each one streams independently to its own
local-dir — HF's cache lock is per-repo so they don't contend):

```bash
/tmp/mlxtest/bin/hf download mlx-community/Hermes-3-Llama-3.2-3B-bf16 \
  --local-dir /Users/ent/.lmstudio/models/mlx-community/Hermes-3-Llama-3.2-3B-bf16 &

/tmp/mlxtest/bin/hf download mlx-community/Ministral-3-3B-Instruct-2512-4bit \
  --local-dir /Users/ent/.lmstudio/models/mlx-community/Ministral-3-3B-Instruct-2512-4bit &

/tmp/mlxtest/bin/hf download mlx-community/Llama-3.2-3B-Instruct-4bit \
  --local-dir /Users/ent/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit &

/tmp/mlxtest/bin/hf download mlx-community/whisper-small-mlx-4bit \
  --local-dir /Users/ent/.lmstudio/models/mlx-community/whisper-small-mlx-4bit &

wait
```

## Why these four

- **Hermes-3-Llama-3.2-3B-bf16** — Nous Research Hermes chat-ML style
  with **no native `tool_calls` iteration** in its chat template. Forces
  Chunk F1's content-synthesis fallback to fire (`<tool_call>` XML injected
  into `content`). No other local model exercised this branch end-to-end.
- **Ministral-3-3B-Instruct-2512-4bit** — Mistral's 2512 release. Uses
  `[AVAILABLE_TOOLS]` / `[TOOL_CALLS]` / `[TOOL_RESULTS]` dialect and
  enforces strict user/assistant alternation. This is a different
  template family from any previously-probed model (Gemma uses
  `<|tool_call>`, Qwen/Bonsai use `<tool_call>`, Llama uses inline
  JSON).
- **Llama-3.2-3B-Instruct-4bit** — Meta's Llama-3.2 chat format with
  `<|python_tag|>` / `<|start_header_id|>ipython<|end_header_id|>`
  ipython-flavoured tool schema. Renders assistant tool_calls as
  inline JSON `{"name": ..., "parameters": ...}`.
- **whisper-small-mlx-4bit** — 4-bit MLX port of OpenAI Whisper-small.
  Small enough to load at CI pace while large enough that a 16 kHz TTS
  round-trip against it produces recognisable words. Chunk E-9 shipped
  the `transcribe_with_whisper` method mock-only; this model makes the
  path end-to-end testable.

## Decommissioning

If disk pressure forces a cleanup: these downloads are reproducible
from the commands above and the test suite skips cleanly when any
individual path is missing.
