# unsloth-studio-mlx

> A development fork of [`unslothai/unsloth`](https://github.com/unslothai/unsloth). Everything here is destined for upstream via a single PR. The story of why it exists follows — then the upstream README below.

## How this started

I wanted to put Ternary Bonsai 8B through its paces. LM Studio had the MLX weights loaded in minutes, but what I really wanted was a UI with native web search, agentic tool calling, and the flexibility to push on the internals as models evolve — Unsloth Studio felt like the right fit.

The fit turned out to be aspirational. Studio had the *shape* for this — a structurally extensible backend (`ModelConfig`, the subprocess orchestrator, the OpenAI-compat route), solid GGUF support through `llama-server`, the right primitives — but zero MLX code. No `mlx_lm.py`, no `mlx_vlm.py`, no `mlx_audio.py`, no `is_mlx` detection, no quant variant anywhere, no hook for Apple Silicon's unified memory, no path through the picker for an MLX repo name. Not a configuration gap you could patch in a weekend; the backend physically couldn't load an MLX checkpoint, and the frontend had nothing to show for one even if it could. The three dozen surfaces that would need to light up for an MLX model to feel native in Studio — picker chip, variant submenu, context length, KV dtype selector, tool-call dialect handling, reasoning extraction, streaming progress, GPU telemetry — each needed building, most from scratch.

So I built it. Across ten phases (chunks A–H plus an H2 matrix-coverage closure) an in-process `MlxLmBackend` peer to `LlamaCppBackend` came together, then layered on top:

- **Chunk A** — sampler fidelity (top-k / top-p / min-p / repetition penalty / logit bias) + reasoning (`<think>` passthrough, `enable_thinking` template threading)
- **Chunks B–C** — tool calling (a dialect parser, an agentic loop with hold-back and cancellation)
- **Chunks D–E** — `MlxVlmBackend` (vision-language) and `MlxAudioBackend` (LFM2.5-Audio TTS, `mlx-whisper` ASR)
- **Chunks F–G** — quantized KV cache (`q8_0` / `q4_0`), LoRA adapters, speculative decoding with a draft model, remote HF pulls with progress and memory preflight, plus a `backend_kind` enum migration that retired ad-hoc `is_mlx` / `is_gguf` booleans across schemas, route layer, and frontend
- **Chunk H + H2** — the real-model matrix: end-to-end coverage on a non-Qwen VLM (GLM-4.6V-Flash), a MoE (Qwen3.5-35B-A3B), real `mlx-whisper` ASR, a trained LoRA via `mlx_lm.lora`, and non-Qwen text families (Hermes, Llama-3.2, Ministral). Anthropic `/v1/messages` and OpenAI tool passthrough wired through MLX too — not just GGUF. Each phase shipped with its own tests; the suite added ~425 cases.

By the time the chunks closed out: **126 commits and ~28,000 lines across 100 files** versus the upstream branchpoint. The OpenAI-compat route nearly quintupled, from ~1,000 lines to 4,855. Three net-new backends — `MlxLmBackend` (2,467 lines), `MlxVlmBackend` (1,906), `MlxAudioBackend` (578) — peering `LlamaCppBackend`. An 881-line tool-call parser covering five dialects. A 1,346-line telemetry subsystem.

That's the base layer. What this README is really about is everything that happened *on top* while I dogfooded it — bugs that surfaced once real models were running, UX gaps that moment-by-moment use kept exposing, and the diagnostic infrastructure that turned out to be the most visible UX improvement of the whole chunk.

## What broke while I tested my own work

**Context length silently clamped to 4096.** First thing I caught after loading Ternary Bonsai: max context was stuck at 4k regardless of what the model actually supported. The MLX backend reported its native context fine — the clamp lived entirely in the frontend's runtime hook, where a legacy `max_seq_length=4096` default applied to every non-GGUF backend. GGUF had a "0 = use native" branch nothing was routing MLX through. Fixed by widening that branch via the new `backend_kind` from Chunk G.

**The picker labelled every MLX checkpoint as "Local".** Where GGUF rows showed quant + size, MLX rows just said "Local" — because when I added MLX detection there was nothing for the frontend to display. I wired the parity surface in stages: detection helpers for `-MLX-Nbit` / `mlx-community/*` naming, then a variant submenu enumerating sibling quants (2 / 4 / 5 / 6 / 8-bit / bf16) by querying Hugging Face, then chip detail consistency (`MLX · 4bit · 19.5 GB`), and finally bubbling MLX results in dropdown search alongside GGUF on Apple Silicon. Catching one related bug along the way: the "Downloaded" section was hiding *all* non-GGUF cached repos in chat-only mode, because the gate had been written as a blanket `!chatOnly` to keep plain PyTorch weights out. MLX quants got swept up. Fixed.

**Each new model surfaced a new tool-call dialect.** Ternary Bonsai worked on the first-cut parser because it emits Qwen-style `<tool_call>{JSON}</tool_call>`. Then I tried Gemma 4 E4B and it refused to call tools at all — Gemma emits `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>` with unquoted keys and escaped-quote string delimiters, an entirely different dialect. Wrote a Gemma-4 parser. Some models bled bare `{name, arguments}` JSON through as content; added a loose-JSON envelope dialect. Then GLM-4.6V-Flash showed up with yet another shape, `<tool_call>name\n<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`. Another parser. Plus Gemma's channel-thinking blocks (`<|channel>thought…<channel|>`) had to be extracted to `reasoning_content` so they didn't double-emit on the next agentic iteration. Plus a hardened tool-execution wrapper (cancel_event propagation, 30-second cap on `web_search`, URL-sentinel handling for models that emit `url="None"`).

The Gemma arc alone was six rounds of "now it gets further but breaks differently" before the chat-template root cause surfaced and the search actually returned results to the model.

By the end I had five tool-call dialects and an ~900-line parser. Once it was stable I commissioned an architectural review — [it confirmed](docs/chunk-h2-matrix/sglang-parser-migration.md) what I already suspected: everyone else (vLLM, SGLang, mistral.rs) has converged on the same per-family-parser architecture. I bookmarked vendoring SGLang's framework as a follow-up rather than slip the chunk to chase it.

**Then the UI got stuck after the second tool call, and the backend looked fine.** This took the longest. I kept hitting it across rounds — backend logs clean, GPU went idle, but the UI never updated. Eventually it became obvious it'd be faster to drive the browser headlessly and watch what the React app was actually doing. Building a Chrome-preview driver — DevTools open, an SSE-tee capturing the wire, scripted clicks, error-boundary introspection — surfaced the actual error in seconds:

> `Duplicate key toolCallId-call_0 in tapResources`

The tool-call parser was assigning IDs per-invocation (`call_0`, `call_1`, ...), and the agentic loop re-invoked it fresh every iteration. So iter=0's first tool call and iter=1's first tool call both arrived at assistant-ui with `toolCallId: "call_0"`. assistant-ui keys its active-tool-parts map by that ID; the second collision crashed it silently. The error boundary swallowed the visible crash, but the state tree corrupted mid-stream and iter=2+ content got dropped on the floor.

Before the root cause surfaced I'd hypothesised an SSE transport issue and built a 5-second keepalive (racing the generator against an `asyncio.shield`-wrapped timeout), with a 15-second frontend inactivity watchdog in the same change, then a same-day thread-safety hotfix when the first keepalive implementation spun up concurrent `next(gen)` calls on the same Python generator. That work stands on its own merits — long silent periods *are* a real failure mode for proxies and browser idle heuristics — but it wasn't the bug. The actual fix is a three-line global tool-counter that makes IDs unique across iterations. The Chrome-preview harness I built to find it (the SSE-tee, the DOM snapshot loop, the error-boundary probe) turned out to be far more valuable than the planned feature.

**The UI still gave no signal during the long pauses.** Even with the transport hardened, tool execution and the prompt re-eval after it produced zero bytes for 30+ seconds. Nothing felt broken to the backend; everything felt frozen at the keyboard. So I added structured progress events — `{"type":"progress","phase":"prompt_eval"|"generating","iter":N}` — and rendered them as chips above the composer ("Re-reading conversation…" / "Generating…"). Additive to the SSE event set; external OpenAI clients ignore unknown event types. The chat suddenly feels responsive during what used to be dead air.

**Then I lifted the harness into a product feature.** The DevTools-driven debugging had given me live visibility into what the backend was doing; users would want the same, just rendered as UI chrome instead of console output. So I added a separate `/ws/telemetry` WebSocket — outlives chat turns, fans out to multiple tabs — carrying GPU stats, session state, and pre-filter token counts. The "pre-filter" distinction matters: the counter ticks raw tokens from the generator *before* hold-back mutation, so during reasoning when the parser is holding back the visible text isn't growing but the chip shows `242 tok · 28.4 t/s · iter 0: 287 tok` and you know the model is working. Token telemetry now flows from MLX *and* the three GGUF streaming paths (verbatim passthrough, tool-loop, plain stream) — every chat path lights the chip.

The first GPU sampler used MLX's active-memory counter as a binary 100/0 proxy — inactive vs active. Watching it for ten seconds I saw it stuck at 100% the whole time and decided it had to go. `mactop` uses Apple's private `IOReport.framework` via CGO; I worked through the Go binding and wrote a pure-Python ctypes equivalent against `/usr/lib/libIOReport.dylib`. No sudo, no entitlements. Validated against `powermetrics` in spot checks. With severity colour bands at 50% and 85% the chart actually says something: idle while Chrome composits the UI, spikes to red when inference starts, drops back.

Studio's deployment matrix isn't just Apple Silicon, so I covered the rest: `pynvml` for NVIDIA (Linux x86_64, Linux aarch64 — the DGX Spark case — WSL, and Windows in one package), AMD sysfs (`gpu_busy_percent`), Intel sysfs engine counters, and Windows PDH (`\GPU Engine`) as the any-GPU fallback. Each backend is a 30–60 line probe + reader behind a platform-aware dispatch chain. And because a chart showing frozen data is worse than a chart showing "disconnected", a `WifiOff` badge replaces the chip when the telemetry stream drops or goes silent for >10s.

**The debugging harness is now a product feature.** The token counter, the GPU sparkline, the progress chips, the WebSocket telemetry endpoint — they all came out of chasing the `Duplicate key` ghost. They ended up being the most-visible UX improvements in this chunk, because they answer the question every user of a local inference tool has looking at a spinning icon: *is anything actually happening?* The answer is now: *yes, at 28.4 t/s on a GPU that's 97% busy in iter=2 of the agentic loop, 1234 tokens produced.*

## What's in the fork (tl;dr)

If the story above was too long, here's the shape of it:

- **Full MLX backend stack** — `MlxLmBackend`, `MlxVlmBackend`, `MlxAudioBackend` peering `LlamaCppBackend`. Detection, sampler fidelity, reasoning, agentic tool loop, KV quant, LoRA, speculative decoding, remote HF pulls with memory preflight.
- **Tool-call coverage across 5 dialects** — Qwen-JSON / Claude-XML / Gemma-4 / loose-JSON envelope / GLM-4. Channel-thought extraction. Globally-unique tool-call IDs across iterations. Hardened execution (cancel propagation, web_search caps, URL sentinels).
- **Streaming health** — SSE keepalive + frontend inactivity watchdog. `_fetch_page_text` with wall-clock deadline to fix a CLOSE_WAIT leak found by accident.
- **Structured progress events** — `prompt_eval` / `generating` chips during silent periods.
- **Live telemetry WebSocket** — `/ws/telemetry` carries GPU stats, session state, pre-filter token counts; outlives chat turns, multi-tab fan-out. Token counts flow from MLX *and* GGUF (verbatim passthrough + tool-loop + plain stream).
- **Cross-platform GPU sparkline** — IOReport on Apple Silicon, `pynvml` on NVIDIA anywhere, AMD/Intel sysfs on Linux, PDH on Windows. Severity colour bands. Disconnected badge.
- **Model picker parity** — MLX quants surface alongside GGUF in search, variant submenu for sibling repos, chip shows `MLX · 4bit · 19.5 GB` from any source (HF cache, LM Studio, custom paths).
- **Anthropic `/v1/messages` + OpenAI `/v1/chat/completions` tool passthrough** for MLX too — not just GGUF.
- **Real-model matrix coverage** — non-Qwen VLM (GLM-4.6V), MoE (Qwen3.5-35B-A3B), real `mlx-whisper`, trained LoRA via `mlx_lm.lora`, Hermes/Llama-3.2/Ministral text families.
- **Every new surface is feature-flagged** — six env-var knobs documented with rollback values in [`docs/env-vars.md`](docs/env-vars.md).

## Honest status

126 commits on `mlx-studio-enablement` vs the upstream branchpoint (122 chunk commits + README rewrites + a merge with upstream). Full backend test suite adds ~425 tests; 918 pass, 19 environmental failures that match `unslothai/unsloth:main` exactly (flash-attn Linux, CUDA-gated GPU tests, pre-existing vision cache). Frontend TypeScript + build clean. Manual end-to-end verification via Chrome preview spans the models most people will actually want to run on this stack today: **Qwen3.5** and **Qwen3.6** (text + MoE variants), **Gemma 4** (E2B / E4B), **GLM-4.6V-Flash** for vision, and **Ternary Bonsai 8B** as the original target — with Qwen3-1.7B-GGUF kept as a regression control on the global-tool-counter fix.

Both this README and the recent merge were preceded by audits I commissioned — a commit-log audit and a chat-history audit — to keep claims accurate and credit honest. Each surfaced rounded-off numbers and misattributed fixes that I corrected before publishing.

Deferred follow-ups are documented, not silently skipped:

- [`docs/chunk-h2-matrix/parity-audit.md`](docs/chunk-h2-matrix/parity-audit.md) — smaller P1/P2 gaps (VLM client-tools passthrough, MlxAudio `hf_variant`, GGUF LoRA, etc.)
- [`docs/chunk-h2-matrix/sglang-parser-migration.md`](docs/chunk-h2-matrix/sglang-parser-migration.md) — the next-generation parser architecture

## Relationship to upstream

Canonical Unsloth development lives at [`unslothai/unsloth`](https://github.com/unslothai/unsloth). The `main` branch here is a pristine mirror; `mlx-studio-enablement` (the default branch you're reading this on) is where the work is. License is inherited unchanged — Studio source stays AGPL-3.0, upstream paths keep their licenses. I pulled in the four upstream commits that landed during the chunk (`#5122` OpenAI tools on `/v1/responses`, `#5128` images on Anthropic `/v1/messages`, `#5129` TRL bool-coercion, plus a `model_mappings.py` update) and reconciled one merge conflict in the Anthropic-route tool dispatch. When upstream reviewers are ready, I'll rebase on latest `main` and open one PR.

Install instructions are the same as upstream — see the Unsloth README below.

---

<!-- ↓ Upstream Unsloth README begins below ↓ -->


<h1 align="center" style="margin:0;">
  <a href="https://unsloth.ai/docs"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
    <img alt="Unsloth logo" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="80" style="max-width:100%;">
  </picture></a>
</h1>
<h3 align="center" style="margin: 0; margin-top: 0;">
Unsloth Studio lets you run and train models locally.
</h3>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-install">Quickstart</a> •
  <a href="#-free-notebooks">Notebooks</a> •
  <a href="https://unsloth.ai/docs">Documentation</a>
</p>
<br>
<a href="https://unsloth.ai/docs/new/studio">
<img alt="unsloth studio ui homepage" src="https://github.com/user-attachments/assets/53ae17a9-d975-44ef-9686-efb4ebd0454d" style="max-width: 100%; margin-bottom: 0;"></a>

## ⚡ Get started

#### macOS, Linux, WSL:
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```
#### Windows:
```powershell
irm https://unsloth.ai/install.ps1 | iex
```
#### Community:

- [Discord](https://discord.gg/unsloth)
- [𝕏 (Twitter)](https://x.com/UnslothAI)
- [Reddit](https://reddit.com/r/unsloth)

## ⭐ Features
Unsloth Studio (Beta) lets you run and train text, [audio](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning), [embedding](https://unsloth.ai/docs/new/embedding-finetuning), [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) models on Windows, Linux and macOS.

### Inference
* **Search + download + run models** including GGUF, LoRA adapters, safetensors
* **Export models**: [Save or export](https://unsloth.ai/docs/new/studio/export) models to GGUF, 16-bit safetensors and other formats.
* **Tool calling**: Support for [self-healing tool calling](https://unsloth.ai/docs/new/studio/chat#auto-healing-tool-calling) and web search
* **[Code execution](https://unsloth.ai/docs/new/studio/chat#code-execution)**: lets LLMs test code in Claude artifacts and sandbox environments
* [Auto-tune inference parameters](https://unsloth.ai/docs/new/studio/chat#auto-parameter-tuning) and customize chat templates.
* We work directly with teams behind [gpt-oss](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune#unsloth-fixes-for-gpt-oss), [Qwen3](https://www.reddit.com/r/LocalLLaMA/comments/1kaodxu/qwen3_unsloth_dynamic_ggufs_128k_context_bug_fixes/), [Llama 4](https://github.com/ggml-org/llama.cpp/pull/12889), [Mistral](models/tutorials/devstral-how-to-run-and-fine-tune.md), [Gemma 1-3](https://news.ycombinator.com/item?id=39671146), and [Phi-4](https://unsloth.ai/blog/phi4), where we’ve fixed bugs that improve model accuracy.
* Upload images, audio, PDFs, code, DOCX and more file types to chat with.
### Training
* Train and RL **500+ models** up to **2x faster** with up to **70% less VRAM**, with no accuracy loss.
* Custom Triton and mathematical **kernels**. See some collabs we did with [PyTorch](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) and [Hugging Face](https://unsloth.ai/docs/new/faster-moe).
* **Data Recipes**: [Auto-create datasets](https://unsloth.ai/docs/new/studio/data-recipe) from **PDF, CSV, DOCX** etc. Edit data in a visual-node workflow.
* **[Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)** (RL): The most efficient [RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) library, using **80% less VRAM** for GRPO, [FP8](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) etc.
* Supports full fine-tuning, RL, pretraining, 4-bit, 16-bit and, FP8 training.
* **Observability**: Monitor training live, track loss and GPU usage and customize graphs.
* [Multi-GPU](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) training is supported, with major improvements coming soon.

## 📥 Install
Unsloth can be used in two ways: through **[Unsloth Studio](https://unsloth.ai/docs/new/studio/)**, the web UI, or through **Unsloth Core**, the code-based version. Each has different requirements.

### Unsloth Studio (web UI)
Unsloth Studio (Beta) works on **Windows, Linux, WSL** and **macOS**.

* **CPU:** Supported for Chat and Data Recipes currently
* **NVIDIA:** Training works on RTX 30/40/50, Blackwell, DGX Spark, Station and more
* **macOS:** Currently supports chat and Data Recipes. **MLX training** is coming very soon
* **AMD:** Chat + Data works. Train with [Unsloth Core](#unsloth-core-code-based). Studio support is out soon.
* **Coming soon:** Training support for Apple MLX, AMD, and Intel.
* **Multi-GPU:** Available now, with a major upgrade on the way

#### macOS, Linux, WSL:
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```
#### Windows:
```powershell
irm https://unsloth.ai/install.ps1 | iex
```

#### Launch
```bash
unsloth studio -H 0.0.0.0 -p 8888
```

#### Update
To update, use the same install commands as above. Or run (does not work on Windows):
```bash
unsloth studio update
```

#### Docker
Use our [Docker image](https://hub.docker.com/r/unsloth/unsloth) ```unsloth/unsloth``` container. Run:
```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 8000:8000 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
  ```

#### Developer, Nightly, Uninstall
To see developer, nightly and uninstallation etc. instructions, see [advanced installation](#-advanced-installation).

### Unsloth Core (code-based)
#### Linux, WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```
#### Windows:
```powershell
winget install -e --id Python.Python.3.13
winget install --id=astral-sh.uv  -e
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```
For Windows, `pip install unsloth` works only if you have PyTorch installed. Read our [Windows Guide](https://unsloth.ai/docs/get-started/install/windows-installation).
You can use the same Docker image as Unsloth Studio.

#### AMD, Intel:
For RTX 50x, B200, 6000 GPUs: `uv pip install unsloth --torch-backend=auto`. Read our guides for: [Blackwell](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) and [DGX Spark](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth). <br>
To install Unsloth on **AMD** and **Intel** GPUs, follow our [AMD Guide](https://unsloth.ai/docs/get-started/install/amd) and [Intel Guide](https://unsloth.ai/docs/get-started/install/intel).

## 📒 Free Notebooks

Train for free with our notebooks. You can use our new [free Unsloth Studio notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb) to run and train models for free in a web UI.
Read our [guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide). Add dataset, run, then deploy your trained model.

| Model | Free Notebooks | Performance | Memory use |
|-----------|---------|--------|----------|
| **Gemma 4 (E2B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb)               | 1.5x faster | 50% less |
| **Qwen3.5 (4B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb)               | 1.5x faster | 60% less |
| **gpt-oss (20B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)               | 2x faster | 70% less |
| **Qwen3.5 GSPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb)               | 2x faster | 70% less |
| **gpt-oss (20B): GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)               | 2x faster | 80% less |
| **Qwen3: Advanced GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)               | 2x faster | 70% less |
| **embeddinggemma (300M)**    | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb)               | 2x faster | 20% less |
| **Mistral Ministral 3 (3B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_VL_(3B)_Vision.ipynb)               | 1.5x faster | 60% less |
| **Llama 3.1 (8B) Alpaca**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2x faster | 70% less |
| **Llama 3.2 Conversational**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)               | 2x faster | 70% less |
| **Orpheus-TTS (3B)**     | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5x faster | 50% less |

- See all our notebooks for: [Kaggle](https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks), [GRPO](https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks), [TTS](https://unsloth.ai/docs/get-started/unsloth-notebooks#text-to-speech-tts-notebooks), [embedding](https://unsloth.ai/docs/new/embedding-finetuning) & [Vision](https://unsloth.ai/docs/get-started/unsloth-notebooks#vision-multimodal-notebooks)
- See [all our models](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [all our notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- See detailed documentation for Unsloth [here](https://unsloth.ai/docs)

## 🦥 Unsloth News
- **Qwen3.6**: Qwen3.6-35B-A3B can now be trained and run in Unsloth Studio. [Blog](https://unsloth.ai/docs/models/qwen3.6)
- **Gemma 4**: Run and train Google’s new models directly in Unsloth. [Blog](https://unsloth.ai/docs/models/gemma-4)
- **Introducing Unsloth Studio**: our new web UI for running and training LLMs. [Blog](https://unsloth.ai/docs/new/studio)
- **Qwen3.5** - 0.8B, 2B, 4B, 9B, 27B, 35-A3B, 112B-A10B are now supported. [Guide + notebooks](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Train **MoE LLMs 12x faster** with 35% less VRAM - DeepSeek, GLM, Qwen and gpt-oss. [Blog](https://unsloth.ai/docs/new/faster-moe)
- **Embedding models**: Unsloth now supports ~1.8-3.3x faster embedding fine-tuning. [Blog](https://unsloth.ai/docs/new/embedding-finetuning) • [Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks#embedding-models)
- New **7x longer context RL** vs. all other setups, via our new batching algorithms. [Blog](https://unsloth.ai/docs/new/grpo-long-context)
- New RoPE & MLP **Triton Kernels** & **Padding Free + Packing**: 3x faster training & 30% less VRAM. [Blog](https://unsloth.ai/docs/new/3x-faster-training-packing)
- **500K Context**: Training a 20B model with >500K context is now possible on an 80GB GPU. [Blog](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning)
- **FP8 & Vision RL**: You can now do FP8 & VLM GRPO on consumer GPUs. [FP8 Blog](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl)
- **gpt-oss** by OpenAI: Read our [RL blog](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning), [Flex Attention](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training) blog and [Guide](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune).

## 📥 Advanced Installation
The below advanced instructions are for Unsloth Studio. For Unsloth Core advanced installation, [view our docs](https://unsloth.ai/docs/get-started/install/pip-install#advanced-pip-installation).
#### Developer installs: macOS, Linux, WSL:
```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
./install.sh --local
unsloth studio -H 0.0.0.0 -p 8888
```
Then to update :
```bash
unsloth studio update
```

#### Developer installs: Windows PowerShell:
```powershell
git clone https://github.com/unslothai/unsloth.git
cd unsloth
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 --local
unsloth studio -H 0.0.0.0 -p 8888
```
Then to update :
```bash
unsloth studio update
```

#### Nightly: MacOS, Linux, WSL:
```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
git checkout nightly
./install.sh --local
unsloth studio -H 0.0.0.0 -p 8888
```
Then to launch every time:
```bash
unsloth studio -H 0.0.0.0 -p 8888
```

#### Nightly: Windows:
Run in Windows Powershell:
```bash
git clone https://github.com/unslothai/unsloth.git
cd unsloth
git checkout nightly
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 --local
unsloth studio -H 0.0.0.0 -p 8888
```
Then to launch every time:
```bash
unsloth studio -H 0.0.0.0 -p 8888
```

#### Uninstall
You can uninstall Unsloth Studio by deleting its install folder usually located under `$HOME/.unsloth/studio` on Mac/Linux/WSL and `%USERPROFILE%\.unsloth\studio` on Windows. Using the `rm -rf` commands will **delete everything**, including your history, cache:

* ​ **MacOS, WSL, Linux:** `rm -rf ~/.unsloth/studio`
* ​ **Windows (PowerShell):** `Remove-Item -Recurse -Force "$HOME\.unsloth\studio"`

For more info, [see our docs](https://unsloth.ai/docs/new/studio/install#uninstall).

#### Deleting model files

You can delete old model files either from the bin icon in model search or by removing the relevant cached model folder from the default Hugging Face cache directory. By default, HF uses:

* ​ **MacOS, Linux, WSL:** `~/.cache/huggingface/hub/`
* ​ **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`

## 💚 Community and Links
| Type                                                                                                                                      | Links                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| <img width="16" src="https://cdn.prod.website-files.com/6257adef93867e50d84d30e2/66e3d80db9971f10a9757c99_Symbol.svg" />  **Discord**                       | [Join Discord server](https://discord.com/invite/unsloth)                          |
| <img width="15" src="https://redditinc.com/hs-fs/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" />  **r/unsloth Reddit**                       | [Join Reddit community](https://reddit.com/r/unsloth)                          |
| 📚 **Documentation & Wiki**                                                                                                               | [Read Our Docs](https://unsloth.ai/docs)                                       |
| <img width="13" src="https://upload.wikimedia.org/wikipedia/commons/0/09/X_(formerly_Twitter)_logo_late_2025.svg" />  **Twitter (aka X)** | [Follow us on X](https://twitter.com/unslothai)                                |
| 🔮 **Our Models**                                                                                                                         | [Unsloth Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)   |
| ✍️ **Blog**                                                                                                                               | [Read our Blogs](https://unsloth.ai/blog)                                      |

### Citation

You can cite the Unsloth repo as follows:
```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}
```
If you trained a model with 🦥Unsloth, you can use this cool sticker!   <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" width="200" align="center" />

### License
Unsloth uses a dual-licensing model of Apache 2.0 and AGPL-3.0. The core Unsloth package remains licensed under **[Apache 2.0](https://github.com/unslothai/unsloth?tab=Apache-2.0-1-ov-file)**, while certain optional components, such as the Unsloth Studio UI are licensed under the open-source license **[AGPL-3.0](https://github.com/unslothai/unsloth?tab=AGPL-3.0-2-ov-file)**.

This structure helps support ongoing Unsloth development while keeping the project open source and enabling the broader ecosystem to continue growing.

### Thank You to
- The [llama.cpp library](https://github.com/ggml-org/llama.cpp) that lets users run and save models with Unsloth
- The Hugging Face team and their libraries: [transformers](https://github.com/huggingface/transformers) and [TRL](https://github.com/huggingface/trl)
- The Pytorch and [Torch AO](https://github.com/unslothai/unsloth/pull/3391) team for their contributions
- NVIDIA for their [NeMo DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) library and their contributions
- And of course for every single person who has contributed or has used Unsloth!
