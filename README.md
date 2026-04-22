# unsloth-studio-mlx

> A development fork of [`unslothai/unsloth`](https://github.com/unslothai/unsloth). Everything here is destined for upstream via a single PR. The story of why this exists follows — then the upstream README below.

## How this started

One afternoon we wanted to try Ternary Bonsai 8B locally. The model claims a lot for its 1.58-bit footprint and we had it running via LM Studio on an M-series Mac in minutes. The obvious next step was to drive it from Unsloth Studio — same interface, same chat history, full local UX.

There was no path. Upstream Studio had solid GGUF support through `llama-server` and some structurally extensible backend scaffolding (`ModelConfig`, the subprocess orchestrator, the OpenAI-compat route), but zero MLX code. No `mlx_lm.py`, no `mlx_vlm.py`, no `is_mlx` detection, no variant anywhere. Running an MLX checkpoint on Studio wasn't "configure something" — the backend literally couldn't load one.

So we built it. Across ten phases we added an in-process `MlxLmBackend` peer to `LlamaCppBackend` — detection via the `"quantization": {"bits", "group_size"}` block in `config.json`, load via `mlx_lm.load()`, streaming chat via `mlx_lm.stream_generate()` — then layered on sampler fidelity (top-k / top-p / min-p / repetition penalty / logit bias), reasoning (`<think>` passthrough, `enable_thinking` threading through the chat template), tool calling (a dialect parser, an agentic loop with hold-back and cancellation), quantized KV cache (`q8_0` / `q4_0`), LoRA adapters, speculative decoding with a draft model, remote HF pulls with progress tracking and memory preflight, `MlxVlmBackend` for vision-language models, and `MlxAudioBackend` for LFM2.5-Audio TTS and `mlx-whisper` ASR. Each phase shipped with its own tests; the matrix came out to ~400 new tests across the full backend suite.

That's the base layer. What this README is really about is everything that happened *on top* of that while we dogfooded it — bugs in our own code we found by actually running the models, UX gaps we hit while debugging, and the diagnostic infrastructure we built that turned out to be the most visible UX improvement of the whole chunk.

## What broke while we tested our own work, and what we did about it

**Context length silently clamped to 4096.** Our MLX load path was honouring the frontend's legacy `max_seq_length=4096` default — same clamp GGUF had — even on Ternary Bonsai's 131K-capable context. The frontend had a "0 = use native" branch for GGUF that we hadn't routed MLX through. One-liner fix once we spotted that the UI number and the backend reality diverged.

**The picker labelled every MLX checkpoint as "Local".** When we added MLX detection there was nothing for the frontend to display — only GGUF had quant / size / variant UI. We built out the equivalent: detection helpers for `-MLX-Nbit` / `mlx-community/*` naming conventions, a variant submenu that enumerates sibling quants (2 / 4 / 5 / 6 / 8-bit / bf16) by querying Hugging Face, the same relevance-bubbled search behaviour GGUFs already had, and chip detail consistency (`MLX · 4bit · 19.5 GB`). While building it we caught a latent bug: the existing "Downloaded" section was hiding *all* non-GGUF cached repos in chat-only mode, because the gate had been written as a blanket `!chatOnly` to keep plain PyTorch weights out. MLX quants got swept up in that. Fixed.

**Each new model surfaced a new tool-call dialect.** Ternary Bonsai worked on our first-cut parser because it emits Qwen-style `<tool_call>{JSON}</tool_call>`. The moment we tried Gemma-4 E4B, nothing happened — Gemma emits `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>` with unquoted keys and escaped-quote string delimiters. We wrote a Gemma-4 dialect parser. Then tried GLM-4.6V-Flash: *another* dialect, `<tool_call>name\n<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`. Another parser. Then the channel-thinking blocks (`<|channel>thought…<channel|>`). And the cases where the model emits unparseable JSON envelopes like `{"tool_name":"…","params":{…}}` that had to be recognised as a fallback dialect.

By the end we had five tool-call dialects and a ~900-line parser. [A research agent confirmed what we already suspected](docs/chunk-h2-matrix/sglang-parser-migration.md): everyone else (vLLM, SGLang, mistral.rs) has arrived at the same per-family-parser architecture. The long-term path is to vendor SGLang's framework so we get ~20 families for free; that decision is documented for a future chunk.

**The UI got stuck after the second tool call, and the backend looked fine.** This one took the longest. Reports were "GPU idle but the chat is still thinking." Our first theory was an SSE transport issue — long silent periods during prompt re-eval, browsers idle-closing the connection. We built a 5-second SSE keepalive (racing the generator against a timeout with `asyncio.shield` so the generator doesn't get cancelled). Then a frontend 15-second inactivity watchdog. Then a thread-safety hotfix because our first keepalive implementation spun up concurrent `next(gen)` calls on the same Python generator, which isn't thread-safe and occasionally deadlocked. All of that was correct, but it wasn't the bug.

The actual bug only surfaced when we drove the full repro in Chrome via preview tools: opened DevTools, spawned an SSE-tee to capture the wire, forced a `New Chat` click, and caught this in the React error boundary:

> `Duplicate key toolCallId-call_0 in tapResources`

Our tool-call parser assigned IDs per-invocation (`call_0`, `call_1`, ...), and the agentic loop re-invoked it fresh every iteration. So iter=0's first tool call and iter=1's first tool call both arrived at assistant-ui with `toolCallId: "call_0"`. assistant-ui keys its active-tool-parts map by that ID. The second one crashed it. The error boundary swallowed the visible crash, but the state tree corrupted mid-stream and iter=2+ content got dropped. The keepalives and watchdog covered different failure modes that *didn't* exist in this code path; they're still correct, they just weren't *this* bug.

The fix is a three-line counter that makes IDs globally unique across iterations. The debugging harness we built to find it — the Chrome preview driver, the SSE-tee hook, the DOM snapshot comparisons — turned out to be much more valuable than the original planned feature.

**The UI gave no signal during the 30-second pauses.** During a tool execution, and then during the prompt re-eval after it, the stream produced zero bytes and the UI showed nothing. Even with keepalives fixing the transport layer, the *user* had no way to know anything was happening. So we added structured progress events — `{"type":"progress","phase":"prompt_eval"|"generating","iter":N}` — and rendered them as chips above the composer ("Re-reading conversation…" / "Generating…"). Additive to the SSE event set; external OpenAI clients ignore unknown event types. Zero intrusion, but the chat suddenly feels responsive during what used to be dead air.

**We wanted to see the work happening.** The progress chip was honest but vague. The next question was: *how many tokens has the model actually produced?* So we instrumented the inner generation loop to count raw tokens *before* the parser's hold-back logic strips anything, and built a token-counter chip that shows `242 tok · 28.4 t/s · iter 0: 287 tok`. The "pre-filter" distinction matters — during reasoning that the parser holds back, the visible text isn't growing but the model is absolutely working. The counter ticks anyway. This told us we weren't stuck during multi-second silences; the model was just thinking.

**And then we wanted to see the GPU.** A sparkline next to the stop button, classic `mactop` vibe. The first implementation used MLX's active-memory counter as a proxy and reported a binary "100% while generating, 0% otherwise." Bad — pinned at 100% for whole tool-call-loop sessions, conveys nothing. We went down the `mactop` rabbit hole, discovered it uses Apple's private `IOReport.framework` via CGO, reverse-engineered the Go binding, and wrote a pure-Python ctypes equivalent against `/usr/lib/libIOReport.dylib`. No sudo, no entitlements, matches `powermetrics` to the tenth of a percent. With severity colour bands at 50% and 85% the chart actually says something: idle-ish while Chrome composits the UI, spikes to red when inference starts, drops back.

Then we thought about the 100K+ non-Mac users and added `pynvml` for NVIDIA (covers Linux x86_64, Linux aarch64 — the DGX Spark case — WSL, and Windows in one package), plus AMD sysfs (`gpu_busy_percent`), Intel sysfs engine counters, and Windows PDH as the any-GPU fallback. Each backend is a 30-60 line probe + reader behind a platform-aware dispatch chain. The sparkline still hides gracefully when no backend succeeds. And when the telemetry WebSocket drops or goes silent for >10s, the chip collapses to a `WifiOff` badge — because a chart showing frozen data is worse than a chart showing "disconnected."

**The debugging harness is now a product feature.** The token counter, the GPU sparkline, the progress chips, the WebSocket telemetry endpoint — we built them to see what was happening while we chased the `Duplicate key` ghost. They ended up being the most-visible UX improvements in this chunk, because they answer the question every user of a local inference tool has looking at a spinning icon: *is anything actually happening?* The answer is now: "yes, at 28.4 t/s on a GPU that's 97% busy in iter=2 of the agentic loop, 1234 tokens produced."

## What's in the fork (tl;dr)

If the story above was too long, here's the shape of it:

- **Full MLX tool-calling** — agentic loops on both `MlxLmBackend` and `MlxVlmBackend`, 5-dialect parser (Qwen / Claude-XML / Gemma-4 / loose-JSON / GLM-4), Gemma-4 channel-thought extraction, globally-unique tool-call IDs across iterations.
- **SSE keepalive + inactivity watchdog** — no more idle-closed streams during long tool executions.
- **Structured progress events** — `prompt_eval` / `generating` chips during the otherwise-opaque silent periods.
- **Live telemetry WebSocket** — `/ws/telemetry` carries GPU stats, session state, pre-filter token counts; separate from chat SSE so it outlives turns and fans out to multiple tabs.
- **GPU sparkline** — real utilisation on every platform Studio runs on: IOReport on Apple Silicon, `pynvml` on NVIDIA anywhere, AMD/Intel sysfs on Linux, PDH on Windows. Severity colour bands. Disconnected badge when the stream dies.
- **Pre-filter token counter** — counts raw tokens from the generator before hold-back mutation; keeps ticking during held-back reasoning.
- **Model picker parity** — MLX quants alongside GGUF in search, variant picker for sibling repos, loaded-model chip shows `MLX · 4bit · 19.5 GB`.
- **Hardened fetch path** — `_fetch_page_text` with wall-clock deadline, `cancel_event` propagation, explicit socket close — fixes a CLOSE_WAIT leak we found by accident.
- **Every new surface is feature-flagged** — documented with rollback values in [`docs/env-vars.md`](docs/env-vars.md).

## Honest status

~122 commits on `mlx-studio-enablement` vs the branchpoint. Full backend test suite adds ~400 tests; 965 pass, 22 environmental failures that match `unslothai/unsloth:main` exactly (flash-attn Linux, CUDA-gated GPU tests, pre-existing vision cache). Frontend TypeScript + build clean. Verified end-to-end via Chrome preview against GLM-4.6V-Flash, Gemma-4 E4B, Ternary Bonsai, and Qwen3-1.7B — which is how several of the bugs above got found.

Deferred follow-ups are documented, not silently skipped:
- [`docs/chunk-h2-matrix/sglang-parser-migration.md`](docs/chunk-h2-matrix/sglang-parser-migration.md) — the next-generation parser architecture
- [`docs/chunk-h2-matrix/parity-audit.md`](docs/chunk-h2-matrix/parity-audit.md) — smaller P1/P2 gaps (VLM passthrough, MlxAudio hf_variant, GGUF LoRA, etc.)

## Relationship to upstream

Canonical Unsloth development lives at [`unslothai/unsloth`](https://github.com/unslothai/unsloth). The `main` branch here is a pristine mirror; `mlx-studio-enablement` (the default branch you're reading this on) is where the work is. License is inherited unchanged — Studio source stays AGPL-3.0, upstream paths keep their licenses. When upstream reviewers are ready, we'll rebase on latest `main` and open one PR.

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
