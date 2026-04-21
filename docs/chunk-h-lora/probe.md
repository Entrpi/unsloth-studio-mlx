# Chunk H — LoRA probe

Before committing to a 200-iter training run, we probed whether `mlx_lm.lora`
(v0.31.2) would train against a **2-bit quantized** MLX base. The official
mlx-lm docs advertise QLoRA support for 4 / 6 / 8-bit; 2-bit is not listed.

## Command

```
mkdir -p /tmp/lora-probe/{data,out}
# toy 12-row train.jsonl + 4-row valid.jsonl under /tmp/lora-probe/data/
# (scratch — not committed; see fixture dataset under
#  studio/backend/tests/fixtures/lora_dataset/emotion for the real thing).

/tmp/mlxtest/bin/mlx_lm.lora \
  --train \
  --model /Users/ent/.lmstudio/models/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
  --data /tmp/lora-probe/data \
  --adapter-path /tmp/lora-probe/out \
  --iters 10 --batch-size 1 --num-layers 2 \
  --learning-rate 1e-4 --fine-tune-type lora
```

## Result — PASS

```
Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.036% (0.623M/1720.028M)
Starting training..., iters: 10
Iter 1: Val loss 6.770, Val took 0.702s
Iter 10: Val loss 1.628, Val took 0.107s
Iter 10: Train loss 3.941, Learning Rate 1.000e-04, It/sec 3.110,
  Tokens/sec 45.717, Trained Tokens 147, Peak mem 0.572 GB
Saved final weights to /tmp/lora-probe/out/adapters.safetensors.
```

Loss trajectory: 6.77 → 1.63 over 10 validation measurements — decreasing,
non-NaN, monotonic-ish. Peak mem 0.57 GB. Both files written:

```
-rw-r--r--  2493347  adapters.safetensors   (2.4 MB)
-rw-r--r--      996  adapter_config.json
```

## Decision

**Proceed with Bonsai 1.7B 2-bit as the Chunk H fixture base.** It's already
the Phase-7 speculative-decoding draft, so the rest of the test suite knows
how to locate it via `MLX_TEST_DRAFT_MODEL_PATH`. No fallback needed.

The probe artifacts under `/tmp/lora-probe/` are scratch — NOT committed.
The real adapter is trained on the full 400-row `dair-ai/emotion` fixture
under `studio/backend/tests/fixtures/lora_dataset/emotion/`.
