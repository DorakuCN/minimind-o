# MiniMind-O 1B Scale-Up Training Notes

This folder contains the 1B training wrapper and configs. It does not duplicate the core trainer; it drives the existing `trainer/train_sft_omni.py` through the parameterized dense launcher, so fixes in the main trainer stay shared.

## Current Training Code Map

Core files:

| File | Role |
| --- | --- |
| `trainer/train_sft_omni.py` | SFT entrypoint: DDP, loss, val, checkpoint, metrics, stage arguments |
| `trainer/trainer_utils.py` | model/tokenizer init, LR schedule, checkpoints, DDP helpers |
| `trainer/optimizers.py` | `MuonWithAuxAdam`: Muon for 2D hidden matrices, AdamW fallback for embeddings/head/norm/bias |
| `dataset/omni_dataset.py` | parquet loading, chat prompt construction, text/audio labels, image/audio preprocessing |
| `model/model_omni.py` | Omni Thinker + Talker + frozen SenseVoice/SigLIP bridges |
| `scripts/run_full_train_muon_dense_3gpu.sh` | 7-stage orchestrator, now parameterized for 1B arch/batches/seq/compile |

Data stays unchanged:

```text
dataset/sft_t2a.parquet
dataset/sft_a2a.parquet
dataset/sft_i2t.parquet
dataset/_full_shards/*.rankXX-of03.parquet
dataset/_val/sft_{t2a,a2a,i2t}_val.parquet
```

## Historical Conclusions To Carry Forward

Useful defaults:

- Use `bf16` for eval/training autocast on 5090D; fp16 inference had non-finite sampling failures.
- Keep `DDP_BROADCAST_BUFFERS=0`; Run C only completed after disabling DDP buffer broadcast.
- Keep validation and metrics JSON; raw train loss is not enough for promotion.
- Keep `warmup_ratio=0.02` and `loss_norm=global` unless a targeted ablation says otherwise.
- Keep resume checkpoint weights fp32 and export weights fp16.
- Use rank-sharded parquet for 3-GPU DDP.

Audio lessons:

- Run C's non-uniform RVQ weights improved text/image but damaged A2A audio by Stage3.
- Run E's current hypothesis is that fine RVQ layer downweighting is the main audio regression source.
- Therefore the 1B default reverts to uniform RVQ weights: `1,1,1,1,1,1,1,1`.
- End the first 1B candidate at Stage6 A2A by default. Stage7 vision-only was nearly flat in prior runs and leaves audio unsupervised at the finish.

## Dense vs MoE

Default: **Dense, `USE_MOE=0`**.

Do not make MoE the first 1B run:

- Existing successful comparisons are dense.
- The current MoE path is local FFN MoE, not expert parallel. Under DDP every 5090D still stores every expert.
- MoE increases routing/aux-loss variables exactly where audio behavior is already fragile.
- Same training data means MoE may overfit or route modalities unevenly without adding reliable quality.

Use MoE only after a dense 1B baseline is measured, and preferably as a separate controlled experiment.

## External Backbone Scan

Checked on 2026-06-13:

| Family | Current useful public signal | Fit for this repo |
| --- | --- | --- |
| Qwen3.5 | `Qwen/Qwen3.5-4B` is open on Hugging Face under Apache-2.0. The card describes a 4B post-trained multimodal model with 32 layers, hidden 2560, Gated DeltaNet plus gated attention, and 262K context. Alibaba's release post also describes a much larger `Qwen3.5-397B-A17B` sparse MoE model. | Strong external teacher or future backbone. Not a drop-in MiniMind pretrain because tokenizer, architecture, multimodal path, and attention/linear-attention blocks differ. |
| Qwen3 | Qwen3 has dense sizes including `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B`, plus MoE variants. | `Qwen3-1.7B` or `Qwen3-0.6B` are closer to a local 1B-ish text base than Qwen3.5-4B if we decide to rebuild around Qwen. Still not shape-compatible with MiniMind. |
| Gemma 4 | Google lists E2B, E4B, 12B, 31B, and 26B A4B. E2B/E4B/12B support native audio/vision; 12B is an encoder-free multimodal model. | Good teacher or alternative product path. Not a direct source for current Omni weights; Gemma tokenizer, architecture, license terms, and audio representation differ. |

Primary sources:

- Qwen3.5 Alibaba release: https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894
- Qwen3.5-4B model card: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3 collection: https://huggingface.co/collections/Qwen/qwen3
- Gemma 4 overview: https://ai.google.dev/gemma/docs/core
- Gemma 4 12B developer guide: https://developers.googleblog.com/gemma-4-12b-the-developer-guide/

Decision for this refactor:

1. Keep the current MiniMind-O 1B dense path as the baseline. It is the only path that preserves the existing Mimi-code Talker, dataset labels, loss code, and eval comparability.
2. Treat Qwen/Gemma as teacher candidates for distillation and evaluation, not as direct `from_weight` sources.
3. If switching backbone, create a separate `backbone_qwen/` or `backbone_gemma/` branch of work. That would require tokenizer migration, model-forward rewrite, projector/Talker interface redesign, and checkpoint conversion.

## Default 1B Architecture

`configs/dense_1b.env`:

| Component | Value |
| --- | ---: |
| Thinker hidden | 2048 |
| Thinker layers | 18 |
| Attention heads / KV heads | 16 / 8 |
| Head dim | 128 |
| Thinker FFN | 6464 |
| Talker hidden | 1024 |
| Talker layers | 4 |
| Talker heads / KV heads | 8 / 4 |
| Talker FFN | 3264 |
| Estimated non-frozen params | about 1.044B |

Fallback: `configs/dense_1b_safe.env` uses 16 Thinker layers, about 939M params, and lower micro-batches.

## Launch

Dry-run:

```bash
DRY_RUN=1 bash train_1b/run_1b_dense_3gpu.sh
```

Preflight:

```bash
bash train_1b/preflight_1b.sh
```

Stage-1 memory smoke:

```bash
bash train_1b/smoke_stage1_t2a.sh
```

The smoke script defaults to `STAGE1_FROM_WEIGHT=none` and `MAX_TRAIN_STEPS=20`. It is only for DDP/model/data/loss/memory verification. The full run should use a real 1B text pretrain unless training from scratch is deliberate.

Full run with SwanLab:

```bash
USE_WANDB=1 bash train_1b/run_1b_dense_3gpu.sh
```

Fallback config:

```bash
CONFIG_FILE=train_1b/configs/dense_1b_safe.env DRY_RUN=1 bash train_1b/run_1b_dense_3gpu.sh
```

The wrapper expects a matching text pretrain at:

```text
out/llm_2048.pth
```

Override with `STAGE1_FROM_WEIGHT=<prefix>` if the 1B pretrain has another prefix. Set `ALLOW_SCRATCH_1B=1` only if training from scratch is intentional.

## Memory Strategy

The first run uses:

- activation checkpointing on Transformer blocks,
- compile off for every stage,
- small per-GPU micro-batches,
- `ACCUMULATION_STEPS=8`,
- `MAX_TRAIN_STEPS=0` for full runs, with smoke scripts overriding it,
- rank-sharded parquet,
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Default micro-batches:

| Stage | Per-GPU batch | Seq len | Effective batch with 3 GPUs and accum 8 |
| --- | ---: | ---: | ---: |
| T2A all | 8 | 512 | 192 |
| A2A audio_proj | 4 | 1024 | 96 |
| A2A all | 4 | 1024 | 96 |
| I2T vision_proj | 8 | 768 | 192 |
| I2T all | 8 | 768 | 192 |
| A2A final | 4 | 1024 | 96 |

Tune only one stage at a time:

| Symptom | First adjustment |
| --- | --- |
| OOM in A2A | `BATCH_A2A_FULL=3`, then `BATCH_A2A_PROJ=3` |
| OOM in I2T | `BATCH_I2T_FULL=6`, `BATCH_I2T_PROJ=6` |
| OOM in T2A | `BATCH_T2A=6` |
| Memory below 26G and util healthy | raise that stage batch by 1-2 |
| GPU util low with memory available | increase `NUM_WORKERS`, check parquet IO before increasing model batch |

## Parameter Count Check

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate minimind-o
python train_1b/estimate_params.py
```

Expected default output is about:

```text
params_non_frozen_encoders=1044.03M
```

## Promotion Gate

Do not promote on train loss alone. Use the existing L1 flow:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_validate_omni.py \
  --weight omni1b_dense_d2048l18_t1024_a2a_final \
  --output_dir eval_results/omni1b_dense_d2048l18_t1024_a2a_final_l1_bf16_t160 \
  --test_set dataset/eval_muon_l1.jsonl \
  --dtype bf16 --max_new_tokens 160 --decode_audio

python scripts/asr_eval_generated_audio.py \
  --results eval_results/omni1b_dense_d2048l18_t1024_a2a_final_l1_bf16_t160/results.jsonl \
  --output eval_results/omni1b_dense_d2048l18_t1024_a2a_final_l1_bf16_t160/asr_eval.json \
  --markdown eval_results/omni1b_dense_d2048l18_t1024_a2a_final_l1_bf16_t160/asr_eval.md \
  --backend sensevoice --sensevoice_model model/SenseVoiceSmall --device cpu --batch_size 16
```

Primary comparison anchors remain:

```text
out/sft_full_muon_final_768.pth
out/sft_full_muon_v3_final_768.pth
out/sft_full_muon_v4_audiofix_a2a_final_768.pth
```
