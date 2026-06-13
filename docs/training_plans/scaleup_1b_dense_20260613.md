# MiniMind-O 1B Dense Scale-Up Plan - 2026-06-13

## Decision

Build the first 1B candidate as **Dense MiniMind-O**, not MoE.

Default implementation lives in:

```text
train_1b/
```

Primary launcher:

```bash
DRY_RUN=1 bash train_1b/run_1b_dense_3gpu.sh
bash train_1b/preflight_1b.sh
bash train_1b/smoke_stage1_t2a.sh
```

## Why Dense First

The current known-good training and evaluation history is dense:

- Run A: dense baseline, best original audio anchor.
- Run C: dense + Phase-0 fixes, best text/image candidate but audio mixed.
- Run D-b: dense audio-tail repair, improved repeat/image health but did not restore audio CER.
- Run E diagnosis: audio damage is likely from non-uniform RVQ weights, not from the image tail alone.

MoE should not be the first 1B run:

- Existing MoE code is local FFN MoE, not expert parallel. Under 3-GPU DDP each 5090D still stores all experts.
- It adds routing/aux-loss variables on top of an already sensitive audio pipeline.
- It makes A/B attribution harder because the current strongest conclusions all come from dense runs.

Keep `USE_MOE=0` for the 1B baseline. Revisit MoE only after dense 1B has L1/ASR/human listening anchors.

## External Model Scan

Checked on 2026-06-13:

| Family | Status | Use here |
| --- | --- | --- |
| Qwen3.5 | Qwen3.5-4B is public on Hugging Face; Alibaba also announced large sparse MoE variants. | Good teacher/future backbone, not a direct `from_weight` source. |
| Qwen3 | Has dense 0.6B/1.7B/4B/8B and MoE variants. | More realistic than Qwen3.5 if rebuilding a Qwen-based Omni, but still a separate backbone project. |
| Gemma 4 | Google lists E2B/E4B/12B/31B and 26B-A4B; Gemma 4 12B is encoder-free multimodal. | Good teacher or alternative product path, not MiniMind checkpoint compatible. |

Primary sources:

- https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/collections/Qwen/qwen3
- https://ai.google.dev/gemma/docs/core
- https://developers.googleblog.com/gemma-4-12b-the-developer-guide/

Decision: keep Qwen/Gemma as teacher/backbone research. Do not mix a backbone migration into this 1B MiniMind-O scale-up.

## 1B Architecture

Default dense config:

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
| Estimated non-frozen params | 1044.03M |

Fallback:

```text
train_1b/configs/dense_1b_safe.env
```

This uses 16 Thinker layers, about 939M params, and smaller micro-batches.

## Training Defaults

Carry forward:

- `DDP_BROADCAST_BUFFERS=0`
- `bf16`
- rank-sharded parquet
- validation every 500 steps
- fp32 resume checkpoint, fp16 export checkpoint
- `warmup_ratio=0.02`
- `loss_norm=global`
- finite guards

Revert from Run C:

- Use uniform RVQ weights: `1,1,1,1,1,1,1,1`

Default stage flow:

```text
1 T2A all
2 A2A audio_proj
3 A2A all
4 I2T vision_proj
5 I2T all
6 A2A final
```

Stage7 vision-only is skipped by default because prior runs showed nearly flat value while leaving audio unsupervised at the finish.

## Memory Strategy

Default:

- activation checkpointing on
- compile off
- T2A/I2T micro-batch 8
- A2A micro-batch 4
- accumulation 8

This is intentionally conservative. Raise only one stage at a time after telemetry.

Smoke:

```bash
bash train_1b/smoke_stage1_t2a.sh
```

This defaults to:

```text
STAGE1_FROM_WEIGHT=none
DATA_T2A=../dataset/_calib/sft_t2a_144rows.parquet
MAX_TRAIN_STEPS=20
```

It is only a memory/chain test.

## Pretrain Requirement

The full 1B run expects:

```text
out/llm_2048.pth
```

or another matching 2048-hidden MiniMind text pretrain via:

```bash
STAGE1_FROM_WEIGHT=<prefix>
```

Without it, `train_1b/run_1b_dense_3gpu.sh` stops before launching training unless `ALLOW_SCRATCH_1B=1` or `STAGE1_FROM_WEIGHT=none` is set deliberately.

## Promotion Gate

Do not promote on train loss alone.

Minimum:

- L1 structural eval passes.
- SenseVoice ASR CER compared against A/C/D-b anchors.
- Human listening spot-check for disputed A2A factual/refusal/physics cases.
- Repeat metrics do not regress.
- Image CER stays near or better than Run C/D-b.

Anchors:

```text
out/sft_full_muon_final_768.pth
out/sft_full_muon_v3_final_768.pth
out/sft_full_muon_v4_audiofix_a2a_final_768.pth
```

## Implementation Summary

Added:

- `train_1b/configs/dense_1b.env`
- `train_1b/configs/dense_1b_safe.env`
- `train_1b/run_1b_dense_3gpu.sh`
- `train_1b/preflight_1b.sh`
- `train_1b/smoke_stage1_t2a.sh`
- `train_1b/estimate_params.py`
- `train_1b/README.md`

Extended:

- `model/model_minimind.py`: gradient checkpointing support.
- `model/model_omni.py`: Talker architecture knobs and Omni checkpointing.
- `trainer/train_sft_omni.py`: 1B architecture CLI, `--max_train_steps`, metrics fields.
- `scripts/run_full_train_muon_dense_3gpu.sh`: architecture/batch/seq/compile/MoE/smoke parameterization.

Verification:

- Python compile passed.
- Bash syntax passed.
- Default 1B parameter estimate: `1044.03M`.
- Default and safe dry-runs expanded correctly.
- Preflight passes with `STAGE1_FROM_WEIGHT=none`; full preflight intentionally requires a matching 1B text pretrain.
