# Next Model Comparison Training Plan - 2026-06-10

## Goal

Compare the completed Muon dense 3GPU full-train checkpoint against focused follow-up runs, using the same data split and evaluation prompts where possible.

Baseline checkpoint:

```text
out/sft_full_muon_final_768.pth
```

## Comparison Matrix

| Run | Purpose | Main Change | Expected Signal |
| --- | --- | --- | --- |
| A | Current baseline | Existing 7-stage Muon dense full train | Reference loss and generation quality |
| B | Optimizer ablation | AdamW or existing repo default optimizer | Check whether Muon improves convergence/quality |
| C | Stage5 capacity ablation | Keep Stage5 `batch_size=32` | Measure impact of high-memory I2T all-param training |
| D | A2A stability ablation | Stage3/6 with lower LR or longer final pass | Check if A2A audio loss can be reduced |
| E | Projector schedule ablation | Repeat Stage4/7 with different LR or 2 epochs | Check visual alignment quality |

## Recommended Next Run

Run B first: same 7-stage schedule, same dataset shards, same batch sizes, but use AdamW/default optimizer instead of Muon. This gives the cleanest A/B comparison because it changes one major variable while keeping data, batch, and stage order fixed.

Current code already supports `--optimizer adamw` in `trainer/train_sft_omni.py`. The existing full-train launcher `scripts/run_full_train_muon_dense_3gpu.sh` hardcodes `--optimizer muon`, so the comparison run should use a copied launcher with:

```text
--optimizer adamw
```

and separate output names to avoid overwriting the Muon baseline.

## Fixed Controls

- Same full parquet datasets:
  - `dataset/sft_t2a.parquet`
  - `dataset/sft_a2a.parquet`
  - `dataset/sft_i2t.parquet`
- Same rank shards under `dataset/_full_shards/`
- Same initial base checkpoint where applicable
- Same stage order: T2A -> A2A projector -> A2A all -> I2T projector -> I2T all -> A2A final -> I2T final projector
- Same logging: SwanLab project with a distinct run group/name
- Same final evaluation prompts and random seeds

## Candidate Batch Sizes

| Stage | Current Batch/GPU | Keep/Change |
| --- | ---: | --- |
| T2A all | 48 | Keep |
| A2A audio projector | 32 | Keep |
| A2A all | 24 | Keep; batch 32 already OOMed |
| I2T vision projector | 64 | Keep |
| I2T all | 64 | Keep; best GPU memory utilization |
| A2A final all | 24 | Keep |
| I2T final vision projector | 64 | Keep |

## Evaluation Plan

1. Training metrics:
   - Final loss, text loss, audio loss per stage
   - Loss curve smoothness and instability spikes
   - Tokens/sec or batches/min if available
   - GPU memory and utilization from `gpu_telemetry.csv`

2. Inference samples:
   - T2A: 20 fixed prompts, mixed short/long, Chinese/English
   - A2A: 20 fixed audio prompts, speaker/style variations
   - I2T: 50 fixed image prompts, OCR, captioning, object counting, scene reasoning

3. Objective checks:
   - Text exact/semantic correctness for I2T
   - ASR transcription quality for generated audio
   - Audio duration stability and obvious failure rate
   - Repetition, silence, truncation, and language drift rate

4. Human review:
   - Rank A/B samples blind where possible
   - Track win/tie/loss by modality
   - Keep failure examples with prompt, output, checkpoint, and seed

## Success Criteria

- No OOM or distributed hang.
- Final checkpoint saves cleanly with both `out/` and `checkpoints/` artifacts.
- At least one modality shows measurable improvement without severe regression in another.
- Inference evaluation produces a reusable comparison report.

## Suggested Output Naming

```text
out/sft_full_adamw_compare_final_768.pth
checkpoints/sft_full_adamw_compare_final_768.pth
.run_logs/full_train_adamw_compare_3gpu_YYYYMMDD_HHMMSS/
docs/training_summaries/full_train_adamw_compare_YYYYMMDD.md
```

## Pre-Run Checklist

- Confirm no training process is active with `pgrep -af 'torchrun|train_sft_omni.py'`.
- Confirm all GPUs are idle with `nvidia-smi`.
- Copy the Muon launcher to a new AdamW comparison launcher.
- Change all output/checkpoint prefixes from `sft_full_muon_*` to `sft_full_adamw_compare_*`.
- Keep `BATCH_A2A_FULL=24`; batch 32 already OOMed.
- Keep `BATCH_I2T_FULL=64`; this was the best use of 32G memory.
- Start SwanLab with a new run group/name for clean dashboard comparison.
