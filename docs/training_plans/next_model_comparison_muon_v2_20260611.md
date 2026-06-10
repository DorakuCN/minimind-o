# Next Model Comparison Training Plan - Muon v2 - 2026-06-11

## Goal

Compare the completed Muon dense 3GPU full-train baseline against a Muon v2 schedule tuned from evaluation findings. AdamW ablation is out of scope; Muon is assumed superior for this project.

Baseline checkpoint:

```text
out/sft_full_muon_final_768.pth
```

Candidate checkpoint (after v2 run):

```text
out/sft_full_muon_v2_final_768.pth
```

## Evaluation-Driven Changes

From [docs/evaluation_results/sft_full_muon_final_eval_20260610.md](../evaluation_results/sft_full_muon_final_eval_20260610.md):

| Issue | v2 Response |
| --- | --- |
| I2A grounding weak (hallucinated details) | Stage4 I2T projector 1 -> 2 epochs; Stage5 I2T all 1 -> 2 epochs |
| A2A Stage6 loss rebound (4.68 -> 4.99) | Stage6 LR 5e-6 -> 2e-6, muon_lr 0.001 -> 0.0005 |
| 96-token truncation on long answers | Eval with `--max_new_tokens 160` for qualitative review; keep 96 for baseline parity |
| fp16 inference unstable | Default eval dtype is now `bf16`; sampling logits cast to fp32 in `generate()` |

## Comparison Matrix

| Run | Purpose | Main Change | Expected Signal |
| --- | --- | --- | --- |
| A | Baseline | Existing 7-stage Muon dense full train | Reference loss and generation quality |
| B | Muon v2 schedule | Longer I2T stages + lower A2A final LR | Better I2A grounding, lower A2A final loss rebound |

## Fixed Controls

- Same optimizer: Muon (`MuonWithAuxAdam`)
- Same full parquet datasets and rank shards under `dataset/_full_shards/`
- Same initial base checkpoint: `out/llm_768.pth` (Stage1 `--from_weight llm`)
- Same stage order and batch sizes (T2A 48, A2A proj 32, A2A full 24, I2T proj/full 64)
- Same logging project: SwanLab `MiniMind-O-Full-Train`, distinct run group `muon_v2`
- Same eval set: `dataset/eval_muon_mini.jsonl`, seed 42
- Default eval dtype: `bf16`

## v2 Stage Schedule Differences

| Stage | Baseline | Muon v2 |
| --- | ---: | ---: |
| 4 I2T vision projector epochs | 1 | 2 |
| 5 I2T all epochs | 1 | 2 |
| 6 A2A final learning_rate | 5e-6 | 2e-6 |
| 6 A2A final muon_lr | 0.001 | 0.0005 |
| All other stages | unchanged | unchanged |

## Launch Commands

Dry-run validation (no GPU training):

```bash
DRY_RUN=1 bash scripts/run_full_train_muon_v2_3gpu.sh
```

Full v2 training (when ready):

```bash
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train bash scripts/run_full_train_muon_v2_3gpu.sh
```

Resume from stage N:

```bash
START_STAGE=5 RESUME_STAGE5=1 bash scripts/run_full_train_muon_v2_3gpu.sh
```

## Output Naming

```text
out/sft_full_muon_v2_final_768.pth
checkpoints/sft_full_muon_v2_final_768.pth
.run_logs/full_train_sft_full_muon_v2_3gpu_YYYYMMDD_HHMMSS/
docs/training_summaries/full_train_muon_v2_YYYYMMDD.md
```

Per-stage metrics JSON is written automatically to `.run_logs/.../<stage>.metrics.json`.

## Evaluation Plan

1. Training metrics: compare per-stage `*.metrics.json` (final loss, text/audio loss, samples/sec).
2. Batch validation (baseline parity, bf16 default):

```bash
python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v2_final \
  --output_dir eval_results/sft_full_muon_v2_final_batch_audio_bf16 \
  --dtype bf16 --decode_audio
```

3. Long-output qualitative pass (optional):

```bash
python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v2_final \
  --output_dir eval_results/sft_full_muon_v2_final_batch_audio_bf16_t160 \
  --dtype bf16 --max_new_tokens 160 --decode_audio
```

4. Compare reports:

```bash
python scripts/compare_eval_runs.py \
  --run_a eval_results/sft_full_muon_final_batch_audio_bf16 \
  --run_b eval_results/sft_full_muon_v2_final_batch_audio_bf16 \
  --label_a baseline --label_b muon_v2 \
  --output docs/evaluation_results/compare_muon_v2_vs_baseline.md
```

## Pre-Run Checklist

- Confirm no active training: `pgrep -af 'torchrun|train_sft_omni.py'`
- Confirm GPUs idle: `nvidia-smi`
- Confirm rank shards exist: `ls dataset/_full_shards/`
- Dry-run launcher: `DRY_RUN=1 bash scripts/run_full_train_muon_v2_3gpu.sh`
- Verify v2 weight prefix does not overwrite baseline artifacts in `out/` and `checkpoints/`

## Success Criteria

- No OOM or distributed hang.
- Final checkpoint saves to both `out/` and `checkpoints/`.
- I2A cases show fewer detail hallucinations vs baseline (manual review).
- A2A final stage loss does not rebound above Stage3 level.
- Eval comparison report generated under `docs/evaluation_results/`.

---

## Run C · Phase-0 Correctness Fixes (v3)

Run C combines **v2 schedule** with **training math fixes**. Compare against the **winner of Run B** (not Run A directly), because Run C changes loss normalization and LR warmup.

Candidate checkpoint:

```text
out/sft_full_muon_v3_final_768.pth
```

### Phase-0 Changes (all flag-gated; defaults preserve legacy behavior)

| Fix | Flag / default | v3 value |
| --- | --- | --- |
| LR warmup | `--warmup_ratio` (0) | 0.02 |
| Global token-weighted loss | `--loss_norm local\|global` | global |
| RVQ layer weighting | `--rvq_layer_weights` (all 1) | 2,1.5,1.2,1,0.8,0.7,0.6,0.5 |
| Stop token weight | `--audio_stop_weight` (10) | 10 (unchanged) |
| fp32 resume weights | unconditional in `omni_checkpoint` | always on |
| Val loss monitoring | `--val_data_path` + `--val_interval` | per-stage `_val/*.parquet`, every 500 steps |

### Prepare validation splits (once, before v3)

```bash
python scripts/make_val_split.py --dataset_dir dataset --output_dir dataset/_val
```

Note: val splits are for **curve stability monitoring** only; they may overlap slightly with training shards and are not a held-out generalization metric.

### Launch v3

Dry-run:

```bash
DRY_RUN=1 bash scripts/run_full_train_muon_v3_3gpu.sh
```

Full training:

```bash
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train bash scripts/run_full_train_muon_v3_3gpu.sh
```

### Run C success criteria

- Resume checkpoint stores fp32/bf16 model weights (not half-truncated).
- Val loss curves logged per stage without regression spikes vs train loss.
- S6 A2A final loss stays at or below S3 level.
- `scripts/test_loss_equivalence.py` passes (local/default = legacy loss).
- bf16 eval on `eval_muon_mini.jsonl` shows no regression vs Run B winner on pass rate and I2A grounding (manual review).

### Comparison matrix (updated)

| Run | Variable axis | Status |
| --- | --- | --- |
| A | Baseline 7-stage Muon | Completed |
| B | v2 schedule only | Launcher ready |
| C | v2 + Phase-0 fixes (v3) | Launcher ready |
| D | Throughput (packing/streaming) | Planned separately |
