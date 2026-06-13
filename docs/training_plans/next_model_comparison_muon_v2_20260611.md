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

### Required pre-run evaluation assets

Freeze the current L0 eval set before launching new long runs:

```bash
python scripts/snapshot_eval_manifest.py \
  --test_set dataset/eval_muon_mini.jsonl \
  --output docs/evaluation_results/eval_muon_mini_manifest_20260611.json \
  --markdown docs/evaluation_results/eval_muon_mini_manifest_20260611.md
```

Run the long-output baseline once, so v2's 160-token qualitative pass has a fair reference:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_final \
  --output_dir eval_results/sft_full_muon_final_batch_audio_bf16_t160 \
  --dtype bf16 --max_new_tokens 160 --decode_audio
```

Dry-run validation (no GPU training):

```bash
DRY_RUN=1 bash scripts/run_full_train_muon_v2_3gpu.sh
```

Full v2 training (when ready):

```bash
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train bash scripts/run_full_train_muon_v2_3gpu.sh
```

The default v2 wrapper is the combined **B3** candidate:

```text
B3 = B1 I2T schedule extension + B2 lower A2A final LR
```

If B3 produces mixed results, split it without code changes:

```bash
# B1: isolate I2T grounding schedule only.
WEIGHT_PREFIX=sft_full_muon_b1_i2t \
RUN_GROUP=muon_b1_i2t \
EPOCHS_I2T_PROJ=2 \
EPOCHS_I2T_ALL=2 \
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train \
bash scripts/run_full_train_muon_dense_3gpu.sh

# B2: isolate A2A final LR only.
WEIGHT_PREFIX=sft_full_muon_b2_a2a_lr \
RUN_GROUP=muon_b2_a2a_lr \
LR_A2A_FINAL=2e-6 \
MUON_LR_A2A_FINAL=0.0005 \
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train \
bash scripts/run_full_train_muon_dense_3gpu.sh
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

5. Manual review template:

```bash
python scripts/make_manual_review_template.py \
  --run_a eval_results/sft_full_muon_final_batch_audio_bf16 \
  --run_b eval_results/sft_full_muon_v2_final_batch_audio_bf16 \
  --label_a baseline --label_b muon_v2 \
  --output docs/evaluation_results/review_muon_v2_vs_baseline.md
```

6. Optional ASR round-trip check with SenseVoiceSmall:

```bash
python scripts/asr_eval_generated_audio.py \
  --results eval_results/sft_full_muon_v2_final_batch_audio_bf16/results.jsonl \
  --output eval_results/sft_full_muon_v2_final_batch_audio_bf16/asr_eval.json \
  --markdown eval_results/sft_full_muon_v2_final_batch_audio_bf16/asr_eval.md \
  --backend sensevoice \
  --sensevoice_project /home/genesis/Projects/SenseVoice \
  --sensevoice_model model/SenseVoiceSmall \
  --device cpu \
  --batch_size 16
```

The ASR script now defaults to SenseVoiceSmall. Use CPU batch mode for L0/L1 because the workload is too small to justify occupying a GPU; `--batch_size 16` ran L0 in about 17-19 seconds per 13-case run and L1 in about 1 minute per 57-case run. The legacy `--backend whisper` path remains available only for reproducing older reports.

## Pre-Run Checklist

- Confirm no active training: `pgrep -af 'torchrun|train_sft_omni.py'`
- Confirm GPUs idle: `nvidia-smi`
- Confirm rank shards exist: `ls dataset/_full_shards/`
- Freeze eval manifest: `python scripts/snapshot_eval_manifest.py ...`
- Run A 160-token baseline before B's 160-token comparison
- For strict Run A behavior reproduction, set `FINITE_GUARD=0`; otherwise keep the safer default `FINITE_GUARD=1`
- Dry-run launcher: `DRY_RUN=1 bash scripts/run_full_train_muon_v2_3gpu.sh`
- Verify v2 weight prefix does not overwrite baseline artifacts in `out/` and `checkpoints/`

## Pre-Run Validation Completed

Completed before launching full Run B:

| Item | Artifact / result |
| --- | --- |
| Eval manifest frozen | `docs/evaluation_results/eval_muon_mini_manifest_20260611.{json,md}` |
| Eval manifest missing files | 0 |
| A 160-token baseline | `eval_results/sft_full_muon_final_batch_audio_bf16_t160/`, 13/13 basic pass |
| Manual review template smoke | `docs/evaluation_results/review_baseline_96_vs_160_template.md` |
| B3 dry-run | `.run_logs/dryrun_muon_v2_b3_20260611.log` |
| B1 dry-run | `.run_logs/dryrun_muon_b1_i2t_20260611.log` |
| B2 dry-run | `.run_logs/dryrun_muon_b2_a2a_lr_20260611.log` |
| Trainer smoke with RVQ metrics/finite guard | `.run_logs/smoke_metrics_guard_20260611.metrics.json` |
| Loss equivalence | `text diff=0.00e+00, audio diff=0.00e+00` |
| ASR round-trip | Script defaults changed to SenseVoiceSmall CPU batch; use GPU only for large ASR batches |
| Formal `_val` monitoring splits | `dataset/_val/sft_{t2a,a2a,i2t}_val.parquet`, 1000 rows each |
| v3 dry-run after `_val` generation | `.run_logs/dryrun_muon_v3_after_val_20260611.log` |

## Success Criteria

- No OOM or distributed hang.
- Final checkpoint saves to both `out/` and `checkpoints/`.
- I2A cases show fewer detail hallucinations vs baseline (manual review).
- A2A final stage loss does not rebound above Stage3 level.
- Eval comparison report generated under `docs/evaluation_results/`.

---

## Run C · Phase-0 Correctness Fixes (v3)

Run C combines the prepared **v2 schedule wrapper** with **training math fixes**. After L1, `B3_himem_b88` was not promoted, so use **baseline A** as the primary evaluation anchor and keep B3 as a secondary reference. Do not compare raw training loss between A and C directly, because `loss_norm=global`, RVQ weights, and warmup change the loss scale; use L1 eval, ASR, manual review, val-curve shape, and per-RVQ metrics.

Status update 2026-06-12: Run C training completed successfully with `DDP_BROADCAST_BUFFERS=0`. L1 automatic eval, SenseVoice ASR round-trip, and AI-assisted review are complete. C is the strongest current candidate and clearly better than B3, but it should not directly replace baseline A yet. C is credible on image gains and mostly credible on Chinese text probes; audio remains unresolved because 10/23 audio cases have semantic-review winners opposite to lower-CER direction, plus one CER tie with semantic preference. Before final promotion, run a 5-10 case human listening and semantic spot-check on disputed audio cases such as `a2a_en_food`, `a2a_en_black_hole`, `a2a_zh_health_probe`, `a2a_en_sky_blue`, `a2a_zh_coffee_probe`, and `a2a_en_animals`. Next training should preserve C's image gains while targeting Stage6 audio regression, A2A factuality, refusal stability, and repeat suppression.

Run D follow-up plan: `docs/training_plans/next_model_training_runD_20260613.md`. Run D is now scoped as an audio-recovery quality run, not throughput work. Conceptually, D-a (joint mixed finale) is the most robust fix because it protects A2A + I2T + T2A together; practically, run D-b first because it is a no-code, low-cost isolation experiment that strengthens Stage6 A2A and ends on audio. Evaluation upgrade is a promotion blocker: run human listening spot-checks, expand L1 to 100+ with more A2A factual/refusal prompts, and report semantic correctness separately from CER. Keep the proven defaults: `loss_norm=global`, `warmup_ratio=0.02`, fp32 resume, 500-step val monitoring, `DDP_BROADCAST_BUFFERS=0`, and I2T 2-epoch schedule. The v3 wrapper now defaults `DDP_BROADCAST_BUFFERS=0`, but Run D commands still set it explicitly for auditability. Evaluate the produced `_a2a_final` artifact before creating any `_final` promotion alias.

Completed checkpoint:

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

Before launching v3, confirm the files exist:

```bash
ls dataset/_val/sft_{t2a,a2a,i2t}_val.parquet
```

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

- Resume checkpoint stores fp32/bf16 model weights (not half-truncated): done.
- Val loss curves logged per stage without non-finite failures: done.
- S6 A2A final loss stays at or below S3 level: not met on raw val (`4.6873` vs S3 `4.4796`), likely due changed post-I2T state; use downstream eval for decision.
- `scripts/test_loss_equivalence.py` passes (local/default = legacy loss): done before launch.
- bf16 eval on `eval_muon_l1.jsonl` shows no regression vs baseline A on pass rate and overall CER: done. AI-assisted review favors C overall (A 17 / C 25 / Tie 15), but audio-specific evidence remains mixed; human listening remains pending.

### Comparison matrix (updated)

| Run | Variable axis | Status |
| --- | --- | --- |
| A | Baseline 7-stage Muon | Completed |
| B3_himem_b88 | v2 schedule + Stage1 batch 88 | Completed; not promoted after L1 |
| C | v2 wrapper + Phase-0 fixes (v3) | Completed; strongest text/image candidate, audio unresolved pending listening/semantic review |
| D-a | Joint mixed finale (most robust audio-forgetting fix) | Planned after D-b signal; requires mixed data + launcher stage |
| D-b | Audio-tail strengthening (first no-code isolation run) | Planned first; see [next_model_training_runD_20260613.md](./next_model_training_runD_20260613.md) |
| D-c | Stage6/7 order swap | Optional if Stage7 image micro-tune must be kept before A2A tail |
| D-rvq/decodeguard | RVQ weight flattening + repeat-controlled eval | Planned in parallel to separate decoding loops from model quality |
| D-eval | Human listening + semantic correctness rubric + L1 100+ expansion | Blocking before promoting C or any Run D candidate |
| Run D defaults | Phase-0 + DDP buffer off + I2T 2 epochs | Keep as defaults; do not roll back while fixing audio |
| T | Throughput (packing/streaming) | Planned separately; orthogonal engineering line, renamed from "D" |
