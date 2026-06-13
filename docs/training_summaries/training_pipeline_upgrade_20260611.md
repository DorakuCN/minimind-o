# MiniMind-O Training Pipeline Upgrade Record - 2026-06-11

## Version Scope

This document records code and launcher changes made after the completed **Run A** full Muon 7-stage train (`sft_full_muon_final_768.pth`) and its evaluation report. It is intended for post-run comparison and traceability.

Related plans:

- Comparison schedule: [docs/training_plans/next_model_comparison_muon_v2_20260611.md](../training_plans/next_model_comparison_muon_v2_20260611.md)
- Baseline train summary: [full_train_muon_dense_3gpu_20260610.md](./full_train_muon_dense_3gpu_20260610.md)
- Baseline eval summary: [../evaluation_results/sft_full_muon_final_eval_20260610.md](../evaluation_results/sft_full_muon_final_eval_20260610.md)

## Goals

| Goal | How addressed |
| --- | --- |
| Fix fp16 inference instability | Default eval dtype `bf16`; sampling logits cast to fp32 in `generate()` |
| Prepare Run B (schedule-only ablation) | Parameterized launcher + `run_full_train_muon_v2_3gpu.sh` |
| Prepare Run C (Phase-0 correctness) | Flag-gated loss/warmup/resume/val fixes + `run_full_train_muon_v3_3gpu.sh` |
| Make runs comparable without manual log copying | Per-stage `*.metrics.json`, SwanLab run names, `compare_eval_runs.py` |
| Preserve Run A/B backward compatibility | All Phase-0 fixes default to legacy behavior unless launcher sets env vars |

## Comparison Run Matrix

| Run | Weight prefix | Main change | Launcher | Status |
| --- | --- | --- | --- | --- |
| A | `sft_full_muon` | 7-stage Muon dense baseline | `run_full_train_muon_dense_3gpu.sh` | **Completed** |
| B3_himem_b88 | `sft_full_muon_v2_himem_b88` | v2 schedule plus Stage1 batch 88 | `run_full_train_muon_v2_3gpu.sh` + env override | **Completed; not promoted after L1** |
| C | `sft_full_muon_v3` | v2 wrapper + Phase-0 fixes | `run_full_train_muon_v3_3gpu.sh` | **Completed 2026-06-12; text/image candidate, audio unresolved pending listening** |

After the L1 review, baseline A remains the primary comparison anchor. Run C should still report B3 as secondary context, but raw training loss must not be compared directly against A because loss normalization, RVQ weights, and warmup change the loss scale.

Run C completed successfully on 2026-06-12 with `DDP_BROADCAST_BUFFERS=0` after the earlier NCCL buffer-broadcast timeout. Final artifacts:

- Model: `out/sft_full_muon_v3_final_768.pth`
- Logs: `.run_logs/full_train_sft_full_muon_v3_RunC_after_L1_3gpu_20260612_032548_RunC_v3_after_L1_ddpbuf0_tmux/`
- L1 eval: `eval_results/sft_full_muon_v3_final_l1_bf16_t160/`
- Three-way report: `docs/evaluation_results/compare_muon_v3_l1_t160_vs_baseline_and_b3_20260612.md`
- AI-assisted review: `docs/evaluation_results/review_muon_v3_l1_t160_vs_baseline_l1_t160.md` (A 17 / C 25 / Tie 15 overall; audio-specific verdict unresolved because CER, semantics, and repeat behavior conflict)

---

## Plan Summary

### Phase 1 · Inference & eval infrastructure (pre Run B)

1. Fix `model_omni.py` `stream_generate` sampling numerics (fp32 softmax/multinomial).
2. Default inference dtype `bf16` in `batch_validate_omni.py` and `eval_omni.py`.
3. Parameterize baseline launcher: `WEIGHT_PREFIX`, stage hyperparams, `DRY_RUN=1`.
4. Add training observability: `--wandb_run_name`, `samples/sec` log, `--metrics_path` JSON.
5. Add `scripts/compare_eval_runs.py` for eval A/B reports.

### Phase 2 · Run C Phase-0 correctness (this implementation)

1. LR warmup (`--warmup_ratio`, default 0).
2. Global token-weighted loss (`--loss_norm global`, default `local`).
3. RVQ layer weighting + stop weight parameterization.
4. fp32 resume checkpoint weights (export weights still half).
5. Val loss monitoring (`make_val_split.py`, `--val_data_path`, `--val_interval`).
6. v3 launcher combining v2 schedule + all Phase-0 flags.

MoE aux mask was intentionally skipped (all current runs use dense `use_moe=0`).

---

## Code Changes

### Model / inference

| File | Change |
| --- | --- |
| `model/model_omni.py` | Text and audio sampling: logits `.float()` before temperature/top_p/softmax/multinomial |
| `eval_omni.py` | Add `--dtype` (default `bf16`); replace hardcoded `model.half()` |
| `scripts/batch_validate_omni.py` | Default `--dtype` changed from `fp16` to `bf16` |

### Trainer core

| File | Change |
| --- | --- |
| `trainer/trainer_utils.py` | `get_lr(..., warmup_steps=0)` linear warmup then cosine; resume `model` stored at original precision, export ckpt still fp16 |
| `trainer/train_sft_omni.py` | New args: `warmup_ratio`, `loss_norm`, `rvq_layer_weights`, `audio_stop_weight`, `val_data_path`, `val_interval`; `compute_batch_loss()`; distributed `evaluate_loader()`; metrics JSON extended |

**New CLI defaults (legacy-safe):**

```text
--warmup_ratio 0.0
--loss_norm local
--rvq_layer_weights 1,1,1,1,1,1,1,1
--audio_stop_weight 10.0
--val_data_path ""  (disabled)
--val_interval 0
```

**Run C (v3) effective values:**

```text
--warmup_ratio 0.02
--loss_norm global
--rvq_layer_weights 2,1.5,1.2,1,0.8,0.7,0.6,0.5
--val_interval 500
--val_data_path ../dataset/_val/sft_{t2a,a2a,i2t}_val.parquet  (per stage)
```

### Launchers & scripts

| File | Role |
| --- | --- |
| `scripts/run_full_train_muon_dense_3gpu.sh` | Baseline orchestrator; env-driven stage params + optional Phase-0 passthrough |
| `scripts/run_full_train_muon_v2_3gpu.sh` | Run B: v2 schedule, same training math as A |
| `scripts/run_full_train_muon_v3_3gpu.sh` | Run C: v2 schedule + Phase-0 flags |
| `scripts/make_val_split.py` | Deterministic val parquet sampling into `dataset/_val/` |
| `scripts/compare_eval_runs.py` | Markdown comparison from two eval `summary.json` dirs |
| `scripts/test_loss_equivalence.py` | Asserts local/default loss matches pre-refactor formula |

### Documentation

| File | Content |
| --- | --- |
| `docs/training_plans/next_model_comparison_muon_v2_20260611.md` | Run B/C matrix, launch commands, Run C section added |

---

## v2 Schedule Differences (Run B / C shared)

| Stage | Baseline (A) | v2 / v3 (B/C) |
| --- | ---: | ---: |
| S4 I2T vision projector epochs | 1 | 2 |
| S5 I2T all epochs | 1 | 2 |
| S6 A2A final `learning_rate` | 5e-6 | 2e-6 |
| S6 A2A final `muon_lr` | 0.001 | 0.0005 |
| All other stages | unchanged | unchanged |

---

## Verification Results (pre full train)

All checks run on 2026-06-11 without starting 3-GPU full training.

### Static / unit checks

| Check | Command | Result |
| --- | --- | --- |
| Python compile | `python -m py_compile trainer/*.py scripts/{make_val_split,test_loss_equivalence,compare_eval_runs}.py` | Pass |
| Loss equivalence | `python scripts/test_loss_equivalence.py` | `text diff=0.00e+00, audio diff=0.00e+00` |
| CLI args | `python trainer/train_sft_omni.py --help` | New Phase-0 flags present |
| Launcher syntax | `bash -n scripts/run_full_train_muon_{dense,v2,v3}_3gpu.sh` | Pass |

### DRY_RUN launcher chains

```bash
DRY_RUN=1 bash scripts/run_full_train_muon_v2_3gpu.sh   # Run B
DRY_RUN=1 bash scripts/run_full_train_muon_v3_3gpu.sh   # Run C
```

v3 stage-1 command includes (verified):

```text
--warmup_ratio 0.02 --loss_norm global
--rvq_layer_weights 2,1.5,1.2,1,0.8,0.7,0.6,0.5
--val_data_path ../dataset/_val/sft_t2a_val.parquet --val_interval 500
--save_weight sft_full_muon_v3_t2a
```

Weight chain: `llm → *_t2a → *_a2a_proj → *_a2a_full → *_i2t_proj → *_i2t_full → *_a2a_final → *_final`

### Inference dtype smoke

| Dtype | Test | Result |
| --- | --- | --- |
| bf16 | 1-case T2A via `batch_validate_omni.py` on `sft_full_muon_final` | Pass (1/1) |
| fp16 | Same | **Fail** (non-finite probs at multinomial); use bf16/fp32 for eval |

### Single-GPU Phase-0 smoke (Run C flags)

Command (32-row subset, 8 steps):

```bash
CUDA_VISIBLE_DEVICES=0 python trainer/train_sft_omni.py \
  --data_path dataset/_val_mini/sft_t2a_mini_val.parquet \
  --val_data_path dataset/_val_mini/sft_t2a_mini_val.parquet \
  --warmup_ratio 0.02 --loss_norm global \
  --rvq_layer_weights 2,1.5,1.2,1,0.8,0.7,0.6,0.5 \
  --val_interval 4 --epochs 1 --batch_size 4 --from_weight llm \
  --save_weight smoke_phase0 \
  --metrics_path .run_logs/smoke_phase0.metrics.json
```

Observed metrics (`.run_logs/smoke_phase0.metrics.json`):

| Metric | Value |
| --- | ---: |
| final_loss | 12.032 |
| final_text_loss | 4.469 |
| final_audio_loss | 7.563 |
| last_val_loss | 10.217 |
| last_val_text | 2.878 |
| last_val_audio | 7.340 |
| warmup_ratio | 0.02 |
| loss_norm | global |

fp32 resume check:

```text
checkpoints/smoke_phase0_768_resume.pth  → model dtype: float32
checkpoints/smoke_phase0_768.pth         → export dtype: float16
```

### Eval compare script smoke

```bash
python scripts/compare_eval_runs.py \
  --run_a eval_results/sft_full_muon_final_batch_audio_fp32 \
  --run_b eval_results/sft_full_muon_final_batch_audio_bf16 \
  --label_a fp32 --label_b bf16 \
  --output docs/evaluation_results/compare_fp32_vs_bf16_test.md
```

Output: 13/13 pass for both; report written successfully.

---

## Artifacts for Post-Run Traceability

### Run A (baseline, already exists)

```text
out/sft_full_muon_final_768.pth
checkpoints/sft_full_muon_final_768_resume.pth
.run_logs/full_train_muon_dense_3gpu_20260610_*/
eval_results/sft_full_muon_final_batch_audio_{fp32,bf16}/
docs/training_summaries/full_train_muon_dense_3gpu_20260610.md
docs/evaluation_results/sft_full_muon_final_eval_20260610.md
```

### Expected after Run B

```text
out/sft_full_muon_v2_final_768.pth
.run_logs/full_train_sft_full_muon_v2_3gpu_YYYYMMDD_HHMMSS/
  01_t2a_all.metrics.json … 07_i2t_final_vision_proj.metrics.json
  gpu_telemetry.csv
eval_results/sft_full_muon_v2_final_batch_audio_bf16/
docs/evaluation_results/compare_muon_v2_vs_baseline.md
```

### Expected after Run C

```text
out/sft_full_muon_v3_final_768.pth
.run_logs/full_train_sft_full_muon_v3_3gpu_YYYYMMDD_HHMMSS/
  *.metrics.json  (includes warmup_ratio, loss_norm, rvq weights, val_metrics)
dataset/_val/sft_{t2a,a2a,i2t}_val.parquet
eval_results/sft_full_muon_v3_final_batch_audio_bf16/
docs/evaluation_results/compare_muon_v3_vs_baseline_l1_t160.md
```

### Per-stage metrics JSON fields (all runs using updated trainer)

```json
{
  "save_weight", "mode", "optimizer", "epochs", "batch_size", "global_batch_size",
  "learning_rate", "muon_lr", "data_path", "from_weight",
  "total_steps", "elapsed_seconds", "samples_per_sec",
  "final_loss", "final_text_loss", "final_audio_loss",
  "warmup_ratio", "loss_norm", "rvq_layer_weights", "audio_stop_weight",
  "val_data_path", "val_interval", "last_val_metrics",
  "epoch_metrics": [{ "val_metrics": { "loss", "text", "audio" } }]
}
```

---

## Detailed Review of the Comparison & Validation Plan

This review uses the pipeline map at `/home/genesis/.cursor/projects/home-genesis-Projects-minimind-o/canvases/minimind-o-pipeline-upgrade-map.canvas.tsx` as the broader system reference. The key message from that map is that the next comparison should not only ask "which checkpoint is better", but also preserve attribution across the nine bottleneck areas: data loading, augmentation, collate/packing, frozen encoders, model forward path, loss, optimizer/distributed training, checkpoint/monitoring, and evaluation/RL readiness.

### What the current plan gets right

| Design choice | Why it is good | Remaining caveat |
| --- | --- | --- |
| A is frozen as the completed baseline | It gives a real full-train reference with logs, metrics, checkpoints, and eval artifacts. | A was trained before Phase-0 fixes, so future loss numbers are not always apples-to-apples. |
| B changes schedule only | A vs B remains the cleanest quality comparison because loss math, optimizer, data, batch, and model are held constant. | B still bundles two hypotheses: more I2T training and gentler S6 A2A final. |
| C is separated from B | Warmup, global loss norm, RVQ weights, and fp32 resume are correctness changes; separating them prevents silent attribution errors. | C loss scale differs; C must be compared against B's winner mostly through eval and curve shape, not raw final loss alone. |
| T throughput is postponed | Packing/streaming/offline encoders can change data order and effective token mix; isolating throughput work avoids confusing quality conclusions. | T needs its own invariance tests before any quality claim. |
| bf16 is now the default eval dtype | It directly addresses the fp16 non-finite-probability failure seen in Run A eval. | fp16 should remain a tracked failure mode, not silently ignored. |

The original A/B/C/T split is therefore directionally sound for attribution. After Run C, the quality line now adds **Run D audio recovery** (`docs/training_plans/next_model_training_runD_20260613.md`) before any throughput work.

### Attribution risks in the current B/C matrix

| Risk | Why it matters | Recommendation |
| --- | --- | --- |
| Run B bundles I2T schedule and A2A LR changes | If I2A improves but A2A regresses, or vice versa, it is unclear which change did the work. | Keep B as the practical candidate run. If B is mixed, add B1 = only S4/S5 2 epochs and B2 = only S6 lower LR before moving to C. |
| Run C changes loss scale | `loss_norm=global` and RVQ weights alter numeric loss magnitude and gradient allocation. | Do not compare C raw loss directly with A. Compare C against A on L1 eval/ASR/manual review and use val curve shape, audio CE components, and per-RVQ metrics for training health. |
| Val split may overlap training shards | Current `_val` is for curve monitoring, not held-out generalization. | Add a later `dataset/_heldout/` built from source rows before shard creation or by stable row hash exclusion. Keep `_val` labeled "monitoring only". |
| `eval_muon_mini.jsonl` has only 13 cases | It is enough for smoke regression, but too small to prove quality movement. | Promote it to L0 smoke. Add L1 objective set with at least 50 T2A, 30 A2A, 50 I2A cases before making strong claims. |
| Basic pass is too weak | Text length + audio frames can pass while semantics are wrong. | Add ASR round-trip, image grounding rubric, audio duration/silence checks, and human blind win/tie/loss. |
| Sampling noise can mask small gains | `temperature=0.7`, `top_p=0.85` introduces stochastic variance. | For each candidate run: fixed-seed sample for continuity, plus either greedy/low-temp deterministic pass or 3-seed aggregate for final comparison. |
| 96-token cap truncates some outputs | A/B differences may reflect truncation behavior rather than knowledge or grounding. | Keep 96-token run for historical parity, but always run a secondary `max_new_tokens=160` qualitative set for T2A/A2A/I2A. |
| Audio quality is not scored | MP3 decode success and frame count do not measure intelligibility, naturalness, or speaker stability. | Add SenseVoiceSmall ASR round-trip CER/WER, silence/clipping checks, optional UTMOS, and a small manual listening panel. |
| I2A grounding is under-measured | The two image probes reveal hallucination, but not enough categories. | Add golden image prompts for object identity, count, color, spatial relation, OCR, and "do not hallucinate" negative cases. |

### Recommended run decomposition

The original recommended default was:

```text
A baseline completed -> B v2 schedule -> C v3 Phase-0 fixes -> T throughput
```

2026-06-13 update: after Run C, text/image improved but audio CER and repeat regressed monotonically across A -> B3 -> C. The next quality run is therefore **Run D audio recovery**, while throughput remains **Run T**.

Use the following decision tree after Run B:

| Run B outcome | Next action |
| --- | --- |
| B improves I2A and A2A without T2A regression | Promote B as the new schedule baseline and run C against B. |
| B improves I2A but hurts A2A | Run B1: only S4/S5 epochs 2; keep S6 baseline LR. Then decide whether to keep the I2T extension. |
| B improves A2A but not I2A | Run B2: only S6 lower LR; keep S4/S5 baseline epochs. Then re-evaluate whether extra I2T epochs are worth the cost. |
| B is neutral on eval but lower training loss | Do not promote on loss alone. Run expanded L1 eval before spending on C. |
| B regresses broadly | Stop B line, keep A as baseline, run a smaller C-smoke or revisit schedule assumptions. |

If compute budget allows a cleaner scientific split, run this matrix instead:

| Run | Change | Purpose |
| --- | --- | --- |
| B1 | S4/S5 epochs 1 -> 2 only | Isolate I2A grounding effect. |
| B2 | S6 LR/muon_lr lower only | Isolate A2A final stability effect. |
| B3 | B1 + B2 | Confirm combined schedule has additive value. |
| C | Best B schedule + Phase-0 fixes | Test correctness upgrades after schedule is chosen. |

If compute budget is tight, run B3 first as currently planned, but treat it as an engineering candidate rather than a pure ablation.

### Evaluation gate upgrade

The pipeline map recommends moving from smoke tests to a layered evaluation ladder. For the next comparison, use these gates:

| Gate | Scope | Required for | Metrics |
| --- | --- | --- | --- |
| L0 smoke | Current 13-case `eval_muon_mini.jsonl` | Every checkpoint and every dtype sanity check | pass rate, repeat, special-code rate, MP3 decode, dtype failure |
| L1 objective | Expanded fixed set | B/C promotion | ASR CER/WER, text repeat, language match, silence/clipping, duration, image QA rubric |
| L2 manual | Small blind review | Final winner selection | T2A helpfulness, A2A input understanding, audio intelligibility, I2A grounding win/tie/loss |
| L3 regression | CI-like quick suite | Before long training or release | no fp16/bf16 numeric crash, no missing keys, no empty audio, no severe repetition |

Minimum L1 set recommended before claiming B or C is better:

```text
T2A: 50 prompts
  20 English short, 10 English long, 10 Chinese short, 10 Chinese long
A2A: 30 audio prompts
  10 English, 10 Chinese, 5 noisy/fast/slow, 5 edge cases
I2A: 50 image prompts
  15 object/caption, 10 count/color, 10 spatial relation, 10 OCR/text-in-image, 5 negative/hallucination traps
```

For generated audio, add:

```text
SenseVoiceSmall ASR -> transcript
CER/WER between generated transcript and generated text
silence ratio / clipping ratio / duration
optional UTMOS or human 1-5 MOS
```

For image grounding, add a small human-readable rubric:

```text
0 = mostly wrong or hallucinated
1 = recognizes main object but wrong details
2 = mostly correct with minor omissions
3 = correct and specific
```

The final comparison report should show both automatic metrics and a compact failure gallery. The failure gallery is more important than a single pass-rate number because the current model can pass structurally while still hallucinating.

### Training metric recommendations

The new `*.metrics.json` is a good start. For the next full run, add or manually derive:

| Metric | Why it matters |
| --- | --- |
| Per-stage wall clock and samples/sec | Confirms schedule changes do not hide throughput regressions. |
| Global train loss, text loss, audio loss | Avoids rank-local reporting bias. |
| Val monitoring loss | Catches spikes and overfitting; use as trend only until heldout split is strict. |
| Per-RVQ-layer CE | Needed to know whether layer-0 weighting helps semantic audio codes or only shifts loss. |
| Audio stop-token rate | Detects early stop or overlong audio degeneration. |
| Gradient norm by module | Finds projector/talker/thinker imbalance after schedule or loss changes. |
| Max logit / non-finite guard | Prevents another fp16-style failure from being discovered only during eval. |
| GPU memory + utilization + power | Keeps the 3x5090D utilization story reproducible. |

Short-term implementation priority:

1. Add per-RVQ-layer CE to metrics JSON.
2. Add non-finite logit/loss guard with stage, step, rank, and sample ids.
3. Add ASR round-trip eval script before Run C.
4. Create a strict heldout split before any claim about generalization.

### Acceptance criteria refinements

Run B should be promoted only if all of these hold:

- 7 stages finish without OOM/hang.
- S6 final loss is not worse than A's S6 (`4.9898`) and preferably approaches A's S3 (`4.6759`).
- L0 bf16 pass remains 13/13 with repeat not worse than A.
- I2A manual score improves on both existing image probes, or at least no longer hallucinates obvious object/color details.
- T2A and Chinese probes do not regress versus A.

Run C should be promoted only if:

- C eval is no worse than baseline A on L0 and L1.
- Val curves are smooth enough for monitoring and do not show unexplained spikes.
- fp32 resume has been verified after a real checkpoint, not only smoke.
- Per-RVQ-layer metrics show the weighting is doing something interpretable.

Run T should be accepted only if:

- Same checkpoint-quality path is preserved within tolerance.
- Throughput improves materially, e.g. `samples/sec` or token throughput +20% or better.
- Packed/streamed data does not alter loss trend beyond a small tolerance on a fixed short run.

### Immediate changes recommended before launching Run B

1. Snapshot the eval set and write a manifest with file hashes for prompts, images, and audio.
2. Run the 160-token baseline eval for A before B, so B's long-output eval has a fair baseline.
3. Add a simple manual review template under `docs/evaluation_results/` with per-case win/tie/loss and notes.
4. Keep `bf16` as primary eval dtype; run `fp32` only as a diagnostic parity check.
5. Record the independent Cursor `git add` process separately if it remains active, because it can distort CPU/disk observations even though it does not affect GPU training.

---

## Implementation Update After Plan Review

Reviewer feedback accepted the analysis and highlighted the minimum action sequence:

```text
freeze eval manifest -> run A 160-token baseline -> launch B3 -> split B1/B2 only if mixed -> add ASR/per-layer CE before C
```

This section records the concrete changes and validation run after that review.

### Plan updates

`docs/training_plans/next_model_comparison_muon_v2_20260611.md` now includes:

- pre-run eval manifest command;
- A baseline 160-token eval command;
- B3 default run definition;
- B1 and B2 no-code-change launch commands using existing environment variables;
- manual review template command;
- optional ASR round-trip command.

B split launch mapping:

| Run | Purpose | How to launch |
| --- | --- | --- |
| B1 | isolate I2T schedule | `EPOCHS_I2T_PROJ=2 EPOCHS_I2T_ALL=2 bash scripts/run_full_train_muon_dense_3gpu.sh` |
| B2 | isolate A2A final LR | `LR_A2A_FINAL=2e-6 MUON_LR_A2A_FINAL=0.0005 bash scripts/run_full_train_muon_dense_3gpu.sh` |
| B3 | combined v2 candidate | `bash scripts/run_full_train_muon_v2_3gpu.sh` |

### New scripts

| File | Purpose | Validation |
| --- | --- | --- |
| `scripts/snapshot_eval_manifest.py` | Hash eval jsonl and referenced media files. | Ran on `dataset/eval_muon_mini.jsonl`; missing files = 0. |
| `scripts/make_manual_review_template.py` | Create a manual A/B review markdown table. | Generated `review_baseline_96_vs_160_template.md`. |
| `scripts/asr_eval_generated_audio.py` | Optional SenseVoiceSmall ASR round-trip with CER/WER; legacy faster-whisper backend retained for old reports. | Defaults changed to SenseVoice CPU batch (`--device cpu --batch_size 16`) because L0/L1 underutilize GPU. |

Generated artifacts:

```text
docs/evaluation_results/eval_muon_mini_manifest_20260611.json
docs/evaluation_results/eval_muon_mini_manifest_20260611.md
docs/evaluation_results/review_baseline_96_vs_160_template.md
eval_results/sft_full_muon_final_batch_audio_bf16_t160/
.run_logs/eval/sft_full_muon_final_batch_audio_bf16_t160.log
```

Eval manifest summary:

```text
test_set_sha256 = 7ff2200345dc4f6cd4978a0b41844be52ff05bfb3ce34211331ce95e3b795b11
cases = 13
referenced_files = 6
missing_files = 0
```

### A baseline 160-token eval

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_final \
  --output_dir eval_results/sft_full_muon_final_batch_audio_bf16_t160 \
  --dtype bf16 --max_new_tokens 160 --decode_audio
```

Result:

| Metric | Value |
| --- | ---: |
| basic pass | 13/13 |
| dtype | bf16 |
| max_new_tokens | 160 |
| special_code_rate | 0.0000 for all cases |

Observed note: longer generation exposes more quality issues even when structural pass is 13/13. Example: `t2a_en_space`, `t2a_en_coffee`, and both I2A cases remain semantically weak. This reinforces that B/C promotion should not rely on pass rate alone.

### B dry-run validation

All three B variants were dry-run validated after launcher parameterization:

```text
.run_logs/dryrun_muon_v2_b3_20260611.log
.run_logs/dryrun_muon_b1_i2t_20260611.log
.run_logs/dryrun_muon_b2_a2a_lr_20260611.log
```

Confirmed:

- B3 uses S4/S5 epochs = 2 and S6 `learning_rate=2e-6`, `muon_lr=0.0005`.
- B1 uses S4/S5 epochs = 2 and keeps S6 baseline LR.
- B2 keeps S4/S5 baseline epochs and uses S6 lower LR.
- All variants pass `FINITE_GUARD=1` through the launcher.

### Trainer changes and smoke validation

Trainer changes:

- per-RVQ-layer CE is recorded in `final_detail_metrics` and val metrics;
- per-layer target counts, stop counts, and stop rates are recorded;
- finite loss guard is enabled by default via `--finite_guard 1`;
- optional full logits guard is controlled by `--finite_guard_logits_interval`;
- launcher exposes `FINITE_GUARD` and `FINITE_GUARD_LOGITS_INTERVAL`.

Static checks:

| Check | Result |
| --- | --- |
| `python -m py_compile ...` | Pass |
| `bash -n scripts/run_full_train_muon_{dense,v2,v3}_3gpu.sh` | Pass |
| `scripts/test_loss_equivalence.py` | `text diff=0.00e+00, audio diff=0.00e+00` |

Single-GPU smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft_omni.py \
  --data_path ../dataset/_val_mini/sft_t2a_mini_val.parquet \
  --val_data_path ../dataset/_val_mini/sft_t2a_mini_val.parquet \
  --warmup_ratio 0.02 --loss_norm global \
  --rvq_layer_weights 2,1.5,1.2,1,0.8,0.7,0.6,0.5 \
  --val_interval 4 --epochs 1 --batch_size 4 --from_weight llm \
  --save_weight smoke_metrics_guard_20260611 \
  --metrics_path ../.run_logs/smoke_metrics_guard_20260611.metrics.json \
  --finite_guard 1 --finite_guard_logits_interval 1
```

Smoke result:

| Metric | Value |
| --- | ---: |
| final_loss | 12.2474 |
| final_text_loss | 4.5998 |
| final_audio_loss | 7.6476 |
| last_val_loss | 10.5905 |
| last_val_text | 3.1263 |
| last_val_audio | 7.4643 |
| finite_guard | true |
| finite_guard_logits_interval | 1 |

Final RVQ layer losses from smoke:

```text
[7.642604, 7.436232, 7.686808, 7.69857, 7.717558, 7.712369, 7.749001, 7.781327]
```

Val RVQ layer losses from smoke:

```text
[7.364997, 7.211297, 7.479949, 7.530988, 7.583414, 7.616978, 7.644253, 7.677642]
```

This validates the trainer path needed before Run C. ASR round-trip is now available through SenseVoiceSmall CPU-batch inference.

### Post-review notes

Follow-up review confirmed the added plan and code paths are consistent with the current repository state. Two operational caveats were recorded:

1. **ASR runtime should use SenseVoice CPU batch by default.** L0/L1 ASR batches are too small to occupy a GPU efficiently. `scripts/asr_eval_generated_audio.py` now defaults to SenseVoice and should be run as:

```text
--backend sensevoice --device cpu --batch_size 16
```

2. **Finite guard changes failure behavior, not numerics.** `FINITE_GUARD=1` is now the safe default, so NaN/Inf stops training early instead of allowing a bad run to continue. For strict old Run A behavior reproduction, set:

```bash
FINITE_GUARD=0
```

The formal monitoring splits were generated after fixing `make_val_split.py` to avoid pyarrow nested-list offset overflow during whole-table `take()`:

```bash
python scripts/make_val_split.py --dataset_dir dataset --output_dir dataset/_val
```

Result:

```text
sft_t2a.parquet: wrote 1000/1248923 rows -> dataset/_val/sft_t2a_val.parquet
sft_a2a.parquet: wrote 1000/414024 rows -> dataset/_val/sft_a2a_val.parquet
sft_i2t.parquet: wrote 1000/2904511 rows -> dataset/_val/sft_i2t_val.parquet
```

v3 dry-run after split generation confirmed all seven stages include the expected `--val_data_path` and Phase-0 flags:

```text
.run_logs/dryrun_muon_v3_after_val_20260611.log
```

These `_val` files are still for monitoring only, not held-out generalization.

---

## How to Run Next

### Before Run B or C

```bash
pgrep -af 'torchrun|train_sft_omni.py'   # no active training
nvidia-smi                                # GPUs idle
ls dataset/_full_shards/                  # rank shards exist
```

### Run B (schedule only)

```bash
DRY_RUN=1 bash scripts/run_full_train_muon_v2_3gpu.sh
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train bash scripts/run_full_train_muon_v2_3gpu.sh
```

### Run C (schedule + Phase-0)

```bash
python scripts/make_val_split.py
DRY_RUN=1 bash scripts/run_full_train_muon_v3_3gpu.sh
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train bash scripts/run_full_train_muon_v3_3gpu.sh
```

### Post-run eval (each candidate)

```bash
python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v2_final \
  --output_dir eval_results/sft_full_muon_v2_final_batch_audio_bf16 \
  --dtype bf16 --decode_audio

python scripts/compare_eval_runs.py \
  --run_a eval_results/sft_full_muon_final_batch_audio_bf16 \
  --run_b eval_results/sft_full_muon_v2_final_batch_audio_bf16 \
  --label_a baseline --label_b muon_v2 \
  --output docs/evaluation_results/compare_muon_v2_vs_baseline.md
```

---

## Success Criteria Checklist (fill after full runs)

### Run B vs A

- [ ] All 7 stages complete without OOM/hang
- [ ] S6 final loss ≤ 4.99 (A rebound level) and ideally ≤ 4.68 (A S3)
- [ ] I2A grounding improved on manual review of `eval_muon_mini.jsonl` image cases
- [ ] bf16 eval pass rate ≥ 13/13; repeat score not worse than A (0.0015 fp32 baseline)
- [ ] `compare_muon_v2_vs_baseline.md` generated

### Run C vs baseline A

- [ ] Val loss curves logged every 500 steps; no unexplained spikes
- [ ] Resume checkpoint model dtype float32 (not half-truncated)
- [ ] S6 A2A final loss stable vs S3
- [ ] bf16 L1 eval no regression vs A on pass rate, ASR proxy, and manual review
- [ ] Per-stage `*.metrics.json` archived under `.run_logs/`

---

## Known Limitations

1. **Val splits** (`dataset/_val/`) may overlap slightly with training shards; use for curve monitoring only, not held-out generalization claims.
2. **fp16 inference** remains unsafe for current checkpoints; always evaluate with `--dtype bf16` or `fp32`.
3. **Loss values across runs** are only directly comparable when `loss_norm` and `rvq_layer_weights` match; compare A vs B on eval metrics, B vs C on both loss trends and eval.
4. **Finite guard** is fail-fast behavior. It does not change normal numeric results, but strict legacy replay can set `FINITE_GUARD=0`.
5. **ASR eval** should default to SenseVoice CPU batch for L0/L1. Use GPU only for larger ASR batches where utilization is worthwhile.
6. **MoE aux mask**, packing, streaming data, and FSDP are planned separately (Run T) and not included in this version. Run D is now reserved for audio-recovery quality training after Run C.

---

## Change Log (file-level)

```text
2026-06-11  model/model_omni.py              fp32 sampling in generate()
2026-06-11  eval_omni.py                     --dtype bf16 default
2026-06-11  scripts/batch_validate_omni.py   --dtype bf16 default
2026-06-11  trainer/train_sft_omni.py        metrics, loss refactor, Phase-0 flags, val eval, RVQ layer metrics, finite guard
2026-06-11  trainer/trainer_utils.py           warmup LR, fp32 resume
2026-06-11  scripts/run_full_train_muon_dense_3gpu.sh   parameterized + Phase-0 env
2026-06-11  scripts/run_full_train_muon_v2_3gpu.sh      Run B wrapper
2026-06-11  scripts/run_full_train_muon_v3_3gpu.sh      Run C wrapper
2026-06-11  scripts/make_val_split.py          new; stream-batch sampling fix for nested-list parquet
2026-06-11  scripts/compare_eval_runs.py       new
2026-06-11  scripts/test_loss_equivalence.py   new
2026-06-11  scripts/snapshot_eval_manifest.py  eval-set manifest/hash snapshot
2026-06-11  scripts/make_manual_review_template.py  manual A/B review template
2026-06-11  scripts/asr_eval_generated_audio.py     SenseVoice CPU-batch CER/WER; legacy whisper backend retained
2026-06-11  docs/training_plans/next_model_comparison_muon_v2_20260611.md  Run C section
2026-06-11  docs/training_summaries/training_pipeline_upgrade_20260611.md  this document
```
