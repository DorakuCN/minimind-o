# Next Model Training Plan - Run E - 2026-06-13

## TL;DR

Run D-b disproved the strong form of the original "audio forgotten by the image tail and can be fixed by a stronger Stage6" hypothesis. A stage-localized diagnostic shows the **audio regression in Run C is already visible at Stage3**, before the image-heavy tail finishes. The cause is not proven yet: v3 fine-RVQ-layer downweighting is a plausible mechanism, but `loss_norm=global`, warmup, and A2A-stage interactions remain open confounds. Run E tests uniform RVQ with a cheap Stage1-3 probe before committing to a full retrain.

## Diagnostic Evidence (2026-06-13)

Audio / Text CER (L1, 57 cases, bf16, t160, SenseVoice CER):

| Checkpoint | Audio CER | Text CER |
| --- | ---: | ---: |
| **A Stage3** (`sft_full_muon_a2a_full`) | **0.2965** | 0.2657 |
| **C Stage3** (`sft_full_muon_v3_a2a_full`) | 0.3976 | 0.2378 |
| A final (`sft_full_muon_final`) | 0.3464 | 0.2412 |
| C final (`sft_full_muon_v3_final`) | 0.3577 | 0.2176 |
| D-b final (`sft_full_muon_v4_audiofix_a2a_final`) | 0.3571 | 0.2387 |

Decomposition of the audio gap:

| Effect | Magnitude | Note |
| --- | ---: | --- |
| Phase-0 damage at audio stage (C3 - A3) | **+0.1011** | dominant, ~2/3 of the gap |
| Image-tail forgetting, A (A3 -> A final) | +0.0499 | real for A (uniform RVQ) |
| Image-tail forgetting, C (C3 -> C final) | -0.0399 | C audio actually improved through the tail |
| D-b Stage6 strengthening (C final -> D-b final) | -0.0006 | negligible -> wrong target |

Conclusions:

1. The original tail-forgetting hypothesis is true for **A** but false for **C**; C's audio was already damaged at Stage3, before any image training.
2. Stages 1-3 use **identical epoch schedules** in A and C (T2A 6ep, A2A_proj 1ep, A2A_full 3ep). The v2 schedule only changes Stage4-7. So the Stage3 audio gap can only come from `{rvq_layer_weights, loss_norm, warmup_ratio}`.
3. The v3 RVQ weights `2,1.5,1.2,1,0.8,0.7,0.6,0.5` (sum 8.3) give the coarsest layer 24% weight and the finest layer 6%, versus uniform 12.5% each. Per `compute_batch_loss` (`trainer/train_sft_omni.py:87-89`), this directly downweights the fine RVQ codes that may carry audio intelligibility. This is a plausible mechanistic explanation and the primary Run E target, but it is not yet proven.
4. D-b strengthening Stage6 was addressing the wrong stage, which is why audio CER barely moved (0.3577 -> 0.3571). D-b is still a useful balanced checkpoint (best WER 0.4926, best image CER 0.5232, repetition fixed), and remains the current deployable fallback.
5. Early Stage1 T2A evidence is neutral-to-weak: uniform RVQ wins more cases (`14:9`) and improves WER (`0.5740 -> 0.4920`), but mean CER is slightly worse (`0.2438 -> 0.2536`) because of one large outlier (`t2a_en_space`). Since the original large gap is in A2A Stage3, Stage1 T2A cannot decide the hypothesis.

## Hypothesis Under Test

Primary: fine-RVQ-layer downweighting may be the dominant cause of C's audio-intelligibility regression. Reverting to uniform RVQ weights at the audio-bearing stages should recover Stage3 audio CER toward A's `0.2965` if this hypothesis is correct.

Secondary confounds to clear if needed: `loss_norm=global`, warmup, or A2A projection/full-stage interaction could also contribute. `warmup_ratio` is currently assumed lower-risk, but it is not formally isolated.

## Run E Matrix

| Run | Variable axis | Cost | Status |
| --- | --- | --- | --- |
| **E-probe** | Stages 1-3 only, RVQ uniform, else = C | ~4.5h | Running; Stage1 done, Stage2-3 underway |
| E-probe-ln | + `loss_norm=local` (only if E-probe fails) | ~4.5h | Conditional |
| **E-full** | Full pipeline, RVQ uniform + D-b audio tail | ~9-10h | Planned (after probe passes) |

Keep as defaults (proven beneficial, do not revert): `warmup_ratio=0.02`, val monitoring, fp32 resume, `DDP_BROADCAST_BUFFERS=0`, v2 I2T 2-epoch schedule.

---

## E-probe (do first) - Localize the RVQ cause cheaply

Re-run only Stages 1-3 with uniform RVQ weights; everything else identical to Run C. Then eval the resulting `_a2a_full` audio CER.

Dry-run:

```bash
DRY_RUN=1 \
WEIGHT_PREFIX=sft_full_muon_v5_rvquniform \
RUN_GROUP=muon_v5_rvquniform \
DDP_BROADCAST_BUFFERS=0 \
RVQ_LAYER_WEIGHTS="1,1,1,1,1,1,1,1" \
START_STAGE=1 END_STAGE=3 \
bash scripts/run_full_train_muon_v3_3gpu.sh
```

Full probe:

```bash
WEIGHT_PREFIX=sft_full_muon_v5_rvquniform \
RUN_GROUP=muon_v5_rvquniform \
DDP_BROADCAST_BUFFERS=0 \
RVQ_LAYER_WEIGHTS="1,1,1,1,1,1,1,1" \
START_STAGE=1 END_STAGE=3 \
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train \
bash scripts/run_full_train_muon_v3_3gpu.sh
```

Optional ~2h-cheaper variant: add `EPOCHS_T2A=3`. This lowers absolute audio quality but still reveals the uniform-vs-downweighted direction; compare the trend, not the absolute number.

Eval the probe checkpoint (Stage3 output):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v5_rvquniform_a2a_full \
  --output_dir eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160 \
  --test_set dataset/eval_muon_l1.jsonl \
  --dtype bf16 --max_new_tokens 160 --decode_audio

python scripts/asr_eval_generated_audio.py \
  --results eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160/results.jsonl \
  --output eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160/asr_eval.json \
  --markdown eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160/asr_eval.md \
  --backend sensevoice --sensevoice_project /home/genesis/Projects/SenseVoice \
  --sensevoice_model model/SenseVoiceSmall --device cpu --batch_size 16
```

### E-probe decision gate

Compare probe Stage3 audio CER against the diagnostic anchors (C3 `0.3976`, A3 `0.2965`):

- If audio CER `<= ~0.31` (clearly better than C3, approaching A3): RVQ downweighting confirmed as dominant -> proceed to **E-full**.
- If audio CER stays `~0.38-0.40`: RVQ is not the cause; run **E-probe-ln** with `LOSS_NORM=local` to test the loss-norm confound before any full retrain.
- If `EPOCHS_T2A=3` was used, judge by the gap to a matching 3-epoch C baseline, not the absolute 0.31 threshold.

---

## E-full (after probe passes) - Best-of-all-worlds retrain

Full pipeline with uniform RVQ (A's audio recipe) + C's text/image gains + D-b's stronger audio finale, ending on audio to also counter the +0.05 image-tail forgetting that even A suffers.

```bash
WEIGHT_PREFIX=sft_full_muon_v5 \
RUN_GROUP=muon_v5 \
DDP_BROADCAST_BUFFERS=0 \
RVQ_LAYER_WEIGHTS="1,1,1,1,1,1,1,1" \
EPOCHS_A2A_FINAL=2 \
LR_A2A_FINAL=5e-6 \
MUON_LR_A2A_FINAL=0.001 \
END_STAGE=6 \
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train \
bash scripts/run_full_train_muon_v3_3gpu.sh
```

Rationale for `END_STAGE=6`: D-b showed that ending on Stage6 still yields the best image CER (`0.5232`, from Stage4-5), and Stage7 vision_proj added near-zero val gain. Ending on audio protects the fragile modality. The produced artifact is `sft_full_muon_v5_a2a_final_768.pth`.

### E-full success criteria

- Audio CER `< 0.3464` (strictly better than A final). Stretch goal `<= 0.31`.
- Text CER `<= 0.24` (keep near C/A; do not regress hard).
- Image CER `<= 0.55` (keep C/D-b image gains).
- No OOM / NCCL hang / non-finite loss; Stage6 audio val loss `<=` Stage3.

---

## Evaluation Plan (per candidate)

Identical to the Run C/D pipeline: L1 structural -> SenseVoice ASR -> `compare_eval_runs.py` vs baseline A and vs D-b -> manual review template -> human listening spot-check on audio factuality/refusal cases.

```bash
# compare vs A
python scripts/compare_eval_runs.py \
  --run_a eval_results/sft_full_muon_final_l1_bf16_t160 \
  --run_b eval_results/sft_full_muon_v5_a2a_final_l1_bf16_t160 \
  --label_a baseline_A --label_b v5 \
  --output docs/evaluation_results/compare_v5_vs_baseline_l1_t160.md
```

## Decision Logic

1. E-probe confirms RVQ -> run E-full -> if audio `< 0.3464` and text/image held -> promote v5 as new baseline (after human listening).
2. E-probe fails -> E-probe-ln (`loss_norm=local`) -> if that recovers audio, fold `loss_norm=local` into E-full instead of/in addition to uniform RVQ.
3. If neither recovers audio -> the audio gap is in data/codec/model, not loss weighting; escalate to investigating Mimi decode settings and A2A data quality. Keep D-b as the deployable balanced checkpoint meanwhile.

## Artifacts From This Diagnostic (for traceability)

```text
eval_results/diag_A_a2a_full_l1_bf16_t160/      # A Stage3 audio CER 0.2965
eval_results/diag_v3_a2a_full_l1_bf16_t160/     # C Stage3 audio CER 0.3976
eval_results/sft_full_muon_v4_audiofix_l1_bf16_t160/         # D-b (my run)
eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160/  # D-b (duplicate, correct naming)
docs/evaluation_results/compare_v4_audiofix_vs_baseline_l1_t160.md
```

## Output Naming

```text
out/sft_full_muon_v5_rvquniform_a2a_full_768.pth   # E-probe Stage3 output
out/sft_full_muon_v5_a2a_final_768.pth             # E-full final (ends at Stage6)
.run_logs/full_train_sft_full_muon_v5_*_3gpu_YYYYMMDD_HHMMSS/
docs/training_summaries/full_train_muon_v5_YYYYMMDD.md
docs/evaluation_results/compare_v5_vs_baseline_l1_t160.md
```
