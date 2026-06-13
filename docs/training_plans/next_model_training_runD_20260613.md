# Next Model Training Plan - Run D - 2026-06-13

## Goal

Run C (`sft_full_muon_v3`) is the strongest candidate so far on text and image, but **audio regressed across all three runs** while text/image improved. Run D targets the one remaining structural blocker: **audio capability is forgotten by the image-heavy tail of the curriculum.** This is a scheduling problem, not an optimizer or Phase-0 problem, so Phase-0 fixes are kept as defaults.

Anchor checkpoint for comparison:

```text
out/sft_full_muon_v3_final_768.pth   # Run C (current best text/image)
out/sft_full_muon_final_768.pth      # Run A (still best audio fallback)
```

Candidate checkpoint (after Run D):

```text
out/sft_full_muon_v4_audiofix_a2a_final_768.pth
```

Important: the primary D-b plan intentionally ends at Stage6. The produced training artifact is therefore `_a2a_final`, not `_final`. Do not create a `_final` alias yet; D-b did not meet the primary audio-CER success criterion.

## Status Update - D-b Completed

D-b fast isolation was run from Run C Stage5 by copying:

```text
out/sft_full_muon_v3_i2t_full_768.pth
-> out/sft_full_muon_v4_audiofix_i2t_full_768.pth
```

Then only Stage6 was re-run:

```text
epochs=2, learning_rate=5e-6, muon_lr=0.001, END_STAGE=6, DDP_BROADCAST_BUFFERS=0
```

Training-side result:

| Checkpoint/stage | Val loss | Val text | Val audio |
| --- | ---: | ---: | ---: |
| Run C Stage3 | 4.4796 | 1.1062 | 3.3734 |
| Run C Stage6 | 4.6873 | 1.3113 | 3.3760 |
| D-b Stage6 last | 4.4929 | 1.1391 | 3.3538 |
| D-b Stage6 best | 4.4750 | 1.1231 | 3.3519 |

L1/SenseVoice result (`dataset/eval_muon_l1.jsonl`, bf16, t160, seed 42):

| Model | Overall CER | Text CER | Audio CER | Image CER | Audio repeat | Image repeat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 0.3497 | 0.2412 | **0.3464** | 0.5834 | **0.0017** | 0.0075 |
| C v3 | **0.3364** | **0.2176** | 0.3577 | 0.5401 | 0.0052 | 0.0178 |
| D-b audiofix | 0.3414 | 0.2387 | 0.3571 | **0.5232** | 0.0025 | 0.0080 |

Calibrated conclusion:

- **D-b did not solve the original audio-quality target.** Audio CER is `0.3571`, essentially tied with C (`0.3577`) and still behind A (`0.3464`). The val recovery was mostly text-side and only weakly moved audio val (`3.3760 -> 3.3538`).
- **D-b did produce a cleaner checkpoint than C.** It fixed most of C's repeat regression, preserved or improved the image gain, delivered the best image CER, and achieved the best WER (`0.4926`, with the usual caution that short Chinese WER is unreliable).
- **Do not promote D-b as the new baseline yet.** It is a useful candidate for analysis and possibly a healthier image/text checkpoint, but it did not meet the audio-CER restoration gate.
- **Root-cause update:** audio quality appears bounded earlier than Stage6. Tail-only low-LR recovery cannot fully repair audio after the image-heavy Stage4-5 updates. The next quality experiment should prevent audio damage during image training or strengthen the earlier A2A core, not merely add a stronger final A2A tail.

## Evidence-Driven Problem Statement

Cross-run L1 metrics (`dataset/eval_muon_l1.jsonl`, 57 cases, bf16, t160, seed 42):

| Capability | A | B3 | C v3 | Trend |
| --- | ---: | ---: | ---: | --- |
| Text CER | 0.2412 | 0.2302 | **0.2176** | improving |
| Image CER | 0.5834 | 0.6028 | **0.5401** | C best |
| **Audio CER** | **0.3464** | 0.3568 | 0.3577 | **monotonic regression** |
| Audio repeat | 0.0017 | 0.0042 | 0.0052 | **monotonic regression** |
| Image repeat | 0.0075 | 0.0129 | 0.0178 | **monotonic regression** |

Two hard pieces of evidence for the root cause:

1. **Stage6 A2A-final val loss `4.6873` > Stage3 A2A-all val loss `4.4796`.** The 5 epochs of pure image training in Stage4–5 (audio loss `0.0000`) erode audio; the single Stage6 epoch at the v3-lowered LR `2e-6` cannot recover it.
2. **Stage7 (the final stage) is `vision_proj`-only and its val loss is flat at `2.01–2.02` from step 3500 to 15000** (~48 min of near-zero gain). The model's last substantial update is the vision projector, leaving audio unsupervised at the finish line.

Stage-flow diagnosis:

```text
1 T2A -> 2 A2A_proj -> 3 A2A_all -> 4 I2T_proj -> 5 I2T_all -> 6 A2A_final -> 7 I2T_proj(final)
                      ^ last full audio training          ^ 5 pure-image epochs     ^ 1 low-LR audio ep   ^ image-only tail
```

This is a curriculum-tail forgetting pattern:

- Stage3 is the last substantial audio training block.
- Stage4-5 add 5 image-only epochs where audio loss is `0.0000`.
- Stage6 gives audio only 1 recovery epoch, and v3 lowers the recovery LR from `5e-6` to `2e-6`.
- Stage7 then ends on a `vision_proj`-only update whose validation curve is nearly flat, so audio is again unsupervised at the finish line.

This explains why C wins text/image but loses to A on audio-prompt factuality/refusal. The Phase-0 fixes are not the culprit; they produced measurable text/image and overall-CER gains. The remaining issue is that the final curriculum order lets the fragile audio behavior be overwritten or under-refreshed after the image-heavy tail.

## Comparison Matrix

| Run | Variable axis | Status |
| --- | --- | --- |
| A | Baseline 7-stage Muon | Completed |
| B3_himem_b88 | v2 schedule + Stage1 batch 88 | Completed; not promoted |
| C | v2 wrapper + Phase-0 fixes (v3) | Completed; best text/image, audio unresolved |
| **D-a** | **Joint anti-forgetting finale (steady-state anti-forgetting, not expected to fix audio ceiling alone)** | **Still useful, but lower confidence after D-b** |
| **D-b** | **Audio-tail strengthening (first isolation run, no code change)** | **Completed; failed audio-CER gate, passed repeat/image-health gates** |
| **D-mix** | **A2A replay during I2T full stage** | **Recommended next training variable** |
| **D-core** | **Strengthen earlier A2A core before image-heavy stages** | **Recommended if D-mix is insufficient** |
| D-c | Stage6/7 order swap (if keeping Stage7 image micro-tune) | Planned if needed |
| D-rvq | RVQ weight flattening + decode-guard eval (repetition control) | Planned in parallel |

Note: the previously earmarked "Run D = throughput (packing/streaming)" is an orthogonal engineering optimization. It is renamed **Run T** and must not be mixed into this quality round, to keep the variable axis clean.

## Recommended Experiment Matrix

| Experiment | Only variable | Command cost | Expected signal | Risk |
| --- | --- | --- | --- | --- |
| D-b audio-tail strengthening | Stage6 becomes 2 epochs, LR returns to `5e-6`, Muon LR returns to `0.001`, and training ends on audio | Completed fast re-run from C Stage5 | Val loss and repeat improve, but audio CER does **not** return to `<= 0.3464` | Marked as partial success, not promotion-ready |
| D-mix image-stage replay | Keep C/D-b defaults but inject A2A replay during Stage5 I2T-all | Requires mixed parquet or launcher/data mixing | Preserve image gain while preventing audio damage before it happens | Adds a data-mix variable; must keep manifest fixed |
| D-core earlier A2A | Tune Stage2-3 A2A training, then rerun image stages with replay | Full run | Raise the audio ceiling before image-heavy stages | More expensive and may trade off text/image |
| D-rvq repetition control | RVQ weights flattened to `1.5,1.3,1.1,1,0.9,0.8,0.7,0.6` | Env-only full run | Audio/image repeat decreases | May slightly hurt coarse-code fidelity |
| D-a joint mixed finale | Add a mixed final stage over A2A + I2T + T2A | Small launcher/data change | No modality is forgotten at the finish | Requires mixed parquet construction |
| D-eval evaluation upgrade | More cases, human listening, and separate semantic score | No training | Future promotion decisions become trustworthy | No model-quality risk |

Run T note: throughput work such as packing/streaming/offline encoders is a separate engineering line. Do not mix it into Run D quality experiments, because it changes data order and effective token mix.

## Discriminative Takeaways and Next Improvements

Adopt:

- Keep the root-cause framing: C's failure mode is curriculum-tail audio forgetting, not Phase-0 failure.
- Keep D-b as the first executable experiment because it isolates the audio-tail hypothesis with no code change.
- Keep D-a as a useful anti-forgetting finish, but after D-b it should not be treated as sufficient to fix audio intelligibility by itself.
- Move the next audio-quality experiment earlier: either A2A replay during Stage5 I2T-all, or stronger Stage2-3 A2A core before image-heavy training.
- Keep D-eval as a promotion gate, not an afterthought.

Do not adopt blindly:

- Do not promote D-b only because audio CER improves; semantic correctness and human listening must agree.
- Do not run D-rvq as the next full training before D-b unless repetition is judged a primary blocker in human listening.
- Do not implement Run T throughput changes in the same candidate, because packing/streaming changes data order and attribution.
- Do not create a `_final` alias for D-b until the `_a2a_final` artifact passes L1, ASR, semantic spot-check, and human listening.

Next concrete improvements:

1. Keep the D-b wrapper for reproducibility, but mark D-b as a partial success: useful checkpoint, failed primary audio-CER gate.
2. Add a mixed-parquet builder for D-mix/D-a, with a fixed seed and manifest, so the A2A/I2T/T2A mix is reproducible.
3. Add optional mixed-stage support to the launcher, gated by env vars, instead of hardcoding a new Run D script.
4. Expand L1 to 100+ cases before final promotion, with an emphasis on A2A factuality, refusal, physics/science, procedural prompts, and Chinese probes.
5. Add a lightweight semantic-score report that stays separate from CER/WER and from human naturalness scores.
6. Run raw and decode-guarded eval for A, C, and D candidates so repeat suppression is measured as a decoding effect, not silently mixed into model-quality claims.

## Priority 1 (MUST DO) - Fix Audio Forgetting

This is the only current blocker preventing Run C from replacing baseline A. There are three candidate fixes:

1. **D-a, most recommended as the steady-state solution:** add a joint mixed finale so A2A + I2T + T2A are all refreshed at the end.
2. **D-b, run first because it is cheapest and cleanest:** strengthen the existing A2A tail and stop at audio.
3. **D-c, fallback if Stage7 must be preserved:** move the image micro-tune earlier and let A2A be the final stage.

Recommended path after D-b: do **not** spend another run on tail-only Stage6 strengthening. D-b showed that a stronger audio tail improves val loss and repetition health but does not recover audio CER. The next run should prevent the damage earlier by adding A2A replay during the image-heavy Stage5 updates, or by strengthening Stage2-3 A2A before the image-heavy section.

### D-a (Most Recommended) - Joint Anti-Forgetting Finale

Add a mixed mini-epoch after Stage7, or replace Stage7 with a mixed finale, using small proportions of all three supervised modalities. This is the standard mitigation for sequential multi-task SFT forgetting: the last stage sees and protects every modality instead of ending on image-only updates.

Suggested mix:

```text
60% A2A / 25% I2T / 15% T2A
```

Expected effect:

- Audio should stop regressing because A2A is present in the final updates.
- C's image gains should be preserved because I2T remains in the tail.
- T2A/text should not silently drift because T2A is included as a smaller stabilizer.

Implementation requirement: build a mixed parquet from existing shards and add a gated mixed-finale stage to the launcher. Do not mix this code change into the first isolation run; use D-b first to confirm the root cause cheaply.

### D-b (First Run) - Audio-Tail Strengthening

Lowest-cost, cleanest isolation of "can audio be recovered without hurting image?". This requires only launcher environment variables and no code change.

### Changes vs Run C

| Knob | Run C (v3) | Run D-b | Why |
| --- | ---: | ---: | --- |
| Stage6 A2A-final epochs | 1 | **2** | One epoch cannot undo 5 image epochs of forgetting |
| Stage6 `learning_rate` | 2e-6 | **5e-6** | v3 lowered LR to avoid rebound, but that also blocks recovery; direction was wrong |
| Stage6 `muon_lr` | 0.0005 | **0.001** | Same reasoning as above |
| End stage | 7 (`i2t vision_proj`) | **6 (A2A)** | Stage7 val is flat (~0 gain); ending on audio protects the fragile modality |

### Launch (env-only, no code change)

CRITICAL: the dense launcher defaults `DDP_BROADCAST_BUFFERS` to `1`, and dry-run confirmed that the v3 wrapper also printed `1` unless this env var was set. Run C only completed because it was launched with `DDP_BROADCAST_BUFFERS=0` (run dir `...ddpbuf0...`, confirmed in the v3 summary); the default `1` is exactly what caused the earlier NCCL buffer-broadcast hang. The v3 wrapper now defaults `DDP_BROADCAST_BUFFERS=0`, and every command below also sets it explicitly for auditability. Dry-run verified locally: Stage6 prints `--epochs 2`, `--learning_rate 5e-6`, `--muon_lr 0.001`, `--ddp_broadcast_buffers 0`, and Stage7 is skipped.

Dry-run first:

```bash
DRY_RUN=1 \
WEIGHT_PREFIX=sft_full_muon_v4_audiofix \
RUN_GROUP=muon_v4_audiofix \
DDP_BROADCAST_BUFFERS=0 \
EPOCHS_A2A_FINAL=2 \
LR_A2A_FINAL=5e-6 \
MUON_LR_A2A_FINAL=0.001 \
END_STAGE=6 \
bash scripts/run_full_train_muon_v3_3gpu.sh
```

Full run:

```bash
WEIGHT_PREFIX=sft_full_muon_v4_audiofix \
RUN_GROUP=muon_v4_audiofix \
DDP_BROADCAST_BUFFERS=0 \
EPOCHS_A2A_FINAL=2 \
LR_A2A_FINAL=5e-6 \
MUON_LR_A2A_FINAL=0.001 \
END_STAGE=6 \
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train \
bash scripts/run_full_train_muon_v3_3gpu.sh
```

Fast iteration option (recommended): Stages 1-5 are identical to Run C, so copy C's Stage5 artifact under the D prefix and only re-run Stage6. Use the wrapper below so the artifact copy, `START_STAGE=6`, `END_STAGE=6`, Stage6 LR/epoch overrides, SwanLab/W&B logging, and `DDP_BROADCAST_BUFFERS=0` stay together:

```bash
DRY_RUN=1 bash scripts/run_full_train_muon_v4_audiofix_stage6_from_c.sh
```

```bash
nohup bash scripts/run_full_train_muon_v4_audiofix_stage6_from_c.sh \
  > .run_logs/runD_v4_audiofix_stage6_from_c_nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The wrapper copies:

```text
out/sft_full_muon_v3_i2t_full_768.pth
-> out/sft_full_muon_v4_audiofix_i2t_full_768.pth
```

Caveat: Stage6 `--from_weight` loads from `out/${WEIGHT_PREFIX}_i2t_full_768.pth`. Do not set `WEIGHT_PREFIX=sft_full_muon_v3` for this run, because it would overwrite C's `_a2a_final` artifacts. If `RESUME_STAGE6=1` is used later, also manage the matching checkpoint under `checkpoints/`; plain `from_weight` only needs the `out/` file.

### D-b success criteria

- Audio CER `<= 0.3464` (back to A level or better) on `eval_muon_l1.jsonl`.
- Image CER does not regress above `0.55` (preserve C's image gains).
- Stage6 final A2A val loss `<=` Stage3 A2A val loss (`4.4796`).
- No OOM / no NCCL hang / no non-finite loss.

D-b outcome:

- Audio CER gate: **failed** (`0.3571`, still behind A `0.3464`).
- Image preservation gate: **passed** (`0.5232`, best so far).
- Stage6 val gate: **passed** (best `4.4750`, last `4.4929`).
- Stability gate: **passed** (no OOM, no NCCL hang, no non-finite loss).
- Repeat-health gate: **passed** relative to C (audio `0.0052 -> 0.0025`, image `0.0178 -> 0.0080`).

Because the primary audio gate failed, do not run the promotion alias command below unless a later human-review decision deliberately promotes D-b for image/text/repeat health rather than audio recovery:

```bash
cp -n out/sft_full_muon_v4_audiofix_a2a_final_768.pth \
  out/sft_full_muon_v4_audiofix_final_768.pth
cp -n checkpoints/sft_full_muon_v4_audiofix_a2a_final_768.pth \
  checkpoints/sft_full_muon_v4_audiofix_final_768.pth
```

---

## Priority 2 (PARALLEL) - Repetition Regression Control

Audio/image repeat ratios rose monotonically A -> B3 -> C, with a pathological loop in `a2a_zh_cat_story_probe`. The v3 RVQ weights `2,1.5,1.2,1,0.8,0.7,0.6,0.5` heavily emphasize coarse codes, a plausible contributor.

Treat repetition as two separate problems:

1. **Training-side D-rvq:** test whether flatter RVQ weights reduce repeated audio/image behavior.
2. **Eval-side decode guard:** add repetition controls during generation so CER and human review are not dominated by decoding loops.

### Training-side D-rvq

```bash
WEIGHT_PREFIX=sft_full_muon_v4_rvqflat \
RUN_GROUP=muon_v4_rvqflat \
DDP_BROADCAST_BUFFERS=0 \
RVQ_LAYER_WEIGHTS="1.5,1.3,1.1,1,0.9,0.8,0.7,0.6" \
EPOCHS_A2A_FINAL=2 LR_A2A_FINAL=5e-6 MUON_LR_A2A_FINAL=0.001 END_STAGE=6 \
USE_WANDB=1 WANDB_PROJECT=MiniMind-O-Full-Train \
bash scripts/run_full_train_muon_v3_3gpu.sh
```

Success signal:

- Audio/image repeat decreases versus C and D-b.
- Audio CER does not worsen versus D-b.
- Image CER stays near C (`<= 0.55` target).

### Eval-side decoupling

`scripts/batch_validate_omni.py` supports text-token repetition controls:

- `--repetition_penalty`: passed to Omni generation as `rp`
- `--no_repeat_ngram_size`: bans repeated text token n-grams during generation

Run raw decoding first for continuity, then run a secondary decode-guarded pass for repeat contamination analysis:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v4_audiofix_a2a_final \
  --output_dir eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160_decodeguard \
  --test_set dataset/eval_muon_l1.jsonl \
  --dtype bf16 --max_new_tokens 160 --decode_audio \
  --repetition_penalty 1.08 \
  --no_repeat_ngram_size 4
```

Use the same decode-guard settings for baseline A, C, and every Run D candidate before comparing them. Do **not** use the decode-guarded run as the only promotion metric; it is a diagnostic view that separates model quality from decoding pathology. If raw repeat is bad but decode-guarded CER/review improves sharply, prioritize decoding controls in evaluation and product inference. If both raw and decode-guarded outputs repeat, treat it as a training-side model regression.

---

## Priority 3 (BLOCKING) - Evaluation System Upgrade

The audio verdict is currently unreliable: in 10 of 23 audio cases the AI-review winner disagrees with the lower-CER model. Example: `a2a_en_sky_blue` has C CER `0.0046` versus A CER `0.3833`, yet the AI-assisted review judged A better on semantic wording. Before using audio evaluation to promote C or steer Run D, split audio evaluation into intelligibility, semantic correctness, naturalness, and factual/refusal behavior.

Required before promoting C or any Run D candidate:

1. **Human listening spot-check on 8-10 audio cases.** Seed set: `a2a_en_food`, `a2a_en_black_hole`, `a2a_zh_health_probe`, `a2a_en_sky_blue`, `a2a_zh_coffee_probe`, `a2a_en_animals`, `a2a_zh_cat_story_probe`, `a2a_en_ai_fields`, `a2a_zh_ai_fields_probe`, `a2a_zh_snow_probe`.
2. **Expand L1 beyond 57 cases, target 100+.** Add more A2A factuality/refusal prompts because the current audio subset is too small for robust decisions.
3. **Report semantic correctness separately from CER.** CER/WER remain ASR intelligibility proxies; semantic correctness should be a separate 0-3 rubric, LLM-assisted if needed, with human spot-checks on disputed cases.

Generate the current A-vs-C spot-check template:

```bash
python scripts/make_audio_spotcheck_template.py \
  --run_a eval_results/sft_full_muon_final_l1_bf16_t160 \
  --run_b eval_results/sft_full_muon_v3_final_l1_bf16_t160 \
  --label_a baseline_A \
  --label_b runC_v3 \
  --output docs/evaluation_results/audio_spotcheck_runC_vs_baseline_l1_t160.md
```

For each future Run D candidate, regenerate the same template against baseline A and against Run C. A candidate cannot be promoted if it improves CER by making semantically wrong, over-short, or unnatural audio answers.

---

## Priority 4 (DEFAULTS) - Preserve Proven Fixes

These settings are now default controls for Run D and should not be rolled back while investigating audio. They are already validated by Run C: training completed without NaN or NCCL timeout, and text/image metrics improved.

- Optimizer: Muon (`MuonWithAuxAdam`)
- Same full parquet datasets and rank shards under `dataset/_full_shards/`
- Same base checkpoint chain (Stage1 `--from_weight llm`)
- Phase-0 defaults kept: `loss_norm=global`, `warmup_ratio=0.02`, fp32 resume weights, val monitoring every 500 steps.
- `DDP_BROADCAST_BUFFERS=0` stays mandatory; it fixed the prior NCCL buffer-broadcast hang.
- I2T 2-epoch schedule stays enabled; it is the likely source of C's image gains.
- Eval set `dataset/eval_muon_l1.jsonl`, seed 42, bf16

---

## Priority 5 (OPTIONAL) - D-c: Swap Stage6/Stage7 Order

Use this only if we still want to keep the Stage7 image micro-tune but do not want to build a mixed finale yet. The idea is simple: run the image-only `vision_proj` micro-tune before the final A2A recovery, so A2A remains the last supervised modality.

Proposed order:

```text
1 T2A -> 2 A2A_proj -> 3 A2A_all -> 4 I2T_proj -> 5 I2T_all -> 6 I2T_proj(final) -> 7 A2A_final
```

Expected effect:

- Keeps Stage7's image projector cleanup if it is later shown to matter.
- Avoids ending the whole run on an image-only stage.
- Still weaker than D-a because the final stage is single-modality A2A, so text/image can still drift slightly.

Implementation requirement: the current launcher cannot express this swap with only `START_STAGE`/`END_STAGE` because Stage7 currently loads from `${WEIGHT_PREFIX}_a2a_final` and Stage6 loads from `${WEIGHT_PREFIX}_i2t_full`. Add a small launcher variant or stage-order flag only if D-b and D-a results justify it.

## Evaluation Plan (per candidate)

L1 structural + ASR + comparison + review, identical to the Run C pipeline:

```bash
# 1. L1 structural eval (bf16, t160)
CUDA_VISIBLE_DEVICES=0 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v4_audiofix_a2a_final \
  --output_dir eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160 \
  --test_set dataset/eval_muon_l1.jsonl \
  --dtype bf16 --max_new_tokens 160 --decode_audio

# 2. SenseVoice ASR round-trip (CPU batch)
python scripts/asr_eval_generated_audio.py \
  --results eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160/results.jsonl \
  --output eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160/asr_eval.json \
  --markdown eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160/asr_eval.md \
  --backend sensevoice \
  --sensevoice_project /home/genesis/Projects/SenseVoice \
  --sensevoice_model model/SenseVoiceSmall \
  --device cpu --batch_size 16

# 3. Compare vs baseline A and candidate C
python scripts/compare_eval_runs.py \
  --run_a eval_results/sft_full_muon_final_l1_bf16_t160 \
  --run_b eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160 \
  --label_a baseline --label_b v4_audiofix \
  --output docs/evaluation_results/compare_v4_audiofix_vs_baseline_l1_t160.md

# 4. Manual review template
python scripts/make_manual_review_template.py \
  --run_a eval_results/sft_full_muon_final_l1_bf16_t160 \
  --run_b eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160 \
  --label_a baseline --label_b v4_audiofix \
  --output docs/evaluation_results/review_v4_audiofix_vs_baseline_l1_t160.md
```

Secondary decode-guarded pass for repeat analysis:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v4_audiofix_a2a_final \
  --output_dir eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160_decodeguard \
  --test_set dataset/eval_muon_l1.jsonl \
  --dtype bf16 --max_new_tokens 160 --decode_audio \
  --repetition_penalty 1.08 \
  --no_repeat_ngram_size 4

python scripts/asr_eval_generated_audio.py \
  --results eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160_decodeguard/results.jsonl \
  --output eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160_decodeguard/asr_eval.json \
  --markdown eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160_decodeguard/asr_eval.md \
  --backend sensevoice \
  --sensevoice_project /home/genesis/Projects/SenseVoice \
  --sensevoice_model model/SenseVoiceSmall \
  --device cpu --batch_size 16
```

## Pre-Run Checklist

- Confirm no active training: `pgrep -af 'torchrun|train_sft_omni.py'`
- Confirm GPUs idle: `nvidia-smi`
- Confirm rank shards exist: `ls dataset/_full_shards/`
- Confirm val splits exist: `ls dataset/_val/sft_{t2a,a2a,i2t}_val.parquet`
- Dry-run launcher and inspect the printed Stage6 command (epochs/LR/end-stage)
- Verify `WEIGHT_PREFIX=sft_full_muon_v4_audiofix` does not overwrite A/C artifacts in `out/` and `checkpoints/`

## Decision Logic

1. Run **D-b first** because it is the cheapest clean isolation of the audio-forgetting hypothesis.
2. If **D-b** meets all success criteria -> promote `v4_audiofix_a2a_final` to a new candidate baseline only after human listening; optionally implement **D-a** later as the more robust all-modality finale.
3. If **D-b** restores audio but harms image/text -> implement **D-a**, because mixed finale is designed to protect all modalities simultaneously.
4. If **D-b** still leaves audio below A -> implement **D-a** before spending more runs on pure A2A LR/epoch tuning.
5. Use **D-c** only if Stage7 image micro-tuning must be preserved and we want an intermediate solution before mixed-finale code.
6. Run **D-rvq/decodeguard** in parallel as a diagnostic: if decode guard fixes repetition, keep it as an eval/inference control; if flatter RVQ weights reduce raw repeat without hurting CER, consider it for the next training candidate.
7. Do not promote C or any Run D candidate without Eval-L2 completion: human listening spot-check, semantic correctness score separated from CER, and a plan to expand L1 to 100+ cases.
8. Do not roll back the proven defaults while debugging audio: keep `loss_norm=global`, `warmup_ratio=0.02`, fp32 resume weights, 500-step val monitoring, `DDP_BROADCAST_BUFFERS=0`, and I2T 2-epoch schedule.

## Output Naming

```text
out/sft_full_muon_v4_audiofix_a2a_final_768.pth      # primary D-b artifact
checkpoints/sft_full_muon_v4_audiofix_a2a_final_768.pth
out/sft_full_muon_v4_audiofix_final_768.pth          # optional promotion alias
checkpoints/sft_full_muon_v4_audiofix_final_768.pth  # optional promotion alias
.run_logs/full_train_sft_full_muon_v4_audiofix_3gpu_YYYYMMDD_HHMMSS/
docs/training_summaries/full_train_muon_v4_audiofix_YYYYMMDD.md
docs/evaluation_results/compare_v4_audiofix_vs_baseline_l1_t160.md
```
