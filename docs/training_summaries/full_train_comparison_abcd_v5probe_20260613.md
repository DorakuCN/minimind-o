# MiniMind-O Full-Train Comparison Summary - 2026-06-13

## Scope

This summary consolidates the current 113M dense MiniMind-O comparison runs:

- **A baseline**: `sft_full_muon_final`
- **B3_himem_b88**: v2 schedule + high-memory Stage1 batch
- **C v3**: v2 schedule + Phase-0 correctness fixes
- **D-b audiofix**: C Stage5 -> stronger Stage6-only A2A tail
- **E-probe / v5 RVQ-uniform**: Stage-local diagnostic currently in progress

All final-candidate automatic metrics below use the same L1 setup unless noted:

```text
dataset/eval_muon_l1.jsonl
57 cases
bf16
max_new_tokens=160
seed=42
ASR: SenseVoiceSmall on CPU
```

## Executive Summary

The current best interpretation is:

- **C v3 is still the strongest text-CER / overall-CER checkpoint**, but it introduced repeat pathology and did not solve audio-input quality.
- **D-b is a healthier, more balanced checkpoint than C**, because it fixes most of C's repeat regression and gives the best image CER / WER.
- **D-b did not solve the original audio-CER target.** Audio CER is still `0.3571`, essentially tied with C (`0.3577`) and still behind A (`0.3464`).
- **Stage1 RVQ-uniform evidence is neutral-to-weak.** It improves WER and wins more cases, but average CER is slightly worse because of one large outlier. It does not decide the A2A question.
- **The decisive test is the v5 Stage3 `_a2a_full` checkpoint.** The large historical gap is at A2A Stage3 (`A3 audio CER 0.2965` vs `C3 0.3976`), not at T2A Stage1.

Decision state:

- Do **not** promote B3.
- Do **not** promote C as a universal baseline because audio remains unresolved.
- Do **not** promote D-b as an audio recovery success; keep it as the current best image/repeat-health candidate.
- Continue E-probe through Stage3 before deciding whether RVQ weights, `loss_norm`, or A2A-stage interactions are the true audio bottleneck.

## Final Candidate L1 Results

| Model | Artifact | Basic pass | Overall CER | Overall WER | Text CER | Audio CER | Image CER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | `out/sft_full_muon_final_768.pth` | 57/57 | 0.3497 | 0.5459 | 0.2412 | **0.3464** | 0.5834 |
| B3_himem_b88 | `out/sft_full_muon_v2_himem_b88_final_768.pth` | 57/57 | 0.3532 | 0.5805 | 0.2302 | 0.3568 | 0.6028 |
| C v3 | `out/sft_full_muon_v3_final_768.pth` | 57/57 | **0.3364** | 0.5553 | **0.2176** | 0.3577 | 0.5401 |
| D-b audiofix | `out/sft_full_muon_v4_audiofix_a2a_final_768.pth` | 57/57 | 0.3414 | **0.4926** | 0.2387 | 0.3571 | **0.5232** |

Interpretation:

- A remains the **audio CER anchor**.
- C remains the **text and overall-CER anchor**.
- D-b is the **image and repeat-health anchor**, but not an audio-CER fix.
- B3 is dominated by C/D-b on most useful dimensions and should not be promoted.

## Repeat Health

| Model | Text repeat avg / max | Audio repeat avg / max | Image repeat avg / max |
| --- | ---: | ---: | ---: |
| A baseline | 0.0037 / 0.0597 | **0.0017 / 0.0400** | **0.0075 / 0.0390** |
| B3_himem_b88 | 0.0096 / 0.1071 | 0.0042 / 0.0299 | 0.0129 / 0.0690 |
| C v3 | **0.0014 / 0.0326** | 0.0052 / 0.0556 | 0.0178 / 0.1125 |
| D-b audiofix | 0.0022 / 0.0500 | 0.0025 / 0.0303 | 0.0080 / 0.0274 |

D-b's clearest win over C is not audio CER; it is repeat health:

- Audio repeat: `0.0052 -> 0.0025`
- Image repeat: `0.0178 -> 0.0080`
- Previously suspicious C cases such as `a2a_zh_cat_story_probe` no longer show the same repeat signature in the structural metric, although semantic quality still needs review.

## Pairwise CER Wins

Lower CER wins per case:

| Pair | All | Text | Audio | Image |
| --- | --- | --- | --- | --- |
| A vs C | A 29 / C 25 / tie 3 | A 12 / C 10 / tie 1 | A 11 / C 10 / tie 2 | A 6 / C 5 / tie 0 |
| A vs D-b | A 23 / D-b 30 / tie 4 | A 8 / D-b 13 / tie 2 | A 11 / D-b 10 / tie 2 | A 4 / D-b 7 / tie 0 |
| C vs D-b | C 27 / D-b 25 / tie 5 | C 12 / D-b 8 / tie 3 | C 10 / D-b 11 / tie 2 | C 5 / D-b 6 / tie 0 |

This is why the D-b decision is nuanced:

- D-b beats A on more total cases (`30:23`) and image cases (`7:4`).
- D-b does **not** clearly beat A on audio cases (`10:11`, two ties).
- C and D-b are effectively split overall (`27:25`, five ties), with C better on text and D-b cleaner on image/repeat.

## D-b Training-Side Result

D-b copied C Stage5 and reran only Stage6:

```text
out/sft_full_muon_v3_i2t_full_768.pth
-> out/sft_full_muon_v4_audiofix_i2t_full_768.pth

Stage6 only:
epochs=2
learning_rate=5e-6
muon_lr=0.001
END_STAGE=6
DDP_BROADCAST_BUFFERS=0
```

Training/val result:

| Checkpoint/stage | Val loss | Val text | Val audio |
| --- | ---: | ---: | ---: |
| Run C Stage3 | 4.4796 | 1.1062 | 3.3734 |
| Run C Stage6 | 4.6873 | 1.3113 | 3.3760 |
| D-b Stage6 last | 4.4929 | 1.1391 | 3.3538 |
| D-b Stage6 best | **4.4750** | 1.1231 | **3.3519** |

Interpretation:

- The Stage6 val-loss regression was repaired.
- But the audio val component moved only slightly (`3.3760 -> 3.3538`).
- L1 audio CER also moved only slightly (`0.3577 -> 0.3571`).
- Therefore D-b is a **training-health / repeat-health success**, not an audio intelligibility recovery success.

## Stage-Localized A2A Diagnostic

The most important audio gap is already visible at Stage3:

| Diagnostic checkpoint | Weight | Basic pass | Overall CER | Text CER | Audio CER | Image CER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A Stage3 | `out/sft_full_muon_a2a_full_768.pth` | 56/57 | **0.3006** | 0.2657 | **0.2965** | **0.3824** |
| C Stage3 | `out/sft_full_muon_v3_a2a_full_768.pth` | 57/57 | 0.3673 | **0.2378** | 0.3976 | 0.5747 |

Gap decomposition:

| Effect | Audio CER delta | Interpretation |
| --- | ---: | --- |
| C Stage3 - A Stage3 | **+0.1011** | The large audio gap appears before image training finishes |
| A final - A Stage3 | +0.0499 | A also suffers some later audio degradation |
| C final - C Stage3 | -0.0399 | C improves from its damaged Stage3 but never recovers to A |
| D-b - C final | -0.0006 | Stronger tail barely changes audio CER |

This shifts the likely bottleneck earlier:

- The original hypothesis, "image-heavy tail forgets audio and Stage6 can fix it", is only partially supported.
- The stronger statement, "C's audio problem is mostly tail forgetting", is **not** supported.
- The decisive stage is A2A Stage2-3, where `{RVQ weighting, loss_norm, warmup, A2A interaction}` differ from the A baseline behavior.

## Stage1 RVQ-Uniform Probe

The first v5 RVQ-uniform probe has only reached Stage1 T2A so far:

| Stage1 checkpoint | RVQ weights | T2A Audio CER | WER | Pairwise wins |
| --- | --- | ---: | ---: | ---: |
| v3_t2a | `2,1.5,1.2,1,0.8,0.7,0.6,0.5` | **0.2438** | 0.5740 | 9 |
| v5_t2a | `1,1,1,1,1,1,1,1` | 0.2536 | **0.4920** | **14** |

Detailed read:

- Uniform RVQ wins more cases (`14:9`) and has much better WER (`-0.0820`).
- Mean CER is slightly worse for uniform RVQ (`+0.0098`).
- The CER result is dominated by one outlier: `t2a_en_space`, where v3 is `0.0147` and v5 is `0.5887`.
- Excluding `t2a_en_space`, v5's average CER becomes better (`0.2384` vs v3 `0.2542`).

Conclusion:

- This is **neutral-to-weak evidence** in favor of uniform RVQ, not a proof.
- It is only T2A Stage1, while the large observed gap is A2A Stage3.
- Stage1 does not reproduce the full audio-CER failure, so it weakens any claim that RVQ weights alone have already been proven causal.
- The `_a2a_full` Stage3 v5 checkpoint is required before deciding whether to run a full v5.

## Current E-Probe Status

Current active run:

```text
.run_logs/full_train_sft_full_muon_v5_rvquniform_3gpu_20260613_071717_Eprobe_stage23/
```

Current stage at last check:

```text
02_a2a_audio_proj
from_weight=sft_full_muon_v5_rvquniform_t2a
save_weight=sft_full_muon_v5_rvquniform_a2a_proj
RVQ_LAYER_WEIGHTS=1,1,1,1,1,1,1,1
loss_norm=global
warmup_ratio=0.02
DDP_BROADCAST_BUFFERS=0
```

Run reproducibility:

```text
.run_logs/full_train_sft_full_muon_v5_rvquniform_3gpu_20260613_071717_Eprobe_stage23/frozen_scripts/
```

Frozen script hashes:

```text
07cdd23d8307d7ad841fd32f8220fabb4c13a4224f85be29db55a27253b1feca  run_full_train_muon_dense_3gpu.sh
5be5a44082081fc3f96aacd217ff8233b6aeb0392c0cc6a12757f8eec4979945  run_full_train_muon_v3_3gpu.sh
01229936418c4e48a86c70a8ad47c6116cd766c7096af87de1480a56987f2bf0  train_sft_omni.py
```

Next required artifact:

```text
out/sft_full_muon_v5_rvquniform_a2a_full_768.pth
```

Next required eval:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/batch_validate_omni.py \
  --weight sft_full_muon_v5_rvquniform_a2a_full \
  --output_dir eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160 \
  --test_set dataset/eval_muon_l1.jsonl \
  --dtype bf16 --max_new_tokens 160 --seed 42 --decode_audio

python scripts/asr_eval_generated_audio.py \
  --results eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160/results.jsonl \
  --output eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160/asr_eval.json \
  --markdown eval_results/diag_v5_rvquniform_a2a_full_l1_bf16_t160/asr_eval.md \
  --backend sensevoice --sensevoice_project /home/genesis/Projects/SenseVoice \
  --sensevoice_model model/SenseVoiceSmall --device cpu --batch_size 16
```

## Updated Hypotheses

### Supported

- Phase-0/v3 improves text and image metrics.
- D-b's stronger Stage6 repairs repeat pathology and training-health indicators.
- C's audio issue is already visible before final image stages.

### Not Supported

- "Just add a stronger final A2A Stage6 and audio CER will recover."
- "D-b should replace A for audio-input use."
- "Stage1 T2A is enough to prove RVQ uniform fixes A2A audio."

### Still Open

- Whether uniform RVQ fixes the Stage3 A2A audio gap.
- Whether `loss_norm=global` is a hidden audio-quality confound.
- Whether warmup or A2A projection/full-stage interaction changes are contributing.
- Whether some CER movement is ASR artifact rather than perceptual audio quality.

## Recommended Next Actions

1. Let current v5 E-probe finish Stage2-3.
2. Evaluate `sft_full_muon_v5_rvquniform_a2a_full` on L1 with SenseVoice.
3. Compare v5 Stage3 directly against:
   - A Stage3 audio CER `0.2965`
   - C Stage3 audio CER `0.3976`
4. Decision gate:
   - If v5 Stage3 audio CER is near A (`<= ~0.31`): uniform RVQ is strongly supported -> run v5 full.
   - If v5 Stage3 stays near C (`~0.38-0.40`): test `LOSS_NORM=local` before a full retrain.
   - If v5 Stage3 is mixed: inspect per-case wins and outliers before spending a full run.
5. Keep D-b as a useful balanced checkpoint, but do not label it as an audio recovery success.

## Artifact Index

Final evals:

```text
eval_results/sft_full_muon_final_l1_bf16_t160/
eval_results/sft_full_muon_v2_himem_b88_final_l1_bf16_t160/
eval_results/sft_full_muon_v3_final_l1_bf16_t160/
eval_results/sft_full_muon_v4_audiofix_a2a_final_l1_bf16_t160/
```

Stage diagnostics:

```text
eval_results/diag_A_a2a_full_l1_bf16_t160/
eval_results/diag_v3_a2a_full_l1_bf16_t160/
eval_results/diag_v3_t2a_only/
eval_results/diag_v5_t2a_only/
```

Key reports:

```text
docs/evaluation_results/compare_v4_audiofix_vs_baseline_l1_t160.md
docs/evaluation_results/compare_v4_audiofix_vs_runc_l1_t160.md
docs/training_plans/next_model_training_runD_20260613.md
docs/training_plans/next_model_training_runE_20260613.md
```
