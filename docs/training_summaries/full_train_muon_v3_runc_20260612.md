# Run C v3 Full Training Summary · 2026-06-12

## 1. Run Identity

本次训练是 MiniMind-O 全量对比训练中的 **Run C / v3**，目标是在 baseline A 和 B3_himem_b88 之后验证 Phase-0 训练正确性改动。

| Item | Value |
| --- | --- |
| Final model | `out/sft_full_muon_v3_final_768.pth` |
| Final checkpoint | `checkpoints/sft_full_muon_v3_final_768.pth` |
| Resume checkpoint | `checkpoints/sft_full_muon_v3_final_768_resume.pth` |
| Training logs | `.run_logs/full_train_sft_full_muon_v3_RunC_after_L1_3gpu_20260612_032548_RunC_v3_after_L1_ddpbuf0_tmux/` |
| L1 eval | `eval_results/sft_full_muon_v3_final_l1_bf16_t160/` |
| Three-way report | `docs/evaluation_results/compare_muon_v3_l1_t160_vs_baseline_and_b3_20260612.md` |
| AI-assisted review | `docs/evaluation_results/review_muon_v3_l1_t160_vs_baseline_l1_t160.md` |
| Environment | `conda env minimind-o`, CUDA 13.3 |
| Hardware | `3x5090D 32G` |
| Wall time | `10:11:33` |

## 2. What Changed vs Earlier Runs

Run C uses the v2 schedule family but adds the Phase-0 correctness changes:

- LR warmup: `warmup_ratio=0.02`
- Global token-weighted loss: `loss_norm=global`
- RVQ layer weights: `2,1.5,1.2,1,0.8,0.7,0.6,0.5`
- Validation loss every 500 steps on `dataset/_val`
- Finite loss guard enabled
- DDP buffer broadcast disabled for this launch: `DDP_BROADCAST_BUFFERS=0`

`DDP_BROADCAST_BUFFERS=0` was added after a prior Run C launch failed during Stage1 with a rank2 `SIGKILL` followed by NCCL buffer-broadcast timeout. The completed run had no NCCL watchdog timeout, no `FloatingPointError`, and no Python traceback.

## 3. Stage Metrics

| Stage | Duration | Mode | Epochs | Batch | Steps | Final train loss | Final text | Final audio | Last val loss | Last val text | Last val audio | Samples/s | Peak mem | Avg GPU util |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 T2A all | 2:33:07 | all | 6 | 48 | 52044 | 4.0578 | 1.5346 | 2.5232 | 4.9584 | 1.4501 | 3.5084 | 818.0 | 17.4G | 94.3% |
| 02 A2A audio_proj | 23:49 | audio_proj | 1 | 32 | 4313 | 6.3668 | 1.5559 | 4.8109 | 6.4461 | 1.7303 | 4.7158 | 293.1 | 21.4G | 93.3% |
| 03 A2A all | 1:32:15 | all | 3 | 24 | 17253 | 4.2796 | 0.7362 | 3.5435 | 4.4796 | 1.1062 | 3.3734 | 225.1 | 24.2G | 95.6% |
| 04 I2T vision_proj | 1:37:31 | vision_proj | 2 | 64 | 30256 | 3.9614 | 3.9614 | 0.0000 | 2.9121 | 2.9121 | 0.0000 | 995.9 | 20.9G | 93.1% |
| 05 I2T all | 2:45:13 | all | 2 | 64 | 30256 | 2.4702 | 2.4702 | 0.0000 | 1.9460 | 1.9460 | 0.0000 | 587.1 | 27.2G | 97.2% |
| 06 A2A final all | 30:55 | all | 1 | 24 | 5751 | 5.0044 | 1.3915 | 3.6128 | 4.6873 | 1.3113 | 3.3760 | 225.2 | 24.0G | 94.1% |
| 07 I2T final vision_proj | 48:43 | vision_proj | 1 | 64 | 15128 | 2.3329 | 2.3329 | 0.0000 | 2.0157 | 2.0157 | 0.0000 | 999.5 | 20.8G | 93.8% |

Notes:

- I2T stages have no audio token supervision, so audio loss is `0.0000` and RVQ-layer detail metrics are `None`; this is expected.
- Stage5 was the highest-memory stage at about `27.2G/32G`, with average GPU utilization `97.2%`.
- Stage6 A2A final val loss (`4.6873`) is higher than Stage3 A2A all val loss (`4.4796`). This is a warning sign for audio-prompt behavior and must be judged with downstream eval rather than raw loss only.
- Root-cause reading: this is not evidence that Phase-0 fixes are bad. Phase-0 helped text/image and overall CER. The audio regression is more consistent with curriculum-tail forgetting: 5 pure-image epochs in Stage4-5, only 1 low-LR audio recovery epoch in Stage6, then an image-only Stage7 finish whose val curve is nearly flat.

## 4. L1 Structural Eval

All three comparison runs used:

- Test set: `dataset/eval_muon_l1.jsonl`
- Cases: 57
- Generation: bf16, `max_new_tokens=160`, `temperature=0.7`, `top_p=0.85`, seed 42

| Run | Weight | Basic pass |
| --- | --- | ---: |
| A baseline | `out/sft_full_muon_final_768.pth` | 57/57 |
| B3 himem_b88 | `out/sft_full_muon_v2_himem_b88_final_768.pth` | 57/57 |
| C v3 | `out/sft_full_muon_v3_final_768.pth` | 57/57 |

The L1 structural gate is saturated. It only confirms that the model generates text/audio with acceptable shape; it no longer separates model quality.

## 5. SenseVoice ASR Round-Trip

ASR uses SenseVoiceSmall through `funasr`, CPU batch mode (`--device cpu --batch_size 16`). Lower CER/WER is better.

| Run | Overall CER | Overall WER | Text CER | Audio CER | Image CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| A baseline | 0.3497 | 0.5459 | 0.2412 | 0.3464 | 0.5834 |
| B3 himem_b88 | 0.3532 | 0.5805 | 0.2302 | 0.3568 | 0.6028 |
| C v3 | **0.3364** | 0.5553 | **0.2176** | 0.3577 | **0.5401** |

Interpretation:

- C has the best overall CER and the best text/image CER.
- C is slightly worse than A on audio CER (`0.3577` vs `0.3464`).
- C WER is slightly worse than A but better than B3. Chinese short probes inflate WER, so CER is the more useful ASR proxy here.
- Case-level CER does not fully agree with the mean: A has lower CER on 29 cases, C on 25 cases, with 3 ties. The lower overall CER for C is therefore a mean-score signal, not a majority-case win.
- Audio CER and semantic audio judgments conflict on several cases, so the audio result should be treated as unresolved until human listening and semantic annotation are done.

Repeat metrics also show a mixed picture:

| Type | A repeat | C repeat | Direction |
| --- | ---: | ---: | --- |
| text | 0.0037 | 0.0014 | C better |
| audio | 0.0017 | 0.0052 | C worse |
| image | 0.0075 | 0.0178 | C worse |

The audio repeat increase is visible in cases such as `a2a_zh_cat_story_probe`, where C produces a clear "咬咬咬" repetition pattern. This is a real regression signal even though C wins some aggregate CER metrics.

## 6. AI-Assisted Review

Review source: text output inspection, image grounding against known eval images, and SenseVoice CER as audio intelligibility proxy. This is **not** a human listening panel. Audio winners in this table should be read as provisional because several audio cases show disagreement between CER, transcript intelligibility, and semantic correctness.

| Type | A wins | C wins | Ties |
| --- | ---: | ---: | ---: |
| audio | 13 | 8 | 2 |
| image | 1 | 8 | 2 |
| text | 3 | 9 | 11 |
| total | 17 | 25 | 15 |

Qualitative pattern:

- C improves image grounding strongly: panda, dinosaur toy, dog/scarf, fruit, and cat-question cases are much better than A.
- C improves several Chinese/text probes: identity, health, black-hole, and some instruction-following responses.
- Audio is mixed and should not be finalized from ASR/CER alone. The AI-assisted review favors A on more audio prompts, but some of those judgments conflict with lower CER for C. Examples needing human adjudication include food-preference refusal, English black-hole A2A, coffee procedure, Chinese story repetition, and science/audio prompts.
- Both models still fail simple counting probes and still hallucinate in physical/science explanations.

Trustworthy examples:

- `i2a_cat_question_probe`: C correctly identifies the cat/book-like scene while A says "cat, specifically a dog"; CER also drops from `0.5013` to `0.0158`.
- Panda/dinosaur/fruit image cases: C usually identifies the main subject; A often drifts to wrong objects such as "cowbell".
- `t2a_zh_who_probe` and `t2a_zh_health_probe`: C gives more direct Chinese answers with lower CER.
- `a2a_en_food`: A correctly refuses personal food preference, while C invents a sandwich preference.
- `a2a_zh_cat_story_probe`: C shows a clear repetition pathology.

Important counterexample:

- `a2a_en_black_hole`: C's CER is `0.0000`, but the generated answer is "The black holster is the male black holster." CER is perfect only because ASR transcribes the wrong answer perfectly. Semantic review correctly favors A here.

High-dispute audio cases:

- 10/23 audio cases have semantic-review winners opposite to the lower-CER direction; one more has tied CER but a semantic preference. The audio subtotal (`A 13 / C 8 / Tie 2`) should not be used as a reliable rank.
- `a2a_zh_coffee_probe`: C was marked as winner, but this is low confidence because C has audio score 0, worse CER (`0.6759` vs A `0.5669`), and odd details such as `200℃` baking paper.
- `a2a_en_sky_blue`: C is far more intelligible by CER (`0.0046` vs A `0.3833`), while semantic wording preference favored A.
- `a2a_en_animals`: C has lower CER but says "five species", so semantic review favors A.
- `a2a_zh_ai_fields_probe`, `a2a_en_ai_fields`, and `a2a_en_health` are weak or content-richness-based preferences rather than clean audio-quality wins.
- Full opposite-CER list is recorded in `docs/evaluation_results/review_muon_v3_l1_t160_vs_baseline_l1_t160.md`; until those cases are human-audited, the Run C audio result is unresolved rather than a final win/loss call.

## 7. Decision

Run C is the strongest candidate so far for **text and image behavior** and should be considered a candidate baseline, but it should not be promoted as the final overall baseline until the audio conflict is resolved.

Calibrated readout:

| Dimension | Result | Confidence |
| --- | --- | --- |
| Training completion | 7 stages completed successfully; Phase-0 fixes were active | High |
| Stage6 audio validation | A2A final val loss rebounded above Stage3 (`4.6873` vs `4.4796`) | High as a risk signal |
| L1 structural gate | A/B3/C all pass 57/57, so the gate is saturated | High, but not discriminative |
| Overall ASR | C has the best overall CER (`0.3364`) | High |
| Image understanding | C is clearly better than A and B3 | High |
| Text T2A | C is slightly better, with many ties | Medium |
| Audio A2A | A remains stronger on factual/refusal/physical prompts; C has several disputed intelligibility wins | Medium-high, pending human listening |
| AI-assisted review | C leads `25:17`, mainly from image/text gains | Medium because audio judgments are provisional |
| B3 comparison | C clearly improves over B3 (`0.3532 -> 0.3364` overall CER; `0.6028 -> 0.5401` image CER) | High |

Recommended decision:

1. Treat Run C as the strongest current candidate and as clearly better than B3.
2. Do not directly replace baseline A yet, especially for audio-input scenarios such as refusal, physical/factual prompts, and English A2A.
3. Trust the AI-assisted review selectively: C's image lead is credible, Chinese text probes are mostly credible, and the audio win/loss subtotal is only reference material because 10/23 audio cases conflict with CER direction.
4. Before final overall promotion, run a 5-10 case human listening and semantic spot-check on disputed audio cases, including `a2a_en_food`, `a2a_en_black_hole`, `a2a_zh_health_probe`, `a2a_en_sky_blue`, `a2a_zh_coffee_probe`, and `a2a_en_animals`.
5. If the next training round targets improvement, preserve C's image-grounding gains while prioritizing Stage6 audio regression, A2A factuality, refusal stability, and repeat suppression.

One-line summary: Run C's training and L1 quantitative data are reliable, and it improves text/image over A and broad metrics over B3; however, the AI-review total `C 25 : A 17` is pulled upward by image wins and has multiple audio contradictions, so the direction is usable but the audio details require human review.

## 8. Remaining Risks

- ASR CER is only a proxy for audio intelligibility and does not measure naturalness, speaker stability, prosody, or semantic correctness. In this run, several audio cases directly conflict with CER-based ranking.
- L1 has only 57 cases. It is enough to catch large regressions but not enough to declare small wins statistically robust.
- Stage6 A2A final val loss is worse than Stage3. This aligns with the audio-risk signal, but it is not by itself enough to rank final audio quality.
- C improves image grounding but still has notable visual hallucinations, especially rabbit/robot-chef style cases.
