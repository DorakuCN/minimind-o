# Run C v3 L1 Evaluation: Baseline A / B3 / C

- Generated: 2026-06-12 18:02:37
- Test set: `dataset/eval_muon_l1.jsonl` (57 cases)
- Generation: `bf16`, `max_new_tokens=160`, `temperature=0.7`, `top_p=0.85`, `seed=42`
- ASR: SenseVoiceSmall via `funasr`, CPU batch, `--device cpu --batch_size 16`
- Primary anchor: baseline A. B3 remains secondary context because L1 did not promote it.

## Overall

| Run | Weight | Basic pass | ASR CER | ASR WER | Text CER | Audio CER | Image CER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | `out/sft_full_muon_final_768.pth` | 57/57 | 0.3497 | 0.5459 | 0.2412 | 0.3464 | 0.5834 |
| B3 himem_b88 | `out/sft_full_muon_v2_himem_b88_final_768.pth` | 57/57 | 0.3532 | 0.5805 | 0.2302 | 0.3568 | 0.6028 |
| C v3 | `out/sft_full_muon_v3_final_768.pth` | 57/57 | 0.3364 | 0.5553 | 0.2176 | 0.3577 | 0.5401 |

## Calibrated Conclusion

Run C completed all 7 stages successfully, and the Phase-0 fixes produced measurable gains over B3 and baseline A. The strongest evidence is in image understanding and overall CER; the weakest and most disputed evidence is audio A2A, where semantic correctness, CER, and repeat behavior often disagree.

| Dimension | C v3 vs baseline A | Confidence |
| --- | --- | --- |
| Training completion | Successful, no hard error | High |
| Phase-0 changes | Warmup/global loss/RVQ weights/val monitoring completed without NaN or NCCL timeout | High |
| Structure/generation | Equal, 57/57 basic pass | High, but not discriminative |
| Overall ASR | C better, CER `0.3497 -> 0.3364` | High |
| Image understanding | C clearly better, image CER `0.5834 -> 0.5401` | High |
| Text T2A | C slightly better, many ties | Medium |
| Audio A2A | A still looks stronger on factual/refusal/physical prompts; C has conflicting intelligibility wins | Medium-high, pending human listening |
| AI-assisted total | C leads `25:17`, mostly pulled by image/text | Medium; audio subtotal is provisional |
| B3 comparison | C clearly better than B3, overall CER `0.3532 -> 0.3364` and image CER `0.6028 -> 0.5401` | High |

Conservative decision:

- Run C training and quantitative evaluation are reliable. C is the strongest current candidate and is clearly better than B3.
- Do not directly replace baseline A yet, especially for audio-input scenarios such as refusal, physical/factual prompts, and English A2A.
- Trust the AI-assisted review selectively: C's image lead is credible; Chinese text probes are mostly credible; the audio win/loss subtotal is only reference material because 10/23 audio cases conflict with CER direction.
- Required before promotion: human listening spot-check on 5-10 disputed audio cases, including `a2a_en_food`, `a2a_en_black_hole`, `a2a_zh_health_probe`, `a2a_en_sky_blue`, `a2a_zh_coffee_probe`, and `a2a_en_animals`.
- Next training focus: preserve C's image gains while addressing Stage6 audio regression, A2A factuality, refusal stability, and repeat suppression.

One-line summary: Run C's training and L1 quantitative data are reliable, and it improves text/image over A and broad metrics over B3; however, the AI-review total `C 25 : A 17` is pulled upward by image wins and has multiple audio contradictions, so the direction is usable but the audio details require human review.

## Structural Metrics

### text

| Run | Pass | Avg chars | Avg frames | Avg repeat |
| --- | ---: | ---: | ---: | ---: |
| A baseline | 23/23 | 200.5 | 104.6 | 0.0037 |
| B3 himem_b88 | 23/23 | 192.6 | 116.4 | 0.0096 |
| C v3 | 23/23 | 186.9 | 104.7 | 0.0014 |

### audio

| Run | Pass | Avg chars | Avg frames | Avg repeat |
| --- | ---: | ---: | ---: | ---: |
| A baseline | 23/23 | 259.3 | 127.9 | 0.0017 |
| B3 himem_b88 | 23/23 | 256.3 | 132.0 | 0.0042 |
| C v3 | 23/23 | 252.1 | 127.4 | 0.0052 |

### image

| Run | Pass | Avg chars | Avg frames | Avg repeat |
| --- | ---: | ---: | ---: | ---: |
| A baseline | 11/11 | 471.7 | 152.0 | 0.0075 |
| B3 himem_b88 | 11/11 | 482.9 | 152.0 | 0.0129 |
| C v3 | 11/11 | 450.5 | 151.2 | 0.0178 |

## ASR CER Pairwise: A baseline vs C v3

- Case-level lower-CER count: `A baseline` 29 / `C v3` 25 / tie 3

| Type | Ref wins | C wins | Ties |
| --- | ---: | ---: | ---: |
| audio | 11 | 10 | 2 |
| image | 6 | 5 | 0 |
| text | 12 | 10 | 1 |

## ASR CER Pairwise: B3 himem_b88 vs C v3

- Case-level lower-CER count: `B3 himem_b88` 27 / `C v3` 30 / tie 0

| Type | Ref wins | C wins | Ties |
| --- | ---: | ---: | ---: |
| audio | 12 | 11 | 0 |
| image | 4 | 7 | 0 |
| text | 11 | 12 | 0 |

## Largest ASR CER Changes vs Baseline A

| Direction | id | type | A CER | C CER | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| C better | `i2a_cat_question_probe` | image | 0.5013 | 0.0158 | -0.4855 |
| C better | `t2a_zh_health_probe` | text | 0.5545 | 0.1190 | -0.4355 |
| C better | `a2a_zh_black_hole_probe` | audio | 0.4844 | 0.0714 | -0.4130 |
| C better | `a2a_en_sky_blue` | audio | 0.3833 | 0.0046 | -0.3787 |
| C better | `a2a_en_stars` | audio | 0.6604 | 0.2885 | -0.3719 |
| C better | `t2a_en_sky_blue` | text | 0.5411 | 0.2308 | -0.3103 |
| C better | `a2a_en_black_hole` | audio | 0.2796 | 0.0000 | -0.2796 |
| C better | `t2a_en_snow_winter` | text | 0.5327 | 0.3957 | -0.1370 |
| C better | `t2a_en_space` | text | 0.1200 | 0.0000 | -0.1200 |
| C better | `t2a_en_health` | text | 0.6506 | 0.5579 | -0.0927 |
| C worse | `a2a_zh_health_probe` | audio | 0.0930 | 0.9302 | +0.8372 |
| C worse | `a2a_en_food` | audio | 0.0000 | 0.5595 | +0.5595 |
| C worse | `t2a_zh_count_probe` | text | 0.0417 | 0.2500 | +0.2083 |
| C worse | `a2a_zh_ai_fields_probe` | audio | 0.0000 | 0.1935 | +0.1935 |
| C worse | `t2a_en_coffee` | text | 0.3648 | 0.5257 | +0.1609 |
| C worse | `t2a_en_stars_twinkle` | text | 0.4891 | 0.6039 | +0.1148 |
| C worse | `a2a_zh_coffee_probe` | audio | 0.5669 | 0.6759 | +0.1090 |
| C worse | `t2a_zh_summer_probe` | text | 0.0000 | 0.0833 | +0.0833 |
| C worse | `t2a_zh_intro_probe` | text | 0.0333 | 0.1081 | +0.0748 |
| C worse | `i2a_cat_untrained_probe` | image | 0.5619 | 0.6328 | +0.0709 |

## AI-Assisted Review vs Baseline A

See `docs/evaluation_results/review_muon_v3_l1_t160_vs_baseline_l1_t160.md`.

This review is based on text output inspection, image grounding against the known eval images, and SenseVoice CER as an audio intelligibility proxy. It is **not** a human listening panel.
For audio cases, this table is provisional: several case-level judgments conflict with ASR/CER rankings, so it should not be used as the final audio-quality verdict.

| Type | A wins | C wins | Ties |
| --- | ---: | ---: | ---: |
| audio | 13 | 8 | 2 |
| image | 1 | 8 | 2 |
| text | 3 | 9 | 11 |
| total | 17 | 25 | 15 |

### Trustworthy Case Patterns

Some review calls are well supported because the raw output and ASR/CER direction agree:

- Image grounding: C's image win is credible. `i2a_cat_question_probe` moves from A describing "cat, specifically a dog" to C describing a cat/book-like scene, with CER `0.5013 -> 0.0158`. Similar image cases such as panda, dinosaur toy, and fruit show C identifying the main subject more often, while A frequently drifts to wrong objects such as "cowbell".
- Chinese text probes: C's wins on `t2a_zh_who_probe` and `t2a_zh_health_probe` are credible because the responses are more direct and the CER is lower.
- Audio factual/refusal cases favoring A are also credible when semantics are clear: `a2a_en_food` has A correctly refusing a personal food preference while C invents a sandwich preference; `a2a_zh_cat_story_probe` shows C's severe "咬咬咬" repetition.

### ASR/CER Counterexamples

Some audio cases show why CER must not be treated as the final judge:

- `a2a_en_black_hole`: C has CER `0.0000` because ASR perfectly transcribes the generated phrase, but the generated answer is semantically wrong ("The black holster is the male black holster."). A has worse CER (`0.2796`) but keeps the black-hole concept, so semantic review favors A.
- Audio repeat and semantic drift are not reliably captured by aggregate CER. C's audio repeat is worse than A (`0.0052` vs `0.0017`) despite C's better overall CER.
- Audio conflict audit: 10/23 audio cases have a semantic review winner opposite to the lower-CER direction; one additional case has tied CER but a semantic preference. The audio subtotal (`A 13 / C 8 / Tie 2`) is therefore a triage signal, not a dependable rank. The full 10-case list is in `docs/evaluation_results/review_muon_v3_l1_t160_vs_baseline_l1_t160.md`.

High-dispute examples:

| Case | Review winner | CER direction | Why disputed |
| --- | --- | --- | --- |
| `a2a_zh_coffee_probe` | C | A lower CER | C has audio score 0, worse CER, and odd procedural details such as `200℃` baking paper. |
| `a2a_en_sky_blue` | A | C much lower CER | C is far more intelligible by ASR (`0.0046` vs A `0.3833`), while A was preferred on semantic wording. |
| `a2a_en_animals` | A | C lower CER | C's "five species" is absurd despite perfect transcription. |
| `a2a_zh_ai_fields_probe` | C | A lower CER | C is richer, but A has perfect CER; this is not a clean quality win. |
| `a2a_en_ai_fields` | C | A lower CER | Both are weak and high-CER; C win is low confidence. |
| `a2a_en_health` | C | A slightly lower CER | CER gap is tiny, so semantic preference is weak. |

## Notes

- All three runs pass the L1 structural gate at 57/57, so structure is saturated and no longer discriminative.
- C has the best overall SenseVoice CER among the three, mainly from text and image improvements; A remains slightly better on audio CER.
- The ASR signal is mixed: C has the better mean overall CER, but A has lower CER on more individual A-vs-C cases (29 vs 25, with 3 ties).
- C has slightly worse overall WER than A but better WER than B3; Chinese one-token/short probes still distort WER, so CER is the more useful ASR proxy here.
- Repeat metrics are mixed: C improves text repeat but worsens audio repeat (`0.0052` vs A `0.0017`) and image repeat (`0.0178` vs A `0.0075`). `a2a_zh_cat_story_probe` shows a clear repeated "咬咬咬" pattern.
- AI-assisted review favors C overall, mostly due to image grounding and Chinese/text probes. The audio portion is unresolved because ASR/CER, semantic correctness, and repeat behavior conflict on several cases.
- Human listening and stricter audio semantic spot-checks are still pending before promoting C as a final overall baseline; current audio evidence should be treated as unresolved, not as an A-over-C or C-over-A final verdict.
