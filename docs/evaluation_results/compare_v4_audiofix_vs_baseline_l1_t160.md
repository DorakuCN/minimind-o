# Eval Comparison: baseline_A vs v4_audiofix

- Generated: 2026-06-12 22:44:31
- Run A: `out/sft_full_muon_final_768.pth`
- Run B: `out/sft_full_muon_v4_audiofix_a2a_final_768.pth`
- Test set: `dataset/eval_muon_l1.jsonl`

## Calibrated Verdict

D-b is a cleaner and healthier checkpoint than Run C, but it does **not** meet the original audio-recovery gate against baseline A.

Key L1/SenseVoice numbers:

| Metric | Baseline A | D-b audiofix | Direction |
| --- | ---: | ---: | --- |
| Overall CER | 0.3497 | **0.3414** | D-b better |
| Text CER | 0.2412 | **0.2387** | D-b slight better |
| Audio CER | **0.3464** | 0.3571 | A still better |
| Image CER | 0.5834 | **0.5232** | D-b much better |
| Overall WER | 0.5459 | **0.4926** | D-b better, but Chinese short WER is noisy |
| Audio repeat | **0.0017** | 0.0025 | A still slightly cleaner |
| Image repeat | **0.0075** | 0.0080 | near-tie |

Case-level CER says D-b wins more rows overall (`30` vs `23`, `4` ties), but the audio subset remains split (`A 11 / D-b 10 / tie 2`). Therefore D-b should not replace A for audio-input scenarios yet.

Decision:

- Do not promote `sft_full_muon_v4_audiofix_a2a_final` as a new universal baseline.
- Keep it as the best current image/repeat-health candidate.
- Use human spot-checks for disputed A2A cases before any audio-facing promotion.
- The next training change should move earlier than Stage6: prevent audio damage during Stage5 I2T-all with A2A replay, or strengthen Stage2-3 A2A core.

## Overall

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| Basic pass | 57/57 | 57/57 | +0 |
| Pass rate | 100.00% | 100.00% | 0 |
| Dtype | bf16 | bf16 | |
| max_new_tokens | 160 | 160 | |
| temperature | 0.7 | 0.7 | |
| top_p | 0.85 | 0.85 | |
| repetition_penalty | 1.0 | 1.0 | |
| no_repeat_ngram_size | 0 | 0 | |

## By Type

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A avg repeat | B avg repeat | A avg sec | B avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 23/23 | 23/23 | 259.3 | 273.8 | 127.9 | 133.3 | 0.0017 | 0.0025 | 0.976 | 1.100 |
| image | 11/11 | 11/11 | 471.7 | 420.9 | 152.0 | 152.0 | 0.0075 | 0.0080 | 1.091 | 1.159 |
| text | 23/23 | 23/23 | 200.5 | 210.1 | 104.6 | 113.7 | 0.0037 | 0.0022 | 0.804 | 1.080 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_ai_fields | True | True | 588 | 604 | 0.0 | 0.0 |  |
| a2a_en_animals | True | True | 55 | 175 | 0.0 | 0.0 |  |
| a2a_en_black_hole | True | True | 304 | 277 | 0.0 | 0.0 |  |
| a2a_en_cat_story | True | True | 452 | 431 | 0.0 | 0.0 |  |
| a2a_en_cats_mice | True | True | 462 | 405 | 0.0 | 0.0 |  |
| a2a_en_coffee | True | True | 374 | 460 | 0.0 | 0.0 |  |
| a2a_en_food | True | True | 94 | 201 | 0.0 | 0.0 |  |
| a2a_en_health | True | True | 504 | 505 | 0.0 | 0.0127 |  |
| a2a_en_quantum | True | True | 519 | 553 | 0.0 | 0.0 |  |
| a2a_en_sky_blue | True | True | 347 | 386 | 0.04 | 0.0303 |  |
| a2a_en_snow | True | True | 464 | 579 | 0.0 | 0.0 |  |
| a2a_en_spring | True | True | 88 | 146 | 0.0 | 0.0 |  |
| a2a_en_stars | True | True | 533 | 458 | 0.0 | 0.0135 |  |
| a2a_en_study | True | True | 553 | 448 | 0.0 | 0.0 |  |
| a2a_zh_ai_fields_probe | True | True | 37 | 29 | 0.0 | 0.0 |  |
| a2a_zh_black_hole_probe | True | True | 128 | 159 | 0.0 | 0.0 |  |
| a2a_zh_cat_story_probe | True | True | 146 | 173 | 0.0 | 0.0 |  |
| a2a_zh_coffee_probe | True | True | 127 | 129 | 0.0 | 0.0 |  |
| a2a_zh_food_probe | True | True | 37 | 48 | 0.0 | 0.0 |  |
| a2a_zh_health_probe | True | True | 43 | 46 | 0.0 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 40 | 37 | 0.0 | 0.0 |  |
| a2a_zh_snow_probe | True | True | 34 | 30 | 0.0 | 0.0 |  |
| a2a_zh_spring_probe | True | True | 34 | 19 | 0.0 | 0.0 |  |
| i2a_astronaut_bicycle_probe | True | True | 502 | 451 | 0.0 | 0.0 |  |
| i2a_cat_question_probe | True | True | 389 | 247 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 452 | 445 | 0.0 | 0.0 |  |
| i2a_coffee_laptop_probe | True | True | 467 | 435 | 0.0 | 0.0267 |  |
| i2a_dinosaur_toy_probe | True | True | 486 | 480 | 0.0 | 0.0 |  |
| i2a_dog_scarf_snow_probe | True | True | 473 | 442 | 0.0114 | 0.0 |  |
| i2a_fruit_question_probe | True | True | 502 | 300 | 0.0 | 0.0 |  |
| i2a_fruit_untrained_probe | True | True | 440 | 494 | 0.0 | 0.0108 |  |
| i2a_panda_sign_probe | True | True | 450 | 456 | 0.039 | 0.0 |  |
| i2a_rabbit_grass_probe | True | True | 561 | 418 | 0.0319 | 0.0274 |  |
| i2a_robot_chef_probe | True | True | 467 | 462 | 0.0 | 0.0233 |  |
| t2a_en_black_hole | True | True | 515 | 514 | 0.0 | 0.0 |  |
| t2a_en_cat_story | True | True | 483 | 481 | 0.0116 | 0.05 |  |
| t2a_en_coffee | True | True | 318 | 467 | 0.0 | 0.0 |  |
| t2a_en_colors | True | True | 38 | 39 | 0.0 | 0.0 |  |
| t2a_en_count | True | True | 67 | 176 | 0.0 | 0.0 |  |
| t2a_en_health | True | True | 498 | 466 | 0.0597 | 0.0 |  |
| t2a_en_intro | True | True | 99 | 59 | 0.0 | 0.0 |  |
| t2a_en_intro_v2 | True | True | 123 | 154 | 0.0 | 0.0 |  |
| t2a_en_intro_v3 | True | True | 103 | 155 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_sky_blue | True | True | 401 | 440 | 0.0 | 0.0 |  |
| t2a_en_snow_winter | True | True | 428 | 428 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 175 | 149 | 0.0 | 0.0 |  |
| t2a_en_stars_twinkle | True | True | 411 | 343 | 0.0141 | 0.0 |  |
| t2a_en_study | True | True | 574 | 558 | 0.0 | 0.0 |  |
| t2a_zh_black_hole_probe | True | True | 51 | 36 | 0.0 | 0.0 |  |
| t2a_zh_count_probe | True | True | 24 | 94 | 0.0 | 0.0 |  |
| t2a_zh_health_probe | True | True | 101 | 44 | 0.0 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 30 | 27 | 0.0 | 0.0 |  |
| t2a_zh_sky_blue_probe | True | True | 47 | 43 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 27 | 32 | 0.0 | 0.0 |  |
| t2a_zh_summer_probe | True | True | 22 | 37 | 0.0 | 0.0 |  |
| t2a_zh_who_probe | True | True | 11 | 24 | 0.0 | 0.0 |  |
