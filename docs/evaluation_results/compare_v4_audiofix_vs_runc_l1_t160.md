# Eval Comparison: runC_v3 vs v4_audiofix_D_b

- Generated: 2026-06-12 22:40:04
- Run A: `out/sft_full_muon_v3_final_768.pth`
- Run B: `out/sft_full_muon_v4_audiofix_a2a_final_768.pth`
- Test set: `dataset/eval_muon_l1.jsonl`

## Calibrated Verdict

D-b is healthier than Run C, but not because it fixed audio intelligibility.

Key L1/SenseVoice numbers:

| Metric | Run C | D-b audiofix | Direction |
| --- | ---: | ---: | --- |
| Overall CER | **0.3364** | 0.3414 | C slight better |
| Text CER | **0.2176** | 0.2387 | C better |
| Audio CER | 0.3577 | **0.3571** | effectively tied |
| Image CER | 0.5401 | **0.5232** | D-b better |
| Overall WER | 0.5553 | **0.4926** | D-b better, but Chinese short WER is noisy |
| Audio repeat | 0.0052 | **0.0025** | D-b much cleaner |
| Image repeat | 0.0178 | **0.0080** | D-b much cleaner |

Interpretation:

- D-b repaired much of C's repeat pathology and improved image CER.
- D-b did not materially improve audio CER (`0.3577 -> 0.3571`).
- The Stage6 val-loss recovery therefore should be read as a training-health signal, not as proof of audio quality recovery.
- D-b is preferable to C when repeat health and image CER matter; C remains stronger on text CER.

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
| audio | 23/23 | 23/23 | 252.1 | 273.8 | 127.4 | 133.3 | 0.0052 | 0.0025 | 0.949 | 0.997 |
| image | 11/11 | 11/11 | 450.5 | 420.9 | 151.2 | 152.0 | 0.0178 | 0.0080 | 1.068 | 1.083 |
| text | 23/23 | 23/23 | 186.9 | 210.1 | 104.7 | 113.7 | 0.0014 | 0.0022 | 0.795 | 0.864 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_ai_fields | True | True | 632 | 604 | 0.0 | 0.0 |  |
| a2a_en_animals | True | True | 43 | 175 | 0.0 | 0.0 |  |
| a2a_en_black_hole | True | True | 44 | 277 | 0.0 | 0.0 |  |
| a2a_en_cat_story | True | True | 484 | 431 | 0.022 | 0.0 |  |
| a2a_en_cats_mice | True | True | 473 | 405 | 0.0133 | 0.0 |  |
| a2a_en_coffee | True | True | 366 | 460 | 0.0 | 0.0 |  |
| a2a_en_food | True | True | 454 | 201 | 0.0 | 0.0 |  |
| a2a_en_health | True | True | 533 | 505 | 0.0 | 0.0127 |  |
| a2a_en_quantum | True | True | 519 | 553 | 0.0556 | 0.0 |  |
| a2a_en_sky_blue | True | True | 217 | 386 | 0.0294 | 0.0303 |  |
| a2a_en_snow | True | True | 465 | 579 | 0.0 | 0.0 |  |
| a2a_en_spring | True | True | 131 | 146 | 0.0 | 0.0 |  |
| a2a_en_stars | True | True | 305 | 458 | 0.0 | 0.0135 |  |
| a2a_en_study | True | True | 549 | 448 | 0.0 | 0.0 |  |
| a2a_zh_ai_fields_probe | True | True | 62 | 29 | 0.0 | 0.0 |  |
| a2a_zh_black_hole_probe | True | True | 70 | 159 | 0.0 | 0.0 |  |
| a2a_zh_cat_story_probe | True | True | 129 | 173 | 0.0 | 0.0 |  |
| a2a_zh_coffee_probe | True | True | 145 | 129 | 0.0 | 0.0 |  |
| a2a_zh_food_probe | True | True | 38 | 48 | 0.0 | 0.0 |  |
| a2a_zh_health_probe | True | True | 43 | 46 | 0.0 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 29 | 37 | 0.0 | 0.0 |  |
| a2a_zh_snow_probe | True | True | 41 | 30 | 0.0 | 0.0 |  |
| a2a_zh_spring_probe | True | True | 27 | 19 | 0.0 | 0.0 |  |
| i2a_astronaut_bicycle_probe | True | True | 444 | 451 | 0.1125 | 0.0 |  |
| i2a_cat_question_probe | True | True | 190 | 247 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 531 | 445 | 0.0 | 0.0 |  |
| i2a_coffee_laptop_probe | True | True | 476 | 435 | 0.0 | 0.0267 |  |
| i2a_dinosaur_toy_probe | True | True | 480 | 480 | 0.0 | 0.0 |  |
| i2a_dog_scarf_snow_probe | True | True | 477 | 442 | 0.0 | 0.0 |  |
| i2a_fruit_question_probe | True | True | 433 | 300 | 0.0256 | 0.0 |  |
| i2a_fruit_untrained_probe | True | True | 492 | 494 | 0.0 | 0.0108 |  |
| i2a_panda_sign_probe | True | True | 488 | 456 | 0.0 | 0.0 |  |
| i2a_rabbit_grass_probe | True | True | 489 | 418 | 0.0 | 0.0274 |  |
| i2a_robot_chef_probe | True | True | 456 | 462 | 0.0575 | 0.0233 |  |
| t2a_en_black_hole | True | True | 485 | 514 | 0.0 | 0.0 |  |
| t2a_en_cat_story | True | True | 492 | 481 | 0.0 | 0.05 |  |
| t2a_en_coffee | True | True | 409 | 467 | 0.0 | 0.0 |  |
| t2a_en_colors | True | True | 39 | 39 | 0.0 | 0.0 |  |
| t2a_en_count | True | True | 68 | 176 | 0.0 | 0.0 |  |
| t2a_en_health | True | True | 484 | 466 | 0.0 | 0.0 |  |
| t2a_en_intro | True | True | 82 | 59 | 0.0 | 0.0 |  |
| t2a_en_intro_v2 | True | True | 105 | 154 | 0.0 | 0.0 |  |
| t2a_en_intro_v3 | True | True | 58 | 155 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_sky_blue | True | True | 273 | 440 | 0.0 | 0.0 |  |
| t2a_en_snow_winter | True | True | 326 | 428 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 79 | 149 | 0.0 | 0.0 |  |
| t2a_en_stars_twinkle | True | True | 515 | 343 | 0.0326 | 0.0 |  |
| t2a_en_study | True | True | 525 | 558 | 0.0 | 0.0 |  |
| t2a_zh_black_hole_probe | True | True | 32 | 36 | 0.0 | 0.0 |  |
| t2a_zh_count_probe | True | True | 4 | 94 | 0.0 | 0.0 |  |
| t2a_zh_health_probe | True | True | 42 | 44 | 0.0 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 37 | 27 | 0.0 | 0.0 |  |
| t2a_zh_sky_blue_probe | True | True | 48 | 43 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 47 | 32 | 0.0 | 0.0 |  |
| t2a_zh_summer_probe | True | True | 48 | 37 | 0.0 | 0.0 |  |
| t2a_zh_who_probe | True | True | 35 | 24 | 0.0 | 0.0 |  |
