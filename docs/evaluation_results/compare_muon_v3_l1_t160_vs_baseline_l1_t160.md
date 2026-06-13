# Eval Comparison: baseline_A_l1_t160 vs RunC_v3_l1_t160

- Generated: 2026-06-12 18:00:27
- Run A: `out/sft_full_muon_final_768.pth`
- Run B: `out/sft_full_muon_v3_final_768.pth`
- Test set: `dataset/eval_muon_l1.jsonl`

## Overall

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| Basic pass | 57/57 | 57/57 | +0 |
| Pass rate | 100.00% | 100.00% | 0 |
| Dtype | bf16 | bf16 | |
| max_new_tokens | 160 | 160 | |

## By Type

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A avg repeat | B avg repeat | A avg sec | B avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 23/23 | 23/23 | 259.3 | 252.1 | 127.9 | 127.4 | 0.0017 | 0.0052 | 0.976 | 0.949 |
| image | 11/11 | 11/11 | 471.7 | 450.5 | 152.0 | 151.2 | 0.0075 | 0.0178 | 1.091 | 1.068 |
| text | 23/23 | 23/23 | 200.5 | 186.9 | 104.6 | 104.7 | 0.0037 | 0.0014 | 0.804 | 0.795 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_ai_fields | True | True | 588 | 632 | 0.0 | 0.0 |  |
| a2a_en_animals | True | True | 55 | 43 | 0.0 | 0.0 |  |
| a2a_en_black_hole | True | True | 304 | 44 | 0.0 | 0.0 |  |
| a2a_en_cat_story | True | True | 452 | 484 | 0.0 | 0.022 |  |
| a2a_en_cats_mice | True | True | 462 | 473 | 0.0 | 0.0133 |  |
| a2a_en_coffee | True | True | 374 | 366 | 0.0 | 0.0 |  |
| a2a_en_food | True | True | 94 | 454 | 0.0 | 0.0 |  |
| a2a_en_health | True | True | 504 | 533 | 0.0 | 0.0 |  |
| a2a_en_quantum | True | True | 519 | 519 | 0.0 | 0.0556 |  |
| a2a_en_sky_blue | True | True | 347 | 217 | 0.04 | 0.0294 |  |
| a2a_en_snow | True | True | 464 | 465 | 0.0 | 0.0 |  |
| a2a_en_spring | True | True | 88 | 131 | 0.0 | 0.0 |  |
| a2a_en_stars | True | True | 533 | 305 | 0.0 | 0.0 |  |
| a2a_en_study | True | True | 553 | 549 | 0.0 | 0.0 |  |
| a2a_zh_ai_fields_probe | True | True | 37 | 62 | 0.0 | 0.0 |  |
| a2a_zh_black_hole_probe | True | True | 128 | 70 | 0.0 | 0.0 |  |
| a2a_zh_cat_story_probe | True | True | 146 | 129 | 0.0 | 0.0 |  |
| a2a_zh_coffee_probe | True | True | 127 | 145 | 0.0 | 0.0 |  |
| a2a_zh_food_probe | True | True | 37 | 38 | 0.0 | 0.0 |  |
| a2a_zh_health_probe | True | True | 43 | 43 | 0.0 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 40 | 29 | 0.0 | 0.0 |  |
| a2a_zh_snow_probe | True | True | 34 | 41 | 0.0 | 0.0 |  |
| a2a_zh_spring_probe | True | True | 34 | 27 | 0.0 | 0.0 |  |
| i2a_astronaut_bicycle_probe | True | True | 502 | 444 | 0.0 | 0.1125 |  |
| i2a_cat_question_probe | True | True | 389 | 190 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 452 | 531 | 0.0 | 0.0 |  |
| i2a_coffee_laptop_probe | True | True | 467 | 476 | 0.0 | 0.0 |  |
| i2a_dinosaur_toy_probe | True | True | 486 | 480 | 0.0 | 0.0 |  |
| i2a_dog_scarf_snow_probe | True | True | 473 | 477 | 0.0114 | 0.0 |  |
| i2a_fruit_question_probe | True | True | 502 | 433 | 0.0 | 0.0256 |  |
| i2a_fruit_untrained_probe | True | True | 440 | 492 | 0.0 | 0.0 |  |
| i2a_panda_sign_probe | True | True | 450 | 488 | 0.039 | 0.0 |  |
| i2a_rabbit_grass_probe | True | True | 561 | 489 | 0.0319 | 0.0 |  |
| i2a_robot_chef_probe | True | True | 467 | 456 | 0.0 | 0.0575 |  |
| t2a_en_black_hole | True | True | 515 | 485 | 0.0 | 0.0 |  |
| t2a_en_cat_story | True | True | 483 | 492 | 0.0116 | 0.0 |  |
| t2a_en_coffee | True | True | 318 | 409 | 0.0 | 0.0 |  |
| t2a_en_colors | True | True | 38 | 39 | 0.0 | 0.0 |  |
| t2a_en_count | True | True | 67 | 68 | 0.0 | 0.0 |  |
| t2a_en_health | True | True | 498 | 484 | 0.0597 | 0.0 |  |
| t2a_en_intro | True | True | 99 | 82 | 0.0 | 0.0 |  |
| t2a_en_intro_v2 | True | True | 123 | 105 | 0.0 | 0.0 |  |
| t2a_en_intro_v3 | True | True | 103 | 58 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_sky_blue | True | True | 401 | 273 | 0.0 | 0.0 |  |
| t2a_en_snow_winter | True | True | 428 | 326 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 175 | 79 | 0.0 | 0.0 |  |
| t2a_en_stars_twinkle | True | True | 411 | 515 | 0.0141 | 0.0326 |  |
| t2a_en_study | True | True | 574 | 525 | 0.0 | 0.0 |  |
| t2a_zh_black_hole_probe | True | True | 51 | 32 | 0.0 | 0.0 |  |
| t2a_zh_count_probe | True | True | 24 | 4 | 0.0 | 0.0 |  |
| t2a_zh_health_probe | True | True | 101 | 42 | 0.0 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 30 | 37 | 0.0 | 0.0 |  |
| t2a_zh_sky_blue_probe | True | True | 47 | 48 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 27 | 47 | 0.0 | 0.0 |  |
| t2a_zh_summer_probe | True | True | 22 | 48 | 0.0 | 0.0 |  |
| t2a_zh_who_probe | True | True | 11 | 35 | 0.0 | 0.0 |  |
