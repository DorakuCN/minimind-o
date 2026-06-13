# Eval Comparison: B3_himem_b88_l1_t160 vs RunC_v3_l1_t160

- Generated: 2026-06-12 18:00:27
- Run A: `out/sft_full_muon_v2_himem_b88_final_768.pth`
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
| audio | 23/23 | 23/23 | 256.3 | 252.1 | 132.0 | 127.4 | 0.0042 | 0.0052 | 1.025 | 0.949 |
| image | 11/11 | 11/11 | 482.9 | 450.5 | 152.0 | 151.2 | 0.0129 | 0.0178 | 1.389 | 1.068 |
| text | 23/23 | 23/23 | 192.6 | 186.9 | 116.4 | 104.7 | 0.0096 | 0.0014 | 0.904 | 0.795 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_ai_fields | True | True | 619 | 632 | 0.0 | 0.0 |  |
| a2a_en_animals | True | True | 193 | 43 | 0.0 | 0.0 |  |
| a2a_en_black_hole | True | True | 110 | 44 | 0.0 | 0.0 |  |
| a2a_en_cat_story | True | True | 497 | 484 | 0.0 | 0.022 |  |
| a2a_en_cats_mice | True | True | 530 | 473 | 0.0 | 0.0133 |  |
| a2a_en_coffee | True | True | 414 | 366 | 0.025 | 0.0 |  |
| a2a_en_food | True | True | 75 | 454 | 0.0 | 0.0 |  |
| a2a_en_health | True | True | 493 | 533 | 0.0 | 0.0 |  |
| a2a_en_quantum | True | True | 439 | 519 | 0.0299 | 0.0556 |  |
| a2a_en_sky_blue | True | True | 242 | 217 | 0.0 | 0.0294 |  |
| a2a_en_snow | True | True | 431 | 465 | 0.0139 | 0.0 |  |
| a2a_en_spring | True | True | 148 | 131 | 0.0 | 0.0 |  |
| a2a_en_stars | True | True | 440 | 305 | 0.0274 | 0.0 |  |
| a2a_en_study | True | True | 523 | 549 | 0.0 | 0.0 |  |
| a2a_zh_ai_fields_probe | True | True | 201 | 62 | 0.0 | 0.0 |  |
| a2a_zh_black_hole_probe | True | True | 35 | 70 | 0.0 | 0.0 |  |
| a2a_zh_cat_story_probe | True | True | 172 | 129 | 0.0 | 0.0 |  |
| a2a_zh_coffee_probe | True | True | 95 | 145 | 0.0 | 0.0 |  |
| a2a_zh_food_probe | True | True | 29 | 38 | 0.0 | 0.0 |  |
| a2a_zh_health_probe | True | True | 31 | 43 | 0.0 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 78 | 29 | 0.0 | 0.0 |  |
| a2a_zh_snow_probe | True | True | 42 | 41 | 0.0 | 0.0 |  |
| a2a_zh_spring_probe | True | True | 57 | 27 | 0.0 | 0.0 |  |
| i2a_astronaut_bicycle_probe | True | True | 451 | 444 | 0.026 | 0.1125 |  |
| i2a_cat_question_probe | True | True | 494 | 190 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 533 | 531 | 0.0 | 0.0 |  |
| i2a_coffee_laptop_probe | True | True | 498 | 476 | 0.069 | 0.0 |  |
| i2a_dinosaur_toy_probe | True | True | 483 | 480 | 0.0112 | 0.0 |  |
| i2a_dog_scarf_snow_probe | True | True | 502 | 477 | 0.0 | 0.0 |  |
| i2a_fruit_question_probe | True | True | 457 | 433 | 0.0 | 0.0256 |  |
| i2a_fruit_untrained_probe | True | True | 489 | 492 | 0.0 | 0.0 |  |
| i2a_panda_sign_probe | True | True | 461 | 488 | 0.0 | 0.0 |  |
| i2a_rabbit_grass_probe | True | True | 472 | 489 | 0.023 | 0.0 |  |
| i2a_robot_chef_probe | True | True | 472 | 456 | 0.0122 | 0.0575 |  |
| t2a_en_black_hole | True | True | 524 | 485 | 0.022 | 0.0 |  |
| t2a_en_cat_story | True | True | 480 | 492 | 0.0 | 0.0 |  |
| t2a_en_coffee | True | True | 344 | 409 | 0.0615 | 0.0 |  |
| t2a_en_colors | True | True | 39 | 39 | 0.0 | 0.0 |  |
| t2a_en_count | True | True | 156 | 68 | 0.1071 | 0.0 |  |
| t2a_en_health | True | True | 487 | 484 | 0.0 | 0.0 |  |
| t2a_en_intro | True | True | 141 | 82 | 0.0 | 0.0 |  |
| t2a_en_intro_v2 | True | True | 105 | 105 | 0.0 | 0.0 |  |
| t2a_en_intro_v3 | True | True | 69 | 58 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_sky_blue | True | True | 253 | 273 | 0.0 | 0.0 |  |
| t2a_en_snow_winter | True | True | 326 | 326 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 209 | 79 | 0.0294 | 0.0 |  |
| t2a_en_stars_twinkle | True | True | 392 | 515 | 0.0 | 0.0326 |  |
| t2a_en_study | True | True | 499 | 525 | 0.0 | 0.0 |  |
| t2a_zh_black_hole_probe | True | True | 41 | 32 | 0.0 | 0.0 |  |
| t2a_zh_count_probe | True | True | 11 | 4 | 0.0 | 0.0 |  |
| t2a_zh_health_probe | True | True | 32 | 42 | 0.0 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 55 | 37 | 0.0 | 0.0 |  |
| t2a_zh_sky_blue_probe | True | True | 35 | 48 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 35 | 47 | 0.0 | 0.0 |  |
| t2a_zh_summer_probe | True | True | 47 | 48 | 0.0 | 0.0 |  |
| t2a_zh_who_probe | True | True | 84 | 35 | 0.0 | 0.0 |  |
