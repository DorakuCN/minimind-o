# Eval Comparison: baseline_A_l1_t160 vs B3_himem_b88_l1_t160

- Generated: 2026-06-11 16:19:14
- Run A: `out/sft_full_muon_final_768.pth`
- Run B: `out/sft_full_muon_v2_himem_b88_final_768.pth`
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
| audio | 23/23 | 23/23 | 259.3 | 256.3 | 127.9 | 132.0 | 0.0017 | 0.0042 | 0.976 | 1.025 |
| image | 11/11 | 11/11 | 471.7 | 482.9 | 152.0 | 152.0 | 0.0075 | 0.0129 | 1.091 | 1.389 |
| text | 23/23 | 23/23 | 200.5 | 192.6 | 104.6 | 116.4 | 0.0037 | 0.0096 | 0.804 | 0.904 |

## ASR Round-Trip (updated 2026-06-11)

SenseVoiceSmall through `funasr`, CPU batch inference (`--device cpu --batch_size 16`); CER/WER between generated text and ASR transcript of generated audio.

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| avg CER | 0.3497 | 0.3532 | +0.0035 |
| avg WER | 0.5459 | 0.5805 | +0.0346 |
| text CER | 0.2412 | 0.2302 | -0.0110 |
| text WER | 0.5066 | 0.5065 | -0.0001 |
| audio CER | 0.3464 | 0.3568 | +0.0104 |
| audio WER | 0.5441 | 0.6213 | +0.0772 |
| image CER | 0.5834 | 0.6028 | +0.0194 |
| image WER | 0.6321 | 0.6500 | +0.0179 |

The expanded L1 ASR proxy is near parity, with baseline A slightly ahead overall and on audio/image cases; B is slightly better only on text CER.

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_ai_fields | True | True | 588 | 619 | 0.0 | 0.0 |  |
| a2a_en_animals | True | True | 55 | 193 | 0.0 | 0.0 |  |
| a2a_en_black_hole | True | True | 304 | 110 | 0.0 | 0.0 |  |
| a2a_en_cat_story | True | True | 452 | 497 | 0.0 | 0.0 |  |
| a2a_en_cats_mice | True | True | 462 | 530 | 0.0 | 0.0 |  |
| a2a_en_coffee | True | True | 374 | 414 | 0.0 | 0.025 |  |
| a2a_en_food | True | True | 94 | 75 | 0.0 | 0.0 |  |
| a2a_en_health | True | True | 504 | 493 | 0.0 | 0.0 |  |
| a2a_en_quantum | True | True | 519 | 439 | 0.0 | 0.0299 |  |
| a2a_en_sky_blue | True | True | 347 | 242 | 0.04 | 0.0 |  |
| a2a_en_snow | True | True | 464 | 431 | 0.0 | 0.0139 |  |
| a2a_en_spring | True | True | 88 | 148 | 0.0 | 0.0 |  |
| a2a_en_stars | True | True | 533 | 440 | 0.0 | 0.0274 |  |
| a2a_en_study | True | True | 553 | 523 | 0.0 | 0.0 |  |
| a2a_zh_ai_fields_probe | True | True | 37 | 201 | 0.0 | 0.0 |  |
| a2a_zh_black_hole_probe | True | True | 128 | 35 | 0.0 | 0.0 |  |
| a2a_zh_cat_story_probe | True | True | 146 | 172 | 0.0 | 0.0 |  |
| a2a_zh_coffee_probe | True | True | 127 | 95 | 0.0 | 0.0 |  |
| a2a_zh_food_probe | True | True | 37 | 29 | 0.0 | 0.0 |  |
| a2a_zh_health_probe | True | True | 43 | 31 | 0.0 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 40 | 78 | 0.0 | 0.0 |  |
| a2a_zh_snow_probe | True | True | 34 | 42 | 0.0 | 0.0 |  |
| a2a_zh_spring_probe | True | True | 34 | 57 | 0.0 | 0.0 |  |
| i2a_astronaut_bicycle_probe | True | True | 502 | 451 | 0.0 | 0.026 |  |
| i2a_cat_question_probe | True | True | 389 | 494 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 452 | 533 | 0.0 | 0.0 |  |
| i2a_coffee_laptop_probe | True | True | 467 | 498 | 0.0 | 0.069 |  |
| i2a_dinosaur_toy_probe | True | True | 486 | 483 | 0.0 | 0.0112 |  |
| i2a_dog_scarf_snow_probe | True | True | 473 | 502 | 0.0114 | 0.0 |  |
| i2a_fruit_question_probe | True | True | 502 | 457 | 0.0 | 0.0 |  |
| i2a_fruit_untrained_probe | True | True | 440 | 489 | 0.0 | 0.0 |  |
| i2a_panda_sign_probe | True | True | 450 | 461 | 0.039 | 0.0 |  |
| i2a_rabbit_grass_probe | True | True | 561 | 472 | 0.0319 | 0.023 |  |
| i2a_robot_chef_probe | True | True | 467 | 472 | 0.0 | 0.0122 |  |
| t2a_en_black_hole | True | True | 515 | 524 | 0.0 | 0.022 |  |
| t2a_en_cat_story | True | True | 483 | 480 | 0.0116 | 0.0 |  |
| t2a_en_coffee | True | True | 318 | 344 | 0.0 | 0.0615 |  |
| t2a_en_colors | True | True | 38 | 39 | 0.0 | 0.0 |  |
| t2a_en_count | True | True | 67 | 156 | 0.0 | 0.1071 |  |
| t2a_en_health | True | True | 498 | 487 | 0.0597 | 0.0 |  |
| t2a_en_intro | True | True | 99 | 141 | 0.0 | 0.0 |  |
| t2a_en_intro_v2 | True | True | 123 | 105 | 0.0 | 0.0 |  |
| t2a_en_intro_v3 | True | True | 103 | 69 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_sky_blue | True | True | 401 | 253 | 0.0 | 0.0 |  |
| t2a_en_snow_winter | True | True | 428 | 326 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 175 | 209 | 0.0 | 0.0294 |  |
| t2a_en_stars_twinkle | True | True | 411 | 392 | 0.0141 | 0.0 |  |
| t2a_en_study | True | True | 574 | 499 | 0.0 | 0.0 |  |
| t2a_zh_black_hole_probe | True | True | 51 | 41 | 0.0 | 0.0 |  |
| t2a_zh_count_probe | True | True | 24 | 11 | 0.0 | 0.0 |  |
| t2a_zh_health_probe | True | True | 101 | 32 | 0.0 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 30 | 55 | 0.0 | 0.0 |  |
| t2a_zh_sky_blue_probe | True | True | 47 | 35 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 27 | 35 | 0.0 | 0.0 |  |
| t2a_zh_summer_probe | True | True | 22 | 47 | 0.0 | 0.0 |  |
| t2a_zh_who_probe | True | True | 11 | 84 | 0.0 | 0.0 |  |
