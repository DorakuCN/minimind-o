# Eval Comparison: baseline_96 vs baseline_160

- Generated: 2026-06-10 16:15:43
- Run A: `out/sft_full_muon_final_768.pth`
- Run B: `out/sft_full_muon_final_768.pth`
- Test set: `dataset/eval_muon_mini.jsonl`

## Overall

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| Basic pass | 13/13 | 13/13 | +0 |
| Pass rate | 100.00% | 100.00% | 0 |
| Dtype | bf16 | bf16 | |
| max_new_tokens | 96 | 160 | |

## By Type

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A avg repeat | B avg repeat | A avg sec | B avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 4/4 | 4/4 | 181.0 | 213.8 | 74.8 | 128.8 | 0.0000 | 0.0100 | 0.681 | 1.037 |
| image | 2/2 | 2/2 | 275.5 | 446.0 | 88.0 | 152.0 | 0.0000 | 0.0000 | 0.655 | 1.085 |
| text | 7/7 | 7/7 | 143.1 | 173.3 | 79.7 | 101.3 | 0.0179 | 0.0085 | 0.682 | 0.829 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_coffee | True | True | 274 | 374 | 0.0 | 0.0 |  |
| a2a_en_food | True | True | 38 | 94 | 0.0 | 0.0 |  |
| a2a_en_sky_blue | True | True | 346 | 347 | 0.0 | 0.04 |  |
| a2a_zh_sky_blue_probe | True | True | 66 | 40 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 280 | 452 | 0.0 | 0.0 |  |
| i2a_fruit_untrained_probe | True | True | 271 | 440 | 0.0 | 0.0 |  |
| t2a_en_coffee | True | True | 290 | 318 | 0.0 | 0.0 |  |
| t2a_en_health | True | True | 297 | 498 | 0.0 | 0.0597 |  |
| t2a_en_intro | True | True | 99 | 99 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 156 | 175 | 0.125 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 59 | 30 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 35 | 27 | 0.0 | 0.0 |  |
