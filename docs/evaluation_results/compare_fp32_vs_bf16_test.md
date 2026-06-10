# Eval Comparison: fp32 vs bf16

- Generated: 2026-06-10 15:41:34
- Run A: `out/sft_full_muon_final_768.pth`
- Run B: `out/sft_full_muon_final_768.pth`
- Test set: `dataset/eval_muon_mini.jsonl`

## Overall

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| Basic pass | 13/13 | 13/13 | +0 |
| Pass rate | 100.00% | 100.00% | 0 |
| Dtype | fp32 | bf16 | |
| max_new_tokens | 96 | 96 | |

## By Type

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A avg repeat | B avg repeat | A avg sec | B avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 4/4 | 4/4 | 176.0 | 181.0 | 79.8 | 74.8 | 0.0047 | 0.0000 | 0.687 | 0.681 |
| image | 2/2 | 2/2 | 274.0 | 275.5 | 88.0 | 88.0 | 0.0000 | 0.0000 | 0.615 | 0.655 |
| text | 7/7 | 7/7 | 118.6 | 143.1 | 72.7 | 79.7 | 0.0000 | 0.0179 | 0.603 | 0.682 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_coffee | True | True | 270 | 274 | 0.0 | 0.0 |  |
| a2a_en_food | True | True | 67 | 38 | 0.0 | 0.0 |  |
| a2a_en_sky_blue | True | True | 328 | 346 | 0.0189 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 39 | 66 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 294 | 280 | 0.0 | 0.0 |  |
| i2a_fruit_untrained_probe | True | True | 254 | 271 | 0.0 | 0.0 |  |
| t2a_en_coffee | True | True | 272 | 290 | 0.0 | 0.0 |  |
| t2a_en_health | True | True | 280 | 297 | 0.0 | 0.0 |  |
| t2a_en_intro | True | True | 99 | 99 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 50 | 156 | 0.0 | 0.125 |  |
| t2a_zh_intro_probe | True | True | 34 | 59 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 29 | 35 | 0.0 | 0.0 |  |
