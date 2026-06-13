# Eval Comparison: baseline_A vs B3_himem_b88

- Generated: 2026-06-11 15:53:32
- Run A: `out/sft_full_muon_final_768.pth`
- Run B: `out/sft_full_muon_v2_himem_b88_final_768.pth`
- Test set: `dataset/eval_muon_mini.jsonl`

## Overall

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| Basic pass | 13/13 | 13/13 | +0 |
| Pass rate | 100.00% | 100.00% | 0 |
| Dtype | bf16 | bf16 | |
| max_new_tokens | 96 | 96 | |

## By Type

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A avg repeat | B avg repeat | A avg sec | B avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 4/4 | 4/4 | 181.0 | 225.2 | 74.8 | 88.0 | 0.0000 | 0.0000 | 0.681 | 0.768 |
| image | 2/2 | 2/2 | 275.5 | 278.5 | 88.0 | 88.0 | 0.0000 | 0.0000 | 0.655 | 0.663 |
| text | 7/7 | 7/7 | 143.1 | 146.0 | 79.7 | 83.6 | 0.0179 | 0.0000 | 0.682 | 0.740 |

## ASR Round-Trip (updated 2026-06-11)

SenseVoiceSmall through `funasr`, CPU batch inference (`--device cpu --batch_size 16`); CER/WER between generated text and ASR transcript of generated audio.

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| avg CER | 0.3826 | 0.4333 | +0.0507 |
| avg WER | 0.5821 | 0.6457 | +0.0636 |
| text CER | 0.2824 | 0.3064 | +0.0240 |
| text WER | 0.5571 | 0.5779 | +0.0208 |
| audio CER | 0.4424 | 0.5448 | +0.1024 |
| audio WER | 0.5755 | 0.7379 | +0.1624 |
| image CER | 0.6138 | 0.6542 | +0.0404 |
| image WER | 0.6830 | 0.6990 | +0.0160 |

B is worse than baseline A on every ASR proxy at 96 tokens.

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_coffee | True | True | 274 | 262 | 0.0 | 0.0 |  |
| a2a_en_food | True | True | 38 | 258 | 0.0 | 0.0 |  |
| a2a_en_sky_blue | True | True | 346 | 323 | 0.0 | 0.0 |  |
| a2a_zh_sky_blue_probe | True | True | 66 | 58 | 0.0 | 0.0 |  |
| i2a_cat_untrained_probe | True | True | 280 | 266 | 0.0 | 0.0 |  |
| i2a_fruit_untrained_probe | True | True | 271 | 291 | 0.0 | 0.0 |  |
| t2a_en_coffee | True | True | 290 | 263 | 0.0 | 0.0 |  |
| t2a_en_health | True | True | 297 | 309 | 0.0 | 0.0 |  |
| t2a_en_intro | True | True | 99 | 141 | 0.0 | 0.0 |  |
| t2a_en_joke | True | True | 66 | 76 | 0.0 | 0.0 |  |
| t2a_en_space | True | True | 156 | 165 | 0.125 | 0.0 |  |
| t2a_zh_intro_probe | True | True | 59 | 36 | 0.0 | 0.0 |  |
| t2a_zh_spring_probe | True | True | 35 | 32 | 0.0 | 0.0 |  |
