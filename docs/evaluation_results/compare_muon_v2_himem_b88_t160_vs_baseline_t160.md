# Eval Comparison: baseline_A_t160 vs B3_himem_b88_t160

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
| max_new_tokens | 160 | 160 | |

## By Type

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A avg repeat | B avg repeat | A avg sec | B avg sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 4/4 | 4/4 | 213.8 | 202.2 | 128.8 | 127.8 | 0.0100 | 0.0063 | 1.037 | 1.011 |
| image | 2/2 | 2/2 | 446.0 | 511.0 | 152.0 | 152.0 | 0.0000 | 0.0000 | 1.085 | 1.078 |
| text | 7/7 | 7/7 | 173.3 | 191.0 | 101.3 | 120.1 | 0.0085 | 0.0130 | 0.829 | 0.955 |

## Per-Case

| id | A pass | B pass | A chars | B chars | A repeat | B repeat | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a2a_en_coffee | True | True | 374 | 414 | 0.0 | 0.025 | A wins review: B mixes milk/cooling-glass steps incoherently |
| a2a_en_food | True | True | 94 | 75 | 0.0 | 0.0 | B wins review: cleaner refusal; A repeats "I don't eat" |
| a2a_en_sky_blue | True | True | 347 | 242 | 0.04 | 0.0 | A wins review: B invents "equatorial tilt" mechanism |
| a2a_zh_sky_blue_probe | True | True | 40 | 78 | 0.0 | 0.0 | A wins review: A concise and correct; B self-contradicts |
| i2a_cat_untrained_probe | True | True | 452 | 533 | 0.0 | 0.0 | Tie: A right color wrong setting; B right setting wrong color |
| i2a_fruit_untrained_probe | True | True | 440 | 489 | 0.0 | 0.0 | B wins review: grounding 2 vs 1 (bananas + white surface correct) |
| t2a_en_coffee | True | True | 318 | 344 | 0.0 | 0.0615 | A wins review: B repeatedly freezes the coffee |
| t2a_en_health | True | True | 498 | 487 | 0.0597 | 0.0 | B wins review: more coherent habit list |
| t2a_en_intro | True | True | 99 | 141 | 0.0 | 0.0 | A wins review: B fully off-prompt (same narrative at 96 and 160) |
| t2a_en_joke | True | True | 66 | 66 | 0.0 | 0.0 | Tie: identical output |
| t2a_en_space | True | True | 175 | 209 | 0.0 | 0.0294 | B wins review: on-topic Big Bang fact; A incoherent + mojibake |
| t2a_zh_intro_probe | True | True | 30 | 55 | 0.0 | 0.0 | B wins review: coherent self-description |
| t2a_zh_spring_probe | True | True | 27 | 35 | 0.0 | 0.0 | Tie: both acceptable |

## ASR Round-Trip (updated 2026-06-11)

SenseVoiceSmall through `funasr`, CPU batch inference (`--device cpu --batch_size 16`); CER/WER between generated text and ASR transcript of generated audio.

| Metric | A | B | Delta (B-A) |
| --- | ---: | ---: | ---: |
| avg CER | 0.2453 | 0.2679 | +0.0226 |
| avg WER | 0.3664 | 0.5019 | +0.1355 |
| text CER | 0.1742 | 0.1863 | +0.0121 |
| text WER | 0.3646 | 0.4981 | +0.1335 |
| audio CER | 0.2094 | 0.2449 | +0.0355 |
| audio WER | 0.2349 | 0.4517 | +0.2168 |
| image CER | 0.5662 | 0.5997 | +0.0335 |
| image WER | 0.6352 | 0.6157 | -0.0195 |

At 160 tokens, B only has a small image-WER edge; overall ASR and audio-input ASR favor baseline A.

## Review Verdict (added 2026-06-11)

AI-assisted win/tie/loss on this 160-token pair: **A 5 / B 5 / Tie 3** — no clear winner. Details in `docs/evaluation_results/review_muon_v2_himem_b88_t160_vs_baseline_t160.md`.
