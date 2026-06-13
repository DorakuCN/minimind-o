# sft_full_muon_v2_himem_b88 Evaluation · 2026-06-11

## Scope

This report evaluates the completed high-memory Run B3 variant:

```text
out/sft_full_muon_v2_himem_b88_final_768.pth
```

This is not the strict schedule-only B3 run, because Stage 1 T2A used `batch_size=88` instead of the baseline `48`. Treat it as `B3_himem_b88` in comparisons.

## Artifacts

| Artifact | Path |
| --- | --- |
| Final model | `out/sft_full_muon_v2_himem_b88_final_768.pth` |
| Final checkpoint | `checkpoints/sft_full_muon_v2_himem_b88_final_768.pth` |
| Training logs | `.run_logs/full_train_sft_full_muon_v2_B3_himem_b88_resume_3gpu_20260611_045038_B3_himem_b88_resume_s1/` |
| 96-token eval | `eval_results/sft_full_muon_v2_himem_b88_final_batch_audio_bf16/` |
| 160-token eval | `eval_results/sft_full_muon_v2_himem_b88_final_batch_audio_bf16_t160/` |
| 96-token comparison | `docs/evaluation_results/compare_muon_v2_himem_b88_vs_baseline.md` |
| 160-token comparison | `docs/evaluation_results/compare_muon_v2_himem_b88_t160_vs_baseline_t160.md` |
| Manual review template | `docs/evaluation_results/review_muon_v2_himem_b88_vs_baseline.md` |
| Manual review template, 160 | `docs/evaluation_results/review_muon_v2_himem_b88_t160_vs_baseline_t160.md` |

## Automatic L0 Results

| Run | dtype | max_new_tokens | basic pass | pass rate |
| --- | --- | ---: | ---: | ---: |
| Baseline A | bf16 | 96 | 13/13 | 100.00% |
| B3_himem_b88 | bf16 | 96 | 13/13 | 100.00% |
| Baseline A | bf16 | 160 | 13/13 | 100.00% |
| B3_himem_b88 | bf16 | 160 | 13/13 | 100.00% |

The L0 smoke check passes structurally, including decoded audio generation for all 13 cases.

## Comparison Against Baseline A

### 96-token eval

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A repeat | B repeat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 4/4 | 4/4 | 181.0 | 225.2 | 74.8 | 88.0 | 0.0000 | 0.0000 |
| image | 2/2 | 2/2 | 275.5 | 278.5 | 88.0 | 88.0 | 0.0000 | 0.0000 |
| text | 7/7 | 7/7 | 143.1 | 146.0 | 79.7 | 83.6 | 0.0179 | 0.0000 |

Notable 96-token differences:

- `a2a_en_food` became much longer: 38 chars in A, 258 chars in B.
- `t2a_en_space` repeat score improved from 0.1250 to 0.0000.
- All cases still pass the basic text/audio structure gate.

### 160-token eval

| Type | A pass | B pass | A avg chars | B avg chars | A avg frames | B avg frames | A repeat | B repeat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 4/4 | 4/4 | 213.8 | 202.2 | 128.8 | 127.8 | 0.0100 | 0.0063 |
| image | 2/2 | 2/2 | 446.0 | 511.0 | 152.0 | 152.0 | 0.0000 | 0.0000 |
| text | 7/7 | 7/7 | 173.3 | 191.0 | 101.3 | 120.1 | 0.0085 | 0.0130 |

Notable 160-token differences:

- Image cases produce longer descriptions in B: 511.0 avg chars vs 446.0 in A.
- Audio cases are roughly similar in frame count and repeat behavior.
- Some long text cases still contain semantic or procedural hallucinations despite passing L0.

## ASR Round-Trip

ASR now uses SenseVoiceSmall through `funasr`, with model weights `model/SenseVoiceSmall` and source checkout reference `/home/genesis/Projects/SenseVoice`. The run uses CPU batch inference (`--device cpu --batch_size 16`), because this small eval workload underutilizes GPU while CPU batch is fast enough for L0/L1. These numbers are a rough audio intelligibility proxy, not a semantic correctness metric.

| Eval | avg CER | avg WER | text CER/WER | audio CER/WER | image CER/WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline A 96 | 0.3826 | 0.5821 | 0.2824 / 0.5571 | 0.4424 / 0.5755 | 0.6138 / 0.6830 |
| B3_himem_b88 96 | 0.4333 | 0.6457 | 0.3064 / 0.5779 | 0.5448 / 0.7379 | 0.6542 / 0.6990 |
| Baseline A 160 | 0.2453 | 0.3664 | 0.1742 / 0.3646 | 0.2094 / 0.2349 | 0.5662 / 0.6352 |
| B3_himem_b88 160 | 0.2679 | 0.5019 | 0.1863 / 0.4981 | 0.2449 / 0.4517 | 0.5997 / 0.6157 |

Baseline ASR artifacts: `eval_results/sft_full_muon_final_batch_audio_bf16{,_t160}/asr_eval.{json,md}`.

Reading:

- At 96 tokens, B is **worse** than A on every type (avg CER 0.433 vs 0.383; avg WER 0.646 vs 0.582).
- At 160 tokens, A is still better overall (avg CER 0.245 vs 0.268; avg WER 0.366 vs 0.502), especially on audio-input WER (0.235 vs 0.452). B only has a small image-WER edge, while image CER is worse.
- Net: SenseVoice CPU-batch ASR strengthens the "do not promote B" decision. The audio proxy now favors baseline A, while the semantic win/tie/loss review remains tied.

## Win/Tie/Loss Review (AI-assisted, 2026-06-11)

Both review templates were filled with AI-assisted judgments (text judged from full outputs, I2A grounding judged against the source images, audio column from an ASR-CER proxy). Human listening spot-checks are still recommended before any promotion decision.

| Eval | A wins | B wins | Tie |
| --- | ---: | ---: | ---: |
| 96 tokens | 4 | 4 | 5 |
| 160 tokens | 5 | 5 | 3 |

Where B is better:

- Chinese probes: `t2a_zh_intro_probe` (coherent self-description vs A's broken phrasing), `a2a_zh_sky_blue_probe` at 96, `t2a_zh_spring_probe` at 96 (A contains the corruption "和机勃勃").
- Refusal/helpfulness style: `a2a_en_food` is a clean refusal plus an offer in B at both lengths.
- Fruit-image grounding at 160: B correctly describes fruits on a white surface and identifies bananas (grounding 2 vs A's 1).

Where B is worse:

- `t2a_en_intro` is a hard regression: B emits the same off-prompt narrative sentence at both 96 and 160 tokens, while A answers the prompt perfectly. This looks like a memorized/collapsed continuation, not a sampling accident.
- Physical/causal explanations: B's sky-blue answers invent wrong mechanisms ("equatorial tilt", "opposite effect") at both lengths; A's are closer to actual scattering.
- Procedural answers: B repeatedly freezes or cools coffee (`t2a_en_coffee`, `a2a_en_coffee`).
- Cat-image grounding at 96: B misidentifies the cat as a dog (grounding 0).

Verdict: the longer I2T schedule (B1 component) plausibly helped image grounding at 160 tokens, but overall semantic quality is a wash and there is at least one clear instruction-following regression.

## L1 Expanded Evaluation (2026-06-11)

The L0 tie motivated an expanded L1 set: `dataset/eval_muon_l1.jsonl`, 57 cases (the 13 L0 cases plus 44 new ones built from previously unused `dataset/eval_omni` assets: 16 new text prompts including instruction-following probes, 19 new A2A audio prompts, 9 new I2A cases over 7 untrained images). Manifest frozen at `docs/evaluation_results/eval_muon_l1_manifest_20260611.{json,md}` with 0 missing files.

Both checkpoints were evaluated at bf16 / 160 tokens / seed 42. The 13 L0 cases reproduced byte-identical outputs to the mini-set runs, confirming deterministic per-case seeding.

| Artifact | Path |
| --- | --- |
| A run | `eval_results/sft_full_muon_final_l1_bf16_t160/` |
| B run | `eval_results/sft_full_muon_v2_himem_b88_final_l1_bf16_t160/` |
| Comparison | `docs/evaluation_results/compare_muon_v2_himem_b88_l1_t160_vs_baseline_l1_t160.md` |
| Review (scored) | `docs/evaluation_results/review_muon_v2_himem_b88_l1_t160_vs_baseline_l1_t160.md` |

Results:

- **Basic pass**: 57/57 for both. The structural L0 gate stays saturated even at 57 cases.
- **ASR round-trip** (SenseVoiceSmall CPU batch, `--device cpu --batch_size 16`): A avg CER 0.3497 / WER 0.5459 vs B 0.3532 / 0.5805 — near parity with a slight A lead overall. B is only better on text CER (0.2302 vs A 0.2412); A is better on audio and image ASR proxies.
- **Win/Tie/Loss (AI-assisted)**: **A 20 / B 16 / Tie 21**. By type: text 7/6/10, audio **10/6/7**, image 3/4/4. On the larger set baseline A pulls slightly ahead, driven by audio-input cases.

New findings beyond L0:

- A is systematically better at physics/factual explanations in both languages (sky-blue, black holes, quantum, snow); B invents wrong mechanisms more often.
- B is better on several Chinese instruction probes (`t2a_zh_who_probe`: A fails to answer entirely) and refusal style.
- B has a list-repetition pathology (`a2a_zh_ai_fields_probe` repeats 计算机视觉/自动驾驶 across a 20-item list).
- A leaks Chinese tokens into English output (`a2a_en_health`: "任务").
- Neither model can count from one to five in either language.
- I2A grounding on the 7 new untrained images is mostly 0-1 for both; B's L0 fruit-image advantage does not generalize into a consistent grounding win (image: A3/B4/T4).

## Qualitative Caveats

The model passes the L0 structural checks, but semantic correctness remains mixed:

- `t2a_en_intro` is off-prompt in both 96 and 160 outputs.
- `t2a_en_coffee` still invents odd procedures such as freezing coffee.
- `i2a_cat_untrained_probe` and `i2a_fruit_untrained_probe` still show visual hallucination risk.
- I2T/I2A improvements should therefore be judged by the manual review templates, not by 13/13 pass alone.

## Conclusion

`B3_himem_b88` completed successfully and preserves L0 pass parity with baseline A, but neither the now-complete A-vs-B ASR comparison nor the win/tie/loss review shows a clear overall win: SenseVoice CPU-batch ASR is worse for B at both 96 and 160 tokens, and the review verdict is tied at both lengths (4-4-5 and 5-5-3). The strongest positive signal is improved I2A grounding at 160 tokens (consistent with the B1 I2T schedule extension); the strongest negative signal is the `t2a_en_intro` off-prompt collapse, which baseline A does not have.

Decision (updated 2026-06-11 after L1): **do not promote B3_himem_b88; baseline A remains the comparison anchor.** The 57-case L1 evaluation confirms and strengthens the L0 verdict: A leads 20-16 with 21 ties, SenseVoice ASR is near parity with a slight A lead overall, and B's only clear advantages (Chinese instruction style, some image cases) are offset by worse factual grounding and a list-repetition pathology.

Next steps:

1. ~~Expand to an L1 set~~ — done 2026-06-11 (`dataset/eval_muon_l1.jsonl`, 57 cases; see the L1 section above).
2. Skip the B1/B2 split: with A ahead on L1, isolating B3's components is no longer worth two full training runs. The B1 I2T-schedule idea (image grounding) can be folded into the next candidate instead.
3. Proceed to **Run C (v3)**: warmup, global loss norm, RVQ layer weights, fp32 resume, val monitoring — compared against **baseline A** (not B3) on the L1 set at bf16/t160/seed 42.
4. The L1 review surfaced concrete data gaps worth addressing in future SFT data: counting/enumeration instructions, factual physics QA, and list-formatting without repetition.
5. Human listening spot-checks on a few MP3s are still pending; SenseVoice CER/WER is a better aligned proxy than the previous whisper-tiny check, but still not a substitute for listening.
