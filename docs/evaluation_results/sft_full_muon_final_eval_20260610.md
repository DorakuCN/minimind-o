# MiniMind-O Full Muon Final Evaluation - 2026-06-10

## Evaluation Setup

- Checkpoint: `out/sft_full_muon_final_768.pth`
- Test set: `dataset/eval_muon_mini.jsonl`
- Cases: 13 total
  - T2A text prompt: 7
  - A2A audio prompt: 4
  - I2A/image prompt: 2
- Generation: `max_new_tokens=96`, `temperature=0.7`, `top_p=0.85`, `seed=42`
- Audio decode: enabled, Mimi decoded to 24 kHz mono MP3
- Primary eval dtype: `fp32`
- Additional dtype check: `bf16` pass, `fp16` failed

Raw outputs:

```text
eval_results/sft_full_muon_final_batch_audio_fp32/
eval_results/sft_full_muon_final_batch_audio_bf16/
.run_logs/eval/sft_full_muon_final_batch_audio_fp32.log
.run_logs/eval/sft_full_muon_final_batch_audio_bf16.log
```

## Key Result

The final full-train checkpoint can complete the fixed 13-case Omni validation set in fp32 and bf16. It generates non-empty text and decodable audio for every case, with zero observed special audio-code pollution.

However, default fp16 inference is numerically unstable for this checkpoint: the first T2A prompt produced non-finite logits/probabilities and crashed at `torch.multinomial`. Weight scanning showed no NaN/Inf in the checkpoint, and the same first-step logits are finite in fp32. Deployment/evaluation should therefore use `bf16` or `fp32` until fp16 inference is fixed.

## Automatic Metrics

| Run | Basic Pass | Avg Chars | Avg Audio Frames | Avg Repeat | Avg Special Code Rate | Avg Seconds/Case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| miniTrain baseline | 13/13 | 188.5 | 76.8 | 0.0401 | 0.0000 | n/a |
| full final fp32 | 13/13 | 160.2 | 77.2 | 0.0015 | 0.0000 | 0.631 |
| full final bf16 | 13/13 | 175.2 | 79.5 | 0.0096 | 0.0000 | 0.678 |

By type, fp32:

| Type | Cases | Basic Pass | Avg Chars | Avg Audio Frames | Avg Repeat | Avg Special |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text/T2A | 7 | 7/7 | 118.6 | 72.7 | 0.0000 | 0.0000 |
| audio/A2A | 4 | 4/4 | 176.0 | 79.8 | 0.0047 | 0.0000 |
| image/I2A | 2 | 2/2 | 274.0 | 88.0 | 0.0000 | 0.0000 |

## FP32 Case Results

| id | type | pass | chars | frames | repeat | special | qualitative note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `t2a_en_intro` | text | True | 99 | 70 | 0.0000 | 0.0000 | Coherent short self-introduction. |
| `t2a_en_joke` | text | True | 66 | 46 | 0.0000 | 0.0000 | Good short joke. |
| `t2a_en_space` | text | True | 50 | 43 | 0.0000 | 0.0000 | Fluent but awkward/factually weak. |
| `t2a_en_coffee` | text | True | 272 | 88 | 0.0000 | 0.0000 | On topic, but recipe logic is partially wrong and truncated. |
| `t2a_en_health` | text | True | 280 | 88 | 0.0000 | 0.0000 | Mostly sensible health advice, truncated by token limit. |
| `t2a_zh_intro_probe` | text | True | 34 | 88 | 0.0000 | 0.0000 | Chinese output is stable and on topic. |
| `t2a_zh_spring_probe` | text | True | 29 | 86 | 0.0000 | 0.0000 | Chinese output is stable and on topic. |
| `a2a_en_food` | audio | True | 67 | 55 | 0.0000 | 0.0000 | Understands question and gives valid answer. |
| `a2a_en_sky_blue` | audio | True | 328 | 88 | 0.0189 | 0.0000 | On topic, but explanation contains scientific inaccuracies. |
| `a2a_en_coffee` | audio | True | 270 | 88 | 0.0000 | 0.0000 | Recognizes coffee prompt, but instruction quality is weak. |
| `a2a_zh_sky_blue_probe` | audio | True | 39 | 88 | 0.0000 | 0.0000 | Chinese A2A works and answer is good. |
| `i2a_cat_untrained_probe` | image | True | 294 | 88 | 0.0000 | 0.0000 | Recognizes a cat, but hallucinates color/outfit/details. |
| `i2a_fruit_untrained_probe` | image | True | 254 | 88 | 0.0000 | 0.0000 | Recognizes fruit-like content, but misses apples/bananas/oranges and invents details. |

## Audio File Check

All 13 fp32 MP3 files were generated and can be decoded as 24 kHz mono audio. Durations range from 3.44s to 7.04s. The generated Mimi code stream had `special_code_rate=0.0` for every case.

Audio directory:

```text
eval_results/sft_full_muon_final_batch_audio_fp32/audio/
```

## Comparison Against miniTrain

Observable improvements over the miniTrain validation output:

- Chinese T2A/A2A probes improved strongly. The mini baseline often answered Chinese prompts in English or off-topic; the full model answers the Chinese probes in Chinese and mostly on topic.
- Repetition improved. Average repeat score dropped from 0.0401 to 0.0015 in fp32.
- I2A path is now active. The mini baseline had no meaningful I2T/I2A training; the full model now produces long image-conditioned answers and audio frames.
- A2A is more usable on the Chinese sky-blue probe and remains functional on English audio prompts.

Remaining weaknesses:

- I2A semantic grounding is still weak. It often recognizes the rough object class but hallucinates details.
- Longer English answers can be fluent but logically poor, especially procedural answers such as coffee-making.
- Some outputs hit the 96-token cap, so longer responses may be truncated.
- Audio was decoded and structurally valid, but this report does not include ASR transcription or human listening scores.
- fp16 inference is currently unsafe for this checkpoint.

## Final Assessment

The full-train model is materially better than the miniTrain baseline as an Omni checkpoint: T2A, A2A, and I2A all run end-to-end; Chinese prompt handling improved; audio-code pollution was not observed; and repetition dropped sharply.

The checkpoint is suitable as the current full-train baseline for the next optimizer/model comparison. For fair future comparisons, use the same `dataset/eval_muon_mini.jsonl`, run both `bf16` and fp32 sanity checks, and add a stronger evaluation layer with ASR transcription plus manual listening/image-grounding review.
