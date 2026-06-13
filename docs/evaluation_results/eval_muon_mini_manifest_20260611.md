# Eval Manifest Snapshot

- Generated: 2026-06-10 16:12:53
- Test set: `dataset/eval_muon_mini.jsonl`
- Test set sha256: `7ff2200345dc4f6cd4978a0b41844be52ff05bfb3ce34211331ce95e3b795b11`
- Cases: 13
- Referenced files: 6
- Missing files: 0

## Files

| path | exists | bytes | sha256 |
| --- | ---: | ---: | --- |
| `dataset/eval_omni/audio-en-01_what_do_you_usually_like_to_eat.mp3` | True | 5924 | `963e9b3a3832360c8f828435914c1ebb368aa1061ea2f86cafd11a93aab33a6a` |
| `dataset/eval_omni/audio-en-05_why_is_the_sky_blue.mp3` | True | 5420 | `b2ac22cdfaf81bd65dd6c7055c139d3ae519cf176c7ca2f2d89dd90899d2787f` |
| `dataset/eval_omni/audio-en-12_how_do_i_make_a_good_cup_of_coffee.mp3` | True | 5924 | `1b58b3276281352d4d258ec91bda35d8de224bf61a79124e270f2ca37087cde4` |
| `dataset/eval_omni/audio-zh-05_为什么天空是蓝色的.mp3` | True | 4268 | `6bc9b6ad95892b9fc1040d905eb2e3c0cf79397a4da7d06c314fa1c33a61b1e7` |
| `dataset/eval_omni/image-01-orange-cat-moon-desk.jpg` | True | 17712 | `fd2898ecc7a73cbf16d9c6f3ded1c59f176e6e8868a4c7699f814d6aba18f19a` |
| `dataset/eval_omni/image-02-fruit-basket-apples-bananas-oranges.jpg` | True | 14376 | `b9d11538c3c404aaee7104b0ac8c6de73d14f834e52ad7efe0c34d61928d1359` |

## Cases

| line | id | type | tags | prompt/media |
| ---: | --- | --- | --- | --- |
| 1 | `t2a_en_intro` | text | t2a,english,in_domain | Please introduce yourself in one short sentence. |
| 2 | `t2a_en_joke` | text | t2a,english,in_domain | Tell me a short joke. |
| 3 | `t2a_en_space` | text | t2a,english,in_domain | Tell me one interesting fact about space. |
| 4 | `t2a_en_coffee` | text | t2a,english,in_domain | How do I make a good cup of coffee? |
| 5 | `t2a_en_health` | text | t2a,english,in_domain | How can I maintain a healthy lifestyle? |
| 6 | `t2a_zh_intro_probe` | text | t2a,chinese,out_of_domain_probe | 请用一句话介绍你自己。 |
| 7 | `t2a_zh_spring_probe` | text | t2a,chinese,out_of_domain_probe | 请用一句话介绍一下春天。 |
| 8 | `a2a_en_food` | audio | a2a,english,in_domain | dataset/eval_omni/audio-en-01_what_do_you_usually_like_to_eat.mp3 |
| 9 | `a2a_en_sky_blue` | audio | a2a,english,in_domain | dataset/eval_omni/audio-en-05_why_is_the_sky_blue.mp3 |
| 10 | `a2a_en_coffee` | audio | a2a,english,in_domain | dataset/eval_omni/audio-en-12_how_do_i_make_a_good_cup_of_coffee.mp3 |
| 11 | `a2a_zh_sky_blue_probe` | audio | a2a,chinese,out_of_domain_probe | dataset/eval_omni/audio-zh-05_为什么天空是蓝色的.mp3 |
| 12 | `i2a_cat_untrained_probe` | image | i2a,english,untrained_probe | dataset/eval_omni/image-01-orange-cat-moon-desk.jpg |
| 13 | `i2a_fruit_untrained_probe` | image | i2a,english,untrained_probe | dataset/eval_omni/image-02-fruit-basket-apples-bananas-oranges.jpg |
