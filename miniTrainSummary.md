# Mini 数据集 Muon 训练总结与 Full Train 长跑方案

本文档总结本轮 `sft_zero_muon` mini 数据集训练结果，并给出后续长时间 full_train 的执行建议。当前结论基于 2026-06-09 在本机 `conda env minimind-o`、3 张 RTX 5090D 上完成的训练和批量验证。

## 1. 本轮训练目标

本轮目标不是训练可发布模型，而是验证以下工程链路：

- `train_sft_omni.py` 从随机初始化训练 MiniMind-O。
- 优化器由 AdamW 改为 `MuonWithAuxAdam`。
- 3 卡 DDP 训练稳定性。
- checkpoint / resume / `torch.compile` 兼容性。
- 最终权重能批量推理，能生成文本和 Mimi audio codes。

结论：工程链路跑通，最终权重可加载、可批量生成、可解码音频；模型质量仍是 mini baseline，不应和 full 数据发布权重比较。

## 2. 环境与产物

### 2.1 运行环境

- Conda 环境：`minimind-o`
- PyTorch：`2.13.0a0+git1d81f67`
- CUDA：`13.3`
- GPU：3 x NVIDIA GeForce RTX 5090 D
- `torch.optim.Muon`：环境中可用
- 本项目实际使用：本地实现的 `MuonWithAuxAdam`，位置为 `trainer/optimizers.py`

### 2.2 关键产物

| 类型 | 路径 | 说明 |
|---|---|---|
| 最终推理权重 | `out/sft_zero_muon_768.pth` | 推理和继续训练可用 |
| checkpoint 权重 | `checkpoints/sft_zero_muon_768.pth` | `omni_checkpoint` 保存的模型权重 |
| resume checkpoint | `checkpoints/sft_zero_muon_768_resume.pth` | 包含 optimizer / scaler / epoch / step |
| 成功训练日志 | `.run_logs/train_muon_scratch_mini_3gpu_resume_fixed.log` | 三阶段完整成功日志 |
| mini 批量测试集 | `dataset/eval_muon_mini.jsonl` | 13 条固定验证样本 |
| 快速验证报告 | `eval_results/sft_zero_muon_batch_quick/summary.md` | 不解码音频 |
| 音频验证报告 | `eval_results/sft_zero_muon_batch_audio/summary.md` | 生成 mp3 |
| 生成音频 | `eval_results/sft_zero_muon_batch_audio/audio/` | 13 条 mp3 |

## 3. Mini 训练配置

训练脚本：

```bash
.run_logs/run_muon_scratch_mini.sh
```

### 3.1 分阶段参数

| 阶段 | 数据 | mode | GPU 数 | batch / GPU | global batch | max_seq_len | accum | compile | steps | lr | muon_lr |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Stage 1 | `sft_t2a_mini` | `all` | 3 | 40 | 120 | 512 | 1 | on | 4296 | `5e-4` | `0.02` |
| Stage 2 | `sft_a2a_mini` | `audio_proj` | 3 | 40 | 120 | 640 | 1 | off | 640 | `5e-4` | `0.02` |
| Stage 3 | `sft_a2a_mini` | `all` | 3 | 16 | 48 | 768 | 1 | off | 1600 | `2e-5` | `0.02` |

说明：

- `batch_size` 是每个 DDP rank / 每张 GPU 的 batch。
- `global batch = batch_size * world_size * accumulation_steps`。
- 本轮 `accumulation_steps=1`。
- `use_moe=0`，Dense 模型。
- `hidden_size=768`，`num_hidden_layers=8`。
- 模型参数约 `113.13M`。

### 3.2 优化器分组

| 阶段 | 可训练参数 | Muon 参数 | AdamW fallback 参数 |
|---|---:|---:|---:|
| Stage 1 | 113.13M | 91.96M | 21.17M |
| Stage 2 | 0.99M | 0.98M | 约 0M |
| Stage 3 | 113.13M | 91.96M | 21.17M |

分组规则：

- 2D hidden matrix 使用 Muon。
- embedding / lm_head / norm / bias / 非矩阵参数使用 AdamW fallback。
- 当前训练日志会打印 `torch.optim.Muon=available`，但实际优化器是项目内 `MuonWithAuxAdam`。

### 3.3 GPU 资源使用

以下为训练期间使用 `nvidia-smi` 抽样观察到的值，不是连续 telemetry。

| 阶段 | GPU mem / card | GPU util | 备注 |
|---|---:|---:|---|
| Stage 1 T2A | 约 16.6GB | 96% - 99% | `torch.compile=on`，启动阶段有 compile worker |
| Stage 2 A2A audio_proj | 约 19.3GB | 92% - 98% | A2A 输入含 SenseVoice 音频特征，显存最高 |
| Stage 3 A2A full | 约 15.1GB | 87% - 100% | batch / GPU 降到 16 |
| 训练结束 | 约 15MB | 0% | GPU 已释放 |

观察：

- Stage 2 虽然只训练 `audio_proj`，但显存最高，主要来自 A2A 音频输入、SenseVoice 特征和较长序列。
- Stage 3 是全量训练，但 batch / GPU 从 40 降到 16，因此显存低于 Stage 2。
- 三卡利用率整体健康，未出现 OOM。

## 4. Mini 训练结果

### 4.1 Loss 统计

统计来自日志采样点，不是每个 step 的完整均值。

| 阶段 | steps | first loss | final loss | min logged | text first -> final | audio first -> final | 评价 |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage 1 T2A | 4296 | 9.8708 | 6.2037 | 5.7443 | 3.8099 -> 1.5858 | 6.0609 -> 4.6180 | 收敛明显 |
| Stage 2 A2A audio_proj | 640 | 6.8535 | 6.7538 | 6.5560 | 2.1756 -> 2.0223 | 4.6780 -> 4.7315 | 基本持平 |
| Stage 3 A2A full | 1600 | 6.7999 | 5.9750 | 5.3069 | 2.3895 -> 1.7742 | 4.4105 -> 4.2007 | 有效下降 |

解读：

- Stage 1 是主要收敛阶段，T2A 链路成功建立，文本和音频 loss 都明显下降。
- Stage 2 只训练约 0.99M 的 audio projection，loss 降幅小是合理现象。
- Stage 3 能继续降低 A2A loss，但最终 audio loss 仍在约 4.2，说明 Talker 还不够稳。

### 4.2 Resume 问题与修复

中途发现一个关键问题：`torch.compile` 后直接执行：

```python
model.load_state_dict(ckp_data["model"], strict=False)
```

会把权重加载到 compiled wrapper，`strict=False` 可能静默跳过大量真实权重键，导致 resume 后 loss 从正常 5.x 跳到 11.x。

修复方式：

```python
raw_model = getattr(model, "_orig_mod", model)
load_result = raw_model.load_state_dict(ckp_data["model"], strict=False)
```

修复后从 step 4000 续训：

- step 4100 loss 从异常的 11.9622 恢复到 5.8650。
- step 4200 loss 从异常的 10.7750 恢复到 5.7550。

建议保留该修复；它对 `torch.compile + checkpoint resume` 很关键。

## 5. Mini 批量验证结果

原始报告：

```bash
eval_results/sft_zero_muon_batch_audio/summary.md
```

测试集合：

```bash
dataset/eval_muon_mini.jsonl
```

验证命令：

```bash
conda run -n minimind-o python scripts/batch_validate_omni.py \
  --test_set dataset/eval_muon_mini.jsonl \
  --output_dir eval_results/sft_zero_muon_batch_audio \
  --weight sft_zero_muon \
  --max_new_tokens 96 \
  --device cuda:0 \
  --decode_audio
```

### 5.1 覆盖范围

| 类型 | 数量 | 目的 |
|---|---:|---|
| English T2A | 5 | mini 主能力验证 |
| Chinese T2A probe | 2 | out-of-domain 检查 |
| English A2A | 3 | 语音输入链路验证 |
| Chinese A2A probe | 1 | out-of-domain 检查 |
| Image probe | 2 | 未训练 I2T 的失败形态检查 |

### 5.2 基础通过率

| 类型 | 通过 |
|---|---:|
| text | 7 / 7 |
| audio | 4 / 4 |
| image probe | 2 / 2 |
| total | 13 / 13 |

这里的 pass 只代表：

- 能生成非空文本。
- 能生成足够数量的 audio code frames。
- 重复度没有超过阈值。
- 权重加载无 missing / unexpected keys。

它不代表语义正确、音质优秀或图像理解有效。

### 5.3 原始 batch summary 明细

以下明细同步自 `eval_results/sft_zero_muon_batch_audio/summary.md`。`special=0.0000` 表示这批样本没有观察到大量特殊 audio code 污染；`repeat` 是文本 4-gram 重复度的轻量指标。

| id | type | pass | chars | frames | repeat | special | seconds | 文本摘要 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `t2a_en_intro` | text | True | 80 | 57 | 0.0000 | 0.0000 | 1.048 | Hello! I'm your friendly voice assistant, here to make your day a little easier. |
| `t2a_en_joke` | text | True | 93 | 68 | 0.0000 | 0.0000 | 0.539 | Why did the scarecrow become a successful guitarist? Because he was outstanding in his field! |
| `t2a_en_space` | text | True | 313 | 88 | 0.1636 | 0.0000 | 0.707 | Sure, here's an interesting fact about space: 1. What is the primary purpose of the universe?... |
| `t2a_en_coffee` | text | True | 281 | 88 | 0.0385 | 0.0000 | 0.656 | To make a good cup of coffee, start by filling your coffee grounds with a pinch of salt and pepper... |
| `t2a_en_health` | text | True | 292 | 88 | 0.0930 | 0.0000 | 0.656 | Maintaining a healthy lifestyle involves several key steps... |
| `t2a_zh_intro_probe` | text | True | 99 | 77 | 0.0000 | 0.0000 | 0.601 | How about "Harry Potter and the Sorcerer's Stone"?... |
| `t2a_zh_spring_probe` | text | True | 231 | 88 | 0.1000 | 0.0000 | 0.650 | Sure! Here are some ideas for you: Take a scenic park with a picnic... |
| `a2a_en_food` | audio | True | 66 | 54 | 0.0000 | 0.0000 | 0.793 | As an AI language model, I don't eat, so I don't eat, I don't eat. |
| `a2a_en_sky_blue` | audio | True | 295 | 88 | 0.0200 | 0.0000 | 0.669 | The sky blinds is a vibrant and intense sky... |
| `a2a_en_coffee` | audio | True | 259 | 88 | 0.1064 | 0.0000 | 0.672 | To make a good cup of coffee, start by preheating your oven... |
| `a2a_zh_sky_blue_probe` | audio | True | 145 | 88 | 0.0000 | 0.0000 | 0.668 | The ocean is not abanallel is where its surface temperature is constant... |
| `i2a_cat_untrained_probe` | image | True | 253 | 88 | 0.0000 | 0.0000 | 0.674 | I am a voice assistant and cannot write code... |
| `i2a_fruit_untrained_probe` | image | True | 43 | 39 | 0.0000 | 0.0000 | 0.358 | The image is a bold and unstructured image. |

补充判断：

- 运行耗时大多在 0.5s - 0.8s，说明单卡推理吞吐没有明显异常。
- `frames=88` 的样本通常触达 `max_new_tokens=96` 下的较长回答，长回答更容易暴露幻觉和重复。
- 中文和图像 probe 虽然基础 pass，但文本内容明显偏题，应按失败/弱能力样本看待。

### 5.4 质量观察

- 英文短问答较好，如自我介绍、短笑话。
- 英文长知识问答有明显幻觉和模板化，例如 coffee / space / sky blue。
- A2A 能跑通，但语音理解和回答准确性一般。
- 中文 probe 经常输出英文或跑题，符合 mini 数据主要英文的预期。
- 图像 probe 基本无效，因为本轮没有跑 I2T 训练。
- 批量验证中 `special_code_rate=0.0`，说明没有大量特殊 token 污染音频 codes。

### 5.5 音频发抖问题判断

生成音频容易发抖，主要原因不是非法 code，而是 Talker 预测 8 层 Mimi codes 的时序稳定性不足。

可能原因：

1. mini 数据和训练时长不足。
2. Stage 3 虽然 `learning_rate=2e-5`，但 `muon_lr=0.02`，对全量 A2A 微调偏激进。
3. 推理阶段 audio code 采样较随机：固定 `audio_temperature=0.2`、`top_k=50`，且对最近 audio code 做 repetition penalty。
4. 默认无参考音色生成，声线和韵律更容易漂。

低成本改进建议：

- 增加可配置推理参数：`audio_temperature`、`audio_top_k`、`audio_repetition_penalty`。
- 推理优先试：`audio_temperature=0.08~0.12`、`audio_top_k=10~20`。
- 减弱或关闭 audio code repetition penalty。
- 限制回答长度，如 `max_new_tokens=48~64`，prompt 要求 one short sentence。
- 尝试带 `ref_codes/spk_emb` 的固定音色条件。

训练侧建议：

- Stage 1 可保留较大 `muon_lr=0.02`。
- Stage 2 / Stage 3 应显式降低 `muon_lr`，例如 `0.001~0.005`。
- 增加一个只训练 `talker + audio_proj` 的稳定阶段，冻结 Thinker。

## 6. Full Train 长时间训练方案

full_train 的目标不是只跑通链路，而是接近发布权重的训练流程。根据 README 和 `trainer/train.sh`，full 数据覆盖：

| 数据集 | 输入语音 | 输出语音 | 语言/能力 |
|---|---:|---:|---|
| `sft_t2a` | 无 | 约 1636.01 h | 中英 T2A |
| `sft_a2a` | 约 1711.97 h | 约 423.40 h | 中英 A2A |
| `sft_i2t` | 无 | 无 | 图像 I2T |

语言比例：

- `sft_t2a`：中文 45.7%，英文 46.5%，混合 7.8%。
- `sft_a2a`：中文 70.8%，英文 21.2%，混合 8.0%。

### 6.1 3 卡 Dense full_train 最优参数

本机 full_train 仍按 3 卡执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc_per_node 3
```

README / `trainer/train.sh` 中的 full 流程只作为阶段顺序和基础超参来源，本节后续所有执行参数以 3 卡为准。目标是在 32GB 显存卡上尽量提高 mem util，同时避免 full A2A 的 `max_seq_len=1024` OOM。

推荐先使用下面这组“稳定最优参数”。其中 mem / util 是根据本轮 mini 实测和 full seq_len 线性估计得到的启动前预估，必须在 full_train smoke 阶段重新记录。

| 阶段 | 数据 | mode | epochs | batch / GPU | global batch(3卡) | max_seq_len | compile | lr | muon_lr | 预计 mem / GPU | mem util(32GB) | 预计 GPU util | 3卡估时 |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `sft_t2a` | all | 6 | 40 | 120 | 512 | on | `5e-4` | `0.02` | 16-20GB | 50%-62% | 90%-99% | 4.8-5.5h |
| 2 | `sft_a2a` | audio_proj | 1 | 32 | 96 | 1024 | off | `5e-4` | `0.005` | 24-28GB | 74%-86% | 85%-99% | 30-40m |
| 3 | `sft_a2a` | all | 3 | 32 | 96 | 1024 | off | `5e-5` | `0.002` | 24-30GB | 74%-92% | 85%-99% | 1.5-2.0h |
| 4 | `sft_i2t` | vision_proj | 1 | 32 | 96 | 768 | on | `5e-5` | `0.005` | 20-25GB | 61%-77% | 85%-99% | 50-70m |
| 5 | `sft_i2t` | all | 1 | 32 | 96 | 768 | on | `5e-6` | `0.001` | 20-26GB | 61%-80% | 85%-99% | 50-70m |
| 6 | `sft_a2a` | all | 1 | 32 | 96 | 1024 | off | `5e-6` | `0.001` | 24-30GB | 74%-92% | 85%-99% | 30-40m |
| 7 | `sft_i2t` | vision_proj | 1 | 32 | 96 | 768 | on | `5e-6` | `0.001` | 20-25GB | 61%-77% | 85%-99% | 50-70m |

3 卡总估时：

```text
约 10.0 - 12.0 小时
```

估时说明：

- Stage 1 使用 `batch_size=40`，global batch=120，和常用 128 左右的全局 batch 接近，因此不需要再按卡数线性放大训练时长。
- A2A 阶段使用 `max_seq_len=1024`，显存压力最大，优先保证稳定，不盲目把 batch 拉到 40。
- I2T 阶段如果 smoke 后 mem util 长期低于 65%，可以试探提高到 `batch_size=40`。
- 如果 GPU util 长时间低于 70%，优先排查 dataloader / parquet IO / `num_workers`，而不是继续加 batch。

### 6.2 batch 调参策略与 OOM fallback

推荐从上表参数开始。full_train 开始前，每个阶段都先跑 100 step smoke，并记录 GPU mem / util。

| 场景 | 调整 |
|---|---|
| A2A seq1024 OOM | `batch_size 32 -> 24 -> 16` |
| I2T mem util < 65% 且 GPU util 正常 | `batch_size 32 -> 40` |
| T2A mem util < 55% 且训练稳定 | `batch_size 40 -> 48` 试探 |
| GPU util < 70%，mem util 不高 | 优先提高 `num_workers` 或检查 IO |
| loss 波动变大 | 不再加 batch，优先降低 `muon_lr` |

不建议为了追求显存占满强行把所有阶段 batch 拉满。A2A 的音频输入、SenseVoice 特征和 `seq_len=1024` 会让显存随样本长度波动，保留 2-4GB 余量更稳。

### 6.3 3 卡 full_train 权重命名

每阶段使用独立 `save_weight`，避免跨阶段 resume 混用。

推荐命名：

| 阶段 | save_weight | from_weight |
|---|---|---|
| 1 | `sft_full_muon_t2a` | `llm` 或 `none` |
| 2 | `sft_full_muon_a2a_proj` | `sft_full_muon_t2a` |
| 3 | `sft_full_muon_a2a_full` | `sft_full_muon_a2a_proj` |
| 4 | `sft_full_muon_i2t_proj` | `sft_full_muon_a2a_full` |
| 5 | `sft_full_muon_i2t_full` | `sft_full_muon_i2t_proj` |
| 6 | `sft_full_muon_a2a_final` | `sft_full_muon_i2t_full` |
| 7 | `sft_full_muon_final` | `sft_full_muon_a2a_final` |

这样做的好处：

- 每阶段都有可回退权重。
- resume checkpoint 不会跨数据集误用。
- 出现退化时可以定位是 A2A、I2T 还是最终回灌阶段造成。

### 6.4 Muon lr 建议

当前 `train_sft_omni.py` 的默认 `--muon_lr=0.02` 会覆盖所有 Muon 参数组，即使 `--learning_rate=5e-6`，Muon 大矩阵仍会按 `0.02 -> 0.002` 的比例调度。这对后期小 lr 微调偏激进。

建议 full_train 显式指定每阶段 `muon_lr`：

| 阶段 | 原 lr | 建议 muon_lr | 理由 |
|---|---:|---:|---|
| T2A all | `5e-4` | `0.01~0.02` | 初始对齐，可较激进 |
| A2A audio_proj | `5e-4` | `0.002~0.005` | 只训投影层，稳一点 |
| A2A all | `5e-5` | `0.001~0.003` | 防止 Talker acoustic codes 抖动 |
| I2T vision_proj | `5e-5` | `0.002~0.005` | 只训视觉投影层 |
| I2T all | `5e-6` | `5e-4~0.001` | 低 lr 保留语言/语音能力 |
| A2A final | `5e-6` | `5e-4~0.001` | 稳定语音输入链路 |
| I2T final proj | `5e-6` | `5e-4~0.001` | 最终视觉对齐，避免覆盖 |

如果目标是复现原始 README 曲线，建议同时跑一个 AdamW baseline 或保留 `--optimizer adamw` 对照。Muon 的收敛速度和后期声学稳定性需要用 CER/WER 与试听共同确认。

### 6.5 3 卡 Dense full_train 命令模板

以下模板偏稳健，显式指定每阶段 `muon_lr` 和独立 `save_weight`。

Stage 1: T2A

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-4 --muon_lr 0.02 \
  --data_path ../dataset/sft_t2a.parquet \
  --epochs 6 --batch_size 40 --use_compile 1 \
  --from_weight llm --save_weight sft_full_muon_t2a \
  --use_moe 0 --use_wandb
```

如果要真正从随机初始化开始，将 `--from_weight llm` 改成 `--from_weight none`。代价是语言能力和收敛质量可能明显变差，需要更长训练。

Stage 2: A2A audio_proj

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-4 --muon_lr 0.005 \
  --data_path ../dataset/sft_a2a.parquet \
  --epochs 1 --batch_size 32 --use_compile 0 \
  --from_weight sft_full_muon_t2a --save_weight sft_full_muon_a2a_proj \
  --max_seq_len 1024 --mode audio_proj \
  --use_moe 0 --use_wandb
```

Stage 3: A2A full

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-5 --muon_lr 0.002 \
  --data_path ../dataset/sft_a2a.parquet \
  --epochs 3 --batch_size 32 --use_compile 0 \
  --from_weight sft_full_muon_a2a_proj --save_weight sft_full_muon_a2a_full \
  --max_seq_len 1024 \
  --use_moe 0 --use_wandb
```

Stage 4: I2T vision_proj

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-5 --muon_lr 0.005 \
  --data_path ../dataset/sft_i2t.parquet \
  --epochs 1 --batch_size 32 --use_compile 1 \
  --from_weight sft_full_muon_a2a_full --save_weight sft_full_muon_i2t_proj \
  --max_seq_len 768 --mode vision_proj \
  --use_moe 0 --use_wandb
```

Stage 5: I2T full

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-6 --muon_lr 0.001 \
  --data_path ../dataset/sft_i2t.parquet \
  --epochs 1 --batch_size 32 --use_compile 1 \
  --from_weight sft_full_muon_i2t_proj --save_weight sft_full_muon_i2t_full \
  --max_seq_len 768 \
  --use_moe 0 --use_wandb
```

Stage 6: A2A final

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-6 --muon_lr 0.001 \
  --data_path ../dataset/sft_a2a.parquet \
  --epochs 1 --batch_size 32 --use_compile 0 \
  --from_weight sft_full_muon_i2t_full --save_weight sft_full_muon_a2a_final \
  --max_seq_len 1024 \
  --use_moe 0 --use_wandb
```

Stage 7: I2T final vision_proj

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port 29560 --nproc_per_node 3 train_sft_omni.py \
  --optimizer muon --learning_rate 5e-6 --muon_lr 0.001 \
  --data_path ../dataset/sft_i2t.parquet \
  --epochs 1 --batch_size 32 --use_compile 1 \
  --from_weight sft_full_muon_a2a_final --save_weight sft_full_muon_final \
  --max_seq_len 768 --mode vision_proj \
  --use_moe 0 --use_wandb
```

### 6.6 Full_train 显存风险与 mem util 验收

mini 抽样显存不能直接等价 full，但可以用于估计趋势：

- T2A seq 512、batch 40 已约 16.6GB。
- A2A seq 640、batch 40 已约 19.3GB。
- Full A2A 推荐 seq 1024、batch 32，预计会到 24GB 以上，是最容易 OOM 的阶段。

当前 5090D 32GB 显存建议按 6.1 表格启动。验收标准：

- T2A / I2T 阶段 mem util 低于 55%-65% 时，可以小幅加 batch。
- A2A 阶段 mem util 达到 75%-90% 属于正常，不建议继续硬拉 batch。
- GPU util 稳定在 85% 以上较健康；若长期低于 70%，优先排查 IO / dataloader。
- 预留 2-4GB 显存余量，避免长音频样本导致随机 OOM。

执行建议：

1. 先跑每阶段前 100 step smoke。
2. 如果 Stage 2 / 3 OOM，优先降 `batch_size` 到 24，再到 16。
3. 如果降 batch 后训练不稳，可适当加 `--accumulation_steps`，但需要重新审视学习率。
4. `use_compile=1` 的阶段启动更慢，会出现多个 compile worker；这是正常现象。

### 6.7 Full_train 监控建议

建议把 GPU telemetry 持续写入日志，而不是靠手动 `nvidia-smi` 抽样。

简单版：

```bash
mkdir -p .run_logs
while true; do
  date '+%F %T'
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader
  sleep 30
done | tee .run_logs/gpu_full_train_$(date +%Y%m%d_%H%M%S).log
```

训练日志建议：

```bash
bash scripts/run_full_train_muon_dense_3gpu.sh 2>&1 | tee .run_logs/full_train_muon_dense_3gpu.log
```

每阶段关注：

- loss 是否连续上升。
- audio loss 是否长期不降。
- `lr` 是否符合预期，特别是 Muon 组 lr。
- GPU util 是否长时间低于 70%，可能是 dataloader / IO 瓶颈。
- checkpoint 是否按 `save_interval` 落盘。

### 6.8 Resume 规则

原则：

- 同一阶段中断后，使用相同 `save_weight` + `--from_resume 1`。
- 跨阶段训练时，使用 `--from_weight 上一阶段save_weight`，不要加 `--from_resume 1`。
- 不要把 A2A 的 resume checkpoint 套到 T2A，也不要把 Stage 3 的 step 套到 Stage 1。

示例：

同一阶段续跑：

```bash
--save_weight sft_full_muon_a2a_full --from_resume 1
```

进入下一阶段：

```bash
--from_weight sft_full_muon_a2a_full --save_weight sft_full_muon_i2t_proj
```

### 6.9 Full_train 阶段性评估

建议每阶段结束后都跑一轮固定验证集：

```bash
conda run -n minimind-o python scripts/batch_validate_omni.py \
  --test_set dataset/eval_muon_mini.jsonl \
  --output_dir eval_results/<stage_name> \
  --weight <stage_weight> \
  --max_new_tokens 96 \
  --device cuda:0 \
  --decode_audio
```

建议额外补充 full_train 专用验证集：

- 20 条英文 T2A 短句。
- 20 条中文 T2A 短句。
- 20 条英文 A2A。
- 20 条中文 A2A。
- 9 条 I2T 图像描述。
- 5 条混合 image + audio prompt。
- 5 个内置音色 clone。
- 7 个 unseen 音色 clone。

如果要严肃比较模型，需要 ASR 回转：

1. 生成音频。
2. 用同一个 ASR 模型转写。
3. 将转写文本和模型文本输出、目标文本比较 CER / WER。
4. 记录音频时长、静音比例、异常重复、能量抖动。

## 7. 总体建议

### 7.1 对 mini 结果的判断

本轮 mini 训练证明：

- Muon 优化器接入可用。
- 3 卡 DDP 稳定。
- checkpoint/resume 修复有效。
- T2A/A2A 基础链路可运行。
- 批量生成和音频解码可运行。

但 mini 权重的问题也很清楚：

- 语义幻觉较多。
- 中文能力弱。
- 图像能力没有训练。
- Talker 音频 codes 不够稳，听感容易发抖。
- 长回答容易重复或跑题。

### 7.2 下一步训练优先级

推荐路线：

1. 先做推理采样参数改造，降低 audio code 随机性。
2. 用 mini 权重重跑一个低 `muon_lr` Stage 3 对照实验。
3. 跑 AdamW mini baseline，确认 Muon 是否影响音频抖动。
4. 再进入 full_train Dense。
5. Dense 稳定后再考虑 MoE。

### 7.3 Full_train 启动前 checklist

- 确认 full 数据存在：`dataset/sft_t2a.parquet`、`dataset/sft_a2a.parquet`、`dataset/sft_i2t.parquet`。
- 确认 `out/llm_768.pth` 存在，除非明确要从 `none` 随机初始化。
- 确认 `.run_logs/`、`out/`、`checkpoints/` 有足够磁盘空间。
- 先跑每阶段 100 step smoke。
- 使用独立 `save_weight`，不要复用同一个 resume。
- 持续记录 GPU memory/util/power/temperature。
- 每阶段结束立即跑 batch validation。

## 8. 当前风险与注意事项

1. `run_muon_scratch_mini.sh` 中 Stage 1 当前带 `--from_resume 1`，如果要从头重跑 mini，需要去掉。
2. 默认 `--muon_lr=0.02` 对后期 full fine-tune 可能过大。
3. 目前 batch validation 是基础通过率，不代表音质和语义正确率。
4. I2T 未训练时，image probe 通过基础检查不代表有图像理解。
5. Full_train 时间长，建议不要在没有 telemetry 和分阶段权重命名的情况下直接整夜跑。
