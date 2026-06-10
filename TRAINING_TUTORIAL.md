# MiniMind-O 训练代码全链路新手入门教程

> 适用对象：从零开始接触 MiniMind-O（Omni 多模态大模型）训练代码的同学
> 目标：读懂 `trainer/`、`model/`、`dataset/` 三大模块如何协同完成 SFT（全参 / 投影层）训练
> 阅读时长：建议 1~2 小时；建议配合 `model/model_minimind.py`、`model/model_omni.py`、`trainer/train_sft_omni.py` 一起看

---

## 0. 全局视图：5 分钟先建立一个心智模型

MiniMind-O 的「**训练代码**」实际上由 5 个层次组成，从下到上：

| 层次 | 入口文件 | 职责 |
|---|---|---|
| ① 配置层 | `model/model_minimind.py` (MiniMindConfig) + `model/model_omni.py` (OmniConfig) | 定义模型结构超参（层数、隐藏维度、是否 MoE、音频词表…） |
| ② 模型层 | `model/model_minimind.py` (RMSNorm/Attention/Block/Model) + `model/model_omni.py` (MiniMindOmni) | 实现 Thinker（文本）+ Talker（语音）+ Projector（音/视）的完整前向 |
| ③ 数据层 | `dataset/omni_dataset.py` (OmniDataset) | 把 parquet 数据 + 音频字节 + 图像字节 → 9 路 token 张量 + 标签 |
| ④ 训练引擎层 | `trainer/train_sft_omni.py` + `trainer/trainer_utils.py` + `trainer/optimizers.py` | 装配模型 / 数据 / 优化器 / 分布式 / 检查点，跑 SFT |
| ⑤ 调度脚本 | `trainer/train.sh` | 4 段式 SFT 流水（t2a → a2a-audio_proj → a2a → i2t）的开箱即用命令 |

> 整个训练流程只有一条主入口：`python trainer/train_sft_omni.py …`（或 `torchrun` 多卡）。理解它，就是理解整个项目。

下面我们用 7 个 Step 由外到内、由浅到深地拆解。每一 Step 都会在「代码位置 / 这一步在做什么 / 关键变量 / 新手常见疑问」四个维度展开。

---

## Step 1. 认识入口：`trainer/train_sft_omni.py` 的 9 个段落

文件长度只有 280 行，但 `__main__` 中用 9 个「注释分隔块」把整个训练脚本切成清晰的 9 段：

```python
# ========== 1. 初始化环境和随机种子 ==========
# ========== 2. 配置目录、模型参数、检查ckp ==========
# ========== 3. 设置混合精度 ==========
# ========== 4. 配wandb ==========
# ========== 5. 定义模型、数据、优化器 ==========
# ========== 6. 从ckp恢复状态 ==========
# ========== 7. DDP包模型 ==========
# ========== 8. 开始训练 ==========
# ========== 9. 清理分布进程 ==========
```

这 9 段就是训练流程的「顺序图」。后续 Step 2 ~ Step 6 会按这个顺序逐段细讲。

**新手疑问**：`__package__ = "trainer"` + `sys.path.append(os.path.abspath(...))` 在做什么？
- 让脚本既能 `python trainer/train_sft_omni.py` 跑，也能 `from trainer.X import Y` 这样按包导入。这是项目内多入口脚本（trainer / scripts / webui）共用的"坑位对齐"小技巧。

**关键变量**：
- `args`：所有命令行超参（`--learning_rate`、`--from_weight`、`--mode`…）。它在第 1 段后被建立，是后面所有逻辑的唯一真源。
- `omni_config`：`OmniConfig` 实例，决定模型结构。
- `model, tokenizer, scaler, optimizer, wandb`：5 个核心对象，第 5 段一次性创建好。

---

## Step 2. 命令行参数：所有可旋钮一览

入口函数 141~179 行的 `parser.add_argument(...)` 一长串就是「**可调旋钮**」。把它们按功能分类后非常清晰：

| 类别 | 关键参数 | 默认 | 含义 |
|---|---|---|---|
| 训练时长 | `--epochs` | 15 | 训练轮数 |
| 批 / 优化 | `--batch_size` | 32 | 每 GPU 批大小 |
|  | `--accumulation_steps` | 1 | 梯度累积步数 |
|  | `--learning_rate` | 5e-4 | AdamW / fallback 学习率 |
|  | `--optimizer` | muon | `muon` 或 `adamw`（见 Step 6） |
|  | `--muon_lr` | 0.02 | Muon 学习率（隐藏矩阵） |
|  | `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| 设备 | `--device` | cuda:0 | 训练设备 |
|  | `--dtype` | bfloat16 | 混合精度 |
|  | `--num_workers` | 4 | DataLoader 线程 |
| 模型结构 | `--hidden_size` | 768 | Thinker 隐藏维度 |
|  | `--num_hidden_layers` | 8 | Thinker 层数 |
|  | `--use_moe` | 0 | 是否用 MoE |
| 数据 | `--data_path` | sft_t2a_mini.parquet | parquet 路径，逗号可拼接多份 |
|  | `--max_seq_len` | 512 | 训练最大截断长度 |
| 续训 | `--from_weight` | llm | `llm`/`none`/其他权重前缀 |
|  | `--from_resume` | 0 | 是否自动检测 `_resume.pth` 续训 |
|  | `--freeze_backbone` | none | `none` / `all` / `last1` |
|  | `--mode` | all | `all` / `audio_proj` / `vision_proj` |
| 加速 / 观测 | `--use_compile` | 0 | torch.compile 开关 |
|  | `--use_wandb` | False | swanlab 日志 |
|  | `--log_interval` | 100 | 日志打印间隔 |
|  | `--save_interval` | 1000 | 检查点保存间隔 |

> 小贴士：所有「保存 / 加载」文件名的格式都是 `{save_weight}_{hidden_size}{_moe}.pth` 和 `{...}_resume.pth`，这俩后缀会在 `omni_checkpoint` 工具里用到。

---

## Step 3. 初始化与数据流：第 1 段 + 第 2 段 + 第 3 段

### 3.1 分布式与种子（第 1 段）

```python
local_rank = init_distributed_mode()                 # 来自 trainer_utils.py
if dist.is_initialized(): args.device = f"cuda:{local_rank}"
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

`init_distributed_mode` 干的事情：
- 看环境变量 `RANK` 是否存在
- 有：`dist.init_process_group(backend="nccl")` + `torch.cuda.set_device(local_rank)`，进入 DDP
- 没有：返回 0，单卡跑

> 这一行让脚本「**自适应**」单卡 / DDP，无需写两份。

`setup_seed(42 + rank)` 用 42 + rank 作种子，确保：
- 各 rank 不会因为随机数完全一样导致采样撞车
- 不同 epoch 之间也可控（`setup_seed(42 + epoch)`）

### 3.2 配置 + 目录 + 续训检测（第 2 段）

```python
os.makedirs(args.save_dir, exist_ok=True)
omni_config = OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

注意 `omni_checkpoint` 的「双面性」：
- **保存模式**（传入 `model=`）：写两份——主权重 `*.pth` + 续训包 `*_resume.pth`（含 optimizer / scaler / epoch / step / wandb_id）
- **加载模式**（不传 `model`）：只读 `*_resume.pth`，返回 dict；不存则返回 `None`

### 3.3 混合精度（第 3 段）

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
```

- bf16 走 `bfloat16`：等价于关闭了 `GradScaler`（不需要 loss scaling）
- fp16 走 `float16`：必须开 `GradScaler`
- 区别体现在后面 `scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))`

> 这块代码不复杂，但能解释为什么 `scaler.scale(loss).backward()` 这种写法在 bf16 模式下也安全（GradScaler 在 fp16 下有效，bf16 下空操作）。

---

## Step 4. 模型装配：第 5 段是最核心的「胶水」

第 5 段一口气把 4 个对象拼装起来：

```python
model, tokenizer = init_omni_model(omni_config, from_weight=args.from_weight,
                                    audio_encoder_path=args.audio_encoder_dir,
                                    vision_model_path=args.vision_dir,
                                    save_dir=args.save_dir, device=args.device,
                                    freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)

if args.use_compile == 1:
    model = torch.compile(model)

if model.audio_encoder is not None: model.audio_encoder.to(args.device)
if model.vision_encoder is not None: model.vision_encoder.to(args.device)
```

来拆解 `init_omni_model`（`trainer_utils.py:66-105`）这一函数：

### 4.1 加载 / 构造模型

```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)             # ../model 下的 tokenizer.json
model = MiniMindOmni(omni_config, audio_encoder_path=..., vision_model_path=...)
```

`MiniMindOmni.__init__` 内部会做 4 件事：
1. 调父类 `MiniMindForCausalLM.__init__` → 建 Thinker（`self.model`，含 embed_tokens + N 层 Block + norm + lm_head）
2. 额外建 `self.talker = TalkerModule(config)`：独立的 N'层 Talker + 独立 lm_head + RoPE 缓存
3. 建 projector：`self.audio_proj`、`self.vision_proj`（MLP 投影，把 SenseVoice/SigLIP 的特征塞进 hidden space）
4. 用 `load_sensevoice` 和 `load_vision` 加载冻结的外部 encoder（默认 cpu eval，**不**参与训练）

### 4.2 加载预训练权重

```python
weight_path = f'{save_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
if os.path.exists(weight_path):
    weights = torch.load(weight_path, map_location=device)
    param_shapes = {k: v.shape for k, v in model.named_parameters()}
    incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
    if incompatible: weights = {k: v for k, v in weights.items() if k not in incompatible}
    model.load_state_dict(weights, strict=False)
```

要点：
- 形状不匹配的 key **不会**抛错，而是被静默丢弃（这是个非常友好的设计，方便你换 hidden_size 迁移）
- `strict=False` 意味着 encoder 等"额外结构"缺失也无所谓（它们本来就被冻结）
- 加载完成后，如果 `talker` 在权重中不存在且 `hidden_size` 对齐，会**自动从 thinker 末尾几层复制到 talker**（`from_resume==0` 分支）

### 4.3 冻结策略

```python
if freeze_backbone == 'all':
    for param in model.model.parameters(): param.requires_grad = False
elif freeze_backbone == 'last1':
    for param in model.model.parameters(): param.requires_grad = False
    for param in model.model.layers[-1].parameters(): param.requires_grad = True
```

⚠️ 注意这里只冻结了 `model.model`（也就是 `thinker`），**没**冻结 `talker`/`proj`/`lm_head`。这意味着：
- `freeze_backbone='all'` + `mode='all'`：你还能训 talker + proj + audio-vision 部分
- `freeze_backbone='last1'` + `mode='all'`：同上，但 thinker 最后一层也开

### 4.4 mode 控制（`--mode`）

紧接其后第 225~230 行：

```python
if args.mode == 'audio_proj':
    for p in model.parameters(): p.requires_grad = False
    for p in model.audio_proj.parameters(): p.requires_grad = True
elif args.mode == 'vision_proj':
    for p in model.parameters(): p.requires_grad = False
    for p in model.vision_proj.parameters(): p.requires_grad = True
```

这是一个「**二段开关**」：先用 `freeze_backbone` 决定 thinker 训不训；再用 `mode` 决定只训哪个 projector（更激进）。两两组合就能玩出 4 种训练粒度：

| freeze_backbone | mode | 训什么 |
|---|---|---|
| none | all | 全部（最常见） |
| all | all | 冻结整个 thinker，只训 talker / proj / lm_head |
| last1 | all | 冻结 thinker 除最后一层以外的部分 |
| all | audio_proj | 只训 audio_proj（轻量 warmup） |
| all | vision_proj | 只训 vision_proj |

> 第 5 段最后 `log_model_params(model)` 还会打印总参 / 活跃参（MoE 时显示 `total-Aactive`），让你对模型规模有直观感知。

---

## Step 5. 数据装配：第 5 段里 `OmniDataset` 的「输入构造」与「label 构造」

```python
train_ds = OmniDataset(
    args.data_path, tokenizer,
    audio_processor=model.audio_processor,
    vision_processor=model.vision_processor,
    max_length=args.max_seq_len,
    image_token_len=model.config.image_token_len,
)
```

`OmniDataset.__init__`（`dataset/omni_dataset.py:45-74`）读 parquet → 提取 `tokenizer.pad_token_id`、`audio_pad/stop/spk_token`、`think_end_ids`、`bos_id/eos_id` 等关键 id。

**新手疑问**：为什么要单独存 `bos_id` 和 `eos_id`？
- 数据集构造的 `prompt` 末尾统一是 `f"{tokenizer.bos_token}assistant\n"`（bos_id），答案末尾是 `f"{tokenizer.eos_token}\n"`（eos_id）
- `generate_text_labels` 用这两个 id 切片，把「bos_id 到 eos_id 之间」当作正样本（label = input_id），其余置 -100

### 5.1 `__getitem__` 流程图

把整个 `__getitem__` 拆成 5 个阶段：

```
[1] 随机截断对话
        ↓
[2] 加载图像/音频字节 → 处理器 → tensor
        ↓
[3] 解码最后一轮 assistant 的 8 层 audio codes（如果存在）
        ↓
[4] 用 chat_template 拼 prompt，并把 <|audio_pad|> 展开成 N 个 token
        ↓
[5] 计算 text labels + audio labels + scheduled sampling
```

下面分别看每个阶段的关键代码。

#### [1] 随机截断对话

```python
asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
if len(asst_indices) > 1:
    rand_idx = random.randint(0, len(asst_indices) - 1)
    for i in range(rand_idx, -1, -1):
        conversations = conversations[:asst_indices[i] + 1]
        test_prompt = self.create_chat_prompt(conversations, 0)
        if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
            break
```

这段很有意思：从随机选定的 assistant 轮次开始，**向前**逐轮尝试；如果拼出来 + 100 token 还放得下，就停在这。效果是「同一对话可产生不同长度的训练样本」，让模型在多轮对话中各种长度都见过。

#### [2] 加载图像 / 音频

```python
audio_bytes = question_audios[user_count - 1]   # 取最后一个 user 轮的音频
mel, valid_len = self.load_audio_inputs(audio_bytes)
audio_inputs = mel.unsqueeze(0)
audio_len = valid_len
audio_features_length = valid_len or 1
```

注意 `augment_wav` / `augment_mel` 这一对函数：训练时随机加变速、噪声、音量、混响、低通、SpecAugment——是 SenseVoice encoder 不会被训、但 projector 要训时常用的小数据集增强技巧。

#### [3] 拆 8 层 audio codes

数据集的 `answer_audios` 是一长串 1D 整数；每 8 个一组 = 一帧 8 层 codebook 的 indices。代码把它 reshape 成 `(8, T)`：

```python
audio_codes_8layers = [[] for _ in range(8)]
for i in range(0, len(tokens) - 7, 8):
    for j in range(8): audio_codes_8layers[j].append(tokens[i + j])
for layer in audio_codes_8layers: layer.append(self.audio_stop_token)
```

末尾再追加一个 `audio_stop_token`（id=2050），代表「说完了」的语义 token。

#### [4] 拼 prompt

`create_chat_prompt` 用 `tokenizer.apply_chat_template`，并对最后一轮 user 的 content 做「4 种位置」随机扰动（40% 只放 audio pad / 20% 原内容 / 20% audio 在前 / 20% audio 在后）。这一招防止模型过度依赖 audio pad 在 prompt 中的固定位置。

`<image>` 也类似做 4 种位置扰动。

`<|image_pad|>` 一次出现会被展开为 `image_token_len` 个 token（默认 64），让 image embedding 一次性塞 64 个 hidden vector 进去。

#### [5] 构造 labels（最关键！）

##### text labels

```python
def generate_text_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    ...
    if input_ids[i:i + len(self.bos_id)] == self.bos_id:
        start = i + len(self.bos_id)
        end = start
        while end < len(input_ids):
            if input_ids[end:end + len(self.eos_id)] == self.eos_id: break
            end += 1
        ranges.append((start, end))
        for j in range(start, min(end + len(self.eos_id), self.max_length)):
            labels[j] = input_ids[j]
```

要点：
- 找每段 `bos_id...eos_id`，把这一段（含 eos）作为正样本，其余 -100
- 在 `__getitem__` 末尾再把所有**非最后一个** assistant 段的 label 抹掉：
  ```python
  for start, end in assistant_ranges[:-1]:
      mask_end = min(end + len(self.eos_id), self.max_length)
      text_labels[start:mask_end] = [-100] * (mask_end - start)
  ```
  → SFT 阶段永远只让 loss 来自**最后一个** assistant 的回答（与 `train.sh` 中三阶段目标一致）

##### audio labels

```python
Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(8)]
audio_labels = [[-100] * self.max_length for _ in range(8)]
...
# spk 占位（仅一个位置）
if has_spk and ref_start > 0:
    spk_pos = ref_start - 1
    for layer_idx in range(8): Y_audio_layers[layer_idx][spk_pos] = self.audio_spk_token
# target codes 填充到 assistant_start 之后（参与 loss）
for layer_idx in range(8):
    codes = last_audio_codes[layer_idx]
    start_pos = assistant_start + layer_idx + 1   # 注意：每层起始位置错位 layer_idx+1（MTP 阶梯式）
    for i, code in enumerate(codes):
        if start_pos + i < self.max_length:
            Y_audio_layers[layer_idx][start_pos + i] = code
            audio_labels[layer_idx][start_pos + i] = code
```

`start_pos = assistant_start + layer_idx + 1` 是 **MTP（Multi-Token Prediction）** 的关键：第 0 层从 assistant 之后第 1 位开始预测；第 1 层从第 2 位开始…… 这正是 MTP 让 Talker 一次预测 8 层 codebook 的对位方式。

`ref_codes`（参考音频 codes）以 50% 概率注入 prompt 区，给模型提供「音色条件」。

##### 最终 9 路输入拼装

```python
X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)  # (8, T-1)
X_text = torch.tensor(input_ids[:-1], dtype=torch.long)                            # (T-1,)
input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)                        # (9, T-1)
text_labels = torch.tensor(text_labels[1:], dtype=torch.long)                       # (T-1,)
audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)  # (8, T-1)
```

→ `input_ids` shape = `(9, T-1)`，8 路 audio 通道 + 1 路 text 通道（与 `MiniMindOmni.forward` 的解析一致）

##### Scheduled Sampling

```python
def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
    audio_mask = (audio_labels != -100).any(dim=0) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
    for i in range(8):
        input_ids[i] = torch.where(audio_mask, torch.randint(0, self.audio_vocab_size, input_ids[i].shape), input_ids[i])
    text_mask = (text_labels != -100) & (input_ids[8] != self.image_token_id) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
    input_ids[8] = torch.where(text_mask, torch.randint(0, self.text_vocab_size, input_ids[8].shape), input_ids[8])
    return input_ids
```

- 训练时随机把 **部分 ground-truth audio / text token** 替换成 vocab 内随机采样
- 防止模型「只会顺着正确答案抄」，迫使其学会从错误历史中恢复（也就是 inference 时 teacher-forcing 累积误差的缓解）
- `image_token_id` 永远不被替换，避免破坏图像 token 的连续性

### 5.2 自定义 `collate_fn`：变长批

`__getitem__` 返回 7 个对象，但 batch 内 audio / image 是变长的：

```python
def omni_collate_fn(batch):
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)
    input_ids = torch.stack(input_ids)               # (B, 9, T-1)
    labels = torch.stack(labels)                     # (B, T-1)
    audio_labels = torch.stack(audio_labels)         # (B, 8, T-1)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)   # (B,)

    # audio: pad 到 (B, T_max, 560) 再 cat
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [a if a.size(1) == max_t else torch.nn.functional.pad(a, (0,0,0,max_t - a.size(1))) for a in valid_audios]
        audio_inputs = torch.cat(padded, dim=0)

    # image: 可能是 dict（含 pixel_values/pixel_attention_mask）或 tensor
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        if hasattr(valid_images[0], 'keys'):
            keys = set.intersection(*[set(d.keys()) for d in valid_images])
            pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
        else:
            pixel_values = torch.cat(valid_images, dim=0)

    spk_emb = torch.stack(spk_emb)                   # (B, 192)
    return input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb
```

> 关键 hack：data 里对「无 audio / 无 image」的样本构造了 dummy tensor（`torch.zeros(1,1,560)` / `torch.zeros(1,3,256,256)`），所以 `cat` 时不会出 shape 错。`MiniMindOmni.encode_audio_inputs` 内部用 `batch_mask` 把 dummy 样本过滤掉。

---

## Step 6. 优化器：Muon（隐藏矩阵）+ AdamW（embed/head/norm）混合策略

第 5 段最后一行是：

```python
optimizer = build_optimizer(model, args)
```

`trainer/optimizers.py` 实现了一个比较新颖的「**Muon-with-Aux-AdamW**」混合优化器。先看 `build_optimizer` 入口：

```python
def build_optimizer(model, args):
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        return optim.AdamW([p for _, p in trainable_named_params], lr=args.learning_rate, ...)

    muon_params, adam_params = [], []
    for name, param in trainable_named_params:
        if _is_muon_parameter(name, param):
            muon_params.append(param)
        else:
            adam_params.append(param)
    ...
    optimizer = MuonWithAuxAdam(param_groups)
```

`_is_muon_parameter` 的判定：

```python
def _is_muon_parameter(name, param):
    if (not param.requires_grad) or param.ndim != 2: return False
    lowered = name.lower()
    excluded = ("embed", "lm_head", "head", "norm", "bias", "audio_encoder", "vision_encoder")
    return not any(token in lowered for token in excluded)
```

> 一句话总结：**ndim==2 且名字不含 embed/head/norm/bias/encoder** 的参数走 Muon；其它走 AdamW。

### 6.1 Muon 是什么

Muon 是一种「**矩阵优化器**」：对 2D 矩阵参数，用 Newton–Schulz 迭代把梯度做近似正交化（谱归一化），然后用作 update。直观好处：
- 隐藏层的「各向同性更新」让 loss landscape 更平稳
- 收敛速度通常比 AdamW 更快（2~3×）
- 适合大矩阵，不适合 embedding / 1D 参数（embed/head 用 AdamW 更稳）

`zeropower_via_newtonschulz5` 是 5 步 Newton–Schulz 多项式逼近「零次方」的正交化：

```python
x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)   # 归一
a, b, c = 3.4445, -4.7750, 2.0315
for _ in range(steps):
    xa = x @ x.mT
    xb = b * xa + c * (xa @ xa)
    x = a * x + xb @ x
```

它把 x 朝「半正交」方向推，相当于把谱半径压到 ~1。

### 6.2 `MuonWithAuxAdam.step` 的双策略

```python
@torch.no_grad()
def step(self, closure=None):
    for group in self.param_groups:
        if group["use_muon"]:   self._step_muon_group(group)
        else:                   self._step_adamw_group(group)
```

- **Muon 组**：momentum + Nesterov + Newton–Schulz 5 步正交化
- **AdamW 组**：标准的 exp_avg / exp_avg_sq 累积 + bias correction

两者用 `lr_scale` 解耦，Cosine 调度时 `param_group['lr']` 会跟着 `get_lr(...)` 实时刷新。

### 6.3 学习率调度

`get_lr` 在 `trainer_utils.py:26`：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

- 起 lr = 1.0 * base_lr（cos(0)=1 → 0.1+0.45*2=1.0）
- 终 lr = 0.1 * base_lr（cos(π)=-1 → 0.1+0=0.1）
- 中间平滑下降（与 SGDR / cosine with warmup 类似，但没有显式 warmup）

每步在 `train_epoch` 内手动设到 param_group：

```python
for param_group in optimizer.param_groups:
    param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
```

`lr_scale` 是在 `build_optimizer` 中：
- AdamW 组：`lr_scale = 1.0`
- Muon 组：`lr_scale = muon_lr / learning_rate`（默认 0.02 / 5e-4 = 40）

> 这就是为什么「全局 5e-4」+「Muon 内部 0.02」能和谐工作：所有 param_group 共享同一 cosine 曲线，只是按比例缩放。

---

## Step 7. 训练循环：第 6 段 + 第 8 段

### 7.1 续训恢复（第 6 段）

```python
start_epoch, start_step = 0, 0
if ckp_data:
    model.load_state_dict(ckp_data['model'], strict=False)
    try:
        optimizer.load_state_dict(ckp_data['optimizer'])
    except ValueError as e:
        Logger(f'优化器状态不兼容，已跳过optimizer resume: {e}')
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

注意 `try/except`：当你改完 hidden_size 续训时，optimizer state 的形状可能对不上——这里**容错降级**只 load model，不 load optimizer。

### 7.2 DDP 包装（第 7 段）

```python
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

单卡时不进 DDP。`train_epoch` 内部所有 `model.xxx` 调用都自动 DDP-safe。

> 一个常见坑：`DistributedSampler` 必须配合 `set_epoch(epoch)` 才能保证每个 epoch 的 shuffle 不同——第 8 段第一行 `train_sampler and train_sampler.set_epoch(epoch)` 就是干这个的。

### 7.3 SkipBatchSampler：续训时跳过已跑 step

```python
class SkipBatchSampler(Sampler):
    def __iter__(self):
        batch, skipped = [], 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []; continue
                yield batch
                batch = []
```

> 续训到第 5 epoch 第 3000 步崩溃重启：先 load `start_epoch=4, start_step=3000`，再跳过前 3000 个 batch 直接接 3001。开箱即用。

### 7.4 `train_epoch` 三件事

#### A. 前向 + 三路 loss

```python
with autocast_ctx:
    res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, spk_emb=spk_emb)
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    # 1) Text loss
    text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
    text_mask = (labels.view(-1) != -100).float()
    text_loss = (text_loss_raw * text_mask).sum() / (text_mask.sum() + 1e-9)

    # 2) Audio loss（MTP 8 层加权求和）
    audio_loss = res.audio_logits[0].sum() * 0
    for i, al in enumerate(res.audio_logits):
        al_flat = al.view(-1, al.size(-1))
        target_flat = audio_labels[:, i, :].reshape(-1)
        layer_loss = loss_fct(al_flat, target_flat)
        valid_mask = (target_flat != -100).float()
        stop_mask  = (target_flat == 2050).float()
        weighted_loss = layer_loss * valid_mask * (1 + stop_mask * 9)
        msum = valid_mask.sum()
        if msum > 0:
            audio_loss = audio_loss + weighted_loss.sum() / (msum + 1e-9)
    audio_loss = audio_loss / 8

    loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps
```

`MiniMindOmni.forward` 一次返回两个输出：
- `res.logits`：thinker 文本输出，shape `(B, T, text_vocab)`
- `res.audio_logits`：list of 8 个 tensor，每个 shape `(B, T, audio_vocab)`，对应 8 层 codebook

注意 audio loss 的 `stop_mask`：**停帧 (2050) 权 ×10**，让它训练得"特别准"——这是因为停帧位置错了会让整段语音没法结束。

`res.aux_loss` 来自 MoE 路由的负载均衡损失（`MOEFeedForward.aux_loss`），dense 模型下是 0。

#### B. 反向 + 累积

```python
scaler.scale(loss).backward()
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

> 几个细节：
> - `scaler.scale(loss).backward()` 在 bf16 模式下 `scaler.scale` 是恒等映射，安全
> - `unscale_` + `clip_grad_norm_` 必须在 `scaler.step` 之前，否则会被缩放影响
> - `accumulation_steps=1` 时每步都更新；大于 1 时相当于用 1/accumulation_steps 步长训练
> - 训练循环最后还有一段 `if last_step % accumulation_steps != 0` 的「补刀」逻辑，把没攒满的最后几个 batch 也 step 掉

#### C. 日志 + 保存

```python
if step % args.log_interval == 0 or step == iters:
    Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, text: {text_loss_val:.4f}, audio: {audio_loss_val:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
    if wandb: wandb.log({"loss": current_loss, "text_loss": text_loss_val, "audio_loss": audio_loss_val, "lr": current_lr, "epoch_time": eta_min})

if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    moe_suffix = '_moe' if omni_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.')}
    torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
    omni_checkpoint(omni_config, weight=args.save_weight, model=model, optimizer=optimizer,
                    epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
    model.train()
```

要点：
- 主权重 `*.pth` 只存**主权重**（不存 `audio_encoder` / `vision_encoder` / `audio_processor` / `vision_processor`），**省一半磁盘**；这些是冻结的，推理时按路径再加载即可
- `*_resume.pth` 存 optimizer / scaler / epoch / step / wandb_id（用于断点续训）
- 写盘用 `*.tmp` → `os.replace` 的原子替换，避免崩溃时留下半个文件
- `model.eval()` + 临时切 train 是为了让 state_dict 反映当前权重；`half().cpu()` 让 checkpoint 体积小一半
- 跳过 audio_encoder 的那行非常关键——如果没跳过，1 GB 起步（多卡时 2 GB 起步）

---

## Step 8. 训练入口的「流程骨架」

把脚本最后一段简化成伪代码：

```python
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                        collate_fn=omni_collate_fn,
                        num_workers=args.num_workers, pin_memory=True)
    if skip > 0:
        Logger(f'Epoch [{epoch+1}/{args.epochs}]: 跳过前{start_step}个step...')
        train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
    else:
        train_epoch(epoch, loader, len(loader), 0, wandb)

if dist.is_initialized(): dist.destroy_process_group()
```

3 个细节：
- `setup_seed(42 + epoch)`：每轮重置随机性但带 epoch 偏置
- `torch.randperm(...)`：单卡 / 没有 DistributedSampler 时手动 shuffle
- `pin_memory=True` + `num_workers>0`：数据预取加速

---

## Step 9. `train.sh` 拆解：完整 4 段式 SFT 流水线

`trainer/train.sh` 给出了从 LLM 权重到 Omni 全模态的完整训练路径。新手最该看的是 **mini 数据集版本**（28~30 行）：

```bash
# 1) 文本 → 音频（T2A，6 epoch，从 llm 起步）
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py \
  --learning_rate 5e-4 --data_path ../dataset/sft_t2a_mini.parquet \
  --epochs 1 --batch_size 40 --use_compile 1 \
  --from_weight llm --save_weight sft_zero --max_seq_len 512 \
  --use_wandb --use_moe 0

# 2) 音频 → 音频 + 文本（A2A warm-up：只训 audio_proj，1 epoch）
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py \
  --learning_rate 5e-4 --data_path ../dataset/sft_a2a_mini.parquet \
  --epochs 1 --batch_size 40 --use_compile 0 \
  --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 \
  --mode audio_proj --use_wandb --use_moe 0

# 3) 音频 → 音频 + 文本（A2A 全量微调，1 epoch，更小 lr）
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py \
  --learning_rate 2e-5 --data_path ../dataset/sft_a2a_mini.parquet \
  --epochs 1 --batch_size 16 --use_compile 0 \
  --from_weight sft_zero --save_weight sft_zero --max_seq_len 768 \
  --use_wandb --use_moe 0
```

| 阶段 | 数据 | 关键 flag | 训练目标 |
|---|---|---|---|
| ① T2A | 文本+音频 | `--from_weight llm` `--mode all` | 学会「文本→Mimi 码」基本对齐 |
| ② A2A warm-up | 音频 | `--mode audio_proj` | 把 audio_proj 对齐到 LLM hidden |
| ③ A2A 全量 | 音频 | `--mode all` `--learning_rate 2e-5` | 端到端细调，lr 小一档防破坏 |
| ④ I2T | 图像 | `--mode vision_proj` | 视觉投影对齐 |
| ⑤ I2T 全量 | 图像 | `--mode all` | 视觉全量微调 |

> 这 5 段的设计思想是「**先 projector 对齐 → 再全量微调**」，每个阶段都从上一阶段的权重（`--from_weight sft_zero`）续训，是工业界常见 multi-stage SFT 套路。

---

## Step 10. 推理/评估（`eval_omni.py`）：把训练链路反过来用

`eval_omni.py` 是训完怎么用的入口。逻辑和训练时几乎对称：

```python
# 1) 加载模型（两种方式）
#   a) 原生 torch 权重：MiniMindOmni(...) + load_state_dict(...)
#   b) HF transformers 格式：AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
# 2) 加载 Mimi decoder（把 8 层 codebook → 24kHz 波形）
# 3) 七种 eval 模式
modes = set(args.mode.replace(',', '').replace('-1', '012345'))
# '0' = 纯文本 → {文本, 语音}
# '1' = 多轮对话
# '2' = 音频 → {文本, 语音}
# '3' = 音色克隆
# '4' = 图像 → {文本, 语音}
# '5' = 文本+音频+图像 → {文本, 语音}
```

`eval_sample` 用 `model.generate(..., stream=True, return_audio_codes=True)` 流式输出：
- 每个 step 同时 yield 一段 thinker 文本 token 和（可选）一个 talker 8-码帧
- `mimi_model.decode(codes).audio_values` 把 8 码 → 24kHz 音频
- `pydub` 再编码成 mp3 落盘

`scripts/convert_omni.py` 的作用：
- `convert_torch2transformers`：把训练保存的原生 `*.pth` 转成 HF `pytorch_model.bin + config.json`，方便 `from_pretrained`
- `convert_transformers2torch`：反向

> 这就是为什么 checkpoint 不存 `audio_encoder`：HF 格式可以重新 load 这个外部 encoder。

---

## Step 11. 完整链路「一图流」

```
┌────────────────────────────────────────────────────────────────────────┐
│                        train_sft_omni.py                                │
│                                                                        │
│  argparse → init_distributed → OmniConfig → omni_checkpoint(load)     │
│       │                                                                │
│       ↓                                                                │
│  init_omni_model  ──→  load llm_768.pth (可)                            │
│       │                                                                │
│       ↓                                                                │
│  OmniDataset(parquet)  ─→  (B,9,T-1) text+audio codes                  │
│       │                ─→  (B,T,560) audio_inputs (SenseVoice fbank)   │
│       │                ─→  (B,3,256,256) pixel_values (SigLIP)         │
│       │                ─→  (B,192) spk_emb                             │
│       ↓                                                                │
│  build_optimizer  ──→  Muon 矩阵 + AdamW fallback                       │
│       │                                                                │
│       ↓                                                                │
│  DDP / 训练循环：                                                       │
│   forward(input_ids)  ─→  thinker ─→ text_logits                       │
│      │                                                                 │
│      └→  talker   ─→  audio_logits[8]                                 │
│   loss = text_loss + audio_loss/8 + aux_loss                           │
│   scaler.scale(loss).backward() → clip → step                          │
│       │                                                                │
│       ↓                                                                │
│  save: {save_weight}_{hidden_size}.pth (half) + _resume.pth (full)    │
└────────────────────────────────────────────────────────────────────────┘
              │                                                │
              ↓                                                ↓
     eval_omni.py                                       scripts/convert_omni.py
     (推理 / 解码 Mimi)                              (原生 ⇄ transformers 格式)
```

---

## Step 12. 新手「**改训练**」实操清单

下面这些是改 MiniMind-O 训练代码时最常见的 10 个动作，附代码锚点：

1. **换数据集**：直接改 `--data_path`，或逗号拼接 `--data_path a.parquet,b.parquet`（`OmniDataset.__init__` 已支持）
2. **改模型大小**：`--hidden_size 512/768/1024` + `--num_hidden_layers 8/16`（**先确认有匹配的 `llm_*.pth`**）
3. **改训练粒度**：`--mode {all, audio_proj, vision_proj}` + `--freeze_backbone {none, all, last1}`（参 Step 4.4 表格）
4. **加 LoRA / IA3**：在 `model_minimind.py` 写 adapter wrapper，改 `init_omni_model` 的冻结策略
5. **替换优化器**：在 `optimizers.py` 加自定义 `Optimizer`，并在 `build_optimizer` 注册（`--optimizer` 选 `[muon, adamw]`）
6. **加自定义 loss**：在 `train_epoch` 的 `with autocast_ctx:` 块内、`text_loss + audio_loss` 之后加一行
7. **多机多卡**：和单机的 DDP 一样，只在 launch 时改 `torchrun --nnodes=N --node_rank=R --rdzv_id=... --rdzv_endpoint=...`；训练脚本本身无需改动
8. **加新检查点指标**：在 `omni_checkpoint` 调用的 `**kwargs` 传 `my_metric=...`（`trainer_utils.py:108-149` 已预留）
9. **改 lr 调度**：替换 `get_lr` 函数（`trainer_utils.py:26`）
10. **加新模态**（比如视频）：在 `MiniMindOmni.__init__` 加 `video_encoder` + `video_proj`；在 `__getitem__` 加 video 解析；扩展 `input_ids` 维度（目前 9 路）

---

## Step 13. 常见踩坑 FAQ

**Q1：训练时报 `RuntimeError: Found dtype Float but expected BFloat16`？**
- 90% 是 `dtype` 不一致。检查 `--dtype` 是不是和 model load 的 dtype 对齐。`model_omni.py:222-223` 显式 `model.audio_encoder.to(args.device)` 与 `model.vision_encoder.to(args.device)`，可保持 fp32；其余用 autocast。

**Q2：DDP 启动卡住不收敛？**
- 八成是 `DistributedSampler` 没 `set_epoch`，导致每个 epoch shuffle 一致。

**Q3：`from_weight llm` 报 shape 不匹配？**
- 检查 `omni_config.hidden_size` 和 `llm_*.pth` 的 hidden_size 是否一致。`init_omni_model` 会**静默丢弃**不一致的 key 然后报错日志很「温和」，但你看到 `跳过shape不匹配的权重: {...}` 就懂了。

**Q4：`audio_loss` 一开始是 NaN？**
- `target == 2050` 那一项权重 ×10，初期 logits 巨大可能爆；可暂时把 `1 + stop_mask * 9` 改成 `1 + stop_mask * 1` 训练几步看是否恢复。

**Q5：`scheduled_sampling_prob` 调到 0.1 后训练变差？**
- 它是「让模型在错误历史中恢复」的正则项，过高会引入过多噪声。建议 0.05 起步。

**Q6：保存的 checkpoint 比预期大 1 倍？**
- 可能是 `audio_encoder` 没被剔除。检查 `omni_checkpoint` 调用的 `clean_state_dict` 是否过滤 `audio_encoder.` / `vision_encoder.`。如果用 DDP 包装，必须先 `model.module` 再过滤。

**Q7：MoE 训练很慢？**
- `MOEFeedForward` 是 for-loop 遍历 expert；n_experts=4 时是 4 倍 dense FFN 的耗时。如果不是必要，可以先用 dense 跑通，再切 `--use_moe 1`。

**Q8：`use_compile=1` 第一次 step 巨慢？**
- 正常。torch.compile 在第一次跑时会 trace + compile，之后的 step 才会加速。建议 warmup 3~5 step 再开始记时。

**Q9：推理时报 `RoPE buffer is all zero`？**
- `transformers>=5.x` 在 meta-device init 时会丢掉 RoPE buffer。`model_omni.py:258-263` 在 forward 内做了「检查 → 重算」修复，第一次 forward 时自动恢复。

**Q10：怎么确认 talker 真的在学？**
- 看 `audio_loss` 是否平稳下降；如果文本 loss 下降而 audio loss 一直 ~3（接近 audio_vocab_size 的均匀分布 log），说明 talker 没接收到梯度——很可能是 `--mode audio_proj` 把 talker 冻了或没正确 freeze。

---

## Step 14. 关键函数「背诵清单」

> 如果只能记 10 个函数名 + 它在哪 + 它做什么，就背下面这 10 个。

| # | 函数 / 类 | 文件 | 作用 |
|---|---|---|---|
| 1 | `OmniConfig` | `model/model_omni.py:10` | 全模态超参（继承 MiniMindConfig） |
| 2 | `MiniMindOmni.forward` | `model/model_omni.py:245` | Thinker+Talker+Projector 一次跑完 |
| 3 | `OmniDataset.__getitem__` | `dataset/omni_dataset.py:211` | 单条样本 → 9 路 token + 标签 |
| 4 | `OmniDataset.create_chat_prompt` | `dataset/omni_dataset.py:155` | chat_template + 4 种位置扰动 |
| 5 | `omni_collate_fn` | `trainer/train_sft_omni.py:25` | 变长 audio/image 拼 batch |
| 6 | `init_omni_model` | `trainer/trainer_utils.py:66` | 加载 / 构造 / 冻结策略 |
| 7 | `omni_checkpoint` | `trainer/trainer_utils.py:108` | 双面 checkpoint：保存 / 加载 |
| 8 | `train_epoch` | `trainer/train_sft_omni.py:52` | 三路 loss + 累积 + 存盘 |
| 9 | `MuonWithAuxAdam.step` | `trainer/optimizers.py:60` | 混合优化器：Muon 矩阵 + AdamW fallback |
| 10 | `get_lr` | `trainer/trainer_utils.py:26` | Cosine 退火（0.1→1.0→0.1） |

---

## Step 15. 推荐阅读顺序（30 分钟极简版）

如果时间紧，按这个顺序看：

1. `trainer/train.sh` 末尾三行 mini pipeline（1 分钟，建立「这是分阶段训练」的直觉）
2. `trainer/train_sft_omni.py` 的 `__main__` + 9 段注释（5 分钟，看清流程骨架）
3. `model/model_omni.py` 的 `MiniMindOmni.forward`（15 分钟，搞懂 9 路 token 怎么流到 text + audio logits）
4. `dataset/omni_dataset.py` 的 `__getitem__`（10 分钟，理解 prompt + label 怎么构造）
5. `trainer/optimizers.py` 的 `MuonWithAuxAdam`（5 分钟，知道优化器为什么这么分）

最后用 `eval_omni.py` 跑一次，看看自己改的训练能不能正常工作。

---

## 附录 A：术语表

| 术语 | 含义 |
|---|---|
| **Thinker** | MiniMind-O 中负责文本理解 / 生成的 LLM 主干（= `self.thinker` == `self.model`） |
| **Talker** | 独立于 Thinker 的「自回归音频码」解码器，结构像 Thinker 但小一圈 |
| **MTP** | Multi-Token Prediction：一次前向输出 8 个 codebook 的码 |
| **Mimi** | Kyutai 开源的神经音频 codec，8 层 codebook × 12.5 Hz → 24 kHz 波形 |
| **Projector** | 音 / 视 特征 → hidden space 的两层 MLP 投影 |
| **SenseVoice** | FunASR 提供的 ASR encoder，提供 fbank (T,560) 特征 |
| **SigLIP2** | Google 的视觉-语言预训练模型，提供 patch embedding (1, 64, 768) |
| **spk_emb** | 192 维说话人 embedding，用于音色克隆 |
| **ref_codes** | 参考音频的 8 层 code，作为音色条件 |
| **scheduled sampling** | 训练时随机把 GT 替换为 vocab 随机采样，让模型学会「纠错」 |
| **DDP** | DistributedDataParallel，多卡数据并行 |
| **NCCL** | NVIDIA Collective Communications Library，DDP 的后端 |
| **autocast** | PyTorch 混合精度上下文管理器 |

## 附录 B：文件 → 行号速查表

| 你想找的东西 | 位置 |
|---|---|
| argparse 全部超参 | `trainer/train_sft_omni.py:141-179` |
| 9 路 input 拼装 | `dataset/omni_dataset.py:321-323` |
| 文本 label 构造 | `dataset/omni_dataset.py:179-197` |
| 音频 label 构造 | `dataset/omni_dataset.py:283-318` |
| MTP 起始位置 `+ layer_idx + 1` | `dataset/omni_dataset.py:314` |
| Stop token 权 ×10 | `trainer/train_sft_omni.py:90` |
| 跳过 audio_encoder 存盘 | `trainer/train_sft_omni.py:124`、`trainer/trainer_utils.py:119` |
| MoE 辅助 loss | `model/model_omni.py:309` |
| Muon Newton–Schulz 5 步 | `trainer/optimizers.py:7-25` |
| Cosine lr 调度 | `trainer/trainer_utils.py:26-28` |
| MoE 路由 loss 公式 | `model/model_minimind.py:170-173` |
| 续训 step 跳过 | `trainer/trainer_utils.py:177-200` |
| 训练 4 段式 pipeline | `trainer/train.sh:28-30` |

---

> 写完这份教程，你应该能：
> 1. 在不查任何资料的情况下讲清楚「MiniMind-O 训练时一个 batch 经过了什么」；
> 2. 找到 90% 常见超参 / 训练策略对应的代码行；
> 3. 安全地修改训练粒度（`--mode`、`--freeze_backbone`）和优化器（`--optimizer`）以做消融实验；
> 4. 用 `omni_checkpoint` + `--from_resume 1` 实现断点续训。
>
> 剩下 10% 隐藏知识，靠多看 `model_omni.py` 里的 forward——它才是所有"为什么"的最终答案。
