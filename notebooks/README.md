# MiniMind-O 新人上手 Notebook 教程

本目录提供一组 Jupyter Notebook，帮助新成员从 0 开始理解 MiniMind-O 的代码结构、训练流程与调试方法。

## 环境要求

- Python 3.10+
- PyTorch 2.x with CUDA
- 已安装项目 `requirements.txt`
- 已下载 SenseVoiceSmall、SigLIP2、Mimi、CAMPPlus 等预训练模型
- Jupyter 或 VS Code Notebook 支持

## Notebook 顺序

| 编号 | 文件名 | 内容 |
| --- | --- | --- |
| 00 | `00_setup_and_overview.ipynb` | 项目结构、依赖检查、环境准备、快速验证 |
| 01 | `01_model_architecture.ipynb` | MiniMind Thinker + Talker + Projectors 逐层解析 |
| 02 | `02_data_flow.ipynb` | OmniDataset、数据增强、collate、token 对齐 |
| 03 | `03_training_loop.ipynb` | 训练循环、loss 计算、Muon 优化器、DDP、checkpoint |
| 04 | `04_inference_eval.ipynb` | 推理、批量验证、ASR 回译评估 |
| 05 | `05_debugging.ipynb` | 常见错误、调试技巧、显存分析、问题清单 |

## 使用建议

1. 按编号顺序阅读，每个 notebook 尽量独立可运行。
2. 阅读时对照仓库真实源码，不要只看 notebook 中的节选。
3. 修改代码后，优先在 `dataset/_calib/` 或 `_smoke/` 小数据上跑通，再上 full。
4. 遇到训练问题先看 `05_debugging.ipynb` 的 checklist。

## 验证状态

- 所有 notebook 已通过 JSON 格式校验与 Python AST 语法检查。
- 以下核心代码片段已在 `conda env minimind-o` 中实测通过：
  - `00_setup_and_overview.ipynb`：OmniConfig 创建、MiniMindOmni 实例化、参数统计（113.13M）。
  - `01_model_architecture.ipynb`：Thinker 前向、RoPE buffer、Omni 输出 logits 形状。
  - `02_data_flow.ipynb`：`omni_collate_fn` 变长音频/图像拼接。
  - `03_training_loop.ipynb`：`compute_batch_loss` local loss norm 与 RVQ 加权。
- 其余 notebook（04/05）主要为命令模板与辅助函数，已做静态语法检查。

## 注意事项

- 所有 notebook 假设工作目录为仓库根目录 `/home/genesis/Projects/minimind-o`。
- 部分单元格需要 GPU；若无 GPU，可将 `device='cuda'` 改为 `cpu`，但部分操作会非常慢。
- 不要直接在大模型权重上随意运行训练单元格，避免覆盖已有 checkpoint。
- 若运行代码时提示缺少依赖，请先激活 `minimind-o` conda 环境或安装 `requirements.txt`。
