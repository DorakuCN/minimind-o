# MiniMind-O Full Train Summary - Muon Dense 3GPU - 2026-06-10

## Overview

- Project: MiniMind-O
- Final run: `full_train_muon_dense_3gpu_20260610_045204_stage5_restart_b64_swanlab`
- Hardware: 3 x NVIDIA GeForce RTX 5090D 32G
- CUDA/NCCL: CUDA 13.3, NCCL 2.30.7+cuda13.3
- Conda env: `minimind-o`
- Optimizer: `MuonWithAuxAdam`
- Model size: 113.13M params
- Data mode: 3-way rank-sharded parquet for DDP
- Tracking: SwanLab project `MiniMind-O-Full-Train`
- Final status: completed at 2026-06-10 07:33:40 PDT

Batch size below is per GPU. Effective global batch is approximately `batch_size * 3`.

## Stage Results

| Stage | Task | Batch/GPU | Epochs | Trainable Params | Avg Mem/GPU | Peak Mem/GPU | Avg GPU Util | Final Loss |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | T2A all params | 48 | 6 | 113.13M | 19228 MiB | 19279 MiB | 96.8% | 5.0492 |
| 2 | A2A audio projector | 32 | 1 | 0.99M | 24990 MiB | 25985 MiB | 97.4% | 7.9235 |
| 3 | A2A all params | 24 | 3 | 113.13M | 24802 MiB | 24883 MiB | 97.0% | 4.6759 |
| 4 | I2T vision projector | 64 | 1 | 1.18M | 23291 MiB | 23301 MiB | 94.7% | 2.9710 |
| 5 | I2T all params, restarted | 64 | 1 | 113.13M | 29448 MiB | 29535 MiB | 97.3% | 2.0811 |
| 6 | A2A final all params | 24 | 1 | 113.13M | 24782 MiB | 24843 MiB | 96.9% | 4.9898 |
| 7 | I2T final vision projector | 64 | 1 | 1.18M | 23156 MiB | 23301 MiB | 95.2% | 2.3108 |

## Final Loss Details

```text
Stage1 T2A:
loss 5.0492, text 1.4755, audio 3.5737

Stage2 A2A audio projector:
loss 7.9235, text 1.9974, audio 5.9261

Stage3 A2A all params:
loss 4.6759, text 0.8557, audio 3.8203

Stage4 I2T vision projector:
loss 2.9710, text 2.9710

Stage5 I2T all params:
loss 2.0811, text 2.0811

Stage6 A2A final all params:
loss 4.9898, text 1.3576, audio 3.6322

Stage7 I2T final vision projector:
loss 2.3108, text 2.3108
```

## Key Adjustments

- Stage3 A2A all-param training initially tried `batch_size=32`, hit OOM, and was restarted with `batch_size=24`.
- Stage5 I2T all-param training originally had low memory usage at `batch_size=32`. It was restarted from Stage5 with `batch_size=64`, reaching about 29.5 GiB/GPU peak memory without OOM.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was used to reduce fragmentation risk.
- SwanLab logging was enabled for the final Stage5/6/7 run.

## SwanLab Runs

- Stage5: https://swanlab.cn/@realbtech/MiniMind-O-Full-Train/runs/bvozryscormusca420skl
- Stage6: https://swanlab.cn/@realbtech/MiniMind-O-Full-Train/runs/wuk5dahfyfdxos1cpiwxu
- Stage7: https://swanlab.cn/@realbtech/MiniMind-O-Full-Train/runs/7hbp1qd826nlez1sxorxj

## Final Artifacts

```text
out/sft_full_muon_final_768.pth
checkpoints/sft_full_muon_final_768.pth
checkpoints/sft_full_muon_final_768_resume.pth
```

Important intermediate artifacts:

```text
out/sft_full_muon_t2a_768.pth
out/sft_full_muon_a2a_proj_768.pth
out/sft_full_muon_a2a_full_768.pth
out/sft_full_muon_i2t_proj_768.pth
out/sft_full_muon_i2t_full_768.pth
out/sft_full_muon_a2a_final_768.pth
```

## Effect Assessment

The full training completed successfully and the loss curves show healthy convergence. T2A reduced from high initial loss to 5.0492; A2A all-param training reduced to 4.6759 with audio loss around 3.82; I2T all-param training reached 2.0811 and was the cleanest converging stage. The final Stage6/Stage7 passes served as multimodal consolidation and final projector alignment.

This confirms training-side convergence, but not final generation quality by itself. The next step should run controlled inference and benchmark comparisons on the final checkpoint against the miniTrain baseline and at least one ablation.
