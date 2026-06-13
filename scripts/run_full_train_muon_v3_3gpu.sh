#!/usr/bin/env bash
# Run C: v2 schedule + Phase-0 training correctness fixes.
set -euo pipefail

ROOT_DIR="/home/genesis/Projects/minimind-o"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WEIGHT_PREFIX="${WEIGHT_PREFIX:-sft_full_muon_v3}"
export RUN_GROUP="${RUN_GROUP:-muon_v3}"
export WANDB_PROJECT="${WANDB_PROJECT:-MiniMind-O-Full-Train}"
export DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-0}"

# v2 schedule
export EPOCHS_I2T_PROJ="${EPOCHS_I2T_PROJ:-2}"
export EPOCHS_I2T_ALL="${EPOCHS_I2T_ALL:-2}"
export LR_A2A_FINAL="${LR_A2A_FINAL:-2e-6}"
export MUON_LR_A2A_FINAL="${MUON_LR_A2A_FINAL:-0.0005}"

# Phase-0 fixes
export WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
export LOSS_NORM="${LOSS_NORM:-global}"
export RVQ_LAYER_WEIGHTS="${RVQ_LAYER_WEIGHTS:-2,1.5,1.2,1,0.8,0.7,0.6,0.5}"
export AUDIO_STOP_WEIGHT="${AUDIO_STOP_WEIGHT:-10}"
export VAL_INTERVAL="${VAL_INTERVAL:-500}"
export VAL_DATA_T2A="${VAL_DATA_T2A:-../dataset/_val/sft_t2a_val.parquet}"
export VAL_DATA_A2A="${VAL_DATA_A2A:-../dataset/_val/sft_a2a_val.parquet}"
export VAL_DATA_I2T="${VAL_DATA_I2T:-../dataset/_val/sft_i2t_val.parquet}"

exec bash "${SCRIPT_DIR}/run_full_train_muon_dense_3gpu.sh" "$@"
