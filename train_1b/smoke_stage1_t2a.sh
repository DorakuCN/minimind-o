#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/train_1b/configs/dense_1b.env}"

export CONFIG_FILE
export START_STAGE="${START_STAGE:-1}"
export END_STAGE="${END_STAGE:-1}"
export EPOCHS_T2A="${EPOCHS_T2A:-1}"
export DATA_T2A="${DATA_T2A:-../dataset/_calib/sft_t2a_144rows.parquet}"
export VAL_INTERVAL="${VAL_INTERVAL:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
export RUN_ID="${RUN_ID:-smoke_1b_t2a_$(date +%Y%m%d_%H%M%S)}"
export WEIGHT_PREFIX="${WEIGHT_PREFIX:-omni1b_smoke}"
export RUN_GROUP="${RUN_GROUP:-omni1b_smoke}"
export STAGE1_FROM_WEIGHT="${STAGE1_FROM_WEIGHT:-none}"
export USE_WANDB="${USE_WANDB:-0}"

echo "Smoke: stage1 T2A, MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}, DATA_T2A=${DATA_T2A}"
echo "This exercises DDP/model/data/loss on a tiny parquet; it is not a quality run."

exec bash "${ROOT_DIR}/train_1b/run_1b_dense_3gpu.sh" "$@"
