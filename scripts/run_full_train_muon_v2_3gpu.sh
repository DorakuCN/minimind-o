#!/usr/bin/env bash
# Muon v2 comparison run: same data/optimizer/batches as baseline, tuned stage schedule
# for I2A grounding (Stage4/5) and A2A stability (Stage6).
set -euo pipefail

ROOT_DIR="/home/genesis/Projects/minimind-o"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WEIGHT_PREFIX="${WEIGHT_PREFIX:-sft_full_muon_v2}"
export RUN_GROUP="${RUN_GROUP:-muon_v2}"
export WANDB_PROJECT="${WANDB_PROJECT:-MiniMind-O-Full-Train}"

# Stage schedule changes vs baseline
export EPOCHS_I2T_PROJ="${EPOCHS_I2T_PROJ:-2}"      # baseline: 1
export EPOCHS_I2T_ALL="${EPOCHS_I2T_ALL:-2}"        # baseline: 1
export LR_A2A_FINAL="${LR_A2A_FINAL:-2e-6}"         # baseline: 5e-6
export MUON_LR_A2A_FINAL="${MUON_LR_A2A_FINAL:-0.0005}"  # baseline: 0.001

exec bash "${SCRIPT_DIR}/run_full_train_muon_dense_3gpu.sh" "$@"
