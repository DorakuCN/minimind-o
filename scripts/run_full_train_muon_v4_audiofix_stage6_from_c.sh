#!/usr/bin/env bash
# Run D-b fast isolation: copy Run C Stage5 and re-run Stage6 only.
set -euo pipefail

ROOT_DIR="/home/genesis/Projects/minimind-o"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_STAGE5_ARTIFACT="${SRC_STAGE5_ARTIFACT:-${ROOT_DIR}/out/sft_full_muon_v3_i2t_full_768.pth}"
DST_STAGE5_ARTIFACT="${DST_STAGE5_ARTIFACT:-${ROOT_DIR}/out/sft_full_muon_v4_audiofix_i2t_full_768.pth}"

if [[ ! -f "${SRC_STAGE5_ARTIFACT}" ]]; then
  echo "Missing Run C Stage5 artifact: ${SRC_STAGE5_ARTIFACT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${DST_STAGE5_ARTIFACT}")"
if [[ -f "${DST_STAGE5_ARTIFACT}" ]]; then
  if [[ "${FORCE_COPY_STAGE5:-0}" == "1" ]]; then
    cp -f "${SRC_STAGE5_ARTIFACT}" "${DST_STAGE5_ARTIFACT}"
    echo "Overwrote Stage5 artifact copy: ${DST_STAGE5_ARTIFACT}"
  elif cmp -s "${SRC_STAGE5_ARTIFACT}" "${DST_STAGE5_ARTIFACT}"; then
    echo "Stage5 artifact copy already exists and matches: ${DST_STAGE5_ARTIFACT}"
  else
    echo "Destination Stage5 artifact exists but differs: ${DST_STAGE5_ARTIFACT}" >&2
    echo "Set FORCE_COPY_STAGE5=1 to overwrite it." >&2
    exit 1
  fi
else
  cp "${SRC_STAGE5_ARTIFACT}" "${DST_STAGE5_ARTIFACT}"
  echo "Copied Stage5 artifact: ${SRC_STAGE5_ARTIFACT} -> ${DST_STAGE5_ARTIFACT}"
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${SRC_STAGE5_ARTIFACT}" "${DST_STAGE5_ARTIFACT}"
fi

export WEIGHT_PREFIX="${WEIGHT_PREFIX:-sft_full_muon_v4_audiofix}"
export RUN_GROUP="${RUN_GROUP:-muon_v4_audiofix_stage6_from_c}"
export WANDB_PROJECT="${WANDB_PROJECT:-MiniMind-O-Full-Train}"
export USE_WANDB="${USE_WANDB:-1}"

export START_STAGE=6
export END_STAGE=6
export EPOCHS_A2A_FINAL="${EPOCHS_A2A_FINAL:-2}"
export LR_A2A_FINAL="${LR_A2A_FINAL:-5e-6}"
export MUON_LR_A2A_FINAL="${MUON_LR_A2A_FINAL:-0.001}"
export DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-0}"

export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_RunD_stage6_fromC_ddpbuf${DDP_BROADCAST_BUFFERS}}"
export LOG_DIR="${LOG_DIR:-${ROOT_DIR}/.run_logs/full_train_${WEIGHT_PREFIX}_RunD_stage6_fromC_3gpu_${RUN_ID}}"

echo "Run D-b fast isolation"
echo "SRC_STAGE5_ARTIFACT=${SRC_STAGE5_ARTIFACT}"
echo "DST_STAGE5_ARTIFACT=${DST_STAGE5_ARTIFACT}"
echo "WEIGHT_PREFIX=${WEIGHT_PREFIX}"
echo "RUN_GROUP=${RUN_GROUP}"
echo "START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE}"
echo "EPOCHS_A2A_FINAL=${EPOCHS_A2A_FINAL}"
echo "LR_A2A_FINAL=${LR_A2A_FINAL}"
echo "MUON_LR_A2A_FINAL=${MUON_LR_A2A_FINAL}"
echo "DDP_BROADCAST_BUFFERS=${DDP_BROADCAST_BUFFERS}"
echo "LOG_DIR=${LOG_DIR}"

exec bash "${SCRIPT_DIR}/run_full_train_muon_v3_3gpu.sh" "$@"
