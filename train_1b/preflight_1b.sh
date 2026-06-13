#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/train_1b/configs/dense_1b.env}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

echo "Config: ${CONFIG_FILE}"
echo "Arch: hidden=${HIDDEN_SIZE}, layers=${NUM_HIDDEN_LAYERS}, heads=${NUM_ATTENTION_HEADS}, kv=${NUM_KEY_VALUE_HEADS}, talker=${TALKER_HIDDEN_SIZE}x${NUM_TALKER_HIDDEN_LAYERS}"
echo "MoE: ${USE_MOE} (recommended first 1B run: 0)"

missing=0
check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "MISSING: ${path}" >&2
    missing=1
  else
    echo "OK: ${path}"
  fi
}

check_file "${ROOT_DIR}/dataset/sft_t2a.parquet"
check_file "${ROOT_DIR}/dataset/sft_a2a.parquet"
check_file "${ROOT_DIR}/dataset/sft_i2t.parquet"
check_file "${ROOT_DIR}/dataset/_val/sft_t2a_val.parquet"
check_file "${ROOT_DIR}/dataset/_val/sft_a2a_val.parquet"
check_file "${ROOT_DIR}/dataset/_val/sft_i2t_val.parquet"

for name in sft_t2a sft_a2a sft_i2t; do
  for rank in 00 01 02; do
    check_file "${ROOT_DIR}/dataset/_full_shards/${name}.rank${rank}-of03.parquet"
  done
done

stage1_prefix="${STAGE1_FROM_WEIGHT:-llm}"
if [[ "${stage1_prefix}" != "none" ]]; then
  pretrain="${ROOT_DIR}/out/${stage1_prefix}_${HIDDEN_SIZE}.pth"
  if [[ -f "${pretrain}" ]]; then
    echo "OK: ${pretrain}"
  else
    echo "MISSING 1B pretrain: ${pretrain}" >&2
    echo "Use STAGE1_FROM_WEIGHT=<prefix>, or set STAGE1_FROM_WEIGHT=none / ALLOW_SCRATCH_1B=1 deliberately." >&2
    missing=1
  fi
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate minimind-o
  python "${ROOT_DIR}/train_1b/estimate_params.py" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_hidden_layers "${NUM_HIDDEN_LAYERS}" \
    --num_attention_heads "${NUM_ATTENTION_HEADS}" \
    --num_key_value_heads "${NUM_KEY_VALUE_HEADS}" \
    --head_dim "${HEAD_DIM}" \
    --intermediate_size "${INTERMEDIATE_SIZE}" \
    --talker_hidden_size "${TALKER_HIDDEN_SIZE}" \
    --num_talker_hidden_layers "${NUM_TALKER_HIDDEN_LAYERS}" \
    --talker_num_attention_heads "${TALKER_NUM_ATTENTION_HEADS}" \
    --talker_num_key_value_heads "${TALKER_NUM_KEY_VALUE_HEADS}" \
    --talker_head_dim "${TALKER_HEAD_DIM}" \
    --talker_intermediate_size "${TALKER_INTERMEDIATE_SIZE}" \
    --use_moe "${USE_MOE}"
else
  echo "conda not found; skipped parameter estimate" >&2
fi

DRY_RUN=1 CONFIG_FILE="${CONFIG_FILE}" bash "${ROOT_DIR}/train_1b/run_1b_dense_3gpu.sh" >/tmp/minimind_1b_dryrun.log
echo "Dry-run command expansion: OK (/tmp/minimind_1b_dryrun.log)"

if [[ "${missing}" != "0" ]]; then
  echo "Preflight finished with missing prerequisites." >&2
  exit 2
fi

echo "Preflight OK."
