#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/genesis/Projects/minimind-o"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WEIGHT_PREFIX="${WEIGHT_PREFIX:-sft_full_muon}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/.run_logs/full_train_${WEIGHT_PREFIX}_3gpu_${RUN_ID}}"

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate minimind-o

cd "${ROOT_DIR}/trainer"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MASTER_PORT="${MASTER_PORT:-29560}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-30}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
RANK_SHARDED_DATA="${RANK_SHARDED_DATA:-1}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-7}"
DRY_RUN="${DRY_RUN:-0}"
USE_MOE="${USE_MOE:-0}"

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-MiniMind-O-SFT}"
RUN_GROUP="${RUN_GROUP:-${WEIGHT_PREFIX}}"

STAGE1_FROM_WEIGHT="${STAGE1_FROM_WEIGHT:-llm}"

BATCH_T2A="${BATCH_T2A:-48}"
BATCH_A2A_PROJ="${BATCH_A2A_PROJ:-32}"
BATCH_A2A_FULL="${BATCH_A2A_FULL:-24}"
BATCH_I2T_PROJ="${BATCH_I2T_PROJ:-64}"
BATCH_I2T_FULL="${BATCH_I2T_FULL:-64}"

HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-}"
HEAD_DIM="${HEAD_DIM:-}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-}"
NUM_TALKER_HIDDEN_LAYERS="${NUM_TALKER_HIDDEN_LAYERS:-}"
TALKER_HIDDEN_SIZE="${TALKER_HIDDEN_SIZE:-}"
TALKER_NUM_ATTENTION_HEADS="${TALKER_NUM_ATTENTION_HEADS:-}"
TALKER_NUM_KEY_VALUE_HEADS="${TALKER_NUM_KEY_VALUE_HEADS:-}"
TALKER_HEAD_DIM="${TALKER_HEAD_DIM:-}"
TALKER_INTERMEDIATE_SIZE="${TALKER_INTERMEDIATE_SIZE:-}"
BRIDGE_LAYER="${BRIDGE_LAYER:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

SEQ_T2A="${SEQ_T2A:-512}"
SEQ_A2A="${SEQ_A2A:-1024}"
SEQ_I2T="${SEQ_I2T:-768}"

DATA_T2A="${DATA_T2A:-../dataset/sft_t2a.parquet}"
DATA_A2A="${DATA_A2A:-../dataset/sft_a2a.parquet}"
DATA_I2T="${DATA_I2T:-../dataset/sft_i2t.parquet}"

COMPILE_T2A="${COMPILE_T2A:-1}"
COMPILE_A2A_PROJ="${COMPILE_A2A_PROJ:-0}"
COMPILE_A2A_FULL="${COMPILE_A2A_FULL:-0}"
COMPILE_I2T_PROJ="${COMPILE_I2T_PROJ:-1}"
COMPILE_I2T_ALL="${COMPILE_I2T_ALL:-1}"
COMPILE_A2A_FINAL="${COMPILE_A2A_FINAL:-0}"
COMPILE_I2T_FINAL="${COMPILE_I2T_FINAL:-1}"

# Stage hyperparameters (defaults match baseline full train)
EPOCHS_T2A="${EPOCHS_T2A:-6}"
EPOCHS_A2A_PROJ="${EPOCHS_A2A_PROJ:-1}"
EPOCHS_A2A_FULL="${EPOCHS_A2A_FULL:-3}"
EPOCHS_I2T_PROJ="${EPOCHS_I2T_PROJ:-1}"
EPOCHS_I2T_ALL="${EPOCHS_I2T_ALL:-1}"
EPOCHS_A2A_FINAL="${EPOCHS_A2A_FINAL:-1}"
EPOCHS_I2T_FINAL="${EPOCHS_I2T_FINAL:-1}"

LR_T2A="${LR_T2A:-5e-4}"
LR_A2A_PROJ="${LR_A2A_PROJ:-5e-4}"
LR_A2A_FULL="${LR_A2A_FULL:-5e-5}"
LR_I2T_PROJ="${LR_I2T_PROJ:-5e-5}"
LR_I2T_ALL="${LR_I2T_ALL:-5e-6}"
LR_A2A_FINAL="${LR_A2A_FINAL:-5e-6}"
LR_I2T_FINAL="${LR_I2T_FINAL:-5e-6}"

MUON_LR_T2A="${MUON_LR_T2A:-0.02}"
MUON_LR_A2A_PROJ="${MUON_LR_A2A_PROJ:-0.005}"
MUON_LR_A2A_FULL="${MUON_LR_A2A_FULL:-0.002}"
MUON_LR_I2T_PROJ="${MUON_LR_I2T_PROJ:-0.005}"
MUON_LR_I2T_ALL="${MUON_LR_I2T_ALL:-0.001}"
MUON_LR_A2A_FINAL="${MUON_LR_A2A_FINAL:-0.001}"
MUON_LR_I2T_FINAL="${MUON_LR_I2T_FINAL:-0.001}"

WARMUP_RATIO="${WARMUP_RATIO:-0}"
LOSS_NORM="${LOSS_NORM:-local}"
RVQ_LAYER_WEIGHTS="${RVQ_LAYER_WEIGHTS:-}"
AUDIO_STOP_WEIGHT="${AUDIO_STOP_WEIGHT:-10}"
VAL_INTERVAL="${VAL_INTERVAL:-0}"
VAL_DATA_T2A="${VAL_DATA_T2A:-}"
VAL_DATA_A2A="${VAL_DATA_A2A:-}"
VAL_DATA_I2T="${VAL_DATA_I2T:-}"
FINITE_GUARD="${FINITE_GUARD:-1}"
FINITE_GUARD_LOGITS_INTERVAL="${FINITE_GUARD_LOGITS_INTERVAL:-0}"
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-1}"

phase0_args=()
if [[ "${WARMUP_RATIO}" != "0" && -n "${WARMUP_RATIO}" ]]; then
  phase0_args+=(--warmup_ratio "${WARMUP_RATIO}")
fi

model_args=(
  --hidden_size "${HIDDEN_SIZE}"
  --num_hidden_layers "${NUM_HIDDEN_LAYERS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
)
append_optional_model_arg() {
  local value="$1"
  local flag="$2"
  if [[ -n "${value}" ]]; then
    model_args+=("${flag}" "${value}")
  fi
}
append_optional_model_arg "${NUM_ATTENTION_HEADS}" --num_attention_heads
append_optional_model_arg "${NUM_KEY_VALUE_HEADS}" --num_key_value_heads
append_optional_model_arg "${HEAD_DIM}" --head_dim
append_optional_model_arg "${INTERMEDIATE_SIZE}" --intermediate_size
append_optional_model_arg "${NUM_TALKER_HIDDEN_LAYERS}" --num_talker_hidden_layers
append_optional_model_arg "${TALKER_HIDDEN_SIZE}" --talker_hidden_size
append_optional_model_arg "${TALKER_NUM_ATTENTION_HEADS}" --talker_num_attention_heads
append_optional_model_arg "${TALKER_NUM_KEY_VALUE_HEADS}" --talker_num_key_value_heads
append_optional_model_arg "${TALKER_HEAD_DIM}" --talker_head_dim
append_optional_model_arg "${TALKER_INTERMEDIATE_SIZE}" --talker_intermediate_size
append_optional_model_arg "${BRIDGE_LAYER}" --bridge_layer
if [[ "${LOSS_NORM}" == "global" ]]; then
  phase0_args+=(--loss_norm global)
fi
if [[ -n "${RVQ_LAYER_WEIGHTS}" ]]; then
  phase0_args+=(--rvq_layer_weights "${RVQ_LAYER_WEIGHTS}")
fi
if [[ "${AUDIO_STOP_WEIGHT}" != "10" ]]; then
  phase0_args+=(--audio_stop_weight "${AUDIO_STOP_WEIGHT}")
fi

stage_val_args() {
  local val_path="$1"
  local -n _out="$2"
  _out=()
  if [[ -n "${val_path}" && "${VAL_INTERVAL}" != "0" ]]; then
    _out+=(--val_data_path "${val_path}" --val_interval "${VAL_INTERVAL}")
  fi
}

wandb_args=()
if [[ "${USE_WANDB}" == "1" ]]; then
  wandb_args=(--use_wandb --wandb_project "${WANDB_PROJECT}")
fi

monitor_gpu() {
  printf 'timestamp,index,memory.used [MiB],utilization.gpu [%%],temperature.gpu,power.draw [W]\n'
  while true; do
    timestamp="$(date '+%F %T')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits | while IFS= read -r line; do
        printf '%s,%s\n' "${timestamp}" "${line}"
      done
    sleep "${GPU_LOG_INTERVAL}"
  done
}

if [[ "${DRY_RUN}" != "1" ]]; then
  monitor_gpu > "${LOG_DIR}/gpu_telemetry.csv" 2>&1 &
  GPU_MONITOR_PID=$!
  trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT
fi

run_stage() {
  local stage_name="$1"
  shift
  echo "[$(date '+%F %T')] START ${stage_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN command: '
    printf '%q ' "$@"
    echo
    echo "[$(date '+%F %T')] DRY_RUN SKIP ${stage_name}"
    return 0
  fi
  "$@" 2>&1 | tee "${LOG_DIR}/${stage_name}.log"
  echo "[$(date '+%F %T')] DONE ${stage_name}"
}

run_stage_numbered() {
  local stage_number="$1"
  shift
  local stage_name="$1"
  shift
  if (( stage_number < START_STAGE || stage_number > END_STAGE )); then
    echo "[$(date '+%F %T')] SKIP ${stage_name}"
    return 0
  fi
  run_stage "${stage_name}" "$@"
}

common_args=(
  --optimizer muon
  --accumulation_steps "${ACCUMULATION_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --log_interval "${LOG_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --rank_sharded_data "${RANK_SHARDED_DATA}"
  --ddp_broadcast_buffers "${DDP_BROADCAST_BUFFERS}"
  --finite_guard "${FINITE_GUARD}"
  --finite_guard_logits_interval "${FINITE_GUARD_LOGITS_INTERVAL}"
  --use_moe "${USE_MOE}"
  "${model_args[@]}"
)

launch=(
  torchrun
  --master_port "${MASTER_PORT}"
  --nproc_per_node "${NPROC_PER_NODE}"
  train_sft_omni.py
)

echo "Run ID: ${RUN_ID}"
echo "Weight prefix: ${WEIGHT_PREFIX}"
echo "Run group: ${RUN_GROUP}"
echo "Logs: ${LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "RANK_SHARDED_DATA=${RANK_SHARDED_DATA}"
echo "START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE}"
echo "DRY_RUN=${DRY_RUN}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "USE_MOE=${USE_MOE}"
echo "Batches: T2A=${BATCH_T2A}, A2A_PROJ=${BATCH_A2A_PROJ}, A2A_FULL=${BATCH_A2A_FULL}, I2T_PROJ=${BATCH_I2T_PROJ}, I2T_FULL=${BATCH_I2T_FULL}"
echo "Accumulation: ${ACCUMULATION_STEPS}"
echo "Max train steps per trainer call: ${MAX_TRAIN_STEPS}"
echo "Arch: hidden=${HIDDEN_SIZE}, layers=${NUM_HIDDEN_LAYERS}, heads=${NUM_ATTENTION_HEADS:-default}, kv=${NUM_KEY_VALUE_HEADS:-default}, ffn=${INTERMEDIATE_SIZE:-default}, talker_hidden=${TALKER_HIDDEN_SIZE:-default}, checkpointing=${GRADIENT_CHECKPOINTING}"
echo "Seq lens: T2A=${SEQ_T2A}, A2A=${SEQ_A2A}, I2T=${SEQ_I2T}"
echo "Compile: T2A=${COMPILE_T2A}, A2A_PROJ=${COMPILE_A2A_PROJ}, A2A_FULL=${COMPILE_A2A_FULL}, I2T_PROJ=${COMPILE_I2T_PROJ}, I2T_ALL=${COMPILE_I2T_ALL}, A2A_FINAL=${COMPILE_A2A_FINAL}, I2T_FINAL=${COMPILE_I2T_FINAL}"
echo "Phase0: WARMUP_RATIO=${WARMUP_RATIO}, LOSS_NORM=${LOSS_NORM}, VAL_INTERVAL=${VAL_INTERVAL}"
echo "Finite guard: FINITE_GUARD=${FINITE_GUARD}, FINITE_GUARD_LOGITS_INTERVAL=${FINITE_GUARD_LOGITS_INTERVAL}"
echo "DDP: DDP_BROADCAST_BUFFERS=${DDP_BROADCAST_BUFFERS}"

stage1_val=()
stage_val_args "${VAL_DATA_T2A}" stage1_val
run_stage_numbered 1 "01_t2a_all" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage1_val[@]}" \
  --learning_rate "${LR_T2A}" \
  --muon_lr "${MUON_LR_T2A}" \
  --data_path "${DATA_T2A}" \
  --epochs "${EPOCHS_T2A}" \
  --batch_size "${BATCH_T2A}" \
  --use_compile "${COMPILE_T2A}" \
  --from_weight "${STAGE1_FROM_WEIGHT}" \
  --from_resume "${RESUME_STAGE1:-0}" \
  --save_weight "${WEIGHT_PREFIX}_t2a" \
  --max_seq_len "${SEQ_T2A}" \
  --wandb_run_name "${RUN_GROUP}_01_t2a_all" \
  --metrics_path "${LOG_DIR}/01_t2a_all.metrics.json" \
  "${wandb_args[@]}"

stage2_val=()
stage_val_args "${VAL_DATA_A2A}" stage2_val
run_stage_numbered 2 "02_a2a_audio_proj" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage2_val[@]}" \
  --learning_rate "${LR_A2A_PROJ}" \
  --muon_lr "${MUON_LR_A2A_PROJ}" \
  --data_path "${DATA_A2A}" \
  --epochs "${EPOCHS_A2A_PROJ}" \
  --batch_size "${BATCH_A2A_PROJ}" \
  --use_compile "${COMPILE_A2A_PROJ}" \
  --from_weight "${WEIGHT_PREFIX}_t2a" \
  --from_resume "${RESUME_STAGE2:-0}" \
  --save_weight "${WEIGHT_PREFIX}_a2a_proj" \
  --max_seq_len "${SEQ_A2A}" \
  --mode audio_proj \
  --wandb_run_name "${RUN_GROUP}_02_a2a_audio_proj" \
  --metrics_path "${LOG_DIR}/02_a2a_audio_proj.metrics.json" \
  "${wandb_args[@]}"

stage3_val=()
stage_val_args "${VAL_DATA_A2A}" stage3_val
run_stage_numbered 3 "03_a2a_all" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage3_val[@]}" \
  --learning_rate "${LR_A2A_FULL}" \
  --muon_lr "${MUON_LR_A2A_FULL}" \
  --data_path "${DATA_A2A}" \
  --epochs "${EPOCHS_A2A_FULL}" \
  --batch_size "${BATCH_A2A_FULL}" \
  --use_compile "${COMPILE_A2A_FULL}" \
  --from_weight "${WEIGHT_PREFIX}_a2a_proj" \
  --from_resume "${RESUME_STAGE3:-0}" \
  --save_weight "${WEIGHT_PREFIX}_a2a_full" \
  --max_seq_len "${SEQ_A2A}" \
  --wandb_run_name "${RUN_GROUP}_03_a2a_all" \
  --metrics_path "${LOG_DIR}/03_a2a_all.metrics.json" \
  "${wandb_args[@]}"

stage4_val=()
stage_val_args "${VAL_DATA_I2T}" stage4_val
run_stage_numbered 4 "04_i2t_vision_proj" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage4_val[@]}" \
  --learning_rate "${LR_I2T_PROJ}" \
  --muon_lr "${MUON_LR_I2T_PROJ}" \
  --data_path "${DATA_I2T}" \
  --epochs "${EPOCHS_I2T_PROJ}" \
  --batch_size "${BATCH_I2T_PROJ}" \
  --use_compile "${COMPILE_I2T_PROJ}" \
  --from_weight "${WEIGHT_PREFIX}_a2a_full" \
  --from_resume "${RESUME_STAGE4:-0}" \
  --save_weight "${WEIGHT_PREFIX}_i2t_proj" \
  --max_seq_len "${SEQ_I2T}" \
  --mode vision_proj \
  --wandb_run_name "${RUN_GROUP}_04_i2t_vision_proj" \
  --metrics_path "${LOG_DIR}/04_i2t_vision_proj.metrics.json" \
  "${wandb_args[@]}"

stage5_val=()
stage_val_args "${VAL_DATA_I2T}" stage5_val
run_stage_numbered 5 "05_i2t_all" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage5_val[@]}" \
  --learning_rate "${LR_I2T_ALL}" \
  --muon_lr "${MUON_LR_I2T_ALL}" \
  --data_path "${DATA_I2T}" \
  --epochs "${EPOCHS_I2T_ALL}" \
  --batch_size "${BATCH_I2T_FULL}" \
  --use_compile "${COMPILE_I2T_ALL}" \
  --from_weight "${WEIGHT_PREFIX}_i2t_proj" \
  --from_resume "${RESUME_STAGE5:-0}" \
  --save_weight "${WEIGHT_PREFIX}_i2t_full" \
  --max_seq_len "${SEQ_I2T}" \
  --wandb_run_name "${RUN_GROUP}_05_i2t_all" \
  --metrics_path "${LOG_DIR}/05_i2t_all.metrics.json" \
  "${wandb_args[@]}"

stage6_val=()
stage_val_args "${VAL_DATA_A2A}" stage6_val
run_stage_numbered 6 "06_a2a_final_all" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage6_val[@]}" \
  --learning_rate "${LR_A2A_FINAL}" \
  --muon_lr "${MUON_LR_A2A_FINAL}" \
  --data_path "${DATA_A2A}" \
  --epochs "${EPOCHS_A2A_FINAL}" \
  --batch_size "${BATCH_A2A_FULL}" \
  --use_compile "${COMPILE_A2A_FINAL}" \
  --from_weight "${WEIGHT_PREFIX}_i2t_full" \
  --from_resume "${RESUME_STAGE6:-0}" \
  --save_weight "${WEIGHT_PREFIX}_a2a_final" \
  --max_seq_len "${SEQ_A2A}" \
  --wandb_run_name "${RUN_GROUP}_06_a2a_final_all" \
  --metrics_path "${LOG_DIR}/06_a2a_final_all.metrics.json" \
  "${wandb_args[@]}"

stage7_val=()
stage_val_args "${VAL_DATA_I2T}" stage7_val
run_stage_numbered 7 "07_i2t_final_vision_proj" \
  "${launch[@]}" \
  "${common_args[@]}" \
  "${phase0_args[@]}" \
  "${stage7_val[@]}" \
  --learning_rate "${LR_I2T_FINAL}" \
  --muon_lr "${MUON_LR_I2T_FINAL}" \
  --data_path "${DATA_I2T}" \
  --epochs "${EPOCHS_I2T_FINAL}" \
  --batch_size "${BATCH_I2T_PROJ}" \
  --use_compile "${COMPILE_I2T_FINAL}" \
  --from_weight "${WEIGHT_PREFIX}_a2a_final" \
  --from_resume "${RESUME_STAGE7:-0}" \
  --save_weight "${WEIGHT_PREFIX}_final" \
  --max_seq_len "${SEQ_I2T}" \
  --mode vision_proj \
  --wandb_run_name "${RUN_GROUP}_07_i2t_final_vision_proj" \
  --metrics_path "${LOG_DIR}/07_i2t_final_vision_proj.metrics.json" \
  "${wandb_args[@]}"

echo "[$(date '+%F %T')] Full Muon dense 3GPU training finished (${WEIGHT_PREFIX})"
