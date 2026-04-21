#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/opa_common_mx/wsy/Math_RL_QWen2.5_0.5B"
VERL_ENV_PATH="/home/work/opa_common_mx/wsy/Search-R1-Qwen3/.venv-sglang-migrated"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TIMESTAMP}_qwen25_15b_dapo_full_sftmerged_sglang_b512_dyn.log}"

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

mkdir -p "$LOG_DIR"
echo "LOG_FILE=${LOG_FILE}"

VERL_ENV_PATH="$VERL_ENV_PATH" \
ROLLOUT_ENGINE=sglang \
SGLANG_ATTENTION_BACKEND=triton \
USE_LORA=false \
BASE_MODEL=/home/work/opa_common_mx/model/Qwen2.5-1.5B-Instruct-sft-merged-strict-main \
PROJECT_NAME=verl_strict_main_math \
EXPERIMENT_NAME=qwen25_15b_dapo_full_sftmerged_sglang_b512_dyn_u08 \
OUTPUT_DIR="$ROOT_DIR/outputs/verl/qwen25_15b_dapo_full_sftmerged_sglang_b512_dyn_u08" \
TRAIN_FILE="$ROOT_DIR/data/processed/verl/strict_main/train.parquet" \
VALID_FILE="$ROOT_DIR/data/processed/verl/strict_main/valid.parquet" \
TRAIN_BATCH_SIZE=512 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=1024 \
ROLLOUT_N=6 \
PPO_MINI_BATCH_SIZE=128 \
USE_DYNAMIC_BSZ=true \
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=12000 \
LOGPROB_MAX_TOKEN_LEN_PER_GPU=12000 \
ACTOR_PARAM_OFFLOAD=false \
REF_PARAM_OFFLOAD=false \
REWARD_MANAGER_NAME=dapo \
ACTOR_USE_KL_LOSS=false \
ACTOR_LOSS_AGG_MODE=token-mean \
ACTOR_CLIP_RATIO_LOW=0.2 \
ACTOR_CLIP_RATIO_HIGH=0.28 \
TOTAL_EPOCHS=1 \
TEST_FREQ=100 \
SAVE_FREQ=50 \
GPU_MEMORY_UTILIZATION=0.8 \
RESUME_MODE=disable \
bash "$ROOT_DIR/verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh" 2>&1 | tee "$LOG_FILE"
