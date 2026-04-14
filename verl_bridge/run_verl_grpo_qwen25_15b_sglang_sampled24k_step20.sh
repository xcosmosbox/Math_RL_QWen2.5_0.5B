#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/opa_common_mx/wsy/Math_RL_QWen2.5_0.5B"
VERL_ENV_PATH="/home/work/opa_common_mx/wsy/Search-R1-Qwen3/.venv-sglang-migrated"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TIMESTAMP}_qwen25_15b_grpo_sampled24k_step20_sglang.log}"

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

mkdir -p "$LOG_DIR"
echo "LOG_FILE=${LOG_FILE}"

VERL_ENV_PATH="$VERL_ENV_PATH" \
ROLLOUT_ENGINE=sglang \
USE_LORA=false \
BASE_MODEL=/home/work/opa_common_mx/model/Qwen2.5-1.5B-Instruct-grpo-step20 \
PROJECT_NAME=verl_strict_main_math \
EXPERIMENT_NAME=qwen25_15b_grpo_sampled24k_step20_b128_mini128_micro16_u08 \
OUTPUT_DIR="$ROOT_DIR/outputs/verl/qwen25_15b_grpo_sampled24k_step20_b128_mini128_micro16_u08" \
TRAIN_FILE="$ROOT_DIR/data/processed/verl/strict_sampled_24k/train.parquet" \
VALID_FILE="$ROOT_DIR/data/processed/verl/strict_sampled_24k/valid.parquet" \
TRAIN_BATCH_SIZE=256 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=1024 \
ROLLOUT_N=6 \
PPO_MINI_BATCH_SIZE=128 \
PPO_MICRO_BATCH_SIZE_PER_GPU=32 \
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=32 \
TOTAL_EPOCHS=1 \
TEST_FREQ=100 \
SAVE_FREQ=50 \
GPU_MEMORY_UTILIZATION=0.8 \
RESUME_MODE=disable \
bash "$ROOT_DIR/verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh" 2>&1 | tee "$LOG_FILE"
