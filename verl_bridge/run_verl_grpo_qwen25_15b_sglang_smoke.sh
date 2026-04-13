#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/opa_common_mx/wsy/Math_RL_QWen2.5_0.5B"
VERL_ENV_PATH="/home/work/opa_common_mx/wsy/Search-R1-Qwen3/.venv-sglang-migrated"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TIMESTAMP}_qwen25_15b_grpo_smoke_sglang.log}"

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

mkdir -p "$LOG_DIR"
echo "LOG_FILE=${LOG_FILE}"

VERL_ENV_PATH="$VERL_ENV_PATH" \
ROLLOUT_ENGINE=sglang \
USE_LORA=false \
PROJECT_NAME=verl_strict_main_math \
EXPERIMENT_NAME=qwen25_15b_grpo_smoke_sglang_single_gpu \
OUTPUT_DIR="$ROOT_DIR/outputs/verl/qwen25_15b_grpo_smoke_sglang_single_gpu" \
TRAIN_FILE="$ROOT_DIR/data/processed/verl/strict_main/smoke/train.parquet" \
VALID_FILE="$ROOT_DIR/data/processed/verl/strict_main/smoke/valid.parquet" \
TRAIN_BATCH_SIZE=16 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=1024 \
ROLLOUT_N=6 \
PPO_MINI_BATCH_SIZE=8 \
PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4 \
TOTAL_EPOCHS=1 \
TEST_FREQ=100 \
SAVE_FREQ=100 \
GPU_MEMORY_UTILIZATION=0.35 \
bash "$ROOT_DIR/verl_bridge/run_verl_grpo_qwen25_15b_single_gpu.sh" 2>&1 | tee "$LOG_FILE"
