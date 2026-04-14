#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/opa_common_mx/wsy/Math_RL_QWen2.5_0.5B"
COMMON_SCRIPTS_DIR="/home/work/opa_common_mx/wsy/common_scripts"
ENTRY_SCRIPT="${ROOT_DIR}/verl_bridge/run_verl_grpo_qwen25_15b_sglang_sampled24k_step20.sh"

export EXPERIMENT_ID="${EXPERIMENT_ID:-383380}"
export INSTANCE_COUNT="${INSTANCE_COUNT:-1}"
export RUN_DESC="${RUN_DESC:-qwen25_15b_grpo_sampled24k_step20_b128_mini128_micro16_u08}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export EXPORT_ENV_VARS="${EXPORT_ENV_VARS:-SWANLAB_API_KEY SWANLAB_MODE}"
export FILESYSTEM_ID="${FILESYSTEM_ID:-43724}"
export ENTRY_SCRIPT

cd "${COMMON_SCRIPTS_DIR}"
bash "${COMMON_SCRIPTS_DIR}/submit_ais.sh"
