#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/work/opa_common_mx/wsy/Math_RL_QWen2.5_0.5B"
COMMON_SCRIPTS_DIR="/home/work/opa_common_mx/wsy/common_scripts"
COMMON_SUBMIT="${COMMON_SUBMIT:-${COMMON_SCRIPTS_DIR}/submit_ais.sh}"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export AIS_TRAIN_SCRIPT="${AIS_TRAIN_SCRIPT:-$ROOT_DIR/verl_bridge/run_verl_grpo_qwen25_15b_sglang_full_sft_b512_dyn24k.sh}"
export VERL_ROOT="${VERL_ROOT:-/home/work/opa_common_mx/wsy/Search-R1-Qwen3/verl}"
export VENV_PATH="${VENV_PATH:-/home/work/opa_common_mx/wsy/Search-R1-Qwen3/.venv-sglang-migrated}"
export SKIP_AIS_REQUIREMENTS="${SKIP_AIS_REQUIREMENTS:-1}"
export AIS_LOG_DIR="${AIS_LOG_DIR:-$ROOT_DIR/logs/ais_qwen25_15b_grpo_dyn24k}"
export AIS_LOG_BASENAME="${AIS_LOG_BASENAME:-qwen25_15b_grpo_full_sftmerged_sglang_b512_dyn24k_u08}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-1}"

export EXPERIMENT_ID="${EXPERIMENT_ID:-383380}"
export INSTANCE_COUNT="${INSTANCE_COUNT:-1}"
export RUN_DESC="${RUN_DESC:-qwen25_15b_grpo_full_sftmerged_sglang_b512_dyn24k_u08}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export FILESYSTEM_ID="${FILESYSTEM_ID:-43724}"

if [[ -z "${SWANLAB_API_KEY:-}" && -f "${HOME}/.swanlab/.netrc" ]]; then
  export SWANLAB_API_KEY="$(
    python - <<'PY'
import netrc
import os

path = os.path.expanduser("~/.swanlab/.netrc")
try:
    hosts = netrc.netrc(path).hosts
except Exception:
    hosts = {}

for _host, auth in hosts.items():
    if not auth:
        continue
    _login, _account, password = auth
    if password:
        print(password)
        break
PY
  )"
fi

ENTRY_SCRIPT="${ENTRY_SCRIPT:-${COMMON_SCRIPTS_DIR}/run_verl_ais.sh}"
EXPORT_ENV_VARS="${EXPORT_ENV_VARS:-SKIP_AIS_REQUIREMENTS VERL_ROOT VENV_PATH PROJECT_ROOT AIS_TRAIN_SCRIPT AIS_LOG_DIR AIS_LOG_BASENAME RAY_NUM_GPUS_PER_NODE SWANLAB_API_KEY SWANLAB_MODE}"

INSTANCE_COUNT="${INSTANCE_COUNT}" \
RUN_DESC="${RUN_DESC}" \
ENTRY_SCRIPT="${ENTRY_SCRIPT}" \
EXPORT_ENV_VARS="${EXPORT_ENV_VARS}" \
bash "${COMMON_SUBMIT}" "$@"
