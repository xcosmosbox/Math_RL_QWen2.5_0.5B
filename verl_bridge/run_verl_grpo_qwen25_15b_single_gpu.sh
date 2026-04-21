#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="/home/work/opa_common_mx/wsy/Math_RL_QWen2.5_0.5B"
VERL_ENV="${VERL_ENV_PATH:-/home/work/opa_common_mx/wsy/Search-R1-Qwen3/.venv}"
VERL_REPO="/home/work/opa_common_mx/wsy/Search-R1-Qwen3/verl"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TIMESTAMP}_${EXPERIMENT_NAME:-qwen25_15b_lora_grpo_single_gpu}.log}"

TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/processed/verl/strict_main/train.parquet}"
VALID_FILE="${VALID_FILE:-$ROOT_DIR/data/processed/verl/strict_main/valid.parquet}"
BASE_MODEL="${BASE_MODEL:-/home/work/opa_common_mx/model/Qwen2.5-1.5B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-verl_strict_main_math}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_15b_lora_grpo_single_gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/verl/$EXPERIMENT_NAME}"
USE_LORA="${USE_LORA:-false}"
LORA_ADAPTER="${LORA_ADAPTER:-$ROOT_DIR/outputs/sft/strict_main_qwen25_15b_bs32_ga1}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$OUTPUT_DIR/swanlog}"
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[\"console\",\"file\",\"swanlab\"]}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
VERL_RUNTIME_PATCH="${VERL_RUNTIME_PATCH:-1}"
VERL_DISABLE_TORCHAO="${VERL_DISABLE_TORCHAO:-0}"
REWARD_MANAGER_NAME="${REWARD_MANAGER_NAME:-naive}"
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-true}"
ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
ACTOR_LOSS_AGG_MODE="${ACTOR_LOSS_AGG_MODE:-}"
ACTOR_CLIP_RATIO_LOW="${ACTOR_CLIP_RATIO_LOW:-}"
ACTOR_CLIP_RATIO_HIGH="${ACTOR_CLIP_RATIO_HIGH:-}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-true}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-50}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER:-wsy}}"
TMPDIR="${TMPDIR:-/tmp/${EXPERIMENT_NAME}}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RAY_TMPDIR"
mkdir -p "$TMPDIR"
echo "LOG_FILE=${LOG_FILE}"
export PATH="$VERL_ENV/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:$VERL_REPO:${PYTHONPATH:-}"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard_log"
export VERL_FILE_LOGGER_PATH="$OUTPUT_DIR/metrics.jsonl"
export SWANLAB_MODE
export SWANLAB_LOG_DIR
export RAY_TMPDIR
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export VERL_RUNTIME_PATCH
export VERL_DISABLE_TORCHAO
export TMPDIR
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export CUDA_TMPDIR="$TMPDIR"

if [[ "${SKIP_RAY_STOP:-0}" != "1" ]]; then
  timeout 15 "$VERL_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
fi

MODEL_ARGS=(
  actor_rollout_ref.model.path="$BASE_MODEL"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

if [[ "$USE_LORA" == "true" ]]; then
  MODEL_ARGS+=(
    actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER"
  )
fi

ACTOR_ARGS=(
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.use_kl_loss="$ACTOR_USE_KL_LOSS"
  actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type="$ACTOR_KL_LOSS_TYPE"
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

if [[ "$USE_DYNAMIC_BSZ" == "true" ]]; then
  ACTOR_ARGS+=(
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  )
  if [[ -n "$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" ]]; then
    ACTOR_ARGS+=(
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU"
    )
  fi
  if [[ -n "$LOGPROB_MAX_TOKEN_LEN_PER_GPU" ]]; then
    ACTOR_ARGS+=(
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN_PER_GPU"
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN_PER_GPU"
    )
  fi
else
  ACTOR_ARGS+=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
  )
fi

if [[ -n "$ACTOR_LOSS_AGG_MODE" ]]; then
  ACTOR_ARGS+=(
    actor_rollout_ref.actor.loss_agg_mode="$ACTOR_LOSS_AGG_MODE"
  )
fi

if [[ -n "$ACTOR_CLIP_RATIO_LOW" ]]; then
  ACTOR_ARGS+=(
    actor_rollout_ref.actor.clip_ratio_low="$ACTOR_CLIP_RATIO_LOW"
  )
fi

if [[ -n "$ACTOR_CLIP_RATIO_HIGH" ]]; then
  ACTOR_ARGS+=(
    actor_rollout_ref.actor.clip_ratio_high="$ACTOR_CLIP_RATIO_HIGH"
  )
fi

ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.name="$ROLLOUT_ENGINE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.load_format=safetensors
)

if [[ -n "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" ]]; then
  ROLLOUT_ARGS+=(
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
  )
fi

if [[ "$ROLLOUT_ENGINE" == "vllm" ]]; then
  ROLLOUT_ARGS+=(
    actor_rollout_ref.rollout.layered_summon=True
  )
fi

if [[ "$ROLLOUT_ENGINE" == "sglang" ]]; then
  ROLLOUT_ARGS+=(
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="$SGLANG_ATTENTION_BACKEND"
  )
fi

RESUME_ARGS=(
  trainer.resume_mode="$RESUME_MODE"
)

if [[ -n "$RESUME_FROM_PATH" ]]; then
  RESUME_ARGS+=(
    trainer.resume_from_path="$RESUME_FROM_PATH"
  )
fi

RUNTIME_ENV_ARGS=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_RUNTIME_PATCH='$VERL_RUNTIME_PATCH'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_DISABLE_TORCHAO='$VERL_DISABLE_TORCHAO'"
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$ROOT_DIR:$VERL_REPO"
)

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  RUNTIME_ENV_ARGS+=(
    +ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_API_KEY="$SWANLAB_API_KEY"
  )
fi

if [[ -n "$SWANLAB_MODE" ]]; then
  RUNTIME_ENV_ARGS+=(
    +ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE="$SWANLAB_MODE"
  )
fi

if [[ -n "$SWANLAB_LOG_DIR" ]]; then
  RUNTIME_ENV_ARGS+=(
    +ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR="$SWANLAB_LOG_DIR"
  )
fi

"$VERL_ENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VALID_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  "${MODEL_ARGS[@]}" \
  "${ACTOR_ARGS[@]}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  "${ROLLOUT_ARGS[@]}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  algorithm.use_kl_in_reward=False \
  reward.reward_manager.name="$REWARD_MANAGER_NAME" \
  reward.custom_reward_function.path="$ROOT_DIR/verl_bridge/reward_fn.py" \
  reward.custom_reward_function.name=compute_score \
  "${RUNTIME_ENV_ARGS[@]}" \
  trainer.val_before_train=False \
  trainer.critic_warmup=0 \
  trainer.logger="$LOGGER_BACKENDS" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  2>&1 | tee "$LOG_FILE"
