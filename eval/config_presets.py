from __future__ import annotations

from datetime import date
from pathlib import Path

from utils.io_utils import resolve_project_root


RUN_DATE = date.today().isoformat()
PROJECT_ROOT = resolve_project_root()
COMMON_GEN_KWARGS = {
    "temperature": 0,
    "do_sample": False,
    "max_gen_toks": 512,
}
TASK_SPECS = {
    "math500": {
        "task_id": "hendrycks_math500",
        "benchmark": "MATH-500",
        "apply_chat_template": True,
        "batch_size": 8,
        "gen_kwargs": COMMON_GEN_KWARGS,
    },
    "gsm8k": {
        "task_id": "gsm8k_cot",
        "benchmark": "GSM8K",
        "apply_chat_template": True,
        "batch_size": 8,
        "gen_kwargs": COMMON_GEN_KWARGS,
    },
    "hellaswag": {
        "task_id": "hellaswag",
        "benchmark": "HellaSwag",
        "apply_chat_template": False,
        "batch_size": 64,
        "gen_kwargs": COMMON_GEN_KWARGS,
    },
}


def _output_root(model_stage: str) -> str:
    return str(Path(PROJECT_ROOT, "eval", "outputs", f"{RUN_DATE}_{model_stage}"))


DEFAULT_PRESET = "base_smoke"
PRESETS = {
    "base_smoke": {
        "model_stage": "base",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "peft_path": None,
        "tasks": ["math500", "gsm8k", "hellaswag"],
        "output_root": _output_root("base"),
        "device": "cuda:0",
        "limit": 50,
        "log_samples": True,
    },
    "base_full": {
        "model_stage": "base",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "peft_path": None,
        "tasks": ["math500", "gsm8k", "hellaswag"],
        "output_root": _output_root("base"),
        "device": "cuda:0",
        "limit": None,
        "log_samples": True,
    },
    "sft_smoke": {
        "model_stage": "sft",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "peft_path": str(Path(PROJECT_ROOT, "outputs", "sft", "strict_smoke")),
        "tasks": ["math500", "gsm8k", "hellaswag"],
        "output_root": _output_root("sft"),
        "device": "cuda:0",
        "limit": 50,
        "log_samples": True,
    },
    "sft_full": {
        "model_stage": "sft",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "peft_path": str(Path(PROJECT_ROOT, "outputs", "sft", "strict_main")),
        "tasks": ["math500", "gsm8k", "hellaswag"],
        "output_root": _output_root("sft"),
        "device": "cuda:0",
        "limit": None,
        "log_samples": True,
    },
    "rl_smoke": {
        "model_stage": "rl",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "peft_path": str(Path(PROJECT_ROOT, "outputs", "grpo", "strict_smoke", "checkpoints", "final")),
        "tasks": ["math500", "gsm8k", "hellaswag"],
        "output_root": _output_root("rl"),
        "device": "cuda:0",
        "limit": 50,
        "log_samples": True,
    },
    "rl_full": {
        "model_stage": "rl",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "peft_path": str(Path(PROJECT_ROOT, "outputs", "grpo", "strict_main", "checkpoints", "final")),
        "tasks": ["math500", "gsm8k", "hellaswag"],
        "output_root": _output_root("rl"),
        "device": "cuda:0",
        "limit": None,
        "log_samples": True,
    },
}
