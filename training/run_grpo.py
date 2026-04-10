from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from rewards.grpo_rewards import batch_compute_rewards
from utils.io_utils import ensure_dir, make_temp_dir, read_jsonl, replace_dir, resolve_project_root, write_json, write_text
from utils.logging_utils import build_run_metadata, configure_logger, write_failure_summary


DEFAULT_PRESET = "strict_main_grpo"
PRESETS = {
    "strict_smoke_grpo": {
        "model_stage": "rl",
        "base_sft_model_path": "outputs/sft/strict_smoke",
        "train_path": "data/processed/grpo_strict_main_train.jsonl",
        "valid_path": "data/processed/grpo_strict_main_valid.jsonl",
        "output_dir": "outputs/grpo/strict_smoke",
        "num_generations": 2,
        "generation_batch_size": 8,
        "max_prompt_length": 1024,
        "max_completion_length": 256,
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "beta": 0.04,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "max_train_samples": 64,
        "max_valid_samples": 32,
    },
    "strict_main_grpo": {
        "model_stage": "rl",
        "base_sft_model_path": "outputs/sft/strict_main",
        "train_path": "data/processed/grpo_strict_main_train.jsonl",
        "valid_path": "data/processed/grpo_strict_main_valid.jsonl",
        "output_dir": "outputs/grpo/strict_main",
        "num_generations": 4,
        "generation_batch_size": 4,
        "max_prompt_length": 1024,
        "max_completion_length": 512,
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "beta": 0.04,
        "num_train_epochs": 1,
        "logging_steps": 5,
    },
    "relaxed_ablation_grpo": {
        "model_stage": "rl",
        "base_sft_model_path": "outputs/sft/relaxed_ablation",
        "train_path": "data/processed/grpo_relaxed_ablation_train.jsonl",
        "valid_path": "data/processed/grpo_relaxed_ablation_valid.jsonl",
        "output_dir": "outputs/grpo/relaxed_ablation_grpo",
        "num_generations": 4,
        "generation_batch_size": 4,
        "max_prompt_length": 1024,
        "max_completion_length": 512,
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "beta": 0.04,
        "num_train_epochs": 1,
        "logging_steps": 5,
    },
    "relaxed_smoke_grpo": {
        "model_stage": "rl",
        "base_sft_model_path": "outputs/sft/relaxed_smoke",
        "train_path": "data/processed/grpo_relaxed_ablation_train.jsonl",
        "valid_path": "data/processed/grpo_relaxed_ablation_valid.jsonl",
        "output_dir": "outputs/grpo/relaxed_smoke",
        "num_generations": 2,
        "generation_batch_size": 8,
        "max_prompt_length": 1024,
        "max_completion_length": 256,
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "beta": 0.04,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "max_train_samples": 64,
        "max_valid_samples": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def load_training_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("datasets, trl, transformers, and peft are required in the target training environment.") from exc
    return Dataset, AutoPeftModelForCausalLM, AutoModelForCausalLM, AutoTokenizer, (GRPOConfig, GRPOTrainer)


def load_sft_model(model_path: Path, auto_peft_cls: Any, auto_model_cls: Any) -> Any:
    try:
        return auto_peft_cls.from_pretrained(str(model_path), is_trainable=True, trust_remote_code=True)
    except Exception:
        return auto_model_cls.from_pretrained(str(model_path), trust_remote_code=True)


def make_reward_function(trace_path: Path):
    def reward_function(completions: list[str], sample_id: list[str], target_final_answer: list[str], **_: Any) -> list[float]:
        samples = [
            {"sample_id": current_id, "target_final_answer": current_target}
            for current_id, current_target in zip(sample_id, target_final_answer)
        ]
        rewards = batch_compute_rewards(samples, completions)
        with trace_path.open("a", encoding="utf-8") as handle:
            for reward in rewards:
                handle.write(json.dumps(reward.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return [reward.total_reward for reward in rewards]

    return reward_function


def build_kl_summary(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    kl_values: list[float] = []
    matched_keys: set[str] = set()
    for row in log_history:
        for key, value in row.items():
            if "kl" in key.lower() and isinstance(value, (int, float)):
                kl_values.append(float(value))
                matched_keys.add(key)
    if not kl_values:
        return {"matched_keys": sorted(matched_keys), "count": 0, "mean": None, "max": None, "min": None}
    return {
        "matched_keys": sorted(matched_keys),
        "count": len(kl_values),
        "mean": round(statistics.mean(kl_values), 6),
        "max": max(kl_values),
        "min": min(kl_values),
    }


def run_grpo_preset(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    logger = configure_logger("training.run_grpo")
    preset = PRESETS[preset_name]
    final_output_dir = Path(project_root, preset["output_dir"])
    temp_output_dir = make_temp_dir(final_output_dir.parent, prefix=f".{final_output_dir.name}_tmp_")
    logs_dir = ensure_dir(temp_output_dir / "logs")
    checkpoints_dir = ensure_dir(temp_output_dir / "checkpoints")
    reward_trace_dir = ensure_dir(temp_output_dir / "reward_traces")
    reward_trace_path = reward_trace_dir / "reward_trace.jsonl"

    try:
        dataset_cls, auto_peft_cls, auto_model_cls, auto_tokenizer_cls, trl_classes = load_training_modules()
        grpo_config_cls, grpo_trainer_cls = trl_classes

        model_path = Path(project_root, preset["base_sft_model_path"])
        model = load_sft_model(model_path, auto_peft_cls, auto_model_cls)
        tokenizer = auto_tokenizer_cls.from_pretrained(str(model_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_rows = read_jsonl(Path(project_root, preset["train_path"]))
        valid_rows = read_jsonl(Path(project_root, preset["valid_path"]))
        if "max_train_samples" in preset:
            train_rows = train_rows[: preset["max_train_samples"]]
        if "max_valid_samples" in preset:
            valid_rows = valid_rows[: preset["max_valid_samples"]]
        train_dataset = dataset_cls.from_list(train_rows)
        valid_dataset = dataset_cls.from_list(valid_rows)
        reward_function = make_reward_function(reward_trace_path)

        rollout_config = {
            "num_generations": preset["num_generations"],
            "generation_batch_size": preset["generation_batch_size"],
            "max_prompt_length": preset["max_prompt_length"],
            "max_completion_length": preset["max_completion_length"],
            "beta": preset["beta"],
            "learning_rate": preset["learning_rate"],
            "per_device_train_batch_size": preset["per_device_train_batch_size"],
            "gradient_accumulation_steps": preset["gradient_accumulation_steps"],
            "num_train_epochs": preset["num_train_epochs"],
            "max_train_samples": preset.get("max_train_samples"),
            "max_valid_samples": preset.get("max_valid_samples"),
        }
        write_json(temp_output_dir / "rollout_config.json", rollout_config)
        write_json(temp_output_dir / "config.json", preset)
        write_json(
            temp_output_dir / "run_metadata.json",
            build_run_metadata(
                preset=preset_name,
                model_path=str(model_path),
                train_path=preset["train_path"],
                valid_path=preset["valid_path"],
            ),
        )
        write_text(temp_output_dir / "run_command.txt", f"python -m training.run_grpo --preset {preset_name}\n")

        training_args = grpo_config_cls(
            output_dir=str(checkpoints_dir),
            learning_rate=preset["learning_rate"],
            per_device_train_batch_size=preset["per_device_train_batch_size"],
            gradient_accumulation_steps=preset["gradient_accumulation_steps"],
            logging_steps=preset["logging_steps"],
            beta=preset["beta"],
            num_generations=preset["num_generations"],
            generation_batch_size=preset["generation_batch_size"],
            max_prompt_length=preset["max_prompt_length"],
            max_completion_length=preset["max_completion_length"],
            num_train_epochs=preset["num_train_epochs"],
            report_to=[],
            bf16=True,
        )
        trainer = grpo_trainer_cls(
            model=model,
            reward_funcs=[reward_function],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(str(checkpoints_dir / "final"))
        trainer.save_state()
        trainer.state.save_to_json(str(temp_output_dir / "trainer_state.json"))

        from analysis.build_grpo_reward_summary import build_reward_summary
        from analysis.build_grpo_comparison import build_grpo_comparison

        reward_summary = build_reward_summary(reward_trace_path, temp_output_dir / "reward_summary.json")
        write_text(logs_dir / "reward_summary_preview.txt", json.dumps(reward_summary, ensure_ascii=False, indent=2))
        kl_summary = build_kl_summary(getattr(trainer.state, "log_history", []))
        write_json(temp_output_dir / "kl_summary.json", kl_summary)
        write_json(
            temp_output_dir / "artifact_manifest.json",
            {
                "required_files": [
                    "config.json",
                    "rollout_config.json",
                    "run_metadata.json",
                    "run_command.txt",
                    "reward_traces/reward_trace.jsonl",
                    "reward_summary.json",
                    "kl_summary.json",
                    "trainer_state.json",
                ],
                "reward_summary": reward_summary,
                "kl_summary": kl_summary,
            },
        )
        build_grpo_comparison(project_root, temp_output_dir, preset_name)
        replace_dir(temp_output_dir, final_output_dir)
        logger.info("GRPO preset %s completed", preset_name)
        return {
            "preset": preset_name,
            "output_dir": str(final_output_dir),
            "artifact_manifest": str(final_output_dir / "artifact_manifest.json"),
        }
    except Exception as exc:
        write_failure_summary(
            temp_output_dir / "failure_summary.json",
            stage="grpo",
            error=exc,
            preset=preset_name,
            output_dir=str(final_output_dir),
        )
        raise


def main() -> None:
    args = parse_args()
    run_grpo_preset(args.preset)


if __name__ == "__main__":
    main()
