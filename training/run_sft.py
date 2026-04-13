from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from utils.io_utils import ensure_dir, list_json_files, make_temp_dir, replace_dir, resolve_project_root, write_json, write_text, write_yaml
from utils.logging_utils import build_run_metadata, configure_logger, run_streaming_command, write_failure_summary


DEFAULT_PRESET = "strict_main"
PRESETS = {
    "strict_smoke": {
        "dataset_name": "strict_main",
        "train_dataset": "strict_main_train",
        "valid_dataset": "strict_main_valid",
        "dataset_dir": "data/processed",
        "dataset_info_path": "data/processed/dataset_info.json",
        "base_model": "Qwen/Qwen2.5-0.5B",
        "template": "qwen",
        "output_dir": "outputs/sft/strict_smoke",
        "num_train_epochs": 1,
        "cutoff_len": 1024,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "bf16": True,
        "logging_steps": 5,
        "save_steps": 50,
        "eval_steps": 50,
        "warmup_ratio": 0.03,
        "max_samples": 256,
    },
    "strict_main": {
        "dataset_name": "strict_main",
        "train_dataset": "strict_main_train",
        "valid_dataset": "strict_main_valid",
        "dataset_dir": "data/processed",
        "dataset_info_path": "data/processed/dataset_info.json",
        "base_model": "Qwen/Qwen2.5-0.5B",
        "template": "qwen",
        "output_dir": "outputs/sft/strict_main",
        "num_train_epochs": 3,
        "cutoff_len": 1024,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "bf16": True,
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "warmup_ratio": 0.03,
    },
    "relaxed_smoke": {
        "dataset_name": "relaxed_ablation",
        "train_dataset": "relaxed_ablation_train",
        "valid_dataset": "relaxed_ablation_valid",
        "dataset_dir": "data/processed",
        "dataset_info_path": "data/processed/dataset_info.json",
        "base_model": "Qwen/Qwen2.5-0.5B",
        "template": "qwen",
        "output_dir": "outputs/sft/relaxed_smoke",
        "num_train_epochs": 1,
        "cutoff_len": 1024,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "bf16": True,
        "logging_steps": 5,
        "save_steps": 50,
        "eval_steps": 50,
        "warmup_ratio": 0.03,
        "max_samples": 256,
    },
    "relaxed_ablation": {
        "dataset_name": "relaxed_ablation",
        "train_dataset": "relaxed_ablation_train",
        "valid_dataset": "relaxed_ablation_valid",
        "dataset_dir": "data/processed",
        "dataset_info_path": "data/processed/dataset_info.json",
        "base_model": "Qwen/Qwen2.5-0.5B",
        "template": "qwen",
        "output_dir": "outputs/sft/relaxed_ablation",
        "num_train_epochs": 3,
        "cutoff_len": 1024,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "bf16": True,
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "warmup_ratio": 0.03,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA SFT with LLaMA-Factory")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--base-model", dest="base_model")
    parser.add_argument("--template", dest="template")
    parser.add_argument("--per-device-train-batch-size", type=int, dest="per_device_train_batch_size")
    parser.add_argument("--per-device-eval-batch-size", type=int, dest="per_device_eval_batch_size")
    parser.add_argument("--gradient-accumulation-steps", type=int, dest="gradient_accumulation_steps")
    parser.add_argument("--max-samples", type=int, dest="max_samples")
    parser.add_argument("--num-train-epochs", type=float, dest="num_train_epochs")
    parser.add_argument("--output-dir", dest="output_dir")
    return parser.parse_args()


def build_llamafactory_config(project_root: Path, preset_name: str, preset: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(project_root, preset["output_dir"])
    config = {
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "model_name_or_path": preset["base_model"],
        "template": preset["template"],
        "dataset_dir": str(Path(project_root, preset["dataset_dir"])),
        "dataset": preset["train_dataset"],
        "eval_dataset": preset["valid_dataset"],
        "finetuning_type": "lora",
        "lora_target": preset["lora_target"],
        "lora_rank": preset["lora_rank"],
        "lora_alpha": preset["lora_alpha"],
        "lora_dropout": preset["lora_dropout"],
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": preset["per_device_train_batch_size"],
        "per_device_eval_batch_size": preset["per_device_eval_batch_size"],
        "gradient_accumulation_steps": preset["gradient_accumulation_steps"],
        "learning_rate": preset["learning_rate"],
        "num_train_epochs": preset["num_train_epochs"],
        "cutoff_len": preset["cutoff_len"],
        "bf16": preset["bf16"],
        "logging_steps": preset["logging_steps"],
        "save_steps": preset["save_steps"],
        "eval_steps": preset["eval_steps"],
        "warmup_ratio": preset["warmup_ratio"],
        "lr_scheduler_type": "cosine",
        "plot_loss": True,
        "save_only_model": False,
        "report_to": "none",
        "val_size": 0.0,
        "ddp_timeout": 180000,
        "run_name": preset_name,
    }
    if "max_samples" in preset:
        config["max_samples"] = preset["max_samples"]
    return config


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    override_fields = [
        "base_model",
        "template",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "max_samples",
        "num_train_epochs",
        "output_dir",
    ]
    overrides: dict[str, Any] = {}
    for field in override_fields:
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    return overrides


def run_command(config_path: Path, *, project_root: Path, logger: Any) -> Any:
    command = [sys.executable, "-m", "llamafactory.cli", "train", str(config_path)]
    return run_streaming_command(command, cwd=project_root, logger=logger)


def locate_latest_path(base_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(base_dir.rglob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def collect_sft_artifacts(stage_dir: Path) -> dict[str, Any]:
    trainer_state_path = locate_latest_path(stage_dir, "trainer_state.json")
    if trainer_state_path is None:
        raise FileNotFoundError("trainer_state.json was not produced by LLaMA-Factory")

    if trainer_state_path != stage_dir / "trainer_state.json":
        shutil.copyfile(trainer_state_path, stage_dir / "trainer_state.json")
        trainer_state_path = stage_dir / "trainer_state.json"

    with trainer_state_path.open("r", encoding="utf-8") as handle:
        trainer_state = json.load(handle)

    log_history = trainer_state.get("log_history", [])
    loss_values = [item["loss"] for item in log_history if "loss" in item]
    eval_loss_values = [item["eval_loss"] for item in log_history if "eval_loss" in item]
    checkpoint_candidates = sorted(stage_dir.rglob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    best_checkpoint = trainer_state.get("best_model_checkpoint")
    if best_checkpoint is None and checkpoint_candidates:
        best_checkpoint = str(checkpoint_candidates[-1])

    train_metrics = {
        "final_train_loss": float(loss_values[-1]) if loss_values else None,
        "best_eval_loss": float(min(eval_loss_values)) if eval_loss_values else None,
        "num_log_steps": len(log_history),
        "best_model_checkpoint": best_checkpoint,
        "global_step": trainer_state.get("global_step"),
        "epoch": trainer_state.get("epoch"),
    }
    write_json(stage_dir / "train_metrics.json", train_metrics)
    write_json(
        stage_dir / "best_checkpoint_info.json",
        {
            "best_model_checkpoint": best_checkpoint,
            "checkpoint_candidates": [str(path) for path in checkpoint_candidates],
        },
    )
    return {
        "trainer_state_path": str(trainer_state_path),
        "train_metrics": train_metrics,
    }


def run_sft_preset(preset_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    if importlib.util.find_spec("llamafactory") is None:
        raise RuntimeError("llamafactory is required in the target training environment.")

    project_root = resolve_project_root()
    logger = configure_logger("training.run_sft")
    preset = dict(PRESETS[preset_name])
    if overrides:
        preset.update(overrides)
    final_output_dir = Path(project_root, preset["output_dir"])
    temp_output_dir = make_temp_dir(final_output_dir.parent, prefix=f".{final_output_dir.name}_tmp_")
    logs_dir = ensure_dir(temp_output_dir / "logs")
    ensure_dir(temp_output_dir / "checkpoints")

    config = build_llamafactory_config(project_root, preset_name, preset)
    config["output_dir"] = str(temp_output_dir)
    config_path = temp_output_dir / "llamafactory_sft_config.yaml"
    write_yaml(config_path, config)
    write_json(temp_output_dir / "config.json", config)
    write_json(
        temp_output_dir / "run_metadata.json",
        build_run_metadata(preset=preset_name, dataset_info_path=preset["dataset_info_path"], config=config),
    )
    write_text(temp_output_dir / "run_command.txt", f"{sys.executable} -m llamafactory.cli train {config_path}\n")

    try:
        result = run_command(config_path, project_root=project_root, logger=logger)
        write_text(logs_dir / "stdout.log", result.stdout)
        write_text(logs_dir / "stderr.log", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"SFT command failed with exit code {result.returncode}")

        collected = collect_sft_artifacts(temp_output_dir)
        from analysis.build_sft_loss_curve import build_loss_curve
        from analysis.build_sft_comparison import build_sft_comparison

        try:
            build_loss_curve(temp_output_dir / "trainer_state.json", temp_output_dir / "loss_curve.png")
        except ValueError as exc:
            logger.warning("Skipping loss curve generation: %s", exc)
        build_sft_comparison(project_root, temp_output_dir, preset_name)
        write_json(
            temp_output_dir / "artifact_manifest.json",
            {
                "required_files": [
                    "config.json",
                    "run_metadata.json",
                    "run_command.txt",
                    "logs/stdout.log",
                    "trainer_state.json",
                    "train_metrics.json",
                    "best_checkpoint_info.json",
                    "loss_curve.png",
                    "loss_accuracy_comparison.json",
                ],
                "collected": collected,
            },
        )
        replace_dir(temp_output_dir, final_output_dir)
        logger.info("SFT preset %s completed", preset_name)
        return {
            "preset": preset_name,
            "output_dir": str(final_output_dir),
            "artifact_manifest": str(final_output_dir / "artifact_manifest.json"),
        }
    except Exception as exc:
        write_failure_summary(
            temp_output_dir / "failure_summary.json",
            stage="sft",
            error=exc,
            preset=preset_name,
            output_dir=str(final_output_dir),
        )
        raise


def main() -> None:
    args = parse_args()
    run_sft_preset(args.preset, overrides=collect_overrides(args))


if __name__ == "__main__":
    main()
