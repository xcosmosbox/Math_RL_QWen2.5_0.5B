from __future__ import annotations

import argparse
from pathlib import Path

from training.prompt_templates import build_math_instruction_prompt
from utils.io_utils import read_jsonl, resolve_project_root, write_json, write_jsonl
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "all"
PRESETS = {
    "all": [
        {
            "dataset_name": "strict_main_grpo",
            "train_path": "data/processed/filtered_strict_train.jsonl",
            "valid_path": "data/processed/filtered_strict_valid.jsonl",
            "train_output": "data/processed/grpo_strict_main_train.jsonl",
            "valid_output": "data/processed/grpo_strict_main_valid.jsonl",
        },
        {
            "dataset_name": "relaxed_ablation_grpo",
            "train_path": "data/processed/filtered_relaxed_train.jsonl",
            "valid_path": "data/processed/filtered_relaxed_valid.jsonl",
            "train_output": "data/processed/grpo_relaxed_ablation_train.jsonl",
            "valid_output": "data/processed/grpo_relaxed_ablation_valid.jsonl",
        },
        {
            "dataset_name": "bigmath_verified_grpo",
            "train_path": "data/processed/bigmath_verified_clean_train.jsonl",
            "valid_path": "data/processed/bigmath_verified_clean_valid.jsonl",
            "train_output": "data/processed/grpo_bigmath_verified_train.jsonl",
            "valid_output": "data/processed/grpo_bigmath_verified_valid.jsonl",
        },
    ],
    "strict_main_grpo": [
        {
            "dataset_name": "strict_main_grpo",
            "train_path": "data/processed/filtered_strict_train.jsonl",
            "valid_path": "data/processed/filtered_strict_valid.jsonl",
            "train_output": "data/processed/grpo_strict_main_train.jsonl",
            "valid_output": "data/processed/grpo_strict_main_valid.jsonl",
        }
    ],
    "relaxed_ablation_grpo": [
        {
            "dataset_name": "relaxed_ablation_grpo",
            "train_path": "data/processed/filtered_relaxed_train.jsonl",
            "valid_path": "data/processed/filtered_relaxed_valid.jsonl",
            "train_output": "data/processed/grpo_relaxed_ablation_train.jsonl",
            "valid_output": "data/processed/grpo_relaxed_ablation_valid.jsonl",
        }
    ],
    "bigmath_verified_grpo": [
        {
            "dataset_name": "bigmath_verified_grpo",
            "train_path": "data/processed/bigmath_verified_clean_train.jsonl",
            "valid_path": "data/processed/bigmath_verified_clean_valid.jsonl",
            "train_output": "data/processed/grpo_bigmath_verified_train.jsonl",
            "valid_output": "data/processed/grpo_bigmath_verified_valid.jsonl",
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GRPO datasets")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def build_grpo_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": record["sample_id"],
        "prompt": build_math_instruction_prompt(str(record["prompt"])),
        "raw_prompt": record["prompt"],
        "target_final_answer": record["target_final_answer"],
        "difficulty": record["difficulty"],
        "topic": record["topic"],
        "task_type": record["task_type"],
        "split": record["split"],
    }


def export_variant(project_root: Path, spec: dict[str, str]) -> dict[str, object]:
    train_records = [row for row in read_jsonl(Path(project_root, spec["train_path"])) if row["is_verifiable"]]
    valid_records = [row for row in read_jsonl(Path(project_root, spec["valid_path"])) if row["is_verifiable"]]
    train_payload = [build_grpo_record(row) for row in train_records]
    valid_payload = [build_grpo_record(row) for row in valid_records]
    write_jsonl(Path(project_root, spec["train_output"]), train_payload)
    write_jsonl(Path(project_root, spec["valid_output"]), valid_payload)
    return {
        "dataset_name": spec["dataset_name"],
        "train_count": len(train_payload),
        "valid_count": len(valid_payload),
        "train_output": spec["train_output"],
        "valid_output": spec["valid_output"],
    }


def run_export_grpo(preset_name: str) -> dict[str, object]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    logger = configure_logger("training.export_grpo_dataset")
    exports = [export_variant(project_root, spec) for spec in PRESETS[preset_name]]
    write_json(
        Path(project_root, "data/metadata/grpo_export_summary.json"),
        build_run_metadata(preset=preset_name, exports=exports),
    )
    logger.info("Exported %s GRPO dataset variants", len(exports))
    return {"preset": preset_name, "exports": exports}


def main() -> None:
    args = parse_args()
    run_export_grpo(args.preset)


if __name__ == "__main__":
    main()
