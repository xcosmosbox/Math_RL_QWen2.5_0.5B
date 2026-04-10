from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from training.prompt_templates import build_math_instruction_prompt
from utils.io_utils import read_jsonl, resolve_project_root, write_json
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "all"
PRESETS = {
    "all": [
        {
            "dataset_name": "strict_main",
            "train_path": "data/processed/filtered_strict_train.jsonl",
            "valid_path": "data/processed/filtered_strict_valid.jsonl",
            "train_output": "data/processed/sft_strict_main_train.json",
            "valid_output": "data/processed/sft_strict_main_valid.json",
        },
        {
            "dataset_name": "relaxed_ablation",
            "train_path": "data/processed/filtered_relaxed_train.jsonl",
            "valid_path": "data/processed/filtered_relaxed_valid.jsonl",
            "train_output": "data/processed/sft_relaxed_ablation_train.json",
            "valid_output": "data/processed/sft_relaxed_ablation_valid.json",
        },
    ],
    "strict_main": [
        {
            "dataset_name": "strict_main",
            "train_path": "data/processed/filtered_strict_train.jsonl",
            "valid_path": "data/processed/filtered_strict_valid.jsonl",
            "train_output": "data/processed/sft_strict_main_train.json",
            "valid_output": "data/processed/sft_strict_main_valid.json",
        }
    ],
    "relaxed_ablation": [
        {
            "dataset_name": "relaxed_ablation",
            "train_path": "data/processed/filtered_relaxed_train.jsonl",
            "valid_path": "data/processed/filtered_relaxed_valid.jsonl",
            "train_output": "data/processed/sft_relaxed_ablation_train.json",
            "valid_output": "data/processed/sft_relaxed_ablation_valid.json",
        }
    ],
}
DATASET_INFO_PATH = "data/processed/dataset_info.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SFT dataset for LLaMA-Factory")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def build_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["sample_id"],
        "sample_id": record["sample_id"],
        "difficulty": record["difficulty"],
        "topic": record["topic"],
        "task_type": record["task_type"],
        "target_final_answer": record["target_final_answer"],
        "messages": [
            {"role": "user", "content": build_math_instruction_prompt(record["prompt"])},
            {"role": "assistant", "content": record["chosen_solution"]},
        ],
    }


def export_dataset(project_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    train_records = read_jsonl(Path(project_root, spec["train_path"]))
    valid_records = read_jsonl(Path(project_root, spec["valid_path"]))
    train_payload = [build_sft_record(row) for row in train_records if row.get("chosen_solution")]
    valid_payload = [build_sft_record(row) for row in valid_records if row.get("chosen_solution")]

    train_output = Path(project_root, spec["train_output"])
    valid_output = Path(project_root, spec["valid_output"])
    write_json(train_output, train_payload)
    write_json(valid_output, valid_payload)

    dataset_info = {
        f"{spec['dataset_name']}_train": {
            "file_name": train_output.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        f"{spec['dataset_name']}_valid": {
            "file_name": valid_output.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    return {
        "dataset_name": spec["dataset_name"],
        "train_count": len(train_payload),
        "valid_count": len(valid_payload),
        "dataset_info": dataset_info,
    }


def run_export_sft(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    logger = configure_logger("training.export_sft_dataset")

    merged_dataset_info: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for spec in PRESETS[preset_name]:
        summary = export_dataset(project_root, spec)
        merged_dataset_info.update(summary["dataset_info"])
        summaries.append(summary)

    write_json(Path(project_root, DATASET_INFO_PATH), merged_dataset_info)
    write_json(
        Path(project_root, "data/metadata/sft_export_summary.json"),
        build_run_metadata(preset=preset_name, exports=summaries, dataset_info_path=DATASET_INFO_PATH),
    )
    logger.info("Exported %s SFT dataset variants", len(summaries))
    return {"preset": preset_name, "exports": summaries, "dataset_info_path": DATASET_INFO_PATH}


def main() -> None:
    args = parse_args()
    run_export_sft(args.preset)


if __name__ == "__main__":
    main()
