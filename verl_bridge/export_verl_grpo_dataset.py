from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRAIN_PATH = "data/processed/grpo_strict_main_train.jsonl"
DEFAULT_VALID_PATH = "data/processed/grpo_strict_main_valid.jsonl"
DEFAULT_OUTPUT_DIR = "data/processed/verl/strict_main"
DEFAULT_DATA_SOURCE = "math_rl_qwen/strict_main"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current GRPO jsonl data into verl parquet format")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_row(row: dict[str, Any], *, index: int, data_source: str) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": str(row["prompt"])}],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": str(row["target_final_answer"]),
                "sample_id": str(row["sample_id"]),
            },
        },
        "extra_info": {
            "split": str(row.get("split", "")),
            "index": index,
            "sample_id": str(row["sample_id"]),
            "task_type": str(row.get("task_type", "")),
            "topic": str(row.get("topic", "")),
            "difficulty": str(row.get("difficulty", "")),
            "raw_prompt": str(row.get("raw_prompt", "")),
        },
    }


def export_split(input_path: Path, output_path: Path, *, data_source: str) -> int:
    rows = read_jsonl(input_path)
    payload = [build_row(row, index=index, data_source=data_source) for index, row in enumerate(rows)]
    frame = pd.DataFrame(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return len(payload)


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)
    output_dir = Path(args.output_dir)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    if not valid_path.is_absolute():
        valid_path = PROJECT_ROOT / valid_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = export_split(train_path, output_dir / "train.parquet", data_source=args.data_source)
    valid_count = export_split(valid_path, output_dir / "valid.parquet", data_source=args.data_source)

    summary = {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "output_dir": str(output_dir),
        "data_source": args.data_source,
        "train_count": train_count,
        "valid_count": valid_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
