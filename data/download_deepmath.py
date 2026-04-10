from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.io_utils import resolve_project_root, write_json, write_jsonl
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "deepmath_full"
PRESETS = {
    "deepmath_full": {
        "dataset_name": "zwhe99/DeepMath-103K",
        "split": "train",
        "output_path": "data/raw/deepmath_103k_raw.jsonl",
        "metadata_path": "data/metadata/deepmath_103k_raw_metadata.json",
        "limit": None,
    },
    "deepmath_smoke": {
        "dataset_name": "zwhe99/DeepMath-103K",
        "split": "train",
        "output_path": "data/raw/deepmath_103k_raw_smoke.jsonl",
        "metadata_path": "data/metadata/deepmath_103k_raw_smoke_metadata.json",
        "limit": 128,
    },
}
RAW_FIELDS = [
    "question",
    "final_answer",
    "difficulty",
    "topic",
    "r1_solution_1",
    "r1_solution_2",
    "r1_solution_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DeepMath-103K raw data")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def load_dataset_records(dataset_name: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install it in the target training environment.") from exc

    dataset = load_dataset(dataset_name, split=split)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        records.append({field: row.get(field) for field in RAW_FIELDS})
    return records


def run_download_preset(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    preset = PRESETS[preset_name]
    project_root = resolve_project_root()
    output_path = Path(project_root, preset["output_path"])
    metadata_path = Path(project_root, preset["metadata_path"])
    logger = configure_logger("data.download_deepmath")

    records = load_dataset_records(
        dataset_name=preset["dataset_name"],
        split=preset["split"],
        limit=preset["limit"],
    )
    write_jsonl(output_path, records)
    write_json(
        metadata_path,
        build_run_metadata(
            preset=preset_name,
            dataset_name=preset["dataset_name"],
            split=preset["split"],
            output_path=str(output_path),
            record_count=len(records),
            raw_fields=RAW_FIELDS,
            limit=preset["limit"],
        ),
    )
    logger.info("Wrote %s raw records to %s", len(records), output_path)
    return {
        "preset": preset_name,
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "record_count": len(records),
    }


def main() -> None:
    args = parse_args()
    run_download_preset(args.preset)


if __name__ == "__main__":
    main()
