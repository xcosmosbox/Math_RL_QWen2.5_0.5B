from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.io_utils import ensure_dir, resolve_project_root, write_json
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "bigmath_verified_full"
PRESETS = {
    "bigmath_verified_full": {
        "dataset_candidates": [
            "SynthLabsAI/Big-Math-RL-Verified",
            "ZhuofengLi/Big-Math-RL-Verified",
        ],
        "split": "train",
        "output_path": "data/raw/bigmath_rl_verified_raw.jsonl",
        "metadata_path": "data/metadata/bigmath_rl_verified_raw_metadata.json",
        "limit": None,
    },
    "bigmath_verified_smoke": {
        "dataset_candidates": [
            "SynthLabsAI/Big-Math-RL-Verified",
            "ZhuofengLi/Big-Math-RL-Verified",
        ],
        "split": "train",
        "output_path": "data/raw/bigmath_rl_verified_raw_smoke.jsonl",
        "metadata_path": "data/metadata/bigmath_rl_verified_raw_smoke_metadata.json",
        "limit": 2048,
    },
}
RAW_FIELDS = ["problem", "answer", "source", "domain", "llama8b_solve_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Big-Math-RL-Verified raw data")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def load_dataset_with_fallback(
    dataset_candidates: list[str],
    split: str,
) -> tuple[Any, str, list[dict[str, str]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install it in the target training environment.") from exc

    errors: list[dict[str, str]] = []
    for dataset_name in dataset_candidates:
        try:
            dataset = load_dataset(dataset_name, split=split)
            return dataset, dataset_name, errors
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "dataset_name": dataset_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    error_lines = [f"{item['dataset_name']}: {item['error_type']} {item['error_message']}" for item in errors]
    raise RuntimeError("Unable to load Big-Math-RL-Verified from any configured source.\n" + "\n".join(error_lines))


def write_raw_records(
    dataset: Any,
    output_path: Path,
    *,
    limit: int | None,
) -> int:
    ensure_dir(output_path.parent)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            payload = {field: row.get(field) for field in RAW_FIELDS}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def run_download_preset(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    preset = PRESETS[preset_name]
    project_root = resolve_project_root()
    output_path = Path(project_root, preset["output_path"])
    metadata_path = Path(project_root, preset["metadata_path"])
    logger = configure_logger("data.download_bigmath_verified")

    dataset, resolved_dataset_name, errors = load_dataset_with_fallback(
        dataset_candidates=preset["dataset_candidates"],
        split=preset["split"],
    )
    record_count = write_raw_records(dataset, output_path, limit=preset["limit"])
    write_json(
        metadata_path,
        build_run_metadata(
            preset=preset_name,
            dataset_candidates=preset["dataset_candidates"],
            resolved_dataset_name=resolved_dataset_name,
            split=preset["split"],
            output_path=str(output_path),
            record_count=record_count,
            raw_fields=RAW_FIELDS,
            limit=preset["limit"],
            load_errors=errors,
        ),
    )
    logger.info("Wrote %s raw records to %s", record_count, output_path)
    return {
        "preset": preset_name,
        "resolved_dataset_name": resolved_dataset_name,
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "record_count": record_count,
    }


def main() -> None:
    args = parse_args()
    run_download_preset(args.preset)


if __name__ == "__main__":
    main()
