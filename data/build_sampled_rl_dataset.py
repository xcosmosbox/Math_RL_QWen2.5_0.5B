from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import read_jsonl, resolve_project_root, write_json, write_jsonl
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "strict_24k"
PRESETS = {
    "strict_24k": {
        "train_input": "data/processed/rl_strict_train.jsonl",
        "valid_input": "data/processed/rl_strict_valid.jsonl",
        "train_output": "data/processed/rl_strict_sampled_train.jsonl",
        "valid_output": "data/processed/rl_strict_sampled_valid.jsonl",
        "target_train_size": 24_000,
        "seed": 42,
    }
}

DIFFICULTY_BUCKETS = {
    "low": {"weight": 0.10, "min": 1200},
    "mid": {"weight": 0.45, "min": 8000},
    "high": {"weight": 0.30, "min": 5000},
    "very_high": {"weight": 0.15, "min": 2500},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sampled RL dataset with difficulty and task-type stratification")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def difficulty_to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def difficulty_bucket(record: dict[str, Any]) -> str:
    score = difficulty_to_float(record.get("difficulty"))
    if score is None:
        return "very_high"
    if score <= 3.0:
        return "low"
    if score <= 6.0:
        return "mid"
    if score <= 8.0:
        return "high"
    return "very_high"


def compute_bucket_targets(total: int) -> dict[str, int]:
    raw = {name: total * config["weight"] for name, config in DIFFICULTY_BUCKETS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    for name, config in DIFFICULTY_BUCKETS.items():
        counts[name] = max(counts[name], config["min"])
    overflow = sum(counts.values()) - total
    if overflow > 0:
        for name in ("mid", "high", "very_high", "low"):
            reducible = counts[name] - DIFFICULTY_BUCKETS[name]["min"]
            delta = min(overflow, max(0, reducible))
            counts[name] -= delta
            overflow -= delta
            if overflow == 0:
                break
    return counts


def allocate_counts(weights: dict[str, int], target: int) -> dict[str, int]:
    total_weight = sum(weights.values())
    if target <= 0 or total_weight <= 0:
        return {key: 0 for key in weights}
    raw = {key: target * value / total_weight for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = target - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def sample_bucket(records: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(records) <= target:
        return list(records)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record.get("task_type") or "unknown")].append(record)

    base_targets = allocate_counts({task: len(items) for task, items in by_task.items()}, target)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for task, items in sorted(by_task.items()):
        ordered = sorted(items, key=lambda row: row["sample_id"])
        rng.shuffle(ordered)
        take = min(len(ordered), max(1, base_targets[task]))
        selected.extend(ordered[:take])
        leftovers.extend(ordered[take:])

    if len(selected) > target:
        rng.shuffle(selected)
        selected = selected[:target]
    elif len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])

    selected.sort(key=lambda row: row["sample_id"])
    return selected


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket = Counter()
    by_difficulty = Counter()
    by_task = Counter()
    for record in records:
        by_bucket[difficulty_bucket(record)] += 1
        by_difficulty[str(record.get("difficulty") or "unknown")] += 1
        by_task[str(record.get("task_type") or "unknown")] += 1
    return {
        "count": len(records),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_difficulty": dict(sorted(by_difficulty.items(), key=lambda item: item[0])),
        "by_task_type": dict(sorted(by_task.items())),
    }


def run_build(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    preset = PRESETS[preset_name]
    project_root = resolve_project_root()
    logger = configure_logger(f"data.build_sampled_rl_dataset.{preset_name}")

    train_records = read_jsonl(Path(project_root, preset["train_input"]))
    valid_records = read_jsonl(Path(project_root, preset["valid_input"]))
    rng = random.Random(preset["seed"])

    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        bucketed[difficulty_bucket(record)].append(record)

    bucket_targets = compute_bucket_targets(preset["target_train_size"])
    sampled_train: list[dict[str, Any]] = []
    for bucket_name, target in bucket_targets.items():
        sampled_train.extend(sample_bucket(bucketed[bucket_name], min(target, len(bucketed[bucket_name])), rng))

    if len(sampled_train) > preset["target_train_size"]:
        sampled_train = sorted(sampled_train, key=lambda row: row["sample_id"])
        rng.shuffle(sampled_train)
        sampled_train = sampled_train[: preset["target_train_size"]]

    sampled_train.sort(key=lambda row: row["sample_id"])

    train_output = Path(project_root, preset["train_output"])
    valid_output = Path(project_root, preset["valid_output"])
    write_jsonl(train_output, sampled_train)
    write_jsonl(valid_output, valid_records)

    metadata = build_run_metadata(
        preset=preset_name,
        target_train_size=preset["target_train_size"],
        seed=preset["seed"],
        bucket_targets=bucket_targets,
        train_summary=summarize(sampled_train),
        valid_summary=summarize(valid_records),
    )
    write_json(Path(project_root, "data/metadata/rl_sampled_summary.json"), metadata)
    logger.info("Built sampled RL dataset %s with %s train records", preset_name, len(sampled_train))
    return metadata


def main() -> None:
    args = parse_args()
    result = run_build(args.preset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
