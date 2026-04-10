from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.task_type_mapping import map_topic_to_task_type
from rewards.answer_extraction import build_final_answer_suffix, normalize_answer_text
from utils.hash_utils import build_sample_id, stable_bucket
from utils.io_utils import ensure_dir, read_json, resolve_project_root, write_json
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "bigmath_verified_full"
PRESETS = {
    "bigmath_verified_full": {
        "raw_input_path": "data/raw/bigmath_rl_verified_raw.jsonl",
        "raw_metadata_path": "data/metadata/bigmath_rl_verified_raw_metadata.json",
        "output_path": "data/processed/bigmath_verified_clean.jsonl",
        "metadata_path": "data/metadata/bigmath_verified_clean_summary.json",
        "split_manifest_path": "data/metadata/bigmath_verified_split_manifest.json",
    },
    "bigmath_verified_smoke": {
        "raw_input_path": "data/raw/bigmath_rl_verified_raw_smoke.jsonl",
        "raw_metadata_path": "data/metadata/bigmath_rl_verified_raw_smoke_metadata.json",
        "output_path": "data/processed/bigmath_verified_clean_smoke.jsonl",
        "metadata_path": "data/metadata/bigmath_verified_clean_smoke_summary.json",
        "split_manifest_path": "data/metadata/bigmath_verified_split_manifest_smoke.json",
    },
}
MAX_PROMPT_CHARS = 6_000
MAX_ANSWER_CHARS = 256
SPLIT_BUCKETS = {
    "train": range(0, 8_000),
    "valid": range(8_000, 9_000),
    "test": range(9_000, 10_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Big-Math-RL-Verified into project format")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_domains(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    normalized = clean_text(value)
    return [normalized] if normalized else []


def parse_solve_rate(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate < 0:
        return 0.0
    if rate > 1:
        return 1.0
    return rate


def solve_rate_band(rate: float | None) -> str:
    if rate is None:
        return "unknown"
    if rate < 0.10:
        return "very_hard"
    if rate < 0.25:
        return "hard"
    if rate < 0.50:
        return "medium"
    if rate < 0.75:
        return "easy"
    return "very_easy"


def assign_split(sample_id: str) -> str:
    bucket = stable_bucket(sample_id, modulo=10_000)
    for split_name, bucket_range in SPLIT_BUCKETS.items():
        if bucket in bucket_range:
            return split_name
    return "test"


def build_record(raw_record: dict[str, Any], *, resolved_dataset_name: str) -> tuple[dict[str, Any] | None, str | None]:
    problem = clean_text(raw_record.get("problem"))
    raw_answer = clean_text(raw_record.get("answer"))
    answer = normalize_answer_text(raw_answer)
    if not problem:
        return None, "missing_problem"
    if not answer:
        return None, "missing_answer"
    if len(problem) > MAX_PROMPT_CHARS:
        return None, "problem_too_long"
    if len(answer) > MAX_ANSWER_CHARS:
        return None, "answer_too_long"

    domains = normalize_domains(raw_record.get("domain"))
    source = clean_text(raw_record.get("source")) or "unknown"
    topic = " | ".join(domains) if domains else source
    task_type = map_topic_to_task_type(topic)
    solve_rate = parse_solve_rate(raw_record.get("llama8b_solve_rate"))
    difficulty = solve_rate_band(solve_rate)
    chosen_solution = build_final_answer_suffix(answer)
    sample_id = build_sample_id(problem, answer)

    return (
        {
            "sample_id": sample_id,
            "prompt": problem,
            "target_final_answer": answer,
            "difficulty": difficulty,
            "topic": topic,
            "task_type": task_type,
            "candidate_solutions": [chosen_solution],
            "chosen_solution": chosen_solution,
            "is_verifiable": True,
            "split": assign_split(sample_id),
            "raw_question": problem,
            "raw_final_answer": raw_answer,
            "raw_r1_solution_1": "",
            "raw_r1_solution_2": "",
            "raw_r1_solution_3": "",
            "filter_flags": [
                "dataset_family:bigmath_verified",
                f"resolved_dataset:{resolved_dataset_name}",
                f"source:{source}",
                "chosen_solution:final_answer_only",
            ],
            "drop_reason": None,
            "raw_source": source,
            "raw_domain": domains,
            "llama8b_solve_rate": solve_rate,
        },
        None,
    )


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def summarize_split_counters(
    split_counts: Counter[str],
    by_split_task_type: dict[str, Counter[str]],
    by_split_difficulty: dict[str, Counter[str]],
) -> dict[str, Any]:
    return {
        "counts": dict(sorted(split_counts.items())),
        "by_task_type": {split: dict(sorted(counter.items())) for split, counter in sorted(by_split_task_type.items())},
        "by_difficulty": {split: dict(sorted(counter.items())) for split, counter in sorted(by_split_difficulty.items())},
    }


def run_build(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    logger = configure_logger("data.build_bigmath_verified_dataset")
    preset = PRESETS[preset_name]

    raw_input_path = Path(project_root, preset["raw_input_path"])
    raw_metadata_path = Path(project_root, preset["raw_metadata_path"])
    output_path = Path(project_root, preset["output_path"])
    metadata_path = Path(project_root, preset["metadata_path"])
    split_manifest_path = Path(project_root, preset["split_manifest_path"])
    if not raw_input_path.exists():
        raise FileNotFoundError(f"Raw input not found: {raw_input_path}")

    raw_metadata = read_json(raw_metadata_path) if raw_metadata_path.exists() else {}
    resolved_dataset_name = str(raw_metadata.get("resolved_dataset_name") or "unknown")

    ensure_dir(output_path.parent)
    split_paths = {
        split: output_path.with_name(f"{output_path.stem}_{split}.jsonl")
        for split in ("train", "valid", "test")
    }

    kept_count = 0
    dropped_count = 0
    duplicate_count = 0
    seen_sample_ids: set[str] = set()
    drop_reasons = Counter()
    source_counts = Counter()
    task_type_counts = Counter()
    domain_counts = Counter()
    split_counts = Counter()
    by_split_task_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_split_difficulty: dict[str, Counter[str]] = defaultdict(Counter)

    with (
        raw_input_path.open("r", encoding="utf-8") as input_handle,
        output_path.open("w", encoding="utf-8") as full_handle,
        split_paths["train"].open("w", encoding="utf-8") as train_handle,
        split_paths["valid"].open("w", encoding="utf-8") as valid_handle,
        split_paths["test"].open("w", encoding="utf-8") as test_handle,
    ):
        split_handles = {"train": train_handle, "valid": valid_handle, "test": test_handle}
        for line in input_handle:
            line = line.strip()
            if not line:
                continue
            raw_record = json.loads(line)
            record, drop_reason = build_record(raw_record, resolved_dataset_name=resolved_dataset_name)
            if record is None:
                dropped_count += 1
                drop_reasons[drop_reason or "unknown_drop"] += 1
                continue
            sample_id = str(record["sample_id"])
            if sample_id in seen_sample_ids:
                duplicate_count += 1
                dropped_count += 1
                drop_reasons["duplicate_sample_id"] += 1
                continue
            seen_sample_ids.add(sample_id)

            split = str(record["split"])
            kept_count += 1
            source_counts[str(record["raw_source"])] += 1
            task_type_counts[str(record["task_type"])] += 1
            split_counts[split] += 1
            by_split_task_type[split][str(record["task_type"])] += 1
            by_split_difficulty[split][str(record["difficulty"])] += 1
            for domain in record["raw_domain"]:
                domain_counts[str(domain)] += 1

            write_jsonl_row(full_handle, record)
            write_jsonl_row(split_handles[split], record)

    metadata = build_run_metadata(
        preset=preset_name,
        raw_input_path=str(raw_input_path),
        resolved_dataset_name=resolved_dataset_name,
        output_path=str(output_path),
        raw_count=kept_count + dropped_count,
        kept_count=kept_count,
        dropped_count=dropped_count,
        duplicate_count=duplicate_count,
        config={
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_answer_chars": MAX_ANSWER_CHARS,
            "split_buckets": {"train": "0-7999", "valid": "8000-8999", "test": "9000-9999"},
        },
        drop_reason_counts=dict(sorted(drop_reasons.items())),
        kept_by_source=dict(sorted(source_counts.items())),
        kept_by_task_type=dict(sorted(task_type_counts.items())),
        kept_by_domain_top_20=dict(domain_counts.most_common(20)),
    )
    split_manifest = build_run_metadata(
        preset=preset_name,
        input_path=str(output_path),
        record_count=kept_count,
        split_summary=summarize_split_counters(split_counts, by_split_task_type, by_split_difficulty),
    )
    write_json(metadata_path, metadata)
    write_json(split_manifest_path, split_manifest)
    logger.info("Cleaned %s / %s Big-Math records into %s", kept_count, kept_count + dropped_count, output_path)
    return {
        "preset": preset_name,
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "split_manifest_path": str(split_manifest_path),
        "kept_count": kept_count,
        "dropped_count": dropped_count,
    }


def main() -> None:
    args = parse_args()
    run_build(args.preset)


if __name__ == "__main__":
    main()
