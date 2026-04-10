from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.hash_utils import normalize_text_for_hash
from utils.io_utils import read_jsonl, resolve_project_root, write_json, write_jsonl
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "all"
PRESETS = {
    "all": ["strict", "relaxed"],
    "strict": ["strict"],
    "relaxed": ["relaxed"],
}
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
VARIANT_INPUTS = {
    "strict": "data/processed/filtered_strict.jsonl",
    "relaxed": "data/processed/filtered_relaxed.jsonl",
}
VARIANT_MANIFESTS = {
    "strict": "data/metadata/split_manifest_strict.json",
    "relaxed": "data/metadata/split_manifest_relaxed.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic DeepMath splits")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def prompt_group_key(record: dict[str, Any]) -> str:
    return normalize_text_for_hash(record["prompt"])


def compute_target_counts(total: int) -> dict[str, int]:
    raw_counts = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    floored = {split: int(value) for split, value in raw_counts.items()}
    remainder = total - sum(floored.values())
    order = sorted(raw_counts.keys(), key=lambda split: (raw_counts[split] - floored[split]), reverse=True)
    for split in order[:remainder]:
        floored[split] += 1
    return floored


def assign_groups_to_splits(groups: list[list[dict[str, Any]]]) -> dict[str, str]:
    total = sum(len(group) for group in groups)
    target_counts = compute_target_counts(total)
    current_counts = {split: 0 for split in SPLIT_RATIOS}
    assignments: dict[str, str] = {}

    sorted_groups = sorted(groups, key=lambda group: (-len(group), prompt_group_key(group[0]), group[0]["sample_id"]))
    for group in sorted_groups:
        candidate_scores: list[tuple[float, str]] = []
        for split in ("train", "valid", "test"):
            projected = current_counts[split] + len(group)
            target = target_counts[split]
            error = abs(projected - target)
            remaining_capacity_bonus = max(target - current_counts[split], 0)
            candidate_scores.append((error - (remaining_capacity_bonus * 0.001), split))
        candidate_scores.sort(key=lambda item: (item[0], item[1] != "train", item[1]))
        chosen_split = candidate_scores[0][1]
        group_key = prompt_group_key(group[0])
        assignments[group_key] = chosen_split
        current_counts[chosen_split] += len(group)
    return assignments


def stratified_split(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        strata[(record["difficulty"] or "unknown", record["task_type"])][prompt_group_key(record)].append(record)

    assigned_records: list[dict[str, Any]] = []
    for _, group_map in sorted(strata.items()):
        groups = list(group_map.values())
        group_assignments = assign_groups_to_splits(groups)
        for group_key, group_records in group_map.items():
            split = group_assignments[group_key]
            for record in group_records:
                updated = dict(record)
                updated["split"] = split
                assigned_records.append(updated)
    assigned_records.sort(key=lambda item: (item["split"], item["sample_id"]))
    return assigned_records


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["split"] for record in records)
    by_difficulty = defaultdict(Counter)
    by_task_type = defaultdict(Counter)
    verifiable = Counter()
    prompt_signatures = defaultdict(set)

    for record in records:
        split = record["split"]
        by_difficulty[split][record["difficulty"] or "unknown"] += 1
        by_task_type[split][record["task_type"]] += 1
        verifiable[split] += int(bool(record["is_verifiable"]))
        prompt_signatures[split].add(prompt_group_key(record))

    overlap = {
        "train_valid": len(prompt_signatures["train"] & prompt_signatures["valid"]),
        "train_test": len(prompt_signatures["train"] & prompt_signatures["test"]),
        "valid_test": len(prompt_signatures["valid"] & prompt_signatures["test"]),
    }
    return {
        "counts": dict(sorted(counts.items())),
        "verifiable_counts": dict(sorted(verifiable.items())),
        "by_difficulty": {split: dict(sorted(counter.items())) for split, counter in sorted(by_difficulty.items())},
        "by_task_type": {split: dict(sorted(counter.items())) for split, counter in sorted(by_task_type.items())},
        "prompt_signature_overlap": overlap,
    }


def write_split_files(project_root: Path, variant: str, records: list[dict[str, Any]]) -> None:
    base_output = Path(project_root, VARIANT_INPUTS[variant])
    write_jsonl(base_output, records)
    for split in ("train", "valid", "test"):
        split_records = [record for record in records if record["split"] == split]
        split_path = base_output.with_name(f"{base_output.stem}_{split}.jsonl")
        write_jsonl(split_path, split_records)


def run_variant(project_root: Path, variant: str) -> None:
    logger = configure_logger(f"data.build_splits.{variant}")
    input_path = Path(project_root, VARIANT_INPUTS[variant])
    manifest_path = Path(project_root, VARIANT_MANIFESTS[variant])
    records = read_jsonl(input_path)
    assigned_records = stratified_split(records)
    write_split_files(project_root, variant, assigned_records)
    manifest = build_run_metadata(
        variant=variant,
        input_path=str(input_path),
        split_ratios=SPLIT_RATIOS,
        record_count=len(assigned_records),
        split_summary=summarize_split(assigned_records),
    )
    write_json(manifest_path, manifest)
    logger.info("Variant %s split manifest saved to %s", variant, manifest_path)


def run_split_build(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    for variant in PRESETS[preset_name]:
        run_variant(project_root, variant)
    return {"preset": preset_name, "variants": PRESETS[preset_name]}


def main() -> None:
    args = parse_args()
    run_split_build(args.preset)


if __name__ == "__main__":
    main()
