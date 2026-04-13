from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from data.build_splits import summarize_split, stratified_split
from data.task_type_mapping import MAPPING_VERSION, map_topic_to_task_type
from rewards.answer_extraction import normalize_answer_text
from utils.hash_utils import build_sample_id
from utils.io_utils import read_jsonl, resolve_project_root, write_json, write_jsonl
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "all"
PRESETS = {
    "all": ["strict", "relaxed"],
    "strict": ["strict"],
    "relaxed": ["relaxed"],
}
RAW_DATA_PATH = "data/raw/deepmath_103k_raw.jsonl"
RL_VARIANTS = {
    "strict": {
        "max_prompt_chars": 2_200,
        "max_target_answer_chars": 96,
        "allow_missing_final_answer": True,
        "output_path": "data/processed/rl_strict.jsonl",
        "metadata_path": "data/metadata/rl_log_strict.json",
    },
    "relaxed": {
        "max_prompt_chars": 3_500,
        "max_target_answer_chars": 160,
        "allow_missing_final_answer": True,
        "output_path": "data/processed/rl_relaxed.jsonl",
        "metadata_path": "data/metadata/rl_log_relaxed.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RL-ready datasets that do not require chosen solutions")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        value = str(value)
    elif not isinstance(value, str):
        value = str(value)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def validate_target_final_answer(answer: str, max_chars: int) -> tuple[bool, list[str]]:
    flags: list[str] = []
    normalized = normalize_answer_text(answer)
    if not normalized:
        flags.append("empty_final_answer")
    if len(normalized) > max_chars:
        flags.append("final_answer_too_long")

    ambiguity_markers = [
        "answers may vary",
        "answer may vary",
        "cannot be determined",
        "not enough information",
        "depends on",
        "either ",
        "one of",
    ]
    lowered = normalized.lower()
    if any(marker in lowered for marker in ambiguity_markers):
        flags.append("ambiguous_final_answer")
    if "\n" in answer.strip():
        flags.append("multi_line_final_answer")
    return not flags, flags


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for record in sorted(records, key=lambda item: item["sample_id"]):
        sample_id = record["sample_id"]
        if sample_id in deduped:
            duplicates += 1
            continue
        deduped[sample_id] = record
    return list(deduped.values()), duplicates


def build_rl_record(
    raw_record: dict[str, Any],
    *,
    variant_name: str,
    variant_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    question = clean_text(raw_record.get("question"))
    final_answer_raw = clean_text(raw_record.get("final_answer"))
    if not question:
        return None, "missing_question"
    if len(question) > variant_config["max_prompt_chars"]:
        return None, "prompt_too_long"

    target_final_answer = normalize_answer_text(final_answer_raw)
    answer_available = bool(target_final_answer)
    if answer_available:
        target_ok, target_flags = validate_target_final_answer(
            final_answer_raw,
            max_chars=variant_config["max_target_answer_chars"],
        )
        if not target_ok:
            if "ambiguous_final_answer" in target_flags:
                return None, "ambiguous_final_answer"
            if "final_answer_too_long" in target_flags:
                return None, "final_answer_too_long"
            return None, "invalid_final_answer"
    elif not variant_config["allow_missing_final_answer"]:
        return None, "missing_final_answer"

    topic = clean_text(raw_record.get("topic"))
    task_type = map_topic_to_task_type(topic)
    sample_id = build_sample_id(question, final_answer_raw)
    filter_flags = [
        f"rl_variant:{variant_name}",
        "dataset_role:rl_ready",
        f"task_mapping:{MAPPING_VERSION}",
        f"answer_available:{str(answer_available).lower()}",
    ]
    if not answer_available:
        filter_flags.append("reward_mode:format_only")

    return (
        {
            "sample_id": sample_id,
            "prompt": question,
            "target_final_answer": target_final_answer,
            "difficulty": clean_text(raw_record.get("difficulty")),
            "topic": topic,
            "task_type": task_type,
            "split": "",
            "is_verifiable": answer_available,
            "raw_question": question,
            "raw_final_answer": final_answer_raw,
            "filter_flags": filter_flags,
        },
        None,
    )


def run_variant(
    *,
    variant_name: str,
    raw_records: list[dict[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    logger = configure_logger(f"data.build_rl_ready_dataset.{variant_name}")
    variant_config = RL_VARIANTS[variant_name]
    kept_records: list[dict[str, Any]] = []
    drop_reasons = Counter()

    for record in raw_records:
        rl_record, drop_reason = build_rl_record(
            record,
            variant_name=variant_name,
            variant_config=variant_config,
        )
        if rl_record is None:
            drop_reasons[drop_reason or "unknown_drop"] += 1
            continue
        kept_records.append(rl_record)

    deduped_records, duplicate_count = deduplicate_records(kept_records)
    if duplicate_count:
        drop_reasons["duplicate_sample_id"] += duplicate_count

    assigned_records = stratified_split(deduped_records)

    output_path = Path(project_root, variant_config["output_path"])
    metadata_path = Path(project_root, variant_config["metadata_path"])
    write_jsonl(output_path, assigned_records)
    for split in ("train", "valid", "test"):
        split_records = [record for record in assigned_records if record["split"] == split]
        split_path = output_path.with_name(f"{output_path.stem}_{split}.jsonl")
        write_jsonl(split_path, split_records)

    metadata = build_run_metadata(
        variant=variant_name,
        raw_input_path=str(Path(project_root, RAW_DATA_PATH)),
        output_path=str(output_path),
        raw_count=len(raw_records),
        kept_count=len(assigned_records),
        dropped_count=sum(drop_reasons.values()),
        verifiable_count=sum(1 for record in assigned_records if record["is_verifiable"]),
        format_only_count=sum(1 for record in assigned_records if not record["is_verifiable"]),
        drop_reason_counts=dict(sorted(drop_reasons.items())),
        split_summary=summarize_split(assigned_records),
        config=variant_config,
    )
    write_json(metadata_path, metadata)
    logger.info(
        "Built RL-ready %s dataset with %s samples (%s verifiable, %s format-only)",
        variant_name,
        len(assigned_records),
        metadata["verifiable_count"],
        metadata["format_only_count"],
    )
    return metadata


def run_build(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    raw_records = read_jsonl(Path(project_root, RAW_DATA_PATH))
    summaries = {
        variant: run_variant(variant_name=variant, raw_records=raw_records, project_root=project_root)
        for variant in PRESETS[preset_name]
    }
    return {"preset": preset_name, "variants": summaries}


def main() -> None:
    args = parse_args()
    run_build(args.preset)


if __name__ == "__main__":
    main()
