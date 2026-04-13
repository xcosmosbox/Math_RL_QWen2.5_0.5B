from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.task_type_mapping import MAPPING_VERSION, map_topic_to_task_type
from rewards.answer_extraction import extract_final_answer, normalize_answer_text
from rewards.math_verifier import exact_match
from utils.hash_utils import build_sample_id, normalize_text_for_hash
from utils.io_utils import read_jsonl, resolve_project_root, write_json, write_jsonl
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_PRESET = "all"
PRESETS = {
    "all": ["strict", "relaxed"],
    "strict": ["strict"],
    "relaxed": ["relaxed"],
}
FILTER_VARIANTS = {
    "strict": {
        "max_prompt_chars": 2_200,
        "max_solution_chars": 4_000,
        "max_target_answer_chars": 96,
        "require_strict_format_for_chosen_solution": True,
        "allow_generic_answer_fallback": False,
        "answer_format_profile": "deepmath",
    },
    "relaxed": {
        "max_prompt_chars": 3_500,
        "max_solution_chars": 6_000,
        "max_target_answer_chars": 160,
        "require_strict_format_for_chosen_solution": False,
        "allow_generic_answer_fallback": True,
        "answer_format_profile": "deepmath",
    },
}
RAW_DATA_PATH = "data/raw/deepmath_103k_raw.jsonl"
PROCESSED_OUTPUTS = {
    "strict": "data/processed/filtered_strict.jsonl",
    "relaxed": "data/processed/filtered_relaxed.jsonl",
}
METADATA_OUTPUTS = {
    "strict": "data/metadata/filter_log_strict.json",
    "relaxed": "data/metadata/filter_log_relaxed.json",
}


@dataclass(frozen=True)
class CandidateAssessment:
    text: str
    extracted_answer: str | None
    exact_match_pass: bool
    strict_format_pass: bool
    extraction_method: str
    score: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered DeepMath datasets")
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


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    alpha_num_count = sum(char.isalnum() for char in stripped)
    return alpha_num_count < 3


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


def aggregate_candidate_solutions(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for field in ("r1_solution_1", "r1_solution_2", "r1_solution_3"):
        value = clean_text(record.get(field))
        if value and not is_noise_text(value):
            candidates.append(value)
    return candidates


def assess_candidate(
    text: str,
    *,
    target_final_answer: str,
    require_strict_format: bool,
    allow_generic_answer_fallback: bool,
    answer_format_profile: str,
) -> CandidateAssessment | None:
    extraction = extract_final_answer(text, format_profile=answer_format_profile)
    if extraction.extracted_answer is None:
        return None
    if require_strict_format and not extraction.format_pass:
        return None
    if not allow_generic_answer_fallback and extraction.extraction_method == "generic_answer_line":
        return None
    exact = exact_match(extraction.extracted_answer, target_final_answer)
    length_penalty = abs(len(text) - 1_200)
    score = (
        1 if exact else 0,
        1 if extraction.format_pass else 0,
        1 if extraction.extraction_method == "final_answer_line" else 0,
        -length_penalty,
    )
    return CandidateAssessment(
        text=text,
        extracted_answer=extraction.extracted_answer,
        exact_match_pass=exact,
        strict_format_pass=extraction.format_pass,
        extraction_method=extraction.extraction_method,
        score=score,
    )


def choose_solution(
    candidates: list[str],
    *,
    target_final_answer: str,
    variant_config: dict[str, Any],
) -> CandidateAssessment | None:
    assessed: list[CandidateAssessment] = []
    for candidate in candidates:
        if len(candidate) > variant_config["max_solution_chars"]:
            continue
        result = assess_candidate(
            candidate,
            target_final_answer=target_final_answer,
            require_strict_format=variant_config["require_strict_format_for_chosen_solution"],
            allow_generic_answer_fallback=variant_config["allow_generic_answer_fallback"],
            answer_format_profile=variant_config["answer_format_profile"],
        )
        if result is not None:
            assessed.append(result)
    if not assessed:
        return None
    assessed = [item for item in assessed if item.exact_match_pass]
    if not assessed:
        return None
    assessed.sort(key=lambda item: item.score, reverse=True)
    return assessed[0]


def build_filtered_record(
    record: dict[str, Any],
    *,
    variant_name: str,
    variant_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    question = clean_text(record.get("question"))
    final_answer = clean_text(record.get("final_answer"))
    if not question:
        return None, "missing_question"
    if not final_answer:
        return None, "missing_final_answer"

    prompt = clean_text(question)
    if len(prompt) > variant_config["max_prompt_chars"]:
        return None, "prompt_too_long"

    target_final_answer = normalize_answer_text(final_answer)
    target_ok, target_flags = validate_target_final_answer(final_answer, max_chars=variant_config["max_target_answer_chars"])
    if not target_ok:
        if "ambiguous_final_answer" in target_flags:
            return None, "ambiguous_final_answer"
        if "final_answer_too_long" in target_flags:
            return None, "final_answer_too_long"
        return None, "invalid_final_answer"

    candidate_solutions = aggregate_candidate_solutions(record)
    if not candidate_solutions:
        return None, "missing_candidate_solutions"

    chosen_assessment = choose_solution(
        candidate_solutions,
        target_final_answer=target_final_answer,
        variant_config=variant_config,
    )
    if chosen_assessment is None:
        return None, "no_usable_chosen_solution"

    task_type = map_topic_to_task_type(clean_text(record.get("topic")))
    sample_id = build_sample_id(question, final_answer)
    is_verifiable = bool(chosen_assessment.strict_format_pass)
    filter_flags = [
        f"filter_variant:{variant_name}",
        f"task_mapping:{MAPPING_VERSION}",
        f"chosen_solution_method:{chosen_assessment.extraction_method}",
    ]
    if not is_verifiable:
        filter_flags.append("verifier_risk:format_not_strict")

    filtered_record = {
        "sample_id": sample_id,
        "prompt": prompt,
        "target_final_answer": target_final_answer,
        "difficulty": clean_text(record.get("difficulty")),
        "topic": clean_text(record.get("topic")),
        "task_type": task_type,
        "candidate_solutions": candidate_solutions,
        "chosen_solution": chosen_assessment.text,
        "is_verifiable": is_verifiable,
        "split": "",
        "raw_question": question,
        "raw_final_answer": final_answer,
        "raw_r1_solution_1": clean_text(record.get("r1_solution_1")),
        "raw_r1_solution_2": clean_text(record.get("r1_solution_2")),
        "raw_r1_solution_3": clean_text(record.get("r1_solution_3")),
        "filter_flags": filter_flags,
        "drop_reason": None,
    }
    return filtered_record, None


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


def build_retention_summary(
    raw_records: list[dict[str, Any]],
    kept_records: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_difficulty = Counter(clean_text(row.get("difficulty")) or "unknown" for row in raw_records)
    kept_difficulty = Counter(row["difficulty"] or "unknown" for row in kept_records)
    raw_topic = Counter(clean_text(row.get("topic")) or "unknown" for row in raw_records)
    kept_topic = Counter(row["topic"] or "unknown" for row in kept_records)
    kept_task_type = Counter(row["task_type"] for row in kept_records)
    return {
        "raw_by_difficulty": dict(sorted(raw_difficulty.items())),
        "kept_by_difficulty": dict(sorted(kept_difficulty.items())),
        "raw_by_topic_top_20": dict(raw_topic.most_common(20)),
        "kept_by_topic_top_20": dict(kept_topic.most_common(20)),
        "kept_by_task_type": dict(sorted(kept_task_type.items())),
    }


def run_variant(
    *,
    variant_name: str,
    raw_records: list[dict[str, Any]],
    project_root: Path,
) -> None:
    logger = configure_logger(f"data.build_filtered_dataset.{variant_name}")
    variant_config = FILTER_VARIANTS[variant_name]
    kept_records: list[dict[str, Any]] = []
    drop_reasons = Counter()

    for record in raw_records:
        filtered_record, drop_reason = build_filtered_record(
            record,
            variant_name=variant_name,
            variant_config=variant_config,
        )
        if filtered_record is None:
            drop_reasons[drop_reason or "unknown_drop"] += 1
            continue
        kept_records.append(filtered_record)

    deduped_records, duplicate_count = deduplicate_records(kept_records)
    if duplicate_count:
        drop_reasons["duplicate_sample_id"] += duplicate_count
    for record in deduped_records:
        record["filter_flags"].append(f"is_verifiable:{str(record['is_verifiable']).lower()}")

    output_path = Path(project_root, PROCESSED_OUTPUTS[variant_name])
    metadata_path = Path(project_root, METADATA_OUTPUTS[variant_name])
    write_jsonl(output_path, deduped_records)

    metadata = build_run_metadata(
        variant=variant_name,
        raw_input_path=str(Path(project_root, RAW_DATA_PATH)),
        output_path=str(output_path),
        raw_count=len(raw_records),
        kept_count=len(deduped_records),
        dropped_count=len(raw_records) - len(deduped_records),
        duplicate_count=duplicate_count,
        verifiable_count=sum(1 for row in deduped_records if row["is_verifiable"]),
        config=variant_config,
        drop_reason_counts=dict(sorted(drop_reasons.items())),
        retention_summary=build_retention_summary(raw_records, deduped_records),
    )
    write_json(metadata_path, metadata)
    logger.info("Variant %s retained %s / %s samples", variant_name, len(deduped_records), len(raw_records))


def run_filtered_build(preset_name: str) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    project_root = resolve_project_root()
    raw_path = Path(project_root, RAW_DATA_PATH)
    raw_records = read_jsonl(raw_path)
    for variant_name in PRESETS[preset_name]:
        run_variant(variant_name=variant_name, raw_records=raw_records, project_root=project_root)
    return {
        "preset": preset_name,
        "variants": PRESETS[preset_name],
        "raw_input_path": str(raw_path),
    }


def main() -> None:
    args = parse_args()
    run_filtered_build(args.preset)


if __name__ == "__main__":
    main()
