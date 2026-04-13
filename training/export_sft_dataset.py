from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from training.prompt_templates import build_math_instruction_prompt
from rewards.answer_extraction import ANSWER_TAG_RE, build_final_answer_suffix, normalize_answer_text
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
FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^\s*final answer\s*[:：]\s*.+?\s*$")
FINAL_ANSWER_HEADER_RE = re.compile(r"(?im)^\s*(?:\*\*)?\s*final answer\s*(?:\*\*)?\s*$")
DISPLAY_BOXED_AT_END_RE = re.compile(r"(?:\n\s*)?(?:\\\[|\$\$)\s*\\boxed\{.*?\}\s*(?:\\\]|\$\$)\s*$", re.S)
THINK_TAG_RE = re.compile(r"</?think>\s*", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SFT dataset for LLaMA-Factory")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def unwrap_boxed_spans(text: str) -> str:
    if "\\boxed{" not in text:
        return text
    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("\\boxed{", cursor)
        if start < 0:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor:start])
        brace_cursor = start + len("\\boxed{")
        depth = 1
        collected: list[str] = []
        while brace_cursor < len(text):
            char = text[brace_cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parts.append("".join(collected).strip())
                    cursor = brace_cursor + 1
                    break
            if depth > 0:
                collected.append(char)
            brace_cursor += 1
        else:
            parts.append(text[start:])
            break
    return "".join(parts)


def is_redundant_answer_line(line: str, final_answer: str) -> bool:
    normalized_line = normalize_answer_text(unwrap_boxed_spans(line))
    if not normalized_line or not final_answer:
        return False
    if normalized_line == final_answer:
        return True
    return len(normalized_line) <= 160 and final_answer in normalized_line


def drop_leading_redundant_answer_line(text: str, final_answer: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        remainder = "\n".join(lines[index + 1 :]).strip()
        if remainder and len(remainder) >= 120 and is_redundant_answer_line(line, final_answer):
            return remainder
        break
    return text


def extract_reasoning_body(text: str, final_answer: str) -> str:
    matches = list(FINAL_ANSWER_HEADER_RE.finditer(text))
    if not matches:
        return text
    candidate = text[matches[-1].end() :].strip()
    if not candidate:
        return text
    candidate = drop_leading_redundant_answer_line(candidate, final_answer)
    if len(candidate) >= 120:
        return candidate
    return text


def normalize_sft_solution(record: dict[str, Any]) -> str:
    text = str(record.get("chosen_solution") or "")
    final_answer = normalize_answer_text(str(record.get("target_final_answer") or ""))
    text = THINK_TAG_RE.sub("", text).strip()
    text = extract_reasoning_body(text, final_answer)
    text = ANSWER_TAG_RE.sub("", text).strip()
    text = FINAL_ANSWER_HEADER_RE.sub("", text).strip()
    text = FINAL_ANSWER_LINE_RE.sub("", text).strip()
    text = DISPLAY_BOXED_AT_END_RE.sub("", text).strip()
    text = drop_leading_redundant_answer_line(text, final_answer)
    text = unwrap_boxed_spans(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text:
        return f"{text}\n\n{build_final_answer_suffix(final_answer)}"
    return build_final_answer_suffix(final_answer)


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
            {"role": "assistant", "content": normalize_sft_solution(record)},
        ],
    }


def has_reasoning_solution(record: dict[str, Any]) -> bool:
    chosen_solution = str(record.get("chosen_solution") or "").strip()
    if not chosen_solution:
        return False
    flags = set(record.get("filter_flags") or [])
    if "chosen_solution:final_answer_only" in flags:
        return False
    return ("\n" in chosen_solution) or (len(chosen_solution) >= 120)


def export_dataset(project_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    train_records = read_jsonl(Path(project_root, spec["train_path"]))
    valid_records = read_jsonl(Path(project_root, spec["valid_path"]))
    train_reasoning_records = [row for row in train_records if has_reasoning_solution(row)]
    valid_reasoning_records = [row for row in valid_records if has_reasoning_solution(row)]
    train_payload = [build_sft_record(row) for row in train_reasoning_records]
    valid_payload = [build_sft_record(row) for row in valid_reasoning_records]

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
        "train_input_count": len(train_records),
        "train_count": len(train_payload),
        "train_filtered_count": len(train_records) - len(train_payload),
        "valid_input_count": len(valid_records),
        "valid_count": len(valid_payload),
        "valid_filtered_count": len(valid_records) - len(valid_payload),
        "requires_reasoning_solution": True,
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
