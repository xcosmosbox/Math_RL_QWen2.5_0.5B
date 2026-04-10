from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewards.answer_extraction import extract_final_answer
from rewards.math_verifier import exact_match
from utils.io_utils import ensure_dir, resolve_project_root, write_jsonl


MAX_FAILURES_PER_CATEGORY = 5


def extract_completion(sample: dict[str, Any]) -> str:
    for key in ("filtered_resps", "resps"):
        value = sample.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, list) and first:
                return str(first[0])
            return str(first)
    return str(sample.get("completion", ""))


def extract_target(sample: dict[str, Any]) -> str:
    if "target" in sample:
        return str(sample["target"])
    doc = sample.get("doc", {})
    for key in ("answer", "final_answer", "label"):
        if key in doc:
            return str(doc[key])
    return ""


def categorize_failure(completion: str, target: str) -> tuple[str, str | None]:
    if not target:
        return "", None
    extraction = extract_final_answer(completion)
    if not extraction.extracted_answer:
        return "answer_extraction_failed", None
    if extraction.format_pass and not exact_match(extraction.extracted_answer, target):
        if len(completion) > 400:
            return "verbose_wrong_answer", extraction.extracted_answer
        return "format_correct_but_answer_wrong", extraction.extracted_answer
    return "", extraction.extracted_answer


def collect_failures(project_root: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    counts: dict[tuple[str, str, str], int] = {}

    for samples_path in sorted(Path(project_root, "eval", "outputs").glob("*/*/samples.jsonl")):
        task_dir = samples_path.parent
        run_config_path = task_dir / "run_config.json"
        if not run_config_path.exists():
            continue
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                completion = extract_completion(sample)
                target = extract_target(sample)
                if not completion or not target:
                    continue
                error_type, extracted_answer = categorize_failure(completion, target)
                if not error_type:
                    continue
                count_key = (run_config["model_stage"], run_config["benchmark"], error_type)
                if counts.get(count_key, 0) >= MAX_FAILURES_PER_CATEGORY:
                    continue
                counts[count_key] = counts.get(count_key, 0) + 1
                failures.append(
                    {
                        "model_stage": run_config["model_stage"],
                        "benchmark": run_config["benchmark"],
                        "error_type": error_type,
                        "target_answer": target,
                        "extracted_answer": extracted_answer,
                        "completion": completion,
                    }
                )
    return failures


def run_sample_failures() -> dict[str, Any]:
    project_root = resolve_project_root()
    output_dir = ensure_dir(Path(project_root, "analysis", "results"))
    failures = collect_failures(project_root)
    output_path = output_dir / "sample_failures.jsonl"
    write_jsonl(output_path, failures)
    return {"output_path": str(output_path), "failure_count": len(failures)}


def main() -> None:
    run_sample_failures()


if __name__ == "__main__":
    main()
