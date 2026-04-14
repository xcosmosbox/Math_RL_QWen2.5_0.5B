from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List

import datasets


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_PATTERNS = [
    re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"The final answer is\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"The answer is\s*(.*)", re.IGNORECASE | re.DOTALL),
]
OPTION_RE = re.compile(r"\(([a-z])\)|\b([a-z])\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    filtered = dataset.filter(lambda doc: doc.get("Picture") is None)

    def _process_doc(doc: dict) -> dict:
        return {
            "question": doc["Question"],
            "answer": str(doc["Answer"]).strip(),
            "answer_type": str(doc["Answer_type"]).strip(),
            "picture": None,
        }

    return filtered.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    prediction = extract_prediction(results[0], doc["answer_type"])
    target = normalize_target(doc["answer"], doc["answer_type"])
    return {"exact_match": 1 if compare_answer(prediction, target, doc["answer_type"]) else 0}


def extract_prediction(text: str, answer_type: str) -> Any:
    text = text or ""
    candidates: list[str] = []

    match = ANSWER_TAG_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())

    match = BOXED_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())

    for pattern in FINAL_PATTERNS:
        match = pattern.search(text)
        if match:
            candidates.append(match.group(1).strip())

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates.extend(reversed(lines[-6:]))
    candidates.append(text.strip())

    for candidate in candidates:
        parsed = parse_candidate(candidate, answer_type)
        if parsed is not None:
            return parsed
    return None


def parse_candidate(candidate: str, answer_type: str) -> Any:
    candidate = candidate.strip().strip("`").rstrip(".。 ")
    if not candidate:
        return None

    if answer_type == "bool":
        lowered = candidate.lower()
        if "true" in lowered or "yes" in lowered:
            return True
        if "false" in lowered or "no" in lowered:
            return False
        return None

    if answer_type == "option":
        match = OPTION_RE.search(candidate.lower())
        if match:
            return next(group for group in match.groups() if group)
        return None

    if answer_type in {"integer", "float"}:
        matches = NUMBER_RE.findall(candidate.replace(",", ""))
        if not matches:
            return None
        value = _to_finite_float(matches[-1])
        if value is None:
            return None
        return int(value) if answer_type == "integer" else value

    if answer_type in {"list of integer", "list of float"}:
        match = re.search(r"\[[^\]]+\]", candidate)
        if match:
            try:
                value = ast.literal_eval(match.group(0))
            except Exception:
                value = None
            if isinstance(value, list):
                parsed_items = [_to_finite_float(item) for item in value]
                if any(item is None for item in parsed_items):
                    return None
                if answer_type == "list of integer":
                    return [int(item) for item in parsed_items]
                return parsed_items
        matches = NUMBER_RE.findall(candidate.replace(",", ""))
        if not matches:
            return None
        parsed_items = [_to_finite_float(item) for item in matches]
        if any(item is None for item in parsed_items):
            return None
        if answer_type == "list of integer":
            return [int(item) for item in parsed_items]
        return parsed_items

    return candidate


def normalize_target(target: str, answer_type: str) -> Any:
    return parse_candidate(target, answer_type)


def compare_answer(prediction: Any, target: Any, answer_type: str) -> bool:
    if prediction is None or target is None:
        return False
    if answer_type == "bool":
        return bool(prediction) == bool(target)
    if answer_type == "option":
        return str(prediction).lower() == str(target).strip().strip("()").lower()
    if answer_type == "integer":
        return int(prediction) == int(target)
    if answer_type == "float":
        return math.isclose(float(prediction), float(target), rel_tol=1e-4, abs_tol=5e-3)
    if answer_type in {"list of integer", "list of float"}:
        if not isinstance(prediction, list) or not isinstance(target, list):
            return False
        if len(prediction) != len(target):
            return False
        if answer_type == "list of integer":
            return all(int(a) == int(b) for a, b in zip(prediction, target))
        return all(math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=5e-3) for a, b in zip(prediction, target))
    return str(prediction).strip() == str(target).strip()
