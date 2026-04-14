from __future__ import annotations

import re
from typing import Dict, List


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
THE_ANSWER_RE = re.compile(r"The answer is ([\\-\\$0-9\\.,]+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?[$0-9.,]{2,}|-?[0-9]+")


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    prediction = normalize_numeric(extract_prediction(results[0]))
    target = normalize_numeric(str(doc["answer"]).split("####")[-1].strip())
    return {"exact_match": 1 if prediction == target else 0}


def extract_prediction(text: str) -> str:
    text = text or ""
    match = ANSWER_TAG_RE.search(text)
    if match:
        text = match.group(1).strip()

    boxed = BOXED_RE.search(text)
    if boxed:
        return boxed.group(1).strip()

    match = FINAL_ANSWER_RE.search(text)
    if match:
        text = match.group(1).strip()

    match = THE_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()

    matches = NUMBER_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def normalize_numeric(text: str) -> str:
    return text.replace(",", "").replace("$", "").strip().rstrip(".")
