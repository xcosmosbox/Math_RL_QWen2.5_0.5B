from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewards.grpo_rewards import compute_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    target = ground_truth
    sample_id = ""
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", "")
        sample_id = str(ground_truth.get("sample_id", ""))
    if extra_info and not sample_id:
        sample_id = str(extra_info.get("sample_id", ""))

    result = compute_reward(
        sample_id=sample_id or "unknown",
        completion=solution_str,
        target_final_answer=str(target or ""),
    )

    # verl agent-loop postprocess converts each returned field into a numpy array.
    # Variable-length Python lists such as parse flags will raise
    # "ValueError: setting an array element with a sequence", so only return
    # scalars or serialized strings here.
    payload = {
        "score": float(result.total_reward),
        "data_source": data_source,
        "sample_id": result.sample_id,
        "correctness_reward": float(result.correctness_reward),
        "exact_match_pass": bool(result.exact_match_pass),
        "equivalent_match_pass": bool(result.equivalent_match_pass),
        "format_reward": float(result.format_reward),
        "answer_tag_reward": float(result.answer_tag_reward),
        "clean_stop_reward": float(result.clean_stop_reward),
        "no_answer_penalty": float(result.no_answer_penalty),
        "repeat_penalty": float(result.repeat_penalty),
        "role_marker_penalty": float(result.role_marker_penalty),
        "tail_penalty": float(result.tail_penalty),
        "overlength_penalty": float(result.overlength_penalty),
        "total_reward": float(result.total_reward),
        "extracted_answer": result.extracted_answer or "",
        "format_pass": bool(result.format_pass),
        "drop_or_parse_flags": json.dumps(result.drop_or_parse_flags, ensure_ascii=False),
    }
    return payload
