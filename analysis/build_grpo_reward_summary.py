from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from utils.io_utils import read_jsonl, resolve_project_root, write_json


DEFAULT_TRACE_PATHS = {
    "strict_main": "outputs/grpo/strict_main/reward_traces/reward_trace.jsonl",
    "relaxed_ablation_grpo": "outputs/grpo/relaxed_ablation_grpo/reward_traces/reward_trace.jsonl",
}


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 6) if values else 0.0


def build_reward_summary(trace_path: Path, output_path: Path) -> dict[str, Any]:
    rows = read_jsonl(trace_path)
    correctness = [float(row["correctness_reward"]) for row in rows]
    format_rewards = [float(row["format_reward"]) for row in rows]
    answer_tag_rewards = [float(row.get("answer_tag_reward", 0.0)) for row in rows]
    clean_stop_rewards = [float(row.get("clean_stop_reward", 0.0)) for row in rows]
    tail_penalties = [float(row.get("tail_penalty", 0.0)) for row in rows]
    repeat_penalties = [float(row.get("repeat_penalty", 0.0)) for row in rows]
    role_marker_penalties = [float(row.get("role_marker_penalty", 0.0)) for row in rows]
    overlength_penalties = [float(row.get("overlength_penalty", 0.0)) for row in rows]
    total = [float(row["total_reward"]) for row in rows]
    payload = {
        "trace_path": str(trace_path),
        "sample_count": len(rows),
        "correctness_reward_mean": _mean(correctness),
        "format_reward_mean": _mean(format_rewards),
        "answer_tag_reward_mean": _mean(answer_tag_rewards),
        "clean_stop_reward_mean": _mean(clean_stop_rewards),
        "tail_penalty_mean": _mean(tail_penalties),
        "repeat_penalty_mean": _mean(repeat_penalties),
        "role_marker_penalty_mean": _mean(role_marker_penalties),
        "overlength_penalty_mean": _mean(overlength_penalties),
        "total_reward_mean": _mean(total),
        "correctness_pass_rate": round(sum(1 for value in correctness if value > 0) / len(correctness), 6) if correctness else 0.0,
        "format_pass_rate": round(sum(1 for value in format_rewards if value > 0) / len(format_rewards), 6) if format_rewards else 0.0,
        "answer_tag_rate": round(sum(1 for value in answer_tag_rewards if value > 0) / len(answer_tag_rewards), 6) if answer_tag_rewards else 0.0,
        "clean_stop_rate": round(sum(1 for value in clean_stop_rewards if value > 0) / len(clean_stop_rewards), 6) if clean_stop_rewards else 0.0,
        "reward_min": min(total) if total else 0.0,
        "reward_max": max(total) if total else 0.0,
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    project_root = resolve_project_root()
    for run_name, relative_trace_path in DEFAULT_TRACE_PATHS.items():
        trace_path = Path(project_root, relative_trace_path)
        if not trace_path.exists():
            continue
        build_reward_summary(trace_path, trace_path.parents[1] / "reward_summary.json")


if __name__ == "__main__":
    main()
