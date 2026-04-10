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
    total = [float(row["total_reward"]) for row in rows]
    payload = {
        "trace_path": str(trace_path),
        "sample_count": len(rows),
        "correctness_reward_mean": _mean(correctness),
        "format_reward_mean": _mean(format_rewards),
        "total_reward_mean": _mean(total),
        "correctness_pass_rate": round(sum(1 for value in correctness if value > 0) / len(correctness), 6) if correctness else 0.0,
        "format_pass_rate": round(sum(1 for value in format_rewards if value > 0) / len(format_rewards), 6) if format_rewards else 0.0,
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
