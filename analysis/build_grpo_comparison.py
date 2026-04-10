from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from utils.io_utils import read_json, resolve_project_root, write_csv_rows, write_json


PREFERRED_METRICS = ["exact_match", "exact_match,none", "acc_norm", "acc"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_metric(rows: list[dict[str, str]], benchmark: str, model_stage: str) -> tuple[str | None, str | None]:
    filtered = [row for row in rows if row["benchmark"] == benchmark and row["model_stage"] == model_stage]
    if not filtered:
        return None, None
    for metric in PREFERRED_METRICS:
        for row in filtered:
            if row["metric"] == metric:
                return metric, row["score"]
    return filtered[0]["metric"], filtered[0]["score"]


def build_grpo_comparison(project_root: Path, output_dir: Path, preset_name: str) -> dict[str, Any]:
    leaderboard_rows = read_csv_rows(Path(project_root, "eval", "summaries", "leaderboard.csv"))
    reward_summary_path = output_dir / "reward_summary.json"
    reward_summary = read_json(reward_summary_path) if reward_summary_path.exists() else {}
    comparison_rows: list[dict[str, Any]] = []

    for benchmark in ("MATH-500", "GSM8K", "HellaSwag"):
        sft_metric, sft_score = pick_metric(leaderboard_rows, benchmark, "sft")
        rl_metric, rl_score = pick_metric(leaderboard_rows, benchmark, "rl")
        comparison_rows.append(
            {
                "preset": preset_name,
                "benchmark": benchmark,
                "metric": rl_metric or sft_metric,
                "sft_score": sft_score,
                "rl_score": rl_score,
                "total_reward_mean": reward_summary.get("total_reward_mean"),
                "format_pass_rate": reward_summary.get("format_pass_rate"),
            }
        )

    csv_path = output_dir / "sft_vs_rl_comparison.csv"
    json_path = output_dir / "sft_vs_rl_comparison.json"
    write_csv_rows(
        csv_path,
        comparison_rows,
        fieldnames=["preset", "benchmark", "metric", "sft_score", "rl_score", "total_reward_mean", "format_pass_rate"],
    )
    payload = {"preset": preset_name, "reward_summary": reward_summary, "rows": comparison_rows}
    write_json(json_path, payload)
    return payload


def main() -> None:
    project_root = resolve_project_root()
    build_grpo_comparison(project_root, Path(project_root, "outputs", "grpo", "strict_main"), "strict_main_grpo")
    build_grpo_comparison(project_root, Path(project_root, "outputs", "grpo", "relaxed_ablation_grpo"), "relaxed_ablation_grpo")


if __name__ == "__main__":
    main()
