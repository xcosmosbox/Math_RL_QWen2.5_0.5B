from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from utils.io_utils import resolve_project_root, write_csv_rows, write_json


PREFERRED_METRICS = ["exact_match", "exact_match,none", "acc_norm", "acc"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def final_train_loss(output_dir: Path) -> float | None:
    trainer_state_path = output_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return None
    with trainer_state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    losses = [item["loss"] for item in payload.get("log_history", []) if "loss" in item]
    return float(losses[-1]) if losses else None


def pick_metric(rows: list[dict[str, str]], benchmark: str, model_stage: str) -> tuple[str | None, str | None]:
    filtered = [row for row in rows if row["benchmark"] == benchmark and row["model_stage"] == model_stage]
    if not filtered:
        return None, None
    for metric in PREFERRED_METRICS:
        for row in filtered:
            if row["metric"] == metric:
                return metric, row["score"]
    return filtered[0]["metric"], filtered[0]["score"]


def build_sft_comparison(project_root: Path, output_dir: Path, preset_name: str) -> dict[str, Any]:
    leaderboard_path = Path(project_root, "eval", "summaries", "leaderboard.csv")
    rows = read_csv_rows(leaderboard_path)
    final_loss = final_train_loss(output_dir)

    comparison_rows: list[dict[str, Any]] = []
    for benchmark in ("MATH-500", "GSM8K", "HellaSwag"):
        base_metric, base_score = pick_metric(rows, benchmark, "base")
        sft_metric, sft_score = pick_metric(rows, benchmark, "sft")
        comparison_rows.append(
            {
                "preset": preset_name,
                "benchmark": benchmark,
                "metric": sft_metric or base_metric,
                "base_score": base_score,
                "sft_score": sft_score,
                "final_train_loss": final_loss,
            }
        )

    csv_path = output_dir / "loss_accuracy_comparison.csv"
    json_path = output_dir / "loss_accuracy_comparison.json"
    write_csv_rows(
        csv_path,
        comparison_rows,
        fieldnames=["preset", "benchmark", "metric", "base_score", "sft_score", "final_train_loss"],
    )
    payload = {"preset": preset_name, "final_train_loss": final_loss, "rows": comparison_rows}
    write_json(json_path, payload)
    return payload


def main() -> None:
    project_root = resolve_project_root()
    build_sft_comparison(project_root, Path(project_root, "outputs", "sft", "strict_main"), "strict_main")
    build_sft_comparison(project_root, Path(project_root, "outputs", "sft", "relaxed_ablation"), "relaxed_ablation")


if __name__ == "__main__":
    main()
