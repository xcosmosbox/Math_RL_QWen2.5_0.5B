from __future__ import annotations

import csv
from pathlib import Path

from utils.io_utils import ensure_dir, resolve_project_root, write_csv_rows


PREFERRED_METRICS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "acc_norm,none",
    "acc,none",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_metric(rows: list[dict[str, str]], benchmark: str, model_stage: str) -> dict[str, str] | None:
    subset = [row for row in rows if row["benchmark"] == benchmark and row["model_stage"] == model_stage]
    if not subset:
        return None
    for metric in PREFERRED_METRICS:
        for row in subset:
            if row["metric"] == metric:
                return row
    return subset[0]


def run_build_result_table() -> dict[str, str]:
    project_root = resolve_project_root()
    leaderboard_rows = read_csv_rows(Path(project_root, "eval", "summaries", "leaderboard.csv"))
    output_dir = ensure_dir(Path(project_root, "analysis", "results"))

    main_rows: list[dict[str, str]] = []
    for benchmark in ("MATH-500", "GSM8K"):
        for model_stage in ("base", "sft", "rl"):
            chosen = choose_metric(leaderboard_rows, benchmark, model_stage)
            if chosen:
                main_rows.append(chosen)
    regression_rows: list[dict[str, str]] = []
    for model_stage in ("base", "sft", "rl"):
        chosen = choose_metric(leaderboard_rows, "HellaSwag", model_stage)
        if chosen:
            regression_rows.append(chosen)

    fieldnames = leaderboard_rows[0].keys() if leaderboard_rows else ["run_date", "model_stage", "benchmark", "metric", "score", "stderr", "model_path", "gen_config", "harness_version", "runtime_minutes"]
    write_csv_rows(output_dir / "main_result_table.csv", main_rows, fieldnames=list(fieldnames))
    write_csv_rows(output_dir / "regression_table.csv", regression_rows, fieldnames=list(fieldnames))
    return {
        "output_dir": str(output_dir),
        "main_result_table": str(output_dir / "main_result_table.csv"),
        "regression_table": str(output_dir / "regression_table.csv"),
    }


def main() -> None:
    run_build_result_table()


if __name__ == "__main__":
    main()
