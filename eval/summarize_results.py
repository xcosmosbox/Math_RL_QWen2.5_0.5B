from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.config_presets import TASK_SPECS
from utils.io_utils import ensure_dir, resolve_project_root, write_csv_rows, write_json


SUMMARY_FIELDS = [
    "run_date",
    "model_stage",
    "benchmark",
    "metric",
    "score",
    "stderr",
    "model_path",
    "gen_config",
    "harness_version",
    "runtime_minutes",
]


def extract_rows(
    result_payload: dict[str, Any],
    run_config: dict[str, Any],
    runtime_minutes: str,
    run_date: str,
) -> list[dict[str, Any]]:
    benchmark = run_config["benchmark"]
    rows: list[dict[str, Any]] = []
    result_block = result_payload.get("results", {})
    for task_id, metrics in result_block.items():
        for metric_name, metric_value in metrics.items():
            if metric_name.endswith("_stderr"):
                continue
            stderr_key = f"{metric_name}_stderr"
            rows.append(
                {
                    "run_date": run_date,
                    "model_stage": run_config["model_stage"],
                    "benchmark": benchmark,
                    "metric": metric_name,
                    "score": metric_value,
                    "stderr": metrics.get(stderr_key),
                    "model_path": run_config["peft_path"] or run_config["model_path"],
                    "gen_config": json.dumps(run_config["gen_kwargs"], ensure_ascii=False, sort_keys=True),
                    "harness_version": run_config["harness_version"],
                    "runtime_minutes": runtime_minutes.strip(),
                }
            )
    return rows


def summarize_eval_outputs(project_root: Path) -> list[dict[str, Any]]:
    outputs_root = Path(project_root, "eval", "outputs")
    rows: list[dict[str, Any]] = []
    for result_path in sorted(outputs_root.glob("*/*/results.json")):
        task_dir = result_path.parent
        run_config_path = task_dir / "run_config.json"
        runtime_path = task_dir / "runtime.txt"
        if not run_config_path.exists():
            continue
        with result_path.open("r", encoding="utf-8") as handle:
            result_payload = json.load(handle)
        with run_config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        runtime_minutes = runtime_path.read_text(encoding="utf-8") if runtime_path.exists() else ""
        run_date = task_dir.parent.name.split("_")[0]
        rows.extend(extract_rows(result_payload, run_config, runtime_minutes, run_date))
    return rows


def run_summarize_results() -> dict[str, Any]:
    project_root = resolve_project_root()
    summaries_dir = ensure_dir(Path(project_root, "eval", "summaries"))
    rows = summarize_eval_outputs(project_root)
    write_csv_rows(summaries_dir / "leaderboard.csv", rows, fieldnames=SUMMARY_FIELDS)
    regression_rows = [row for row in rows if row["benchmark"] == TASK_SPECS["hellaswag"]["benchmark"]]
    write_csv_rows(summaries_dir / "regression_table.csv", regression_rows, fieldnames=SUMMARY_FIELDS)
    write_json(summaries_dir / "summary_manifest.json", {"row_count": len(rows)})
    return {"row_count": len(rows), "summaries_dir": str(summaries_dir)}


def main() -> None:
    run_summarize_results()


if __name__ == "__main__":
    main()
