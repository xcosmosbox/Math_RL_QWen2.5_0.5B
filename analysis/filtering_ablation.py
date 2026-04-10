from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io_utils import read_json, resolve_project_root, write_json


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def run_filtering_ablation() -> dict[str, str]:
    project_root = resolve_project_root()
    strict_filter = load_optional_json(Path(project_root, "data", "metadata", "filter_log_strict.json"))
    relaxed_filter = load_optional_json(Path(project_root, "data", "metadata", "filter_log_relaxed.json"))
    strict_split = load_optional_json(Path(project_root, "data", "metadata", "split_manifest_strict.json"))
    relaxed_split = load_optional_json(Path(project_root, "data", "metadata", "split_manifest_relaxed.json"))
    strict_sft = load_optional_json(Path(project_root, "outputs", "sft", "strict_main", "loss_accuracy_comparison.json"))
    relaxed_sft = load_optional_json(Path(project_root, "outputs", "sft", "relaxed_ablation", "loss_accuracy_comparison.json"))
    strict_grpo = load_optional_json(Path(project_root, "outputs", "grpo", "strict_main", "sft_vs_rl_comparison.json"))
    relaxed_grpo = load_optional_json(Path(project_root, "outputs", "grpo", "relaxed_ablation_grpo", "sft_vs_rl_comparison.json"))

    summary = {
        "dataset_counts": {
            "strict_kept_count": strict_filter.get("kept_count"),
            "relaxed_kept_count": relaxed_filter.get("kept_count"),
            "strict_verifiable_count": strict_filter.get("verifiable_count"),
            "relaxed_verifiable_count": relaxed_filter.get("verifiable_count"),
        },
        "task_type_distribution": {
            "strict": strict_filter.get("retention_summary", {}).get("kept_by_task_type", {}),
            "relaxed": relaxed_filter.get("retention_summary", {}).get("kept_by_task_type", {}),
        },
        "split_counts": {
            "strict": strict_split.get("split_summary", {}).get("counts", {}),
            "relaxed": relaxed_split.get("split_summary", {}).get("counts", {}),
        },
        "sft_comparison": {
            "strict": strict_sft.get("rows", []),
            "relaxed": relaxed_sft.get("rows", []),
        },
        "grpo_comparison": {
            "strict": strict_grpo.get("rows", []),
            "relaxed": relaxed_grpo.get("rows", []),
        },
    }
    output_path = Path(project_root, "analysis", "results", "filtering_ablation_summary.json")
    write_json(output_path, summary)
    return {"output_path": str(output_path)}


def main() -> None:
    run_filtering_ablation()


if __name__ == "__main__":
    main()
