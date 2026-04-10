from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from analysis.build_result_table import run_build_result_table
from analysis.filtering_ablation import run_filtering_ablation
from analysis.sample_failures import run_sample_failures
from data.build_filtered_dataset import run_filtered_build
from data.build_splits import run_split_build
from data.download_deepmath import PRESETS as DOWNLOAD_PRESETS
from data.download_deepmath import run_download_preset
from data.task_type_mapping import write_task_type_mapping_manifest
from eval.config_presets import PRESETS as EVAL_PRESETS
from eval.run_lm_eval import run_eval_preset
from eval.summarize_results import run_summarize_results
from training.export_grpo_dataset import run_export_grpo
from training.export_sft_dataset import run_export_sft
from training.run_grpo import PRESETS as GRPO_PRESETS
from training.run_grpo import run_grpo_preset
from training.run_sft import PRESETS as SFT_PRESETS
from training.run_sft import run_sft_preset
from utils.io_utils import ensure_dir, read_json, resolve_project_root, write_json
from utils.logging_utils import build_run_metadata, configure_logger


DEFAULT_MODE = "smoke"


@dataclass(frozen=True)
class Stage:
    name: str
    action: Callable[[], dict[str, Any] | None]
    is_complete: Callable[[], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project pipeline with artifact-aware skipping")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["smoke", "full", "ablation"])
    parser.add_argument("--force", action="store_true", help="Rerun all stages even if artifacts already exist")
    return parser.parse_args()


def manifest_is_complete(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    payload = read_json(manifest_path)
    required_files = payload.get("required_files", [])
    for relative_path in required_files:
        if not (manifest_path.parent / relative_path).exists():
            return False
    return True


def files_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def download_complete(project_root: Path, preset_name: str) -> bool:
    preset = DOWNLOAD_PRESETS[preset_name]
    return files_exist(
        [
            Path(project_root, preset["output_path"]),
            Path(project_root, preset["metadata_path"]),
        ]
    )


def filtered_complete(project_root: Path) -> bool:
    return files_exist(
        [
            Path(project_root, "data/processed/filtered_strict.jsonl"),
            Path(project_root, "data/processed/filtered_relaxed.jsonl"),
            Path(project_root, "data/metadata/filter_log_strict.json"),
            Path(project_root, "data/metadata/filter_log_relaxed.json"),
        ]
    )


def splits_complete(project_root: Path) -> bool:
    return files_exist(
        [
            Path(project_root, "data/processed/filtered_strict_train.jsonl"),
            Path(project_root, "data/processed/filtered_strict_valid.jsonl"),
            Path(project_root, "data/processed/filtered_strict_test.jsonl"),
            Path(project_root, "data/processed/filtered_relaxed_train.jsonl"),
            Path(project_root, "data/processed/filtered_relaxed_valid.jsonl"),
            Path(project_root, "data/processed/filtered_relaxed_test.jsonl"),
            Path(project_root, "data/metadata/split_manifest_strict.json"),
            Path(project_root, "data/metadata/split_manifest_relaxed.json"),
        ]
    )


def sft_export_complete(project_root: Path) -> bool:
    return files_exist(
        [
            Path(project_root, "data/processed/sft_strict_main_train.json"),
            Path(project_root, "data/processed/sft_strict_main_valid.json"),
            Path(project_root, "data/processed/sft_relaxed_ablation_train.json"),
            Path(project_root, "data/processed/sft_relaxed_ablation_valid.json"),
            Path(project_root, "data/processed/dataset_info.json"),
            Path(project_root, "data/metadata/sft_export_summary.json"),
        ]
    )


def grpo_export_complete(project_root: Path) -> bool:
    return files_exist(
        [
            Path(project_root, "data/processed/grpo_strict_main_train.jsonl"),
            Path(project_root, "data/processed/grpo_strict_main_valid.jsonl"),
            Path(project_root, "data/processed/grpo_relaxed_ablation_train.jsonl"),
            Path(project_root, "data/processed/grpo_relaxed_ablation_valid.jsonl"),
            Path(project_root, "data/metadata/grpo_export_summary.json"),
        ]
    )


def sft_run_complete(project_root: Path, preset_name: str) -> bool:
    output_dir = Path(project_root, SFT_PRESETS[preset_name]["output_dir"])
    return manifest_is_complete(output_dir / "artifact_manifest.json")


def grpo_run_complete(project_root: Path, preset_name: str) -> bool:
    output_dir = Path(project_root, GRPO_PRESETS[preset_name]["output_dir"])
    return manifest_is_complete(output_dir / "artifact_manifest.json")


def eval_preset_complete(project_root: Path, preset_name: str) -> bool:
    preset = EVAL_PRESETS[preset_name]
    required_paths: list[Path] = []
    for task_alias in preset["tasks"]:
        task_dir = Path(preset["output_root"]) / task_alias
        required_paths.extend(
            [
                task_dir / "results.json",
                task_dir / "samples.jsonl",
                task_dir / "run_metadata.json",
                task_dir / "run_config.json",
            ]
        )
    return files_exist(required_paths)


def eval_summary_complete(project_root: Path) -> bool:
    return files_exist(
        [
            Path(project_root, "eval/summaries/leaderboard.csv"),
            Path(project_root, "eval/summaries/regression_table.csv"),
            Path(project_root, "eval/summaries/summary_manifest.json"),
        ]
    )


def analysis_complete(project_root: Path, name: str) -> bool:
    outputs = {
        "build_result_table": [
            Path(project_root, "analysis/results/main_result_table.csv"),
            Path(project_root, "analysis/results/regression_table.csv"),
        ],
        "filtering_ablation": [Path(project_root, "analysis/results/filtering_ablation_summary.json")],
        "sample_failures": [Path(project_root, "analysis/results/sample_failures.jsonl")],
    }
    return files_exist(outputs[name])


def build_modes(project_root: Path) -> dict[str, list[Stage]]:
    smoke_stages = [
        Stage("download_deepmath", lambda: run_download_preset("deepmath_full"), lambda: download_complete(project_root, "deepmath_full")),
        Stage("task_type_mapping", write_task_type_mapping_manifest, lambda: files_exist([Path(project_root, "data/metadata/task_type_mapping.json")])),
        Stage("build_filtered_dataset", lambda: run_filtered_build("all"), lambda: filtered_complete(project_root)),
        Stage("build_splits", lambda: run_split_build("all"), lambda: splits_complete(project_root)),
        Stage("export_sft_dataset", lambda: run_export_sft("all"), lambda: sft_export_complete(project_root)),
        Stage("run_sft_strict_smoke", lambda: run_sft_preset("strict_smoke"), lambda: sft_run_complete(project_root, "strict_smoke")),
        Stage("export_grpo_dataset", lambda: run_export_grpo("all"), lambda: grpo_export_complete(project_root)),
        Stage("run_grpo_strict_smoke", lambda: run_grpo_preset("strict_smoke_grpo"), lambda: grpo_run_complete(project_root, "strict_smoke_grpo")),
        Stage("eval_base_smoke", lambda: run_eval_preset("base_smoke"), lambda: eval_preset_complete(project_root, "base_smoke")),
        Stage("eval_sft_smoke", lambda: run_eval_preset("sft_smoke"), lambda: eval_preset_complete(project_root, "sft_smoke")),
        Stage("eval_rl_smoke", lambda: run_eval_preset("rl_smoke"), lambda: eval_preset_complete(project_root, "rl_smoke")),
        Stage("summarize_eval", run_summarize_results, lambda: eval_summary_complete(project_root)),
        Stage("build_result_table", run_build_result_table, lambda: analysis_complete(project_root, "build_result_table")),
        Stage("filtering_ablation", run_filtering_ablation, lambda: analysis_complete(project_root, "filtering_ablation")),
        Stage("sample_failures", run_sample_failures, lambda: analysis_complete(project_root, "sample_failures")),
    ]
    full_stages = [
        Stage("download_deepmath", lambda: run_download_preset("deepmath_full"), lambda: download_complete(project_root, "deepmath_full")),
        Stage("task_type_mapping", write_task_type_mapping_manifest, lambda: files_exist([Path(project_root, "data/metadata/task_type_mapping.json")])),
        Stage("build_filtered_dataset", lambda: run_filtered_build("all"), lambda: filtered_complete(project_root)),
        Stage("build_splits", lambda: run_split_build("all"), lambda: splits_complete(project_root)),
        Stage("export_sft_dataset", lambda: run_export_sft("all"), lambda: sft_export_complete(project_root)),
        Stage("run_sft_strict_main", lambda: run_sft_preset("strict_main"), lambda: sft_run_complete(project_root, "strict_main")),
        Stage("export_grpo_dataset", lambda: run_export_grpo("all"), lambda: grpo_export_complete(project_root)),
        Stage("run_grpo_strict_main", lambda: run_grpo_preset("strict_main_grpo"), lambda: grpo_run_complete(project_root, "strict_main_grpo")),
        Stage("eval_base_full", lambda: run_eval_preset("base_full"), lambda: eval_preset_complete(project_root, "base_full")),
        Stage("eval_sft_full", lambda: run_eval_preset("sft_full"), lambda: eval_preset_complete(project_root, "sft_full")),
        Stage("eval_rl_full", lambda: run_eval_preset("rl_full"), lambda: eval_preset_complete(project_root, "rl_full")),
        Stage("summarize_eval", run_summarize_results, lambda: eval_summary_complete(project_root)),
        Stage("build_result_table", run_build_result_table, lambda: analysis_complete(project_root, "build_result_table")),
        Stage("filtering_ablation", run_filtering_ablation, lambda: analysis_complete(project_root, "filtering_ablation")),
        Stage("sample_failures", run_sample_failures, lambda: analysis_complete(project_root, "sample_failures")),
    ]
    ablation_stages = [
        Stage("export_sft_dataset", lambda: run_export_sft("all"), lambda: sft_export_complete(project_root)),
        Stage("run_sft_relaxed_ablation", lambda: run_sft_preset("relaxed_ablation"), lambda: sft_run_complete(project_root, "relaxed_ablation")),
        Stage("export_grpo_dataset", lambda: run_export_grpo("all"), lambda: grpo_export_complete(project_root)),
        Stage("run_grpo_relaxed_ablation", lambda: run_grpo_preset("relaxed_ablation_grpo"), lambda: grpo_run_complete(project_root, "relaxed_ablation_grpo")),
        Stage("filtering_ablation", run_filtering_ablation, lambda: analysis_complete(project_root, "filtering_ablation")),
    ]
    return {"smoke": smoke_stages, "full": full_stages, "ablation": ablation_stages}


def run_pipeline(mode: str, *, force: bool = False) -> dict[str, Any]:
    project_root = resolve_project_root()
    logger = configure_logger("run_pipeline")
    modes = build_modes(project_root)
    stages = modes[mode]
    pipeline_output_dir = ensure_dir(Path(project_root, "outputs", "pipeline"))
    stage_results: list[dict[str, Any]] = []

    for stage in stages:
        logger.info("Stage %s: checking artifacts", stage.name)
        if not force and stage.is_complete():
            logger.info("Stage %s: skipped (artifacts complete)", stage.name)
            stage_results.append({"stage": stage.name, "status": "skipped"})
            continue
        logger.info("Stage %s: running", stage.name)
        try:
            result = stage.action() or {}
            stage_results.append({"stage": stage.name, "status": "completed", "result": result})
            logger.info("Stage %s: completed", stage.name)
        except Exception as exc:
            failure_payload = {
                "stage": stage.name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            stage_results.append(failure_payload)
            write_json(
                pipeline_output_dir / f"{mode}_status.json",
                build_run_metadata(mode=mode, force=force, stages=stage_results),
            )
            raise

    summary = build_run_metadata(mode=mode, force=force, stages=stage_results)
    write_json(pipeline_output_dir / f"{mode}_status.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args.mode, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
