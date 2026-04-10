from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from eval.config_presets import DEFAULT_PRESET, PRESETS, TASK_SPECS
from utils.io_utils import ensure_dir, make_temp_dir, replace_dir, resolve_project_root, write_json, write_text
from utils.logging_utils import build_run_metadata, configure_logger, run_streaming_command, write_failure_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-eval-harness presets")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    return parser.parse_args()


def validate_preset(preset_name: str, preset: dict[str, Any]) -> None:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    if any("theorem" in task.lower() for task in preset["tasks"]):
        raise ValueError("TheoremQA is forbidden in this project")
    missing = [task for task in preset["tasks"] if task not in TASK_SPECS]
    if missing:
        raise ValueError(f"Unknown task aliases: {missing}")


def get_harness_version() -> str:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def print_available_tasks() -> None:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise RuntimeError("lm-eval-harness is not installed") from exc
    manager = TaskManager()
    for task_name in sorted(manager.all_tasks):
        print(task_name)


def build_command(preset: dict[str, Any], task_alias: str, task_dir: Path) -> list[str]:
    task_spec = TASK_SPECS[task_alias]
    model_args_parts = [
        f"pretrained={preset['model_path']}",
        "trust_remote_code=True",
        "dtype=bfloat16",
    ]
    if preset.get("peft_path"):
        model_args_parts.append(f"peft={preset['peft_path']}")
    model_args = ",".join(model_args_parts)
    gen_kwargs = ",".join(f"{key}={value}" for key, value in task_spec["gen_kwargs"].items())
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        task_spec["task_id"],
        "--device",
        preset["device"],
        "--batch_size",
        str(task_spec["batch_size"]),
        "--output_path",
        str(task_dir),
        "--gen_kwargs",
        gen_kwargs,
    ]
    if task_spec.get("apply_chat_template"):
        command.append("--apply_chat_template")
    if preset.get("log_samples"):
        command.append("--log_samples")
    if preset.get("limit") is not None:
        command.extend(["--limit", str(preset["limit"])])
    return command


def standardize_task_artifacts(task_dir: Path) -> None:
    json_candidates = [
        path
        for path in task_dir.rglob("*.json")
        if path.name != "run_config.json" and path.name != "run_metadata.json"
    ]
    jsonl_candidates = list(task_dir.rglob("*.jsonl"))
    if json_candidates:
        latest_json = max(json_candidates, key=lambda path: path.stat().st_mtime)
        shutil.copyfile(latest_json, task_dir / "results.json")
    if jsonl_candidates:
        latest_jsonl = max(jsonl_candidates, key=lambda path: path.stat().st_mtime)
        shutil.copyfile(latest_jsonl, task_dir / "samples.jsonl")


def run_eval_preset(preset_name: str) -> dict[str, Any]:
    preset = PRESETS[preset_name]
    validate_preset(preset_name, preset)
    project_root = resolve_project_root()
    logger = configure_logger("eval.run_lm_eval")
    harness_version = get_harness_version()

    for task_alias in preset["tasks"]:
        task_spec = TASK_SPECS[task_alias]
        final_task_dir = Path(preset["output_root"]) / task_alias
        temp_task_dir = make_temp_dir(Path(preset["output_root"]), prefix=f".{task_alias}_tmp_")
        command = build_command(preset, task_alias, temp_task_dir)
        run_config = {
            "preset": preset_name,
            "model_stage": preset["model_stage"],
            "model_path": preset["model_path"],
            "peft_path": preset.get("peft_path"),
            "task_alias": task_alias,
            "task_id": task_spec["task_id"],
            "benchmark": task_spec["benchmark"],
            "gen_kwargs": task_spec["gen_kwargs"],
            "batch_size": task_spec["batch_size"],
            "limit": preset.get("limit"),
            "apply_chat_template": task_spec.get("apply_chat_template", False),
            "harness_version": harness_version,
        }
        write_json(temp_task_dir / "run_config.json", run_config)
        write_text(temp_task_dir / "run_command.txt", " ".join(command) + "\n")

        start_time = time.time()
        try:
            result = run_streaming_command(command, cwd=project_root, logger=logger)
            runtime_minutes = round((time.time() - start_time) / 60.0, 4)
            write_text(temp_task_dir / "stdout.log", result.stdout)
            write_text(temp_task_dir / "stderr.log", result.stderr)
            write_text(temp_task_dir / "runtime.txt", f"{runtime_minutes}\n")
            if result.returncode != 0:
                raise RuntimeError(f"lm-eval failed for task {task_alias} with exit code {result.returncode}")
            standardize_task_artifacts(temp_task_dir)
            write_json(
                temp_task_dir / "run_metadata.json",
                build_run_metadata(
                    preset=preset_name,
                    model_stage=preset["model_stage"],
                    benchmark=task_spec["benchmark"],
                    task_id=task_spec["task_id"],
                    harness_version=harness_version,
                    runtime_minutes=runtime_minutes,
                ),
            )
            replace_dir(temp_task_dir, final_task_dir)
            logger.info("Completed %s for preset %s", task_alias, preset_name)
        except Exception as exc:
            runtime_minutes = round((time.time() - start_time) / 60.0, 4)
            write_failure_summary(
                temp_task_dir / "failure_summary.json",
                stage="lm_eval",
                error=exc,
                preset=preset_name,
                task_alias=task_alias,
                runtime_minutes=runtime_minutes,
            )
            raise
    return {
        "preset": preset_name,
        "output_root": preset["output_root"],
        "tasks": list(preset["tasks"]),
        "harness_version": harness_version,
    }


def main() -> None:
    args = parse_args()
    run_eval_preset(args.preset)


if __name__ == "__main__":
    main()
