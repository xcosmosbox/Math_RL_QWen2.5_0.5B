from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.io_utils import ensure_dir, resolve_project_root, write_json


DEFAULT_RUNS = {
    "strict_main": {
        "trainer_state": "outputs/sft/strict_main/trainer_state.json",
        "output_png": "outputs/sft/strict_main/loss_curve.png",
        "output_json": "outputs/sft/strict_main/loss_curve_data.json",
    },
    "relaxed_ablation": {
        "trainer_state": "outputs/sft/relaxed_ablation/trainer_state.json",
        "output_png": "outputs/sft/relaxed_ablation/loss_curve.png",
        "output_json": "outputs/sft/relaxed_ablation/loss_curve_data.json",
    },
}


def extract_loss_points(trainer_state: dict[str, Any]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for item in trainer_state.get("log_history", []):
        if "loss" in item and "step" in item:
            points.append({"step": float(item["step"]), "loss": float(item["loss"])})
    return points


def build_loss_curve(trainer_state_path: Path, output_path: Path) -> list[dict[str, float]]:
    with trainer_state_path.open("r", encoding="utf-8") as handle:
        trainer_state = json.load(handle)
    points = extract_loss_points(trainer_state)
    if not points:
        raise ValueError(f"No loss points found in {trainer_state_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render loss curves") from exc

    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 4.5))
    plt.plot([point["step"] for point in points], [point["loss"] for point in points], linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("SFT Training Loss Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return points


def main() -> None:
    project_root = resolve_project_root()
    for run_name, config in DEFAULT_RUNS.items():
        trainer_state_path = Path(project_root, config["trainer_state"])
        if not trainer_state_path.exists():
            continue
        output_path = Path(project_root, config["output_png"])
        points = build_loss_curve(trainer_state_path, output_path)
        write_json(Path(project_root, config["output_json"]), {"run_name": run_name, "points": points})


if __name__ == "__main__":
    main()
