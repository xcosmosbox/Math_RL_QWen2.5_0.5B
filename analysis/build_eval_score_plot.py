from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import ensure_dir, resolve_project_root


SOURCE_TABLE = "EVAL_RESULT_TABLE_20260414.md"
OUTPUT_PNG = "figs/eval_score_comparison_seaborn.png"
OUTPUT_CSV = "analysis/results/eval_score_comparison.csv"


def extract_main_table(markdown_text: str) -> pd.DataFrame:
    lines = markdown_text.splitlines()
    start = lines.index("## 主结果表") + 1

    table_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break

    if len(table_lines) < 3:
        raise ValueError("主结果表不存在或格式不完整")

    content = "\n".join(table_lines)
    frame = pd.read_csv(StringIO(content), sep="|", engine="python")
    frame = frame.drop(columns=[column for column in frame.columns if str(column).startswith("Unnamed")])
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.iloc[1:].copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    frame = frame.rename(columns={"模型阶段": "model_stage", "训练数据量": "train_size"})
    frame["train_size"] = frame["train_size"].astype(int)

    metric_columns = ["GSM8K", "MATH-500", "TheoremQA", "HellaSwag acc"]
    for column in metric_columns:
        frame[column] = frame[column].astype(float)

    return frame[["model_stage", "train_size", *metric_columns]]


def build_plot(data: pd.DataFrame, output_path: Path) -> None:
    metric_order = ["GSM8K", "MATH-500", "TheoremQA", "HellaSwag acc"]
    stage_order = data["model_stage"].tolist()
    long_df = data.melt(
        id_vars=["model_stage", "train_size"],
        value_vars=metric_order,
        var_name="benchmark",
        value_name="score",
    )

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("Set2", n_colors=len(stage_order))

    plot = sns.catplot(
        data=long_df,
        kind="bar",
        x="benchmark",
        y="score",
        hue="model_stage",
        order=metric_order,
        hue_order=stage_order,
        height=6.6,
        aspect=2.25,
        palette=palette,
    )

    ax = plot.axes.flat[0]
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 0.75)
    ax.tick_params(axis="x", labelrotation=12, labelsize=11)
    plot.set_titles("")
    plot.figure.subplots_adjust(top=0.90, bottom=0.18)
    plot.figure.suptitle("Current Eval Score Comparison")
    if plot._legend is not None:
        plot._legend.set_title("Model Stage")

    for container in ax.containers:
        labels = [f"{bar.get_height():.4f}" for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, rotation=90)

    ensure_dir(output_path.parent)
    plot.savefig(output_path, dpi=220, bbox_inches="tight")


def main() -> None:
    project_root = Path(resolve_project_root())
    source_path = project_root / SOURCE_TABLE
    output_png = project_root / OUTPUT_PNG
    output_csv = project_root / OUTPUT_CSV

    frame = extract_main_table(source_path.read_text(encoding="utf-8"))
    ensure_dir(output_csv.parent)
    frame.to_csv(output_csv, index=False)
    build_plot(frame, output_png)


if __name__ == "__main__":
    main()
