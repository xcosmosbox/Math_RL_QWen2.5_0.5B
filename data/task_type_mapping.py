from __future__ import annotations

from pathlib import Path

from utils.io_utils import resolve_project_root, write_json


MAPPING_VERSION = "v1"
TASK_TYPE_RULES = {
    "algebra": [
        "algebra",
        "equation",
        "polynomial",
        "inequality",
        "matrix",
        "linear",
        "quadratic",
    ],
    "geometry": [
        "geometry",
        "triangle",
        "circle",
        "angle",
        "polygon",
        "coordinate geometry",
    ],
    "calculus": [
        "calculus",
        "derivative",
        "integral",
        "limit",
        "differential",
        "series",
    ],
    "number_theory": [
        "number theory",
        "divisibility",
        "prime",
        "modular",
        "congruence",
    ],
    "probability": [
        "probability",
        "statistics",
        "combinatorics probability",
        "expected value",
        "random",
    ],
    "discrete_math": [
        "combinatorics",
        "graph",
        "logic",
        "set theory",
        "recurrence",
        "discrete",
    ],
    "word_problem": [
        "word problem",
        "applied",
        "rate",
        "mixture",
        "work",
        "distance",
        "financial",
    ],
}


def map_topic_to_task_type(topic: str) -> str:
    normalized = (topic or "").strip().lower()
    if not normalized:
        return "other"
    for task_type, keywords in TASK_TYPE_RULES.items():
        for keyword in keywords:
            if keyword in normalized:
                return task_type
    return "other"


def build_mapping_manifest() -> dict[str, object]:
    return {
        "mapping_version": MAPPING_VERSION,
        "task_types": list(TASK_TYPE_RULES.keys()) + ["other"],
        "rules": TASK_TYPE_RULES,
    }


def write_task_type_mapping_manifest() -> dict[str, object]:
    project_root = resolve_project_root()
    output_path = Path(project_root, "data", "metadata", "task_type_mapping.json")
    payload = build_mapping_manifest()
    write_json(output_path, payload)
    print(f"Wrote task type mapping manifest to {output_path}")
    return {"output_path": str(output_path), "mapping_version": MAPPING_VERSION}


def main() -> None:
    write_task_type_mapping_manifest()


if __name__ == "__main__":
    main()
