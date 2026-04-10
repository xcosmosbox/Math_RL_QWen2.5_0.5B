from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path | str, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def write_text(path: Path | str, content: str, encoding: str = "utf-8") -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(content, encoding=encoding)
    return path_obj


def read_json(path: Path | str, encoding: str = "utf-8") -> Any:
    with Path(path).open("r", encoding=encoding) as handle:
        return json.load(handle)


def write_json(
    path: Path | str,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    encoding: str = "utf-8",
) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding=encoding) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
        handle.write("\n")
    return path_obj


def read_jsonl(path: Path | str, encoding: str = "utf-8") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding=encoding) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(
    path: Path | str,
    rows: Iterable[dict[str, Any]],
    *,
    encoding: str = "utf-8",
) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding=encoding) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path_obj


def write_csv_rows(
    path: Path | str,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str],
    encoding: str = "utf-8",
) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path_obj


def list_json_files(path: Path | str) -> list[Path]:
    return sorted(Path(path).glob("*.json"))


def list_jsonl_files(path: Path | str) -> list[Path]:
    return sorted(Path(path).glob("*.jsonl"))


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in [":", "#", "\n", "{", "}", "[", "]", ",", "\"", "'"]):
        escaped = text.replace("\\", "\\\\").replace("\"", "\\\"")
        return f"\"{escaped}\""
    return text


def _to_yaml_lines(payload: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(payload, dict):
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_to_yaml_lines(value, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
        return lines
    if isinstance(payload, list):
        lines = []
        for item in payload:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_to_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(payload)}"]


def write_yaml(path: Path | str, payload: Any, *, encoding: str = "utf-8") -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    content = "\n".join(_to_yaml_lines(payload)) + "\n"
    path_obj.write_text(content, encoding=encoding)
    return path_obj


def make_temp_dir(parent: Path | str, *, prefix: str) -> Path:
    ensure_dir(parent)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))


def replace_dir(src: Path | str, dst: Path | str) -> Path:
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), str(dst_path))
    return dst_path
