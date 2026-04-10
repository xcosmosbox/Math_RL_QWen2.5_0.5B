from __future__ import annotations

import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.io_utils import write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def build_run_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "created_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    metadata.update(extra)
    return metadata


def write_run_metadata(path: Path | str, **extra: Any) -> Path:
    return write_json(path, build_run_metadata(**extra))


def write_failure_summary(path: Path | str, *, stage: str, error: BaseException, **extra: Any) -> Path:
    payload = build_run_metadata(
        stage=stage,
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error),
        **extra,
    )
    return write_json(path, payload)


def run_streaming_command(
    command: list[str],
    *,
    cwd: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    streamed_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        streamed_lines.append(line)
        print(line, end="")
    return_code = process.wait()
    stdout = "".join(streamed_lines)
    if logger is not None:
        logger.info("Command finished with exit code %s: %s", return_code, " ".join(command))
    return subprocess.CompletedProcess(command, return_code, stdout=stdout, stderr="")
