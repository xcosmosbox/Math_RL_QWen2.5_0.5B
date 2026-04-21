from __future__ import annotations

import importlib.util
import os


if os.getenv("VERL_DISABLE_TORCHAO", "0") == "1":
    _orig_find_spec = importlib.util.find_spec

    def _patched_find_spec(name: str, package: str | None = None):
        if name == "torchao" or name.startswith("torchao."):
            return None
        return _orig_find_spec(name, package)

    importlib.util.find_spec = _patched_find_spec
