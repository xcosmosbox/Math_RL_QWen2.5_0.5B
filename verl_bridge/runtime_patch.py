import os

import numpy as np


def apply_patches() -> None:
    if os.environ.get("VERL_RUNTIME_PATCH_APPLIED") == "1":
        return

    from verl import protocol

    original_deep_equal = protocol._deep_equal

    def patched_deep_equal(a, b, visited):
        if isinstance(a, (str, np.str_)) and isinstance(b, (str, np.str_)):
            return str(a) == str(b)
        return original_deep_equal(a, b, visited)

    protocol._deep_equal = patched_deep_equal
    os.environ["VERL_RUNTIME_PATCH_APPLIED"] = "1"
