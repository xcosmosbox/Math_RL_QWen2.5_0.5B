import os


if os.environ.get("VERL_RUNTIME_PATCH") == "1":
    from verl_bridge.runtime_patch import apply_patches

    apply_patches()
