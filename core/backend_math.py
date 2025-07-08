import os

FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

GPU_ENABLED = False

try:
    if FORCE_CPU:
        raise ImportError("Forced CPU override")
    import cupy as xp
    # Runtime check for actual GPU availability
    try:
        _ = xp.zeros((1,))
        GPU_ENABLED = True
    except Exception:
        import numpy as xp
        GPU_ENABLED = False
except ImportError:
    import numpy as xp
    GPU_ENABLED = False

def get_backend():
    return xp

def is_gpu():
    return GPU_ENABLED 