import os

# Force override if explicitly set
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() in ("true", "1", "yes")

try:
    if FORCE_CPU:
        raise ImportError("Forced CPU fallback triggered.")
    import cupy as xp

    GPU_ENABLED = True
except ImportError:
    import numpy as xp

    GPU_ENABLED = False


def get_backend():
    return xp


def is_gpu():
    return GPU_ENABLED


def backend_info():
    return {
        "backend": "CuPy" if GPU_ENABLED else "NumPy",
        "accelerated": GPU_ENABLED,
        "force_cpu": FORCE_CPU,
    }
