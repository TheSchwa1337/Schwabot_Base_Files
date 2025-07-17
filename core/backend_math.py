#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Math Module - GPU/CPU Acceleration Support
=================================================

Provides backend support for mathematical operations with GPU acceleration
when available, falling back to CPU (NumPy) when needed.
"""

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
    """Get the current backend (CuPy or NumPy)."""
    return xp


def is_gpu():
    """Check if GPU acceleration is enabled."""
    return GPU_ENABLED


def backend_info():
    """Get information about the current backend."""
    return {
        "backend": "CuPy" if GPU_ENABLED else "NumPy",
        "accelerated": GPU_ENABLED,
        "force_cpu": FORCE_CPU,
    }