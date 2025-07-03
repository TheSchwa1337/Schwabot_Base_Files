#!/usr/bin/env python3
"""Unified Tensor Algebra — compatibility stub.
Full tensor operations are provided by `core.advanced_tensor_algebra`.
"""
from core.advanced_tensor_algebra import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
