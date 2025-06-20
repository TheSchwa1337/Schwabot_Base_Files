#!/usr/bin/env python3
"""Drift compensator – positional drift correction vector.

Implements the equation:

    Ξ_drift = Δt · (Ξ_now − Ξ_expected)

Used when ghost logic misses an entry window but the opportunity is still
valid.  Returns a vector that can be added to the next trade signal to adjust
for lag-induced error.
"""

from __future__ import annotations

import numpy as np

__all__: list[str] = ["compute_drift_vector"]


def compute_drift_vector(
    current: np.ndarray,
    expected: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Return drift compensation vector Ξ_drift.

    Parameters
    ----------
    current, expected
        1-D NumPy arrays of identical shape representing current and expected
        state vectors.
    delta_t
        Time lag in **seconds** (or ticks).  Must be non-negative.
    """
    if delta_t < 0:
        raise ValueError("delta_t must be non-negative")
    if current.shape != expected.shape:
        raise ValueError("current and expected must share shape")

    return delta_t * (current - expected) 