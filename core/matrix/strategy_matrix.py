"""Strategy matrix for adaptive vector projection."""

from __future__ import annotations
from core.unified_math_system import unified_math
import numpy as np
# from core.unified_math_system import unified_math  # F811: duplicate import


def project(weights: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute adaptive projection Πₓ = Σ wᵢ·Vᵢ.

Perform vectorized dot product for weighted vector combination
supporting both static and dynamic weight updates.

Args:
weights: Weight coefficients array
vectors: Vector matrix (weights axis should align)

Returns:
Projected vector result

Raises:
ValueError: If dimension mismatch occurs
"""
    if weights.shape[0] != vectors.shape[0]:
        raise ValueError(
            f"Weight dimension {weights.shape[0]} != "
f"vector dimension {vectors.shape[0]}"


    return np.tensordot(weights, vectors, axes=1)
