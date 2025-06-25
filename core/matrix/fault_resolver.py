"""Matrix fault resolver for rank consistency checking."""

from __future__ import annotations
from core.unified_math_system import unified_math
import numpy as np
# from core.unified_math_system import unified_math  # F811: duplicate import


def check_rank(matrix: np.ndarray, eps: int = 0) -> None:
    """Check matrix rank consistency and raise if drift exceeds threshold.

    Verify rank stability: δ = rank(A) – rank(A·Aᵀ)
    Raise ValueError if |δ| > eps

    Args:
        matrix: Input matrix to check
        eps: Maximum allowed rank drift (default 0)

    Raises:
        ValueError: If rank drift exceeds threshold
    """
    r1 = np.linalg.matrix_rank(matrix)
    r2 = np.linalg.matrix_rank(matrix @ matrix.T)

    drift = unified_math.abs(r1 - r2)
    if drift > eps:
        raise ValueError(f"Rank drift {r1}->{r2} = {drift} > {eps}")
