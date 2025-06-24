"""Profit cycle allocator for basket distribution."""

from __future__ import annotations
import numpy as np


def allocate(phi: float, alphas: list[float]) -> np.ndarray:
    """Split entry weight across baskets proportionally.

    Compute allocation: alloc_i = α_i·Φ / Σα

    Args:
        phi: Total entry signal strength
        alphas: Per-basket allocation coefficients

    Returns:
        Per-basket allocation array that sums to |phi|

    Raises:
        ValueError: If alphas sum to zero
    """
    if not alphas:
        return np.array([])

    a = np.array(alphas, dtype=float)
    alpha_sum = a.sum()

    if alpha_sum == 0:
        raise ValueError("Alpha coefficients sum to zero")

    return phi * (a / alpha_sum)
