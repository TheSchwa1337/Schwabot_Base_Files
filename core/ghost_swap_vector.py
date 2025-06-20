#!/usr/bin/env python3
"""Ghost swap vector – trade simulation projection.

Implements the Φ₍ghost₎ matrix from the Ghost design doc:

    Φ₍ghost₎ = M(t) · σ(W + B) + Ψ_noise

where
• ``M(t)``   – market-state transformation matrix (time varying).
• ``W`` / ``B`` – learned weight & bias arrays (same shape).
• ``σ``       – element-wise sigmoid.
• ``Ψ_noise`` – optional additive noise (same shape as the sigmoid output).

The helper is intentionally simple – it does *no* learning, gradient updates
or fancy broadcasting.  All arrays must have the same shape so we avoid silent
numpy broadcasting errors.
"""

from __future__ import annotations

import numpy as np
from typing import Final

__all__: list[str] = ["ghost_swap_vector"]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_SIGMOID_K: Final = 1.0  # logistic steepness


def _sigmoid(x: np.ndarray, k: float = _SIGMOID_K) -> np.ndarray:  # noqa: D401
    """Vectorised logistic function 1 / (1 + exp(-k·x))."""
    return 1.0 / (1.0 + np.exp(-k * x))


def ghost_swap_vector(
    market_matrix: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    noise: np.ndarray | None = None,
    sigmoid_k: float = _SIGMOID_K,
) -> np.ndarray:
    """Return ghost trade simulation matrix Φ₍ghost₎.

    Parameters
    ----------
    market_matrix
        ``M(t)`` – current market state features (2-D array).
    weights, bias
        Learned parameters (same shape as ``market_matrix``).  No broadcasting
        is applied – exact shape match is required.
    noise
        Optional additive noise ``Ψ_noise``.  If ``None``, a zero matrix is
        used.
    sigmoid_k
        Steepness parameter *k* of the logistic.  Higher ⇒ harder gate.
    """
    if not (market_matrix.shape == weights.shape == bias.shape):
        raise ValueError("market_matrix, weights and bias must share shape")

    # σ(W + B) term
    activated = _sigmoid(weights + bias, k=sigmoid_k)

    # Core multiplication
    phi = market_matrix * activated

    if noise is None:
        return phi

    if noise.shape != phi.shape:
        raise ValueError("noise must match output shape")
    return phi + noise 