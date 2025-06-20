#!/usr/bin/env python3
"""Entropy flattener – smooths strategy response during uncertain conditions.

Implements the formula:
    η(t) = softmax(−|∂²S(t)/∂t²| · 1/σ_price)

This module detects when strategy signals are experiencing high second-derivative
volatility and applies entropy-based smoothing to prevent erratic switching.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["entropy_flatten", "compute_second_derivative", "adaptive_smooth"]

# ---------------------------------------------------------------------------
# Core flattening logic
# ---------------------------------------------------------------------------


def compute_second_derivative(signal: Sequence[float]) -> np.ndarray:  # noqa: D401
    """Return second derivative ∂²S/∂t² using finite differences.

    Input signal must have at least 3 points for meaningful computation.
    """
    s = np.asarray(signal, dtype=float)
    if len(s) < 3:
        return np.array([0.0])

    # First derivative via central difference
    first_deriv = np.gradient(s)
    # Second derivative via gradient of first derivative
    second_deriv = np.gradient(first_deriv)
    return second_deriv


def _softmax(x: np.ndarray) -> np.ndarray:  # noqa: D401
    """Numerically stable softmax implementation."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def entropy_flatten(
    signal: Sequence[float],
    price_sigma: float,
    *,
    epsilon: float = 1e-9,
) -> float:  # noqa: D401
    """Return η(t) entropy flattening coefficient ∈ [0, 1].

    Parameters
    ----------
    signal
        Time series of strategy values S(t).
    price_sigma
        Current price volatility σ_price.
    epsilon
        Small constant to prevent division by zero.
    """
    if price_sigma <= epsilon:
        return 0.0

    second_deriv = compute_second_derivative(signal)
    if len(second_deriv) == 0:
        return 0.0

    # Compute flattening term: -|∂²S/∂t²| / σ_price
    abs_second_deriv = np.abs(second_deriv)
    flatten_term = -abs_second_deriv / max(price_sigma, epsilon)

    # Apply softmax and return the mean as single coefficient
    smoothed = _softmax(flatten_term)
    return float(np.mean(smoothed))


def adaptive_smooth(
    current_value: float,
    smoothed_value: float,
    entropy_coeff: float,
    *,
    alpha: float = 0.1,
) -> float:  # noqa: D401
    """Apply entropy-weighted smoothing between current and smoothed values.

    Returns:
        (1 - α·η) · current + α·η · smoothed
    where η is the entropy coefficient and α controls smoothing strength.
    """
    weight = alpha * entropy_coeff
    return (1.0 - weight) * current_value + weight * smoothed_value 