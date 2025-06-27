from __future__ import annotations

from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Entropy flattener \\u2013 smooths strategy response during uncertain conditions.

Implements the formula:
    \\u03b7(t) = softmax(\\u2212|\\u2202\\u00b2S(t)/\\u2202t\\u00b2| \\u00b7 1/\\u03c3_price)

This module detects when strategy signals are experiencing high second-derivative
volatility and applies entropy-based smoothing to prevent erratic switching.
"""


from typing import Sequence

from core.unified_math_system import unified_math

__all__: list[str] = [
    "entropy_flatten",
    "compute_second_derivative",
    "adaptive_smooth",
]

# ---------------------------------------------------------------------------
# Core flattening logic
# ---------------------------------------------------------------------------


def compute_second_derivative(
    signal: Sequence[float],
) -> np.ndarray:  # noqa: D401
    """Return second derivative \\u2202\\u00b2S/\\u2202t\\u00b2 using finite differences.

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
    x_shifted = x - unified_math.unified_math.max(x)
    exp_x = unified_math.unified_math.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def entropy_flatten(
    signal: Sequence[float],
    price_sigma: float,
    *,
    epsilon: float = 1e-9,
) -> float:  # noqa: D401
    """Return \\u03b7(t) entropy flattening coefficient \\u2208 [0, 1].

    Parameters
    ----------
    signal
        Time series of strategy values S(t).
    price_sigma
        Current price volatility \\u03c3_price.
    epsilon
        Small constant to prevent division by zero.
    """
    if price_sigma <= epsilon:
        return 0.0

    second_deriv = compute_second_derivative(signal)
    if len(second_deriv) == 0:
        return 0.0

    # Compute flattening term: -|\\u2202\\u00b2S/\\u2202t\\u00b2| / \\u03c3_price
    abs_second_deriv = unified_math.unified_math.abs(second_deriv)
    flatten_term = -abs_second_deriv / unified_math.max(price_sigma, epsilon)

    # Apply softmax and return the mean as single coefficient
    smoothed = _softmax(flatten_term)
    return float(unified_math.unified_math.mean(smoothed))


def adaptive_smooth(
    current_value: float,
    smoothed_value: float,
    entropy_coeff: float,
    *,
    alpha: float = 0.1,
) -> float:  # noqa: D401
    """Apply entropy-weighted smoothing between current and smoothed values.

    Returns:
        (1 - \\u03b1\\u00b7\\u03b7) \\u00b7 current + \\u03b1\\u00b7\\u03b7 \\u00b7 smoothed
    where \\u03b7 is the entropy coefficient and \\u03b1 controls smoothing strength.
    """
    weight = alpha * entropy_coeff
    return (1.0 - weight) * current_value + weight * smoothed_value

"""