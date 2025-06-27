# -*- coding: utf - 8 -*-\\nfrom typing import Sequence
""""""
""""""
""""""
""""""
# -*- coding: utf - 8 -*-\\nfrom typing import Sequence

""""""
""""""
""""""
""""""
# -*- coding: utf - 8 -*-\\nfrom typing import Sequence
# -*- coding: utf - 8 -*-\\nfrom typing import Sequence
from __future__ import annotations
import math


# """Entropy flattener - smooths strategy response during uncertain conditions."""

Implements the formula:
eta(t) = softmax(- | partial**2S(t) / partialt**2 | . 1 / sigma_price)

This module detects when strategy signals are experiencing high second - derivative
volatility and applies entropy - based smoothing to prevent erratic switching.
""""""
""""""
""""""


# from core.unified_math_system import unified_math  # F811: duplicate import

__all__: list[str] = []
"entropy_flatten",
"compute_second_derivative",
"adaptive_smooth",

# ---------------------------------------------------------------------------
# Core flattening logic
# ---------------------------------------------------------------------------

    def compute_second_derivative():

signal: Sequence[float],
    -> np.ndarray:  # noqa: D401
"""Return second derivative partial**2S / partialt**2 using finite differences."""
""""""
""""""

Input signal must have at least 3 points for meaningful computation.
""""""
""""""
""""""
s = np.asarray(signal, dtype=float)
    if len(s) < 3:
#         return np.array([0.0])

# First derivative via central difference
first_deriv = np.gradient(s)
# Second derivative via gradient of first derivative
second_deriv = np.gradient(first_deriv)
#     return second_deriv


def _softmax(x: np.ndarray) -> np.ndarray:  # noqa: D401
    """Numerically stable softmax implementation."""


""""""
""""""


x_shifted = x - unified_math.unified_math.max(x)
    exp_x = unified_math.unified_math.exp(x_shifted)
#     return exp_x / np.sum(exp_x)


def entropy_flatten():


signal: Sequence[float],
price_sigma: float,
*,
epsilon: float = 1e-9,
    -> float:  # noqa: D401
"""Return eta(t) entropy flattening coefficient in [0, 1]."""
""""""
""""""

Parameters
----------
signal
Time series of strategy values S(t).
    price_sigma
Current price volatility sigma_price.
epsilon
Small constant to prevent division by zero.
""""""
""""""
""""""
    if price_sigma <= epsilon:
#         return 0.0

second_deriv = compute_second_derivative(signal)
    if len(second_deriv) == 0:
#         return 0.0

# Compute flattening term: -|partial**2S / partialt**2| / sigma_price
abs_second_deriv = unified_math.unified_math.abs(second_deriv)
    flatten_term = -abs_second_deriv / unified_math.max(price_sigma, epsilon)

# Apply softmax and return the mean as single coefficient
smoothed = _softmax(flatten_term)
#     return float(unified_math.unified_math.mean(smoothed))


def adaptive_smooth():


current_value: float,
smoothed_value: float,
entropy_coeff: float,
*,
alpha: float = 0.1,
    -> float:  # noqa: D401
"""Apply entropy - weighted smoothing between current and smoothed values."""
""""""
""""""

Returns:
(1 - alpha.eta) . current + alpha.eta . smoothed
    where eta is the entropy coefficient and alpha controls smoothing strength.
""""""
""""""
""""""
weight = alpha * entropy_coeff
#     return (1.0 - weight) * current_value + weight * smoothed_value


""""""
""""""
""""""
