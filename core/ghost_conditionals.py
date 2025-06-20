#!/usr/bin/env python3
"""Ghost-conditional helpers.

Implements the routing activation Γᵣ = σ(Δₜ · Ξ_ghost).
A lightweight logistic gate converts the continuous product of *delta_t*
(seconds since last activation) and *xi_ghost* (scalar 0-1 intensity) into a
probability.  Down-stream the router can compare this value against a policy
threshold.
"""

from __future__ import annotations

import math
from typing import Final

__all__: list[str] = ["ghost_route_activation"]

_K: Final = 1.0  # logistic steepness


def _sigmoid(x: float) -> float:  # noqa: D401
    return 1.0 / (1.0 + math.exp(-_K * x))


def ghost_route_activation(delta_t: float, xi_ghost: float) -> float:
    """Return Γᵣ activation probability in (0, 1).

    Parameters
    ----------
    delta_t
        Time delta since last ghost evaluation (seconds).
    xi_ghost
        Scalar intensity of current ghost signal, expected in [0, 1].
    """
    return _sigmoid(delta_t * xi_ghost) 