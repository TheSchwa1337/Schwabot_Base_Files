#!/usr/bin/env python3
"""Phantom exit logic – compute exit score Pₓ.

Approximates the improper integral:

    Pₓ = lim_{T→∞} ∫₀^{T} φ_exit(t) dt / Δ⟨profit⟩

Numerically we evaluate a discrete array *phi_exit* and divide by profit delta.
"""

from __future__ import annotations

import math

__all__: list[str] = ["phantom_exit_score"]


def phantom_exit_score(
    *,
    lambda_trust: float,
    profit_delta: float,
    zeta_derivative: float,
    halt_bias: float = 0.0,
) -> float:
    """Return exit probability P_exit ∈ [0, 1].

    Implements:
        P_exit = sigmoid( λ_trust + Δprofit · dζ/dt − ε_halt )
    where ε_halt is *halt_bias*.
    """
    val = lambda_trust + profit_delta * zeta_derivative - halt_bias
    return 1.0 / (1.0 + math.exp(-val)) 