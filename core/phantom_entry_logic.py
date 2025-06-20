#!/usr/bin/env python3
"""Phantom entry logic – compute entry probability Pₑ.

Formula implemented:

    Pₑ = Σ_i (ζ_i · τ_i) · exp(−λ_entry · t)

The summation is a dot-product between *zeta* and *tau* vectors (same length).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__: list[str] = ["phantom_entry_probability"]


def phantom_entry_probability(
    *,
    alpha_vec: Sequence[float],
    phi_vec: Sequence[float],
    zeta_final: float,
    mu_echo: float,
    price_now: float,
    profit_band: Tuple[float, float],
    lambda_entry: float = 0.1,
    t: float = 0.0,
    mu_threshold: float = 0.5,
) -> float:
    """Return entry probability P_entry ∈ [0, 1].

    Implements the specification:
        P_entry = tanh( Σ α_i Φ_i(x,t) ) · exp(−λ·t)
    and applies validation gates using *zeta_final*, *mu_echo* and
    the current *price_now* relative to the *profit_band* limits.
    """
    alpha = np.asarray(alpha_vec, dtype=float)
    phi = np.asarray(phi_vec, dtype=float)
    if alpha.shape != phi.shape:
        raise ValueError("alpha_vec and phi_vec must share shape")

    # Core activation term
    activation = math.tanh(float(np.dot(alpha, phi)))
    base_prob = activation * math.exp(-lambda_entry * t)

    # Validation gates
    in_band = price_now <= profit_band[0] or price_now >= profit_band[1]
    if zeta_final <= 0.0 or mu_echo < mu_threshold or not in_band:
        return 0.0

    return max(0.0, min(1.0, base_prob)) 