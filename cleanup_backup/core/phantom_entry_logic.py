from __future__ import annotations

from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Phantom entry logic \\u2013 compute entry probability P\\u2091.

Formula implemented:

    P\\u2091 = \\u03a3_i (\\u03b6_i \\u00b7 \\u03c4_i) \\u00b7 exp(\\u2212\\u03bb_entry \\u00b7 t)

The summation is a dot-product between *zeta* and *tau* vectors (same length).
"""


from core.unified_math_system import unified_math
from typing import Sequence, Tuple

from core.unified_math_system import unified_math

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
    """Return entry probability P_entry \\u2208 [0, 1].

    Implements the specification:
        P_entry = tanh( \\u03a3 \\u03b1_i \\u03a6_i(x,t) ) \\u00b7 exp(\\u2212\\u03bb\\u00b7t)
    and applies validation gates using *zeta_final*, *mu_echo* and
    the current *price_now* relative to the *profit_band* limits.
    """
    alpha = np.asarray(alpha_vec, dtype=float)
    phi = np.asarray(phi_vec, dtype=float)
    if alpha.shape != phi.shape:
        raise ValueError("alpha_vec and phi_vec must share shape")

    # Core activation term
    activation = math.tanh(float(unified_math.unified_math.dot_product(alpha, phi)))
    base_prob = activation * unified_math.exp(-lambda_entry * t)

    # Validation gates
    in_band = price_now <= profit_band[0] or price_now >= profit_band[1]
    if zeta_final <= 0.0 or mu_echo < mu_threshold or not in_band:
        return 0.0

    return unified_math.max(0.0, unified_math.min(1.0, base_prob))
