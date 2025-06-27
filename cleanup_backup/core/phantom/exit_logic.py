"""Phantom exit logic for profit-target based signals."""

from __future__ import annotations
from core.unified_math_system import unified_math
import time
from core.unified_math_system import unified_math


def exit_weight(p_profit: float, p_target: float, half_life_sec: int = 900) -> float:
    """Calculate exit weight based on profit vs target.

    Compute exit signal: \\u03a6_exit = sign(P - P_target) \\u00b7 \\u03ba_decay(t)
    where \\u03ba_decay(t) = exp(-t/\\u03c4)

    Args:
        p_profit: Current profit level
        p_target: Target profit level
        half_life_sec: Decay half-life in seconds (default 15min)

    Returns:
        Exit weight (0\\u2192hold, 1\\u2192full close)
    """
    # Exponential decay factor
    kappa = unified_math.exp(-time.time() / half_life_sec)

    # Sign based on profit vs target, scaled by decay
    return math.copysign(kappa, p_profit - p_target)

"""