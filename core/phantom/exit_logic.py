# -*- coding: utf - 8 -*-
"""Phantom exit logic for profit - target based signals."""
"""
"""
"""
"""
"""Phantom exit logic for profit - target based signals."""
# -*- coding: utf - 8 -*-

"""
"""
"""
"""
"""Phantom exit logic for profit - target based signals."""
"""Phantom exit logic for profit - target based signals."""
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from __future__ import annotations
import math


# from core.unified_math_system import unified_math  # F811: duplicate import


def exit_weight()

        p_profit: float,
        p_target: float,
        half_life_sec: int = 900 -> float:
    """Calculate exit weight based on profit vs target."""
"""
"""

    Compute exit signal: \\u03a6_exit = sign(P - P_target) . kappa_decay(t)
    where kappa_decay(t) = exp(-t / tau)

    Args:
        p_profit: Current profit level
        p_target: Target profit level
        half_life_sec: Decay half - life in seconds (default 15min)

    Returns:
        Exit weight (0->hold, 1->full close)
    """"""
"""
"""
# Exponential decay factor
    kappa = unified_math.exp(-time.time() / half_life_sec)

# Sign based on profit vs target, scaled by decay
    return math.copysign(kappa, p_profit - p_target)


# Module exports
__all__ = ["exit_weight"]



"""
"""
"""
"""
