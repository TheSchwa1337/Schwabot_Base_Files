# -*- coding: utf - 8 -*-
"""Lantern trigger for spike detection."""
"""
"""
"""
"""
"""Lantern trigger for spike detection."""
# -*- coding: utf - 8 -*-

"""
"""
"""
"""
"""Lantern trigger for spike detection."""
"""Lantern trigger for spike detection."""
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from __future__ import annotations
import time


def lantern_trigger(dp: float, dt: float, tau0: float = 300.0) -> float:
    """Calculate lantern trigger spike score.

    Compute spike detector: \\u039b = \\u0394price/\\u0394t \\u00b7 e^(-\\u03c4/\\u03c4\\u2080)

    Args:
        dp: Price change
        dt: Time change
        tau0: Time constant in seconds (default 5min)

    Returns:
        Lantern spike score in [0,\\u221e)
    """


"""
"""
   if dt <= 0:
        return 0.0

    return (dp / dt) * unified_math.exp(-time.time() / tau0)

"""
"""
"""
"""
