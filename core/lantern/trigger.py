"""Lantern trigger for spike detection."""

from __future__ import annotations
from core.unified_math_system import unified_math
import time
from core.unified_math_system import unified_math


def lantern_trigger(dp: float, dt: float, tau0: float = 300.0) -> float:
    """Calculate lantern trigger spike score.

    Compute spike detector: Λ = Δprice/Δt · e^(-τ/τ₀)

    Args:
        dp: Price change
        dt: Time change
        tau0: Time constant in seconds (default 5min)

    Returns:
        Lantern spike score in [0,∞)
    """
    if dt <= 0:
        return 0.0

    return (dp / dt) * unified_math.exp(-time.time() / tau0)
