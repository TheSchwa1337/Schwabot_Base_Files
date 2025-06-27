# -*- coding: utf-8 -*-\\n# #!/usr/bin/env python3
"""Lantern trigger - L\\u209c = sigma(delta_price) . partial\\u1d62tau_k."""

from __future__ import annotations

from core.unified_math_system import unified_math
import math
# from core.unified_math_system import unified_math  # F811: duplicate import

__all__: list[str] = ["lantern_trigger"]


def lantern_trigger(delta_price: float, partial_tau_k: float) -> float:
    pass
    """Return lantern trigger strength in (0,1)."""
    sigmoid = 1.0 / (1.0 + unified_math.exp(-delta_price))
    return sigmoid * partial_tau_k


