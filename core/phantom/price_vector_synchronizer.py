# -*- coding: utf-8 -*-\n"""Price vector synchronizer with EMA smoothing."""

from __future__ import annotations


def ema(prices: list[float], tau: int = 12) -> float:

    pass
    pass
    """Calculate exponential moving average of price sequence.

Compute smoothed price: Ψ_sync = EMA(price, τ)

Args:
prices: List of price values (chronological order)
        tau: EMA time constant (default 12 periods)

Returns:
Latest EMA value

Raises:
ValueError: If prices list is empty
"""
    if not prices:
        raise ValueError("empty price list")


alpha = 2 / (tau + 1)
ema_val = prices[0]

for price in prices[1:]:
ema_val = alpha * price + (1 - alpha) * ema_val

return ema_val
