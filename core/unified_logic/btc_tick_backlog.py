# -*- coding: utf-8 -*-
"""BTC Tick Backlog System for 3.75/min, 16/hr intervals with float valuation and phase support."""

from dataclasses import dataclass
from datetime import datetime

@dataclass
class BTCTick:
    timestamp: datetime
    hash_rate: float
    price: float
    float_valuation: float
    phase: str
    unicode_label: str = ""

# Function to backlog/save ticks, to be called every 3.75 min or 16/hr

def save_btc_tick(hash_rate, price, float_valuation, phase, unicode_label=""):
    tick = BTCTick(
        timestamp=datetime.utcnow(),
        hash_rate=hash_rate,
        price=price,
        float_valuation=float_valuation,
        phase=phase,
        unicode_label=unicode_label
    )
    # TODO: Integrate with backchannel memory or database
    return tick 