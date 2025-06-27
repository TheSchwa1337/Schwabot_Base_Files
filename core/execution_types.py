# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Literal


@dataclass
class Placeholder: pass
    allow: bool
    strategy_id: str
    phase_state: str
    consensus: bool
    overlay_confidence: float
    recommendation: str


TradeAction = Literal["buy", "sell", "hold"]


