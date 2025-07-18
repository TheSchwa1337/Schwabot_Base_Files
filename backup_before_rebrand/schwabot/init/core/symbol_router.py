"""
symbol_router.py
----------------
Cycles through a list of trading symbols at a configured time interval.

Usage:
    router = SymbolRouter(['BTC/USDC', 'ETH/USDC', ...], interval=225)
    current_symbol = router.get_symbol()
    # call this each time before fetching ticks or executing trades
"""

from __future__ import annotations

import time
from typing import List


class SymbolRouter:
    """Round-robin router for a list of symbols."""

    def __init__(self,  symbols: List[str], interval: int = 225) -> None:
        """
        symbols: list of symbol strings (e.g. 'BTC/USDC')
        interval: seconds per symbol before rotating (default 3.75min)
        """
        if not symbols:
            raise ValueError("SymbolRouter requires a non-empty symbol list.")
        self.symbols = symbols
        self.interval = interval
        self._start = time.time()
        self._count = len(symbols)

    def get_symbol(self) -> str:
        """Return the symbol for the current time slot."""
        elapsed = time.time() - self._start
        slot = int(elapsed // self.interval)
        idx = slot % self._count
        return self.symbols[idx]
