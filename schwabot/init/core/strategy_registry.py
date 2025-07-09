"""
strategy_registry.py
--------------------
Tracks strategy performance over time, storing PnL results and evolving a
strategy score via exponential weighted average.
"""

from __future__ import annotations

import json
import os
from datetime import datetime


class StrategyRegistry:
    """Persistent registry for strategy results and scores."""

    def __init__(self, filename: str = "strategy_registry.json") -> None:
        """Initialize strategy registry with filename."""
        self.filename = filename
        self.data = self._load()

    def _load(self) -> dict[str, any]:
        if not os.path.exists(self.filename):
            return {}
        with open(self.filename, 'r') as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def register_result(self,  strategy_id: str, pnl: float, tick_time: str | None = None) -> None:
        """Record a strategy PnL result and update its score."""
        tick_time = tick_time or datetime.utcnow().isoformat()
        if strategy_id not in self.data:
            self.data[strategy_id] = {'results': [], 'score': 0.0}

        self.data[strategy_id]['results'].append({'pnl': pnl, 'timestamp': tick_time})

        # Exponential weighted average score update
        prev = self.data[strategy_id]['score']
        self.data[strategy_id]['score'] = round(0.85 * prev + 0.15 * pnl, 5)

        self._save()

    def get_top_strategies(self,  n: int = 5) -> list[tuple[str, dict[str, any]]]:
        """Return top-n strategies by score descending."""
        return sorted(self.data.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:n]
