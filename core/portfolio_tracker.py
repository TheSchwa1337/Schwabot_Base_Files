# -*- coding: utf-8 -*-
"""
Portfolio Tracker
================
Tracks portfolio value, APR, and history for the mathematical relay system.
Exposes data for visualization and API handoff. Integrates with state_connectivity.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.state_connectivity import state_connectivity


class PortfolioTracker:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []  # Each entry: {timestamp, value}
        self.current_value: Decimal = Decimal("0")
        self.last_update: Optional[datetime] = None

    def update_value(self, value: Decimal, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        self.current_value = value
        self.last_update = timestamp
        self.history.append({"timestamp": timestamp, "value": float(value)})

    def get_history(self, as_floats: bool = False) -> List[Any]:
        if as_floats:
            return [entry["value"] for entry in self.history]
        return self.history

    def get_apr(self, periods_per_year: int = 365) -> float:
        values = [entry["value"] for entry in self.history]
        return state_connectivity.apr(values, periods_per_year=periods_per_year)

    def get_summary(self) -> Dict[str, Any]:
        values = [entry["value"] for entry in self.history]
        return {
            "current_value": float(self.current_value),
            "history_length": len(self.history),
            "apr": self.get_apr(),
            "mean": state_connectivity.mean(values) if values else 0.0,
            "median": state_connectivity.median(values) if values else 0.0,
            "deviation": state_connectivity.deviation(values) if values else 0.0,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


# Global instance
portfolio_tracker = PortfolioTracker()
