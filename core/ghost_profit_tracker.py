#!/usr/bin/env python3
"""Ghost profit tracker – recursive Π(t) accumulator.

Tracks realised profit deltas during ghost-mode cycles and provides
summaries for feedback loops (memory reinforcement, drift compensation,
etc.).  The implementation is intentionally small – no persistence layer or
DB – it runs in-memory and can be serialised by the caller if necessary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

__all__: list[str] = ["ProfitTracker", "register_profit", "profit_summary"]


def _safe_float(x: float | int) -> float:
    try:
        return float(x)
    except Exception as exc:  # pragma: no cover – defensive
        raise ValueError("profit value must be numeric") from exc


@dataclass(slots=True)
class ProfitTracker:
    """In-memory list of profit deltas and helper stats."""

    _profits: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, profit: float) -> None:  # noqa: D401
        self._profits.append(_safe_float(profit))

    def total(self) -> float:  # noqa: D401
        return float(np.sum(self._profits))

    def mean(self) -> float:
        return float(np.mean(self._profits)) if self._profits else 0.0

    def variance(self) -> float:
        return float(np.var(self._profits)) if self._profits else 0.0

    def summary(self) -> Tuple[float, float, float]:
        """Return (total, mean, variance)."""
        return self.total(), self.mean(), self.variance()


# -----------------------------------------------------------------------------
# Module-level singleton & functional wrappers
# -----------------------------------------------------------------------------

_tracker = ProfitTracker()


def register_profit(delta: float) -> None:  # noqa: D401
    """Append *delta* to global profit tracker."""
    _tracker.add(delta)


def profit_summary() -> Tuple[float, float, float]:  # noqa: D401
    """Return global tracker summary (total, mean, variance)."""
    return _tracker.summary() 