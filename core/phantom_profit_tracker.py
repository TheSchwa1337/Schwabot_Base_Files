#!/usr/bin/env python3
"""phantom_profit_tracker – thin wrapper around ghost_profit_tracker.

Some legacy modules reference *phantom_profit_tracker*.  Internally we simply
delegate to :pymod:`core.ghost_profit_tracker` so there is a single source of
truth.
"""

from __future__ import annotations

from .ghost_profit_tracker import (
    ProfitTracker as _GhostProfitTracker,  # rename to avoid export clash
    register_profit as _register_profit,
    profit_summary as _profit_summary,
)

__all__: list[str] = [
    "ProfitTracker",
    "register_profit",
    "profit_summary",
]

# Public re-exports
ProfitTracker = _GhostProfitTracker
register_profit = _register_profit
profit_summary = _profit_summary 