# -*- coding: utf - 8 -*-\\n# from __future__ import annotations
"""phantom_profit_tracker - thin wrapper around ghost_profit_tracker."""
"""
"""
"""
"""
"""phantom_profit_tracker - thin wrapper around ghost_profit_tracker."""
# -*- coding: utf - 8 -*-\\n# from __future__ import annotations
"""
"""
"""
"""
"""phantom_profit_tracker - thin wrapper around ghost_profit_tracker."""
"""phantom_profit_tracker - thin wrapper around ghost_profit_tracker."""
# -*- coding: utf - 8 -*-\\n# from __future__ import annotations
# -*- coding: utf - 8 -*-\\n# from __future__ import annotations
from .ghost_profit_tracker import register_profit as _register_profit
from .ghost_profit_tracker import ()
from .ghost_profit_tracker import profit_summary as _profit_summary


Some legacy modules reference * phantom_profit_tracker * .  Internally we simply
delegate to: pymod: `core.ghost_profit_tracker` so there is a single source of
truth.
""""""
"""
"""

ProfitTracker as _GhostProfitTracker,
# rename to avoid export clash

__all__: list[str] = []
"ProfitTracker",
"register_profit",
"profit_summary",

# Public re - exports
ProfitTracker = _GhostProfitTracker
register_profit = _register_profit
profit_summary = _profit_summary
