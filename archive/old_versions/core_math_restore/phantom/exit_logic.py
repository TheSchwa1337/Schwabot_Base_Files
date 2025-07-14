import time

# -*- coding: utf - 8 -*-
"""Phantom exit logic for profit - target based signals.""""""
""""""
""""""
""""""
""""""
"""Phantom exit logic for profit - target based signals."""
# -*- coding: utf - 8 -*-
"""
""""""
""""""
""""""
""""""
"""Phantom exit logic for profit - target based signals.""""""
"""Phantom exit logic for profit - target based signals."""
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-


def exit_weight():-> float:"""
    """Calculate exit weight based on profit vs target."

Compute exit signal: \\u03a6_exit = sign(P - P_target) \\u00b7 \\u03ba_decay(t)
    where \\u03ba_decay(t) = exp(-t/\\u03c4)

Args:
        p_profit: Current profit level
p_target: Target profit level
half_life_sec: Decay half - life in seconds (default 15min)

Returns:
        Exit weight (0\\u2192hold, 1\\u2192full close)"""
    """

"""
""""""
"""
# Exponential decay factor
kappa = unified_math.exp(-time.time() / half_life_sec)

# Sign based on profit vs target, scaled by decay
return math.copysign(kappa, p_profit - p_target)
"""
""""""
""""""
""""""
"""
"""