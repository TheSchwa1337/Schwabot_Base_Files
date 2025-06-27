# -*- coding: utf-8 -*-
"""Phase-based math for extensible phase logic in core and demo modules."""

def phase_adjust(value: float, phase: str = "mid") -> float:
    """
    Adjust a value based on the current phase (low, mid, high).

    Args:
        value: The value to adjust
        phase: 'low', 'mid', or 'high'

    Returns:
        Adjusted value
    """
    phase_factors = {"low": 0.9, "mid": 1.0, "high": 1.1}
    return value * phase_factors[phase] 