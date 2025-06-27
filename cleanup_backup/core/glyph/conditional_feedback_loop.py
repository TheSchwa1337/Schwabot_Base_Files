"""Conditional glyph feedback loop with exponential smoothing."""

from __future__ import annotations


def feedback(g_prev: float, zeta: float, beta: float = 0.9) -> float:
    """Apply exponential moving feedback to glyph weights.

    Compute feedback: g_{t+1} = \\u03b2\\u00b7g_t + (1-\\u03b2)\\u00b7\\u03b6

    Args:
        g_prev: Previous glyph weight
        zeta: Current zeta coefficient from phase integrator
        beta: Smoothing factor (default 0.9)

    Returns:
        Updated glyph weight with feedback applied
    """
    return beta * g_prev + (1 - beta) * zeta

"""