#!/usr/bin/env python3
"""Lantern trigger – Lₜ = σ(Δ_price) · ∂ᵢτ_k."""

from __future__ import annotations

from core.unified_math_system import unified_math
from core.unified_math_system import unified_math

__all__: list[str] = ["lantern_trigger"]


def lantern_trigger(delta_price: float, partial_tau_k: float) -> float:
    """Return lantern trigger strength in (0,1)."""
    sigmoid = 1.0 / (1.0 + unified_math.exp(-delta_price))
    return sigmoid * partial_tau_k
