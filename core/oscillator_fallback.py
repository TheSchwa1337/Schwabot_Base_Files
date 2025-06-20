#!/usr/bin/env python3
"""Oscillator fallback – damped harmonic pulse generator.

This helper provides a mathematically–stable fallback oscillator that produces
bounded amplitude even if upstream signal generators become unstable.  Ferris
wheel and GAN-entropy modules can call this in *safe-mode* to keep phase timers
alive without injecting unbounded energy into the system.

Mathematics
~~~~~~~~~~
Damped harmonic oscillator (underdamped case):

    x(t) = A · e^(−γ t) · cos(2π f t + φ)

where 0 < γ < ∞ is the damping coefficient.

The implementation is intentionally minimal – no dynamic state, no numerical
integrator – just the closed-form expression that guarantees ‖x(t)‖ ≤ A.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = ["fallback_oscillator"]

_PI2: Final = 2.0 * math.pi


def fallback_oscillator(
    t: float,
    *,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    damping: float = 0.1,
    phase: float = 0.0,
) -> float:
    """Return damped cosine value x(t).

    Parameters
    ----------
    t
        Time (seconds) or dimension-less tick.
    amplitude
        Initial amplitude ``A``.  Defaults to **1.0**.
    frequency
        Frequency ``f`` in Hz.  Defaults to **1.0**.
    damping
        Damping coefficient ``γ``.  **0.0** ⇒ no damping.  Must be ≥ 0.
    phase
        Phase offset ``φ`` in radians.

    Returns
    -------
    float
        Damped oscillator value at *t*.
    """
    if damping < 0:
        raise ValueError("damping must be non-negative")
    envelope = math.exp(-damping * t)
    angle = _PI2 * frequency * t + phase
    return amplitude * envelope * math.cos(angle) 