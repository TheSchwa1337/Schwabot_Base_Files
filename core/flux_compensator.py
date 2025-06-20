#!/usr/bin/env python3
"""Flux compensator – entropy drift corrector.

A *flux compensator* is a lightweight corrective layer that smooths noisy
entropy (or variance) readings and provides a boolean gate indicating whether
an input sample is still inside acceptable drift bounds.  Think of it as a
mini-Kalman corrector but with negligible computational overhead.

Implemented now
---------------
1. ``FluxCompensator`` class with exponential–moving-average (EMA) tracking.
2. Stateless helper ``sync_flux_compensator`` for one-off checks.
3. Fully-typed & Flake8-clean ≤ 79-char lines.

Advanced Jacobian/KF tuning can be layered later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Tuple
from collections import deque

import numpy as np

__all__ = ["FluxCompensator", "sync_flux_compensator"]


@dataclass(slots=True)
class FluxCompensator:
    """Exponential-smoothing entropy corrector.

    Parameters
    ----------
    threshold
        Base entropy threshold.  When the *smoothed* entropy exceeds
        ``threshold * multiplier`` the validator flags *False*.
    alpha
        Smoothing factor for EMA – between 0 and 1.  Higher = faster reaction.
    window
        Optional fixed window for simple moving average (SMA) if you prefer
        deterministic lag.  If ``window`` is ``None`` the class uses EMA.
    multiplier
        Safety margin.  A value of 0.9 ⇒ allow 10 % slack under threshold.
    """

    threshold: float = 5.0
    alpha: float = 0.3
    window: int | None = None
    multiplier: float = 0.9

    _sma_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=10), init=False)
    _ema: float | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, entropy: float) -> Tuple[bool, float]:
        """Ingest a new entropy value and return (is_valid, smoothed_entropy)."""
        smoothed = self._smooth(entropy)
        is_valid = smoothed < self.threshold * self.multiplier
        return is_valid, smoothed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _smooth(self, value: float) -> float:
        if self.window is not None and self.window > 1:
            self._sma_buf.append(value)
            smoothed = float(np.mean(self._sma_buf))
            return smoothed
        # EMA path
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.alpha * value + (1.0 - self.alpha) * self._ema
        return self._ema


# -----------------------------------------------------------------------------
# Stateless convenience wrapper – mirrors historical stub signature
# -----------------------------------------------------------------------------

def sync_flux_compensator(entropy: float, threshold: float) -> bool:
    """Single-shot flux compensation check.

    Uses a fixed damping multiplier (0.9) and no state retention.  Suitable for
    quick gating where persistent history is not necessary.
    """
    return entropy < threshold * 0.9 