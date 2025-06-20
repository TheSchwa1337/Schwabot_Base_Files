#!/usr/bin/env python3
"""Thermal delta switch – minimal thermal drift detector.

This helper flags sudden temperature jumps (*thermal shifts*) above a preset
threshold.  It is intentionally lightweight so it can execute inside tight
trading-loop iterations without blocking the GIL.

Current implementation
----------------------
1. ``ThermalShift`` class – exponential-moving-average (EWMA) smoothing with
   :py:meth:`update` returning ``(is_stable, delta)``.
2. Stateless wrapper :func:`thermal_delta_switch`` mirroring the legacy stub
   signature requested by earlier Schwabot code.
3. Fully typed and Flake8-clean (≤ 79-character lines).

Future versions may include adaptive hysteresis or GPU-calibrated drift maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Tuple

__all__ = ["ThermalShift", "thermal_delta_switch"]

_DEFAULT_ALPHA: Final = 0.2
_DEFAULT_THRESHOLD: Final = 2.5  # °C


@dataclass(slots=True)
class ThermalShift:
    """EWMA-based thermal drift detector.

    Parameters
    ----------
    threshold
        Absolute temperature delta (°C) that triggers an *unstable* flag.
    alpha
        EWMA smoothing factor between 0 and 1.  Higher = faster reaction.
    """

    threshold: float = _DEFAULT_THRESHOLD
    alpha: float = _DEFAULT_ALPHA

    _ema: float | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, temp: float) -> Tuple[bool, float]:
        """Process a new temperature reading and return stability status.

        Parameters
        ----------
        temp
            Current temperature reading (°C).

        Returns
        -------
        Tuple[bool, float]
            ``(is_stable, delta)``, where *delta* is the absolute
            temperature change with respect to the EWMA baseline.
        """
        if self._ema is None:
            self._ema = temp
        else:
            self._ema = self.alpha * temp + (1.0 - self.alpha) * self._ema

        delta = abs(temp - self._ema)
        is_stable = delta < self.threshold
        return is_stable, delta


# -----------------------------------------------------------------------------
# Stateless helper – mirrors historical stub signature
# -----------------------------------------------------------------------------


def thermal_delta_switch(
    current: float,
    previous: float,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """Return ``True`` if the temperature delta is below *threshold*.

    Parameters
    ----------
    current
        Current temperature reading (°C).
    previous
        Previous or baseline temperature reading (°C).
    threshold
        Allowed delta before declaring instability.  Defaults to 2.5 °C.
    """
    delta = abs(current - previous)
    return delta < threshold 