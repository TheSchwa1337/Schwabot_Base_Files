# -*- coding: utf-8 -*-\n"""core.phase.drift_phase_weighter
Drift - Phase Weighter
== == == == == == == == == ==

Calculates a λ - decay drift weight that quantifies the * tension * present in the
latest price movement and provides an entropy - gradient helper used by other
phase modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils.math_utils import calculate_entropy, moving_average

__all__ = [
    "DriftWeightReport",
    "DriftPhaseWeighter",
]


@dataclass(slots=True)
class DriftWeightReport:
    drift_weight: float
    smoothed_series: np.ndarray
    lambda_: float

    def as_dict(self) -> dict[str, float]:
        return {
            "drift_weight": self.drift_weight,
            "lambda": self.lambda_,
        }


class DriftPhaseWeighter:
    """Compute λ - decay drift weight from a price vector."""

    def __init__(self, *, lambda_: float = 0.3) -> None:
        if not (0.0 < lambda_ <= 1.0):
            raise ValueError("lambda_ must be in (0, 1]")
        self.lambda_ = lambda_

    # ------------------------------------------------------------------
    def calculate_drift_weight(
    self, prices: Sequence[float]) -> DriftWeightReport:
        """Return drift weight for `prices`.

        Algorithm
        ~~~~~~~~~
        1. Compute abs price differences Δp.
        2. Apply exponential smoothing with factor λ.
        3. Drift weight = mean of last 5 smoothed deltas.
        """
        prices_arr = np.asarray(prices, dtype=float)
        if prices_arr.size < 3:
            raise ValueError("need at least 3 price points")

        delta = np.abs(np.diff(prices_arr))
        # exponential smoothing: s[t] = λ*x[t] + (1-λ)*s[t-1]
        smoothed = np.empty_like(delta)
        smoothed[0] = delta[0]
        for i in range(1, delta.size):
            smoothed[i] = self.lambda_ * delta[i] + \
                (1.0 - self.lambda_) * smoothed[i - 1]

        # use last 5 points (or all if shorter) for current drift weight
        tail = smoothed[-5:]
        drift_weight = float(np.mean(tail))

        return DriftWeightReport(drift_weight, smoothed, self.lambda_)

    # ------------------------------------------------------------------
    @staticmethod
    def gradient_entropy_score(
    prices: Sequence[float],
     window: int = 10) -> float:
        """Directional entropy gradient score for *prices * .

        A simple metric: difference between entropy of the most recent `window`
        samples and the preceding `window` samples.
        """
        arr = np.asarray(prices, dtype=float)
        if arr.size < window * 2:
            raise ValueError("price history too short for entropy gradient")
        ent_now = calculate_entropy(arr[-window:])
        ent_prev = calculate_entropy(arr[-(2 * window) : -window])
        return float(ent_now - ent_prev) 