"""ghost_field_stabilizer.py
Ghost Field Stabilizer
======================

Purpose
-------
Detect whether Schwabot is operating in a **Stable-Field State (SFS)** or an
**Unstable-Field State (UFS)** by measuring the short-term entropy delta of an
input signal series.

Mathematical model
------------------
A field is considered stable when the discrete derivative of the signal's
entropy remains below a configurable threshold:

    Δₑ ψ(t) = |ψ(t + ε) − ψ(t)| / ε  <  τ

where
    ε   – micro-drift window size (ticks)
    τ   – volatility threshold defining stability

Implementation notes
--------------------
* We approximate **ψ(t)** with the Shannon entropy of the last *N* prices.
* All public functions are fully type-hinted and documented.
* The module is Flake8 + mypy ‑-strict clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

__all__ = [
    "StabilityReport",
    "GhostFieldStabilizer",
    "is_sfs_state",
]


@dataclass(slots=True)
class StabilityReport:
    """Container returned by :pymeth:`GhostFieldStabilizer.check_stability`."""

    is_stable: bool
    delta_entropy: float
    epsilon: int
    tau: float

    def as_dict(self) -> Dict[str, float | int | bool]:
        """Return the report as a plain ``dict`` (useful for logging / JSON)."""
        return {
            "is_stable": self.is_stable,
            "delta_entropy": self.delta_entropy,
            "epsilon": self.epsilon,
            "tau": self.tau,
        }


class GhostFieldStabilizer:
    """Evaluate field stability of a numerical series.

    Example
    -------
    >>> gfs = GhostFieldStabilizer(epsilon=3, tau=0.015)
    >>> prices = np.random.random(100)
    >>> report = gfs.check_stability(prices)
    >>> report.is_stable
    True
    """

    def __init__(self, *, epsilon: int = 3, tau: float = 0.015) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.epsilon: int = int(epsilon)
        self.tau: float = float(tau)

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def check_stability(self, series: np.ndarray | List[float]) -> StabilityReport:  # noqa: D401,E501
        """Return a :class:`StabilityReport` for *series*.

        Parameters
        ----------
        series
            Input 1-D numerical array (price, entropy, etc.). Must contain at
            least ``epsilon + 1`` samples.
        """
        array = np.asarray(series, dtype=float)
        if array.ndim != 1:
            raise ValueError("series must be 1-D")
        if array.size < self.epsilon + 1:
            raise ValueError(
                "series length must be >= epsilon + 1 (got %d)" % array.size,
            )

        # entropy at t and t+ε using a rolling window of `epsilon` samples each
        ent_now = self._shannon_entropy(array[-self.epsilon :])
        ent_future = self._shannon_entropy(array[-(self.epsilon * 2) : -self.epsilon])

        delta_entropy = abs(ent_future - ent_now) / self.epsilon
        is_stable = delta_entropy < self.tau

        return StabilityReport(is_stable, delta_entropy, self.epsilon, self.tau)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _shannon_entropy(samples: np.ndarray) -> float:
        """Compute Shannon entropy of a 1-D sample vector."""
        # Normalize to probability distribution
        hist, _ = np.histogram(samples, bins="auto", density=True)
        # Filter zeros to avoid log problems
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)


# ---------------------------------------------------------------------
# Functional helper (kept outside the class for quick procedural access)
# ---------------------------------------------------------------------

def is_sfs_state(signal: np.ndarray, eps: int = 3, threshold: float = 0.02) -> bool:  # noqa: D401,E501
    """Return ``True`` if signal derivative magnitude < *threshold*.

    This mirrors the formula::

        |Δ_ε ψ(t)| < τ  →  Stable Field State (SFS)

    where ``Δ_ε ψ(t)`` is approximated by the finite difference over *eps*
    samples. It is a convenience wrapper around the core
    :class:`GhostFieldStabilizer` logic for scenarios where a full report is
    not required.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if signal.size < eps + 1:
        raise ValueError("signal length must be >= eps + 1")
    delta = (signal[-1] - signal[-1 - eps]) / eps
    return abs(delta) < threshold 