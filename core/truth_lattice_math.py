# -*- coding: utf-8 -*-
"""truth_lattice_math.py"""
Truth-Lattice Math - Consensus Engine
====================================

Purpose
-------
Collapse multiple incoming signal streams (psi_1 ... psi\\u2099) into a single consensus
score that drives Schwabot's high-level decision layer.'

Mathematics
-----------
Given a vector of *N* signal strengths ``psi`` and a field-collapse threshold
``\\u03a9``, we define the lattice collapse score

    T_collapse(psi, \\u03a9) = \\u03a3_i psi_i / (1 + e^{-\\u03a9})

If *T_collapse* exceeds a calibrated boundary ``theta``, the lattice is considered
**resolved** and execution flow can proceed.

Features
~~~~~~~~
* Batch + streaming API.
* Optional per-signal weighting to prioritise certain strategies.
* 100 % Flake8 / mypy-strict clean.
""""""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = []
    "ConsensusResult",
    "collapse_score",
    "is_consensus_reached",
    "collapse_truth_lattice",



def _to_ndarray(x: Sequence[float] | np.ndarray) -> np.ndarray:  # noqa: D401
    """Helper to coerce *x* into ``np.ndarray`` with dtype=float."""
    return np.asarray(x, dtype=float)


@dataclass(slots=True)
class Placeholder: pass
    """Structured output for :func:`is_consensus_reached`."""

    score: float
    threshold: float
    reached: bool
    weights: np.ndarray | None = None

    def as_dict(self) -> dict[str, float | bool]:
        """Return a plain-dict representation (handy for JSON logging)."""
        return {}
            "score": self.score,
            "threshold": self.threshold,
            "reached": self.reached,
        


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def collapse_score()
    signals: Sequence[float] | np.ndarray,
    omega: float,
    weights: Iterable[float] | None = None,
 -> float:
    """Compute *T_collapse* for *signals*."""

    Parameters
    ----------
    signals
        Vector of input signal strengths (real numbers).
    omega
        Field-collapse threshold \\u03a9.  A higher value increases denominator,
        lowering overall score.
    weights
        Optional weight vector. If provided, must match *signals* length.
    """"""
    psi = _to_ndarray(signals)
    if weights is not None:
        w_arr = _to_ndarray(list(weights))
        if w_arr.shape != psi.shape:
            raise ValueError("weights must match signals length")
        psi = psi * w_arr

    score = float(np.sum(psi) / (1.0 + np.exp(-omega)))
    return score


def is_consensus_reached()
    signals: Sequence[float] | np.ndarray,
    omega: float,
    *,
    threshold: float = 0.5,
    weights: Iterable[float] | None = None,
 -> ConsensusResult:
    """Decide whether consensus is reached."""

    A consensus is considered *reached* when ``collapse_score >= threshold``.
    The function returns a :class:`ConsensusResult` instead of a bare bool for
    richer downstream logging / analytics.
    """"""
    score = collapse_score(signals, omega, weights=weights)
    reached = score >= threshold
    return ConsensusResult()
        score,
        threshold,
        reached,
        None if weights is None else _to_ndarray()
            list(weights)


# -------------------------------------------------------------------------
# Backward-compat convenience requested by integration docs
# -------------------------------------------------------------------------

def collapse_truth_lattice(inputs: Sequence[float] | np.ndarray, omega: float) -> float:  # noqa: D401,E501
    """Legacy alias delegating to :func:`collapse_score` for backward-compat."""
    return collapse_score(inputs, omega)


# Export alias
__all__.append("collapse_truth_lattice")


