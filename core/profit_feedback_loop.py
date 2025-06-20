#!/usr/bin/env python3
"""Profit feedback loop – reinforcement signal δ_profit_t.

Implements the summation:
    δ_profit_t = Σ ( ζ_i · τ_i · PnL_i )
where ζ_i is trade-phase weighting, τ_i trade duration (seconds) and PnL_i the
profit/loss of trade *i*.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["profit_feedback_delta"]


def profit_feedback_delta(
    zeta_trades: Sequence[float],
    durations: Sequence[float],
    pnl: Sequence[float],
) -> float:  # noqa: D401
    """Return δ_profit_t scalar.

    All input sequences must share length; missing values raise ValueError.
    """
    if not (len(zeta_trades) == len(durations) == len(pnl)):
        raise ValueError("input sequences must share length")
    arr_zeta = np.asarray(zeta_trades, dtype=float)
    arr_tau = np.asarray(durations, dtype=float)
    arr_pnl = np.asarray(pnl, dtype=float)
    return float(np.dot(arr_zeta * arr_tau, arr_pnl)) 