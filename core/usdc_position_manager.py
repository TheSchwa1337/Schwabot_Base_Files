#!/usr/bin/env python3
"""USDC position manager – exponential decay and position optimization.

Implements the formulas:
    P_usdc(t) = Σ_holdings·e^(−r·Δt)
    T_usdc = α_entry·δ_buy − β_exit·δ_sell
    σ_usdc(t) = ∇P_usdc(t) · log(1 + T_usdc)
    Ψ_usdc = argmax_t(σ_usdc(t) > θ_usdc)

This module manages USDC positions with time-decay modeling and optimal
timing detection for entry/exit decisions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["usdc_position", "usdc_trading", "usdc_sigma", "usdc_optimal_time"]

# ---------------------------------------------------------------------------
# Position management functions
# ---------------------------------------------------------------------------


def usdc_position(
    holdings: Sequence[float],
    rates: Sequence[float],
    time_deltas: Sequence[float],
) -> float:  # noqa: D401
    """Return P_usdc(t) = Σ_holdings·e^(−r·Δt).

    Parameters
    ----------
    holdings
        Current USDC holdings amounts.
    rates
        Decay rates r for each holding.
    time_deltas
        Time deltas Δt since each holding was acquired.
    """
    if not (len(holdings) == len(rates) == len(time_deltas)):
        raise ValueError("all input sequences must have same length")
    
    hold_arr = np.asarray(holdings, dtype=float)
    rate_arr = np.asarray(rates, dtype=float)
    dt_arr = np.asarray(time_deltas, dtype=float)
    
    # Exponential decay: e^(−r·Δt)
    decay_factors = np.exp(-rate_arr * dt_arr)
    
    # Sum of decayed holdings
    decayed_holdings = hold_arr * decay_factors
    
    return float(np.sum(decayed_holdings))


def usdc_trading(
    alpha_entry: float,
    delta_buy: float,
    beta_exit: float,
    delta_sell: float,
) -> float:  # noqa: D401
    """Return T_usdc = α_entry·δ_buy − β_exit·δ_sell.

    Parameters
    ----------
    alpha_entry
        Entry coefficient α_entry.
    delta_buy
        Buy signal magnitude δ_buy.
    beta_exit
        Exit coefficient β_exit.
    delta_sell
        Sell signal magnitude δ_sell.
    """
    entry_term = alpha_entry * delta_buy
    exit_term = beta_exit * delta_sell
    
    return entry_term - exit_term


def usdc_sigma(
    position_gradient: Sequence[float],
    t_usdc: float,
) -> np.ndarray:  # noqa: D401
    """Return σ_usdc(t) = ∇P_usdc(t) · log(1 + T_usdc).

    Parameters
    ----------
    position_gradient
        Gradient ∇P_usdc(t) of the position function.
    t_usdc
        Trading signal T_usdc from usdc_trading().
    """
    grad_arr = np.asarray(position_gradient, dtype=float)
    
    # Compute log(1 + T_usdc), handling negative values safely
    if t_usdc <= -1:
        log_term = np.log(1e-10)  # Avoid log(0) or log(negative)
    else:
        log_term = np.log(1 + t_usdc)
    
    sigma_usdc = grad_arr * log_term
    
    return sigma_usdc


def usdc_optimal_time(
    sigma_series: Sequence[float],
    theta_usdc: float,
    *,
    times: Sequence[float] | None = None,
) -> int:  # noqa: D401
    """Return Ψ_usdc = argmax_t(σ_usdc(t) > θ_usdc).

    Parameters
    ----------
    sigma_series
        Time series of σ_usdc(t) values.
    theta_usdc
        Threshold θ_usdc for optimal timing.
    times
        Optional time indices. If None, uses array indices.

    Returns
    -------
    int
        Index of optimal time when condition is maximally satisfied.
    """
    sigma_arr = np.asarray(sigma_series, dtype=float)
    
    if len(sigma_arr) == 0:
        return 0
    
    # Find indices where σ_usdc(t) > θ_usdc
    above_threshold = sigma_arr > theta_usdc
    
    if not np.any(above_threshold):
        # If no values above threshold, return index of maximum value
        return int(np.argmax(sigma_arr))
    
    # Among values above threshold, find the maximum
    valid_indices = np.where(above_threshold)[0]
    valid_values = sigma_arr[valid_indices]
    max_idx_in_valid = np.argmax(valid_values)
    
    return int(valid_indices[max_idx_in_valid]) 