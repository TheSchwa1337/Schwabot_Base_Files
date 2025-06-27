# -*- coding: utf-8 -*-
from __future__ import annotations

from core.unified_math_system import unified_math
import numpy as np
import math
# #!/usr/bin/env python3
"""BTC vector aggregator - volume-weighted price analysis and FFT filtering."""

Implements the formulas:
V_btc = \\u03a3_i^n [p_exit(i) - p_entry(i)].w_vol(i)
    eta_btc = deltap / deltat . \\u03a3_j vol(j)
    \\u039e_btc(t) = tanh(V_btc . eta_btc)
    A_btc = FFT(\\u039e_btc(t)) . filter(f_peak)

This module aggregates BTC price movements with volume weighting and applies
spectral filtering for enhanced signal quality.
""""""


from typing import Sequence

# from core.unified_math_system import unified_math  # F811: duplicate import

__all__: list[str] = []
    "btc_vector",
    "btc_eta",
    "btc_xi",
    "btc_spectral_aggregate",


# ---------------------------------------------------------------------------
# Core aggregation functions
# ---------------------------------------------------------------------------


def btc_vector()
    exit_prices: Sequence[float],
    entry_prices: Sequence[float],
    volume_weights: Sequence[float],
 -> float:  # noqa: D401
    """Return V_btc = \\u03a3_i^n [p_exit(i) - p_entry(i)].w_vol(i)."""

    Parameters
    ----------
    exit_prices
        Exit prices p_exit(i) for each trade.
    entry_prices
        Entry prices p_entry(i) for each trade.
    volume_weights
        Volume weights w_vol(i) for each trade.
    """"""
    if not (len(exit_prices) == len(entry_prices) == len(volume_weights)):
        raise ValueError("all input sequences must have same length")

    exit_arr = np.asarray(exit_prices, dtype=float)
    entry_arr = np.asarray(entry_prices, dtype=float)
    vol_arr = np.asarray(volume_weights, dtype=float)

    # Compute price differences weighted by volume
    price_diffs = exit_arr - entry_arr
    weighted_diffs = price_diffs * vol_arr

    return float(np.sum(weighted_diffs))


def btc_eta()
    price_delta: float,
    time_delta: float,
    volumes: Sequence[float],
 -> float:  # noqa: D401
    """Return eta_btc = deltap / deltat . \\u03a3_j vol(j)."""

    Parameters
    ----------
    price_delta
        Price change deltap over the time period.
    time_delta
        Time period deltat (must be > 0).
    volumes
        Volume series vol(j) to sum.
    """"""
    if time_delta <= 0:
        raise ValueError("time_delta must be positive")

    vol_sum = float(np.sum(volumes))
    price_velocity = price_delta / time_delta

    return price_velocity * vol_sum


def btc_xi(v_btc: float, eta_btc: float) -> float:  # noqa: D401
    """Return \\u039e_btc(t) = tanh(V_btc . eta_btc)."""

    Parameters
    ----------
    v_btc
        Vector value from btc_vector().
    eta_btc
        Eta value from btc_eta().
    """"""
    product = v_btc * eta_btc
    return float(np.tanh(product))


def btc_spectral_aggregate()
    xi_series: Sequence[float],
    peak_frequency: float,
    *,
    filter_width: float = 1.0,
 -> np.ndarray:  # noqa: D401
    """Return A_btc = FFT(\\u039e_btc(t)) . filter(f_peak)."""

    Parameters
    ----------
    xi_series
        Time series of \\u039e_btc(t) values.
    peak_frequency
        Target frequency f_peak for filtering.
    filter_width
        Width of the frequency filter around f_peak.
    """"""
    xi_arr = np.asarray(xi_series, dtype=float)

    if len(xi_arr) == 0:
        return np.array([], dtype=complex)

    # Compute FFT
    xi_fft = np.fft.fft(xi_arr)

    # Create frequency array
    n = len(xi_arr)
    freqs = np.fft.fftfreq(n)

    # Create Gaussian filter centered at peak_frequency
    filter_mask = unified_math.exp(-((freqs - peak_frequency)))
                                   ** 2 / (2 * filter_width**2)

    # Apply filter
    filtered_fft = xi_fft * filter_mask

    return filtered_fft


