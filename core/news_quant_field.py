#!/usr/bin/env python3
"""News quantization field – weighted news processing and spectral analysis.

Implements the formulas:
    Q_news(t) = Σ_i W_i·N_i(t)
    ∇Q = (∂Q/∂x, ∂Q/∂t)
    Ψ_news = exp(−∇Q² / σ²)
    F_news = FFT(Q_news) → Spectral Field

This module processes financial news streams into quantized fields with
gradient analysis and frequency domain representations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["quantize_news", "news_gradient", "news_psi", "news_spectral_field"]

# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------


def quantize_news(
    weights: Sequence[float],
    news_values: Sequence[Sequence[float]],
) -> np.ndarray:  # noqa: D401
    """Return Q_news(t) = Σ_i W_i·N_i(t) weighted news quantization.

    Parameters
    ----------
    weights
        Weighting factors W_i for each news source.
    news_values
        Sequence of news time series N_i(t), each as array-like.
    """
    if len(weights) != len(news_values):
        raise ValueError("weights and news_values must have same length")
    
    w_array = np.asarray(weights, dtype=float)
    
    # Ensure all news series have same length
    news_arrays = [np.asarray(n, dtype=float) for n in news_values]
    if not news_arrays:
        return np.array([])
    
    length = len(news_arrays[0])
    if not all(len(n) == length for n in news_arrays):
        raise ValueError("all news series must have same length")
    
    # Weighted sum: Σ_i W_i·N_i(t)
    q_news = np.zeros(length, dtype=float)
    for i, n_array in enumerate(news_arrays):
        q_news += w_array[i] * n_array
    
    return q_news


def news_gradient(
    q_news: np.ndarray,
    *,
    dx: float = 1.0,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:  # noqa: D401
    """Return ∇Q = (∂Q/∂x, ∂Q/∂t) using numpy.gradient.

    Parameters
    ----------
    q_news
        Quantized news field Q_news(t).
    dx
        Spatial step size (for spatial derivative).
    dt
        Temporal step size.
    """
    if len(q_news) < 2:
        return np.array([0.0]), np.array([0.0])
    
    # Compute gradient (treating as 1D spatial-temporal field)
    grad_q = np.gradient(q_news, dt)
    
    # For consistency with formula, return (spatial, temporal) components
    # Since we have 1D time series, spatial component is zero
    spatial_grad = np.zeros_like(grad_q)
    temporal_grad = grad_q
    
    return spatial_grad, temporal_grad


def news_psi(
    spatial_grad: np.ndarray,
    temporal_grad: np.ndarray,
    sigma: float,
) -> np.ndarray:  # noqa: D401
    """Return Ψ_news = exp(−∇Q² / σ²) Gaussian-weighted field.

    Parameters
    ----------
    spatial_grad, temporal_grad
        Spatial and temporal components of ∇Q.
    sigma
        Gaussian spread parameter.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    # Compute gradient magnitude squared: |∇Q|²
    grad_mag_sq = spatial_grad**2 + temporal_grad**2
    
    # Gaussian weighting: exp(−|∇Q|² / σ²)
    psi_news = np.exp(-grad_mag_sq / (sigma**2))
    
    return psi_news


def news_spectral_field(q_news: np.ndarray) -> np.ndarray:  # noqa: D401
    """Return F_news = FFT(Q_news) spectral field representation.

    Parameters
    ----------
    q_news
        Time-domain quantized news field.
    """
    if len(q_news) == 0:
        return np.array([], dtype=complex)
    
    # Compute FFT for spectral analysis
    f_news = np.fft.fft(q_news)
    
    return f_news 