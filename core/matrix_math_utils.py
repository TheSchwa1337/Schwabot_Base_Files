"""
Matrix Math Utilities for Schwabot Trading System.

Provides advanced matrix and linear-algebra functions used by
back-testing and self-corrective engines.

Key Features:
1. Covariance & correlation matrix calculation
2. Eigenvalue & condition-number diagnostics
3. Simple risk-parity weight generator
4. Matrix stability scoring for dynamic risk controls

All public helpers are pure functions and NumPy-based so they can be
unit-tested in isolation.

CUDA Integration:
- GPU-accelerated matrix utilities with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = 'cupy (GPU)'
    xp = cp
    la = cp.linalg
except ImportError:
    import numpy as np
    USING_CUDA = False
    _backend = 'numpy (CPU)'
    xp = np
    la = np.linalg

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ MatrixMathUtils using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 MatrixMathUtils using CPU fallback: {_backend}")


def analyze_price_matrix(price_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Analyze a 2-D matrix of prices or returns.

    The input shape is (N, M) where N is the number of samples/timesteps
    and M is the number of assets.

    Returns a dictionary of diagnostics suitable for adaptive
    parameter tuning.

    Args:
        price_matrix: 2D numpy array of prices or returns

    Returns:
        Dictionary containing matrix analysis results

    Raises:
        TypeError: If price_matrix is not a numpy array
        ValueError: If price_matrix is not 2D
    """
    if not isinstance(price_matrix, np.ndarray):
        raise TypeError("price_matrix must be a NumPy array")

    if price_matrix.ndim != 2:
        raise ValueError("price_matrix must be 2D (samples, assets)")

    num_samples, num_assets = price_matrix.shape

    if num_samples < 2:
        raise ValueError("Need at least 2 samples for analysis")

    if num_assets < 1:
        raise ValueError("Need at least 1 asset for analysis")

    # Calculate returns if input is prices
    if np.all(price_matrix > 0):  # Likely prices
        returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]
    else:  # Likely already returns
        returns = price_matrix

    # Basic statistics
    mean_returns = np.mean(returns, axis=0)
    std_returns = np.std(returns, axis=0, ddof=1)

    # Correlation matrix
    correlation_matrix = np.corrcoef(returns.T)

    # Covariance matrix
    covariance_matrix = np.cov(returns.T, ddof=1)

    # Eigenvalue analysis
    try:
        eigenvalues = la.eigvals(covariance_matrix)
        condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
    except la.LinAlgError:
        eigenvalues = np.array([])
        condition_number = np.inf

    # Matrix stability score (lower is more stable)
    stability_score = (
        np.std(eigenvalues) / np.mean(np.abs(eigenvalues)) if len(eigenvalues) > 0 else np.inf
    )

    # Risk metrics
    portfolio_volatility = np.sqrt(np.sum(covariance_matrix))
    max_correlation = np.max(np.abs(correlation_matrix - np.eye(num_assets)))

    return {
        "num_samples": num_samples,
        "num_assets": num_assets,
        "mean_returns": mean_returns.tolist(),
        "std_returns": std_returns.tolist(),
        "correlation_matrix": correlation_matrix.tolist(),
        "covariance_matrix": covariance_matrix.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "condition_number": float(condition_number),
        "stability_score": float(stability_score),
        "portfolio_volatility": float(portfolio_volatility),
        "max_correlation": float(max_correlation),
        "is_stable": stability_score < 1.0,
        "is_well_conditioned": condition_number < 1000.0,
    }


def risk_parity_weights(
    covariance_matrix: np.ndarray,
    target_volatility: Optional[float] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calculate risk parity weights for a given covariance matrix.

    Risk parity aims to equalize the risk contribution of each asset
    to the portfolio.

    Args:
        covariance_matrix: Asset covariance matrix
        target_volatility: Target portfolio volatility (optional)
        max_iterations: Maximum iterations for optimization
        tolerance: Convergence tolerance

    Returns:
        Tuple of (weights, metadata)

    Raises:
        ValueError: If covariance matrix is invalid
    """
    if not isinstance(covariance_matrix, np.ndarray):
        raise TypeError("covariance_matrix must be a NumPy array")

    if covariance_matrix.ndim != 2:
        raise ValueError("covariance_matrix must be 2D")

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be square")

    num_assets = covariance_matrix.shape[0]

    if num_assets < 1:
        raise ValueError("Need at least 1 asset")

    # Initialize equal weights
    weights = np.ones(num_assets) / num_assets

    # Iterative optimization
    for iteration in range(max_iterations):
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)

        if portfolio_vol == 0:
            break

        # Calculate risk contributions
        risk_contributions = (weights * (covariance_matrix @ weights)) / portfolio_vol

        # Calculate target risk contribution (equal for all assets)
        target_risk_contribution = portfolio_vol / num_assets

        # Update weights
        weight_updates = (target_risk_contribution - risk_contributions) / (
            covariance_matrix @ weights
        )
        weights += 0.1 * weight_updates  # Small step size for stability

        # Normalize weights to sum to 1
        weights = np.maximum(weights, 0)  # Ensure non-negative
        weights = weights / np.sum(weights)

        # Check convergence
        risk_contribution_std = np.std(risk_contributions)
        if risk_contribution_std < tolerance:
            break

    # Scale to target volatility if specified
    if target_volatility is not None:
        current_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
        if current_vol > 0:
            weights = weights * (target_volatility / current_vol)

    # Calculate final metrics
    final_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
    final_risk_contributions = (weights * (covariance_matrix @ weights)) / final_vol

    metadata = {
        "iterations": iteration + 1,
        "converged": risk_contribution_std < tolerance,
        "portfolio_volatility": float(final_vol),
        "risk_contributions": final_risk_contributions.tolist(),
        "risk_contribution_std": float(np.std(final_risk_contributions)),
        "weights_sum": float(np.sum(weights)),
    }

    return weights, metadata


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default: 0.0)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    return mean_excess_return / std_return


def calculate_max_drawdown(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.

    Args:
        returns: Array of returns

    Returns:
        Dictionary with drawdown metrics
    """
    if len(returns) == 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "drawdown_duration": 0,
        }

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max

    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    max_drawdown_idx = np.argmin(drawdown)

    # Find peak before maximum drawdown
    peak_idx = np.argmax(cumulative[: max_drawdown_idx + 1])

    # Calculate duration
    drawdown_duration = max_drawdown_idx - peak_idx

    return {
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown * 100),
        "drawdown_duration": int(drawdown_duration),
        "peak_idx": int(peak_idx),
        "trough_idx": int(max_drawdown_idx),
    }


def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 5%)

    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0

    return np.percentile(returns, confidence_level * 100)


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 5%)

    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence_level)
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    return np.mean(tail_returns)


# Export main functions
__all__ = [
    "analyze_price_matrix",
    "risk_parity_weights",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "calculate_cvar",
]
