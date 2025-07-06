#!/usr/bin/env python3
"""
Clean Math Foundation - Core Mathematical Operations

Provides a clean, unified interface for mathematical operations used throughout
the Schwabot system. This module serves as the mathematical foundation for all
trading calculations, ensuring consistency and reliability.

Key Features:
- Unified mathematical operations
- Bit phase management
- Thermal state tracking
- Error handling and validation
- Performance optimization
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = 'cupy (GPU)'
    xp = cp
except ImportError:
    import numpy as np
    USING_CUDA = False
    _backend = 'numpy (CPU)'
    xp = np

# Log backend status
logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ Clean Math Foundation using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 Clean Math Foundation using CPU fallback: {_backend}")


class ThermalState(Enum):
    """Thermal state enumeration for trading system."""

    COOL = "cool"
    WARM = "warm"
    HOT = "hot"


class BitPhase(Enum):
    """Bit phase enumeration for precision control."""

    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"
    SIXTEEN_BIT = "16bit"
    THIRTY_TWO_BIT = "32bit"
    FORTY_TWO_BIT = "42bit"


# Mathematical constants
PI = math.pi
E = math.e
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
SQRT_2 = math.sqrt(2)
LN_2 = math.log(2)


class CleanMathFoundation:
    """
    Clean mathematical foundation providing core mathematical operations.

    This class serves as the foundation for all mathematical computations
    in the Schwabot trading system.
    """

    def __init__(self):
        """Initialize the mathematical foundation."""
        self.version = "1.0.0"
        self.precision = 64

    def get_version_info(self) -> Dict[str, Any]:
        """Get version information."""
        return {
            "version": self.version,
            "precision": self.precision,
            "thermal_states": [state.value for state in ThermalState],
            "bit_phases": [phase.value for phase in BitPhase],
        }


def calculate_vector_norm(vector: np.ndarray, p: float = 2.0) -> float:
    """
    Calculate the p-norm of a vector.

    Args:
        vector: Input vector
        p: Norm order (default: 2.0 for Euclidean norm)

    Returns:
        Vector norm value

    Raises:
        ValueError: If p < 1 or vector is empty
    """
    if len(vector) == 0:
        raise ValueError("Vector cannot be empty")

    if p < 1:
        raise ValueError("Norm order must be >= 1")

    if p == float("inf"):
        return float(np.max(np.abs(vector)))

    return float(np.sum(np.abs(vector) ** p) ** (1 / p))


def calculate_matrix_condition_number(matrix: np.ndarray) -> float:
    """
    Calculate the condition number of a matrix.

    The condition number measures how sensitive the solution of a
    linear system is to changes in the input data.

    Args:
        matrix: Input matrix

    Returns:
        Condition number (infinity if matrix is singular)

    Raises:
        ValueError: If matrix is not square
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    try:
        eigenvalues = np.linalg.eigvals(matrix)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        min_eigenvalue = np.min(np.abs(eigenvalues))

        if min_eigenvalue == 0:
            return float("inf")

        return float(max_eigenvalue / min_eigenvalue)
    except np.linalg.LinAlgError:
        return float("inf")


def calculate_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Calculate the correlation matrix from returns data.

    Args:
        returns: Returns matrix (time x assets)

    Returns:
        Correlation matrix

    Raises:
        ValueError: If returns matrix is invalid
    """
    if returns.ndim != 2:
        raise ValueError("Returns must be a 2D array")

    if returns.shape[0] < 2:
        raise ValueError("Need at least 2 time periods")

    if returns.shape[1] < 1:
        raise ValueError("Need at least 1 asset")

    # Handle NaN values
    returns_clean = returns.copy()
    returns_clean = returns_clean[~np.isnan(returns_clean).any(axis=1)]

    if len(returns_clean) < 2:
        raise ValueError("Insufficient valid data after removing NaN")

    return np.corrcoef(returns_clean.T)


def calculate_covariance_matrix(returns: np.ndarray, ddof: int = 1) -> np.ndarray:
    """
    Calculate the covariance matrix from returns data.

    Args:
        returns: Returns matrix (time x assets)
        ddof: Delta degrees of freedom (default: 1 for sample covariance)

    Returns:
        Covariance matrix

    Raises:
        ValueError: If returns matrix is invalid
    """
    if returns.ndim != 2:
        raise ValueError("Returns must be a 2D array")

    if returns.shape[0] < 2:
        raise ValueError("Need at least 2 time periods")

    if returns.shape[1] < 1:
        raise ValueError("Need at least 1 asset")

    # Handle NaN values
    returns_clean = returns.copy()
    returns_clean = returns_clean[~np.isnan(returns_clean).any(axis=1)]

    if len(returns_clean) < 2:
        raise ValueError("Insufficient valid data after removing NaN")

    return np.cov(returns_clean.T, ddof=ddof)


def calculate_sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)

    Returns:
        Annualized Sharpe ratio

    Raises:
        ValueError: If returns array is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]

    if len(returns_clean) == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = returns_clean - risk_free_rate / periods_per_year

    # Calculate mean and standard deviation
    mean_excess_return = np.mean(excess_returns)
    std_return = np.std(returns_clean, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize
    sharpe_ratio = (mean_excess_return * periods_per_year) / (
        std_return * math.sqrt(periods_per_year)
    )

    return float(sharpe_ratio)


def calculate_sortino_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino ratio for a series of returns.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)

    Returns:
        Annualized Sortino ratio

    Raises:
        ValueError: If returns array is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]

    if len(returns_clean) == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = returns_clean - risk_free_rate / periods_per_year

    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float("inf") if np.mean(excess_returns) > 0 else 0.0

    downside_deviation = np.std(downside_returns, ddof=1)

    if downside_deviation == 0:
        return 0.0

    # Annualize
    sortino_ratio = (np.mean(excess_returns) * periods_per_year) / (
        downside_deviation * math.sqrt(periods_per_year)
    )

    return float(sortino_ratio)


def calculate_max_drawdown(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.

    Args:
        returns: Array of returns

    Returns:
        Dictionary with drawdown metrics

    Raises:
        ValueError: If returns array is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]

    if len(returns_clean) == 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "drawdown_duration": 0,
            "peak_idx": 0,
            "trough_idx": 0,
        }

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns_clean)

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


def calculate_value_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 5%)

    Returns:
        VaR value

    Raises:
        ValueError: If returns array is empty or confidence level is invalid
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]

    if len(returns_clean) == 0:
        return 0.0

    return float(np.percentile(returns_clean, confidence_level * 100))


def calculate_conditional_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 5%)

    Returns:
        CVaR value

    Raises:
        ValueError: If returns array is empty or confidence level is invalid
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]

    if len(returns_clean) == 0:
        return 0.0

    var = calculate_value_at_risk(returns_clean, confidence_level)
    tail_returns = returns_clean[returns_clean <= var]

    if len(tail_returns) == 0:
        return float(var)

    return float(np.mean(tail_returns))


def normalize_vector(vector: np.ndarray, norm_type: str = "l2") -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector
        norm_type: Normalization type ('l1', 'l2', 'max')

    Returns:
        Normalized vector

    Raises:
        ValueError: If vector is empty or norm_type is invalid
    """
    if len(vector) == 0:
        raise ValueError("Vector cannot be empty")

    if norm_type == "l1":
        norm = np.sum(np.abs(vector))
    elif norm_type == "l2":
        norm = np.linalg.norm(vector)
    elif norm_type == "max":
        norm = np.max(np.abs(vector))
    else:
        raise ValueError("Invalid norm_type. Must be 'l1', 'l2', or 'max'")

    if norm == 0:
        return np.zeros_like(vector)

    return vector / norm


def calculate_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate eigenvalues of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Array of eigenvalues

    Raises:
        ValueError: If matrix is not square
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    try:
        eigenvalues = np.linalg.eigvals(matrix)
        return eigenvalues
    except np.linalg.LinAlgError:
        return np.array([])


def calculate_eigenvectors(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate eigenvalues and eigenvectors of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)

    Raises:
        ValueError: If matrix is not square
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors
    except np.linalg.LinAlgError:
        return np.array([]), np.array([])


# Export main functions
__all__ = [
    "calculate_vector_norm",
    "calculate_matrix_condition_number",
    "calculate_correlation_matrix",
    "calculate_covariance_matrix",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_value_at_risk",
    "calculate_conditional_var",
    "normalize_vector",
    "calculate_eigenvalues",
    "calculate_eigenvectors",
]
