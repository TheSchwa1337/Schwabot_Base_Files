"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Mathematical Foundation
============================

Core mathematical operations for the Schwabot trading system.

Features:
- Vector and matrix operations
- Statistical calculations
- Risk metrics
- Performance ratios
- Thermal state tracking
- Error handling and validation
- Performance optimization
"""

import logging
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as cp
    CUPY_AVAILABLE = False

# CUDA Integration with Fallback
try:
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ CleanMathFoundation using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 CleanMathFoundation using CPU fallback: {_backend}")


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
PI = np.pi
E = np.e
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
SQRT_2 = np.sqrt(2)
LN_2 = np.log(2)


class CleanMathFoundation:
    """
    Clean mathematical foundation providing core mathematical operations.

    This class serves as the foundation for all mathematical computations
    in the Schwabot trading system.
    """

    def __init__(self) -> None:
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


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio.

    Args:
        returns: Returns array
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)

    Returns:
        Sharpe ratio

    Raises:
        ValueError: If returns array is invalid
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    
    if len(returns_clean) == 0:
        return 0.0

    excess_returns = returns_clean - risk_free_rate / periods_per_year
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)

    if std_excess_return == 0:
        return 0.0

    return float(mean_excess_return / std_excess_return * np.sqrt(periods_per_year))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino ratio.

    Args:
        returns: Returns array
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)

    Returns:
        Sortino ratio

    Raises:
        ValueError: If returns array is invalid
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    
    if len(returns_clean) == 0:
        return 0.0

    excess_returns = returns_clean - risk_free_rate / periods_per_year
    mean_excess_return = np.mean(excess_returns)
    
    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    
    downside_deviation = np.std(downside_returns, ddof=1)

    if downside_deviation == 0:
        return 0.0

    return float(mean_excess_return / downside_deviation * np.sqrt(periods_per_year))


def calculate_max_drawdown(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.

    Args:
        returns: Returns array

    Returns:
        Dictionary with max drawdown, start index, and end index

    Raises:
        ValueError: If returns array is invalid
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    
    if len(returns_clean) == 0:
        return {"max_drawdown": 0.0, "start_index": 0, "end_index": 0}

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns_clean)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown_idx = np.argmin(drawdown)
    max_drawdown = drawdown[max_drawdown_idx]
    
    # Find start of drawdown (peak before max drawdown)
    peak_idx = np.argmax(cumulative_returns[:max_drawdown_idx + 1])
    
    return {
        "max_drawdown": float(max_drawdown),
        "start_index": int(peak_idx),
        "end_index": int(max_drawdown_idx)
    }


def calculate_value_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Returns array
        confidence_level: Confidence level (default: 0.05 for 95% VaR)

    Returns:
        Value at Risk

    Raises:
        ValueError: If returns array is invalid or confidence level is invalid
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
        returns: Returns array
        confidence_level: Confidence level (default: 0.05 for 95% CVaR)

    Returns:
        Conditional Value at Risk

    Raises:
        ValueError: If returns array is invalid or confidence level is invalid
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    # Remove NaN values
    returns_clean = returns[~np.isnan(returns)]
    
    if len(returns_clean) == 0:
        return 0.0

    var = np.percentile(returns_clean, confidence_level * 100)
    tail_returns = returns_clean[returns_clean <= var]
    
    if len(tail_returns) == 0:
        return float(var)
    
    return float(np.mean(tail_returns))


def normalize_vector(vector: np.ndarray, norm_type: str = "l2") -> np.ndarray:
    """
    Normalize a vector.

    Args:
        vector: Input vector
        norm_type: Normalization type ("l1", "l2", "max", "minmax")

    Returns:
        Normalized vector

    Raises:
        ValueError: If vector is empty or norm_type is invalid
    """
    if len(vector) == 0:
        raise ValueError("Vector cannot be empty")

    if norm_type == "l1":
        norm = np.sum(np.abs(vector))
        if norm == 0:
            return vector
        return vector / norm
    elif norm_type == "l2":
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    elif norm_type == "max":
        max_val = np.max(np.abs(vector))
        if max_val == 0:
            return vector
        return vector / max_val
    elif norm_type == "minmax":
        min_val = np.min(vector)
        max_val = np.max(vector)
        if max_val == min_val:
            return np.zeros_like(vector)
        return (vector - min_val) / (max_val - min_val)
    else:
        raise ValueError("Invalid norm_type. Use 'l1', 'l2', 'max', or 'minmax'")


def calculate_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate eigenvalues of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Eigenvalues array

    Raises:
        ValueError: If matrix is not square
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    try:
        return np.linalg.eigvals(matrix)
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


def create_math_foundation() -> CleanMathFoundation:
    """Create a new mathematical foundation instance."""
    return CleanMathFoundation()


def quick_calculation(operation: str, *args, **kwargs) -> Any:
    """Quick calculation wrapper for common operations."""
    foundation = CleanMathFoundation()
    
    if operation == "vector_norm":
        return calculate_vector_norm(*args, **kwargs)
    elif operation == "correlation":
        return calculate_correlation_matrix(*args, **kwargs)
    elif operation == "covariance":
        return calculate_covariance_matrix(*args, **kwargs)
    elif operation == "sharpe":
        return calculate_sharpe_ratio(*args, **kwargs)
    elif operation == "sortino":
        return calculate_sortino_ratio(*args, **kwargs)
    elif operation == "max_drawdown":
        return calculate_max_drawdown(*args, **kwargs)
    elif operation == "var":
        return calculate_value_at_risk(*args, **kwargs)
    elif operation == "cvar":
        return calculate_conditional_var(*args, **kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")
