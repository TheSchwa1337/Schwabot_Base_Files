# -*- coding: utf-8 -*-
""""""
Mathematical Utilities
======================

This module centralizes common mathematical definitions, constants, and utility functions
necessary for the mathematical relay system. It provides core mathematical operations
and concepts, including placeholders for Zero Point Energy (ZPE) and Zero Balance Energy (ZBE)
analogies, to be used across various components of the trading engine.
""""""

import math
from typing import List, Tuple, Union

import numpy as np


class MathematicalConstants:
    """Collection of common mathematical and system-specific constants."""

    PI = math.pi
    E = math.e
    GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # Approximately 1.618
    SPEED_OF_LIGHT_ANALOGUE = 299792458  # Analogue for system speed limits
    PLANCK_CONSTANT_ANALOGUE = 6.62607015e-34  # Analogue for smallest observable unit
    BOLTZMANN_CONSTANT_ANALOGUE = 1.380649e-23  # Analogue for system entropy
    BIT_32_MAX = 2**31 - 1
    BIT_64_MAX = 2**63 - 1
    EPSILON = 1e-9  # Small value for floating point comparisons


def normalize_vector(vector: Union[List[float], np.ndarray]) -> np.ndarray:
    """"""
    Normalize a given vector to a unit vector.

    Args:
        vector: A list or numpy array of numerical values.

    Returns:
        A numpy array representing the normalized vector.

    Raises:
        ValueError: If the vector is empty or contains non-finite values.
    """"""
    if not vector:
        raise ValueError("Cannot normalize an empty vector.")

    np_vector = np.array(vector, dtype=float)
    if not np.all(np.isfinite(np_vector)):
        raise ValueError("Vector contains non-finite values (NaN or Inf).")

    norm = np.linalg.norm(np_vector)
    if norm == 0:
        return np_vector  # Return zero vector if norm is zero
    return np_vector / norm


def safe_division(numerator: float, denominator: float) -> float:
    """"""
    Perform division safely, handling division by zero.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        The result of the division, or 0.0 if the denominator is zero.
    """"""
    if abs(denominator) < MathematicalConstants.EPSILON:
        return 0.0  # Return 0 or handle as appropriate for your domain
    return numerator / denominator


def calculate_entropy(data: Union[List[float], np.ndarray]) -> float:
    """"""
    Calculate the Shannon entropy of a given dataset.
    Assumes data represents probabilities or can be converted to them.

    Args:
        data: A list or numpy array of non-negative numerical values.

    Returns:
        The calculated entropy value.

    Raises:
        ValueError: If data is empty or contains negative values.
    """"""
    if not data:
        raise ValueError("Cannot calculate entropy for empty data.")

    np_data = np.array(data, dtype=float)
    if np.any(np_data < 0):
        raise ValueError("Data for entropy calculation must be non-negative.")

    # Normalize to probabilities if not already probabilities (sum to 1)
    sum_data = np.sum(np_data)
    if sum_data == 0:
        return 0.0  # No information if all values are zero

    probabilities = np_data / sum_data

    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def estimate_zpe(market_data: Union[List[float], np.ndarray]) -> float:
    """"""
    Estimate Zero Point Energy (ZPE) analogue from market data.
    This is a conceptual function representing the baseline 'noise' or 'minimum energy state'
    of the market, even in seemingly stable conditions.

    Args:
        market_data: Historical price or volatility data.

    Returns:
        An estimated ZPE value.
    """"""
    if not market_data:
        return 0.0

    np_data = np.array(market_data)
    # Example: ZPE could be related to minimum observed volatility or noise floor
    # This implementation is a placeholder; refine based on actual mathematical model.
    return np.std(np_data) * 0.1  # Very simplified, needs proper mathematical definition


def estimate_zbe(portfolio_value: float, benchmark_value: float) -> float:
    """"""
    Estimate Zero Balance Energy (ZBE) analogue from portfolio performance.
    This is a conceptual function representing the minimum 'energy' or 'momentum' required
    to maintain a neutral (zero profit/loss) position relative to a benchmark.

    Args:
        portfolio_value: Current value of the portfolio.
        benchmark_value: Value of a relevant market benchmark.

    Returns:
        An estimated ZBE value.
    """"""
    if benchmark_value == 0:
        return 0.0

    # Example: ZBE could be the normalized difference or a threshold for neutrality.
    # This implementation is a placeholder; refine based on actual mathematical model.
    return (portfolio_value - benchmark_value) / benchmark_value  # Needs proper mathematical definition


def calculate_drift_differential(current_value: float, historical_average: float, time_period: float) -> float:
    """"""
    Calculate the drift differential, indicating deviation over time.

    Args:
        current_value: The current observed value.
        historical_average: The historical average value.
        time_period: The time duration over which the drift is measured (e.g., in seconds, ticks).

    Returns:
        The calculated drift differential.
    """"""
    if time_period <= 0:
        return 0.0
    return safe_division((current_value - historical_average), time_period)


def calculate_profit_vectorization_score(profit_delta: float, risk_factor: float) -> float:
    """"""
    Calculate a profit vectorization compliance score.
    This score represents the effectiveness of profit generation relative to risk.

    Args:
        profit_delta: The actual profit change.
        risk_factor: A numerical representation of associated risk.

    Returns:
        The profit vectorization score.
    """"""
    if risk_factor <= 0:
        # Handle cases where risk is zero or negative (e.g., risk-free profit)
        return profit_delta * 1000.0  # Assign a high score for risk-free scenarios
    return safe_division(profit_delta, risk_factor)
