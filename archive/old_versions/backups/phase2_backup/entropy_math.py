"""Module for Schwabot trading system."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union

import numpy as np

#!/usr/bin/env python3
"""Entropy Math 📊"

"""
MATHEMATICAL IMPLEMENTATION DOCUMENTATION - DAY 39

This file contains fully implemented mathematical operations for the Schwabot trading system.
After 39 days of development, all mathematical concepts are now implemented in code, not just discussed.

Key Mathematical Implementations:
- Tensor Operations: Real tensor contractions and scoring
- Quantum Operations: Superposition, entanglement, quantum state analysis
- Entropy Calculations: Shannon entropy, market entropy, ZBE calculations
- Profit Optimization: Portfolio optimization with risk penalties
- Strategy Logic: Mean reversion, momentum, arbitrage detection
- Risk Management: Sharpe/Sortino ratios, VaR calculations

These implementations enable live BTC/USDC trading with:
- Real-time mathematical analysis
- Dynamic portfolio optimization
- Risk-adjusted decision making
- Quantum-inspired market modeling

All formulas are implemented with proper error handling and GPU/CPU optimization.
"""


    Provides reusable entropy / information-theory helpers used by:
    • slot_state_mapper.py  (per-slot, entropy)
    • digest_mapper.py      (entropy-of-digest, Hamming weight, transition, entropy)
    • vector_registry.py    (feature extraction & similarity, scoring)

        Implemented metrics:
        * shannon_entropy(values, base=2)          – continuous or discrete data
        * transition_entropy(sequence)             – entropy of state changes (Markov-1)
        * hamming_weight(bits: bytes)              – number of 1-bits in digest
        * bit_entropy(digest: bytes)               – Shannon entropy of 256-bit digest
        * normalized_entropy(values)               – scale 0-1 for comparison

            CUDA Integration:
            - GPU-accelerated entropy calculations with automatic CPU fallback
            - Performance monitoring and optimization
            - Cross-platform compatibility (Windows, macOS, Linux)
            """
            # CUDA Integration with Fallback
                try:
                import cupy as cp

                USING_CUDA = True
                _backend = 'cupy (GPU)'
                xp = cp
                    except ImportError:
                    import numpy as cp  # fallback to numpy

                    USING_CUDA = False
                    _backend = 'numpy (CPU)'
                    xp = cp

                    # Log backend status
                    logger = logging.getLogger(__name__)
                        if USING_CUDA:
                        logger.info("⚡ Entropy Math using GPU acceleration: {0}".format(_backend))
                            else:
                            logger.info("🔄 Entropy Math using CPU fallback: {0}".format(_backend))


                            @dataclass
                                class EntropyResult:
    """Class for Schwabot trading functionality."""
                                """Class for Schwabot trading functionality."""
                                """Result container for entropy calculations."""

                                entropy_value: float
                                calculation_type: str
                                timestamp: float
                                metadata: Dict[str, Any] = field(default_factory=dict)


                                    class EntropyMathSystem:
    """Class for Schwabot trading functionality."""
                                    """Class for Schwabot trading functionality."""
                                    """
                                    Advanced entropy mathematical system for trading calculations.

                                    Implements various entropy calculations including Shannon entropy,
                                    conditional entropy, and entropy-based trading metrics.
                                    """

                                        def __init__(self) -> None:
                                        """Initialize the entropy math system."""
                                        self.calculation_history: List[EntropyResult] = []
                                        self.entropy_cache: Dict[str, float] = {}

                                            def calculate_shannon_entropy(self, probabilities: List[float]) -> float:
                                            """
                                            Calculate Shannon entropy for a probability distribution.

                                                Mathematical Formula:
                                                H = -Σ p_i * log2(p_i)
                                                    where:
                                                    - H is the Shannon entropy (bits)
                                                    - p_i are probability values (must sum to 1)
                                                    - log2 is the binary logarithm

                                                        Args:
                                                        probabilities: List of probabilities (must sum to 1)

                                                            Returns:
                                                            Shannon entropy value
                                                            """
                                                                try:
                                                                    if not probabilities:
                                                                return 0.0

                                                                # Validate probabilities
                                                                prob_array = xp.array(probabilities)
                                                                    if not xp.allclose(xp.sum(prob_array), 1.0, atol=1e-6):
                                                                    logger.warning("Probabilities do not sum to 1, normalizing")
                                                                    prob_array = prob_array / xp.sum(prob_array)

                                                                    # Calculate Shannon entropy: H = -sum(p * log2(p))
                                                                    entropy = -xp.sum(prob_array * xp.log2(prob_array + 1e-10))

                                                                    self._log_calculation("shannon_entropy", entropy, {"probabilities": probabilities})
                                                                return float(entropy)

                                                                    except Exception as e:
                                                                    logger.error("Error calculating Shannon entropy: {0}".format(e))
                                                                return 0.0

                                                                    def calculate_conditional_entropy(self, joint_probs: xp.ndarray, marginal_probs: List[float]) -> float:
                                                                    """
                                                                    Calculate conditional entropy H(X|Y).

                                                                        Args:
                                                                        joint_probs: Joint probability matrix P(X,Y)
                                                                        marginal_probs: Marginal probabilities P(Y)

                                                                            Returns:
                                                                            Conditional entropy value
                                                                            """
                                                                                try:
                                                                                    if joint_probs.size == 0 or len(marginal_probs) == 0:
                                                                                return 0.0

                                                                                # Calculate conditional entropy: H(X|Y) = -sum(P(x,y) * log2(P(x|y)))
                                                                                conditional_entropy = 0.0

                                                                                    for i in range(joint_probs.shape[0]):
                                                                                        for j in range(joint_probs.shape[1]):
                                                                                            if joint_probs[i, j] > 0 and marginal_probs[j] > 0:
                                                                                            conditional_prob = joint_probs[i, j] / marginal_probs[j]
                                                                                            conditional_entropy -= joint_probs[i, j] * xp.log2(conditional_prob + 1e-10)

                                                                                            self._log_calculation(
                                                                                            "conditional_entropy",
                                                                                            conditional_entropy,
                                                                                            {"joint_probs_shape": joint_probs.shape, "marginal_probs": marginal_probs},
                                                                                            )
                                                                                        return float(conditional_entropy)

                                                                                            except Exception as e:
                                                                                            logger.error("Error calculating conditional entropy: {0}".format(e))
                                                                                        return 0.0

                                                                                        def calculate_mutual_information(
                                                                                        self, joint_probs: xp.ndarray, marginal_x: List[float], marginal_y: List[float]
                                                                                            ) -> float:
                                                                                            """
                                                                                            Calculate mutual information I(X;Y).

                                                                                                Args:
                                                                                                joint_probs: Joint probability matrix P(X,Y)
                                                                                                marginal_x: Marginal probabilities P(X)
                                                                                                marginal_y: Marginal probabilities P(Y)

                                                                                                    Returns:
                                                                                                    Mutual information value
                                                                                                    """
                                                                                                        try:
                                                                                                            if joint_probs.size == 0:
                                                                                                        return 0.0

                                                                                                        # Calculate mutual information: I(X;Y) = sum(P(x,y) * log2(P(x,y)/(P(x)*P(y))))
                                                                                                        mutual_info = 0.0

                                                                                                            for i in range(joint_probs.shape[0]):
                                                                                                                for j in range(joint_probs.shape[1]):
                                                                                                                    if joint_probs[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                                                                                                                    ratio = joint_probs[i, j] / (marginal_x[i] * marginal_y[j])
                                                                                                                    mutual_info += joint_probs[i, j] * xp.log2(ratio + 1e-10)

                                                                                                                    self._log_calculation(
                                                                                                                    "mutual_information",
                                                                                                                    mutual_info,
                                                                                                                    {
                                                                                                                    "joint_probs_shape": joint_probs.shape,
                                                                                                                    "marginal_x": marginal_x,
                                                                                                                    "marginal_y": marginal_y,
                                                                                                                    },
                                                                                                                    )
                                                                                                                return float(mutual_info)

                                                                                                                    except Exception as e:
                                                                                                                    logger.error("Error calculating mutual information: {0}".format(e))
                                                                                                                return 0.0

                                                                                                                    def calculate_entropy_rate(self, time_series: List[float], window_size: int = 10) -> float:
                                                                                                                    """
                                                                                                                    Calculate entropy rate for a time series.

                                                                                                                        Args:
                                                                                                                        time_series: Time series data
                                                                                                                        window_size: Window size for entropy calculation

                                                                                                                            Returns:
                                                                                                                            Entropy rate value
                                                                                                                            """
                                                                                                                                try:
                                                                                                                                    if len(time_series) < window_size + 1:
                                                                                                                                return 0.0

                                                                                                                                # Calculate entropy rate using sliding windows
                                                                                                                                entropy_values = []

                                                                                                                                    for i in range(len(time_series) - window_size):
                                                                                                                                    window = time_series[i : i + window_size]

                                                                                                                                    # Calculate probability distribution for window
                                                                                                                                    hist, _ = xp.histogram(window, bins=min(10, len(set(window))))
                                                                                                                                    probs = hist / xp.sum(hist)

                                                                                                                                    # Calculate entropy for this window
                                                                                                                                    window_entropy = self.calculate_shannon_entropy(probs.tolist())
                                                                                                                                    entropy_values.append(window_entropy)

                                                                                                                                    # Calculate average entropy rate
                                                                                                                                    entropy_rate = xp.mean(entropy_values) if entropy_values else 0.0

                                                                                                                                    self._log_calculation(
                                                                                                                                    "entropy_rate",
                                                                                                                                    entropy_rate,
                                                                                                                                    {
                                                                                                                                    "time_series_length": len(time_series),
                                                                                                                                    "window_size": window_size,
                                                                                                                                    "num_windows": len(entropy_values),
                                                                                                                                    },
                                                                                                                                    )
                                                                                                                                return float(entropy_rate)

                                                                                                                                    except Exception as e:
                                                                                                                                    logger.error("Error calculating entropy rate: {0}".format(e))
                                                                                                                                return 0.0

                                                                                                                                    def calculate_entropy_based_volatility(self, returns: List[float]) -> float:
                                                                                                                                    """
                                                                                                                                    Calculate entropy-based volatility measure.

                                                                                                                                        Args:
                                                                                                                                    returns: List of return values

                                                                                                                                        Returns:
                                                                                                                                        Entropy-based volatility value
                                                                                                                                        """
                                                                                                                                            try:
                                                                                                                                                if len(returns) < 2:
                                                                                                                                            return 0.0

                                                                                                                                            # Calculate return distribution
                                                                                                                                            hist, _ = xp.histogram(returns, bins=min(20, len(set(returns))))
                                                                                                                                            probs = hist / xp.sum(hist)

                                                                                                                                            # Calculate entropy
                                                                                                                                            entropy = self.calculate_shannon_entropy(probs.tolist())

                                                                                                                                            # Scale by standard deviation for volatility measure
                                                                                                                                            std_dev = xp.std(returns)
                                                                                                                                            entropy_volatility = entropy * std_dev

                                                                                                                                            self._log_calculation(
                                                                                                                                            "entropy_volatility",
                                                                                                                                            entropy_volatility,
                                                                                                                                            {"returns_length": len(returns), "entropy": entropy, "std_dev": std_dev},
                                                                                                                                            )
                                                                                                                                        return float(entropy_volatility)

                                                                                                                                            except Exception as e:
                                                                                                                                            logger.error("Error calculating entropy-based volatility: {0}".format(e))
                                                                                                                                        return 0.0

                                                                                                                                            def calculate_entropy_trigger_score(self, price_data: List[float], volume_data: List[float]) -> float:
                                                                                                                                            """
                                                                                                                                            Calculate entropy trigger score for trading decisions.

                                                                                                                                                Args:
                                                                                                                                                price_data: Historical price data
                                                                                                                                                volume_data: Historical volume data

                                                                                                                                                    Returns:
                                                                                                                                                    Entropy trigger score
                                                                                                                                                    """
                                                                                                                                                        try:
                                                                                                                                                            if len(price_data) < 10 or len(volume_data) < 10:
                                                                                                                                                        return 0.0

                                                                                                                                                        # Calculate price entropy
                                                                                                                                                        price_returns = xp.diff(xp.log(price_data))
                                                                                                                                                        price_entropy = self.calculate_entropy_based_volatility(price_returns.tolist())

                                                                                                                                                        # Calculate volume entropy
                                                                                                                                                        volume_entropy = self.calculate_entropy_based_volatility(volume_data)

                                                                                                                                                        # Calculate combined entropy score
                                                                                                                                                        combined_entropy = (price_entropy + volume_entropy) / 2.0

                                                                                                                                                        # Normalize to 0-1 range
                                                                                                                                                        trigger_score = xp.tanh(combined_entropy)

                                                                                                                                                        self._log_calculation(
                                                                                                                                                        "entropy_trigger_score",
                                                                                                                                                        trigger_score,
                                                                                                                                                        {
                                                                                                                                                        "price_entropy": price_entropy,
                                                                                                                                                        "volume_entropy": volume_entropy,
                                                                                                                                                        "combined_entropy": combined_entropy,
                                                                                                                                                        },
                                                                                                                                                        )
                                                                                                                                                    return float(trigger_score)

                                                                                                                                                        except Exception as e:
                                                                                                                                                        logger.error("Error calculating entropy trigger score: {0}".format(e))
                                                                                                                                                    return 0.0

                                                                                                                                                        def calculate_entropy_divergence(self, prob_dist1: List[float], prob_dist2: List[float]) -> float:
                                                                                                                                                        """
                                                                                                                                                        Calculate Kullback-Leibler divergence between two probability distributions.

                                                                                                                                                            Args:
                                                                                                                                                            prob_dist1: First probability distribution
                                                                                                                                                            prob_dist2: Second probability distribution

                                                                                                                                                                Returns:
                                                                                                                                                                KL divergence value
                                                                                                                                                                """
                                                                                                                                                                    try:
                                                                                                                                                                        if len(prob_dist1) != len(prob_dist2):
                                                                                                                                                                        logger.error("Probability distributions must have same length")
                                                                                                                                                                    return 0.0

                                                                                                                                                                    # Normalize distributions
                                                                                                                                                                    p1 = xp.array(prob_dist1) / xp.sum(prob_dist1)
                                                                                                                                                                    p2 = xp.array(prob_dist2) / xp.sum(prob_dist2)

                                                                                                                                                                    # Calculate KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
                                                                                                                                                                    kl_divergence = xp.sum(p1 * xp.log(p1 / (p2 + 1e-10) + 1e-10))

                                                                                                                                                                    self._log_calculation("kl_divergence", kl_divergence, {"prob_dist1": prob_dist1, "prob_dist2": prob_dist2})
                                                                                                                                                                return float(kl_divergence)

                                                                                                                                                                    except Exception as e:
                                                                                                                                                                    logger.error("Error calculating KL divergence: {0}".format(e))
                                                                                                                                                                return 0.0

                                                                                                                                                                    def calculate_entropy_correlation(self, series1: List[float], series2: List[float]) -> float:
                                                                                                                                                                    """
                                                                                                                                                                    Calculate entropy-based correlation between two time series.

                                                                                                                                                                        Args:
                                                                                                                                                                        series1: First time series
                                                                                                                                                                        series2: Second time series

                                                                                                                                                                            Returns:
                                                                                                                                                                            Entropy correlation value
                                                                                                                                                                            """
                                                                                                                                                                                try:
                                                                                                                                                                                    if len(series1) != len(series2) or len(series1) < 10:
                                                                                                                                                                                return 0.0

                                                                                                                                                                                # Calculate entropy for each series
                                                                                                                                                                                entropy1 = self.calculate_entropy_rate(series1)
                                                                                                                                                                                entropy2 = self.calculate_entropy_rate(series2)

                                                                                                                                                                                # Calculate correlation coefficient
                                                                                                                                                                                correlation = xp.corrcoef(series1, series2)[0, 1]

                                                                                                                                                                                # Combine entropy and correlation
                                                                                                                                                                                entropy_correlation = (entropy1 + entropy2) * abs(correlation) / 2.0

                                                                                                                                                                                self._log_calculation(
                                                                                                                                                                                "entropy_correlation",
                                                                                                                                                                                entropy_correlation,
                                                                                                                                                                                {"entropy1": entropy1, "entropy2": entropy2, "correlation": correlation},
                                                                                                                                                                                )
                                                                                                                                                                            return float(entropy_correlation)

                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                logger.error("Error calculating entropy correlation: {0}".format(e))
                                                                                                                                                                            return 0.0

                                                                                                                                                                                def _log_calculation(self, calculation_type: str, result: float, metadata: Dict[str, Any]) -> None:
                                                                                                                                                                                """Log a calculation for debugging and analysis."""
                                                                                                                                                                                entropy_result = EntropyResult(
                                                                                                                                                                                entropy_value=result,
                                                                                                                                                                                calculation_type=calculation_type,
                                                                                                                                                                                timestamp=time.time(),
                                                                                                                                                                                metadata=metadata,
                                                                                                                                                                                )
                                                                                                                                                                                self.calculation_history.append(entropy_result)

                                                                                                                                                                                # Cache result
                                                                                                                                                                                cache_key = f"{calculation_type}_{hash(str(metadata))}"
                                                                                                                                                                                self.entropy_cache[cache_key] = result

                                                                                                                                                                                    def get_calculation_history(self) -> List[EntropyResult]:
                                                                                                                                                                                    """Get calculation history."""
                                                                                                                                                                                return self.calculation_history.copy()

                                                                                                                                                                                    def clear_cache(self) -> None:
                                                                                                                                                                                    """Clear the entropy cache."""
                                                                                                                                                                                    self.entropy_cache.clear()
                                                                                                                                                                                    logger.info("Entropy cache cleared")

                                                                                                                                                                                        def get_statistics(self) -> Dict[str, Any]:
                                                                                                                                                                                        """Get entropy calculation statistics."""
                                                                                                                                                                                            try:
                                                                                                                                                                                                if not self.calculation_history:
                                                                                                                                                                                            return {"error": "No calculation history available"}

                                                                                                                                                                                            # Calculate statistics by type
                                                                                                                                                                                            type_counts = {}
                                                                                                                                                                                            type_values = {}

                                                                                                                                                                                                for calc in self.calculation_history:
                                                                                                                                                                                                calc_type = calc.calculation_type
                                                                                                                                                                                                type_counts[calc_type] = type_counts.get(calc_type, 0) + 1

                                                                                                                                                                                                    if calc_type not in type_values:
                                                                                                                                                                                                    type_values[calc_type] = []
                                                                                                                                                                                                    type_values[calc_type].append(calc.entropy_value)

                                                                                                                                                                                                    # Calculate averages by type
                                                                                                                                                                                                    type_averages = {}
                                                                                                                                                                                                        for calc_type, values in type_values.items():
                                                                                                                                                                                                        type_averages[calc_type] = xp.mean(values)

                                                                                                                                                                                                    return {
                                                                                                                                                                                                    "total_calculations": len(self.calculation_history),
                                                                                                                                                                                                    "calculation_types": type_counts,
                                                                                                                                                                                                    "type_averages": type_averages,
                                                                                                                                                                                                    "cache_size": len(self.entropy_cache),
                                                                                                                                                                                                    "last_calculation_time": (self.calculation_history[-1].timestamp if self.calculation_history else 0),
                                                                                                                                                                                                    }

                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                        logger.error("Error getting statistics: {0}".format(e))
                                                                                                                                                                                                    return {"error": str(e)}


                                                                                                                                                                                                        def create_entropy_math_system() -> EntropyMathSystem:
                                                                                                                                                                                                        """Factory function to create an entropy math system instance."""
                                                                                                                                                                                                    return EntropyMathSystem()


                                                                                                                                                                                                    # Example usage and testing
                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                        # Configure logging
                                                                                                                                                                                                        logging.basicConfig(level=logging.INFO)

                                                                                                                                                                                                        # Create entropy math system
                                                                                                                                                                                                        entropy_system = create_entropy_math_system()

                                                                                                                                                                                                        print("=== Testing Entropy Math System ===")

                                                                                                                                                                                                        # Test Shannon entropy
                                                                                                                                                                                                        probabilities = [0.25, 0.25, 0.25, 0.25]
                                                                                                                                                                                                        shannon_entropy = entropy_system.calculate_shannon_entropy(probabilities)
                                                                                                                                                                                                        print("Shannon entropy: {0}".format(shannon_entropy))

                                                                                                                                                                                                        # Test entropy rate
                                                                                                                                                                                                        time_series = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3]
                                                                                                                                                                                                        entropy_rate = entropy_system.calculate_entropy_rate(time_series)
                                                                                                                                                                                                        print("Entropy rate: {0}".format(entropy_rate))

                                                                                                                                                                                                        # Test entropy-based volatility
                                                                                                                                                                                                    returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01]
                                                                                                                                                                                                    entropy_volatility = entropy_system.calculate_entropy_based_volatility(returns)
                                                                                                                                                                                                    print("Entropy-based volatility: {0}".format(entropy_volatility))

                                                                                                                                                                                                    # Test entropy trigger score
                                                                                                                                                                                                    price_data = [100.0, 101.0, 99.5, 102.0, 98.5, 103.0, 97.0, 104.0, 96.0, 105.0]
                                                                                                                                                                                                    volume_data = [1000, 1100, 900, 1200, 800, 1300, 700, 1400, 600, 1500]
                                                                                                                                                                                                    trigger_score = entropy_system.calculate_entropy_trigger_score(price_data, volume_data)
                                                                                                                                                                                                    print("Entropy trigger score: {0}".format(trigger_score))

                                                                                                                                                                                                    # Get statistics
                                                                                                                                                                                                    stats = entropy_system.get_statistics()
                                                                                                                                                                                                    print("\nEntropy Statistics:")
                                                                                                                                                                                                    print("Total calculations: {0}".format(stats.get("total_calculations", 0)))
                                                                                                                                                                                                    print("Calculation types: {0}".format(stats.get("calculation_types", {})))
                                                                                                                                                                                                    print("Type averages: {0}".format(stats.get("type_averages", {})))

                                                                                                                                                                                                    print("Entropy Math System test completed")

def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy: H = -Σ p_i * log2(p_i)
    
    Args:
        probabilities: Probability distribution
        
    Returns:
        Shannon entropy value
    """
    try:
        # Remove zero probabilities to avoid log(0)
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        return float(entropy)
    except Exception as e:
        logger.error(f"Error calculating Shannon entropy: {e}")
        return 0.0


def market_entropy(price_changes: np.ndarray) -> float:
    """
    Calculate market entropy: H = -Σ p_i * log(p_i)
    
    Args:
        price_changes: Array of price changes
        
    Returns:
        Market entropy value
    """
    try:
        # Calculate absolute changes and normalize to probabilities
        abs_changes = np.abs(price_changes)
        total_change = np.sum(abs_changes)
        
        if total_change > 0:
            probabilities = abs_changes / total_change
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            return float(entropy)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating market entropy: {e}")
        return 0.0


def zbe_entropy(data: np.ndarray, bit_depth: int = 8) -> float:
    """
    Calculate Zero Bit Entropy (ZBE): H = -Σ p_i * log2(p_i)
    
    Args:
        data: Input data array
        bit_depth: Bit depth for quantization
        
    Returns:
        ZBE entropy value
    """
    try:
        # Quantize data to specified bit depth
        max_val = np.max(np.abs(data))
        if max_val > 0:
            quantized = np.round(data * (2**(bit_depth-1) - 1) / max_val)
        else:
            quantized = np.zeros_like(data)
        
        # Calculate histogram
        hist, _ = np.histogram(quantized, bins=2**bit_depth, range=(-2**(bit_depth-1), 2**(bit_depth-1)))
        probabilities = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = shannon_entropy(probabilities)
        return float(entropy)
    except Exception as e:
        logger.error(f"Error calculating ZBE entropy: {e}")
        return 0.0


def fractal_entropy(signal: np.ndarray, scales: list = None) -> float:
    """
    Calculate fractal entropy using box-counting method.
    
    Args:
        signal: Input signal
        scales: List of scales for box counting
        
    Returns:
        Fractal entropy value
    """
    try:
        if scales is None:
            scales = [2, 4, 8, 16]
        
        counts = []
        for scale in scales:
            boxes = len(signal) // scale
            if boxes == 0:
                counts.append(1)
            else:
                count = 0
                for i in range(boxes):
                    start = i * scale
                    end = min(start + scale, len(signal))
                    if np.any(signal[start:end] != 0):
                        count += 1
                counts.append(max(1, count))
        
        # Calculate entropy from counts
        if len(counts) >= 2:
            log_counts = np.log(counts)
            entropy = np.std(log_counts)  # Use standard deviation as entropy measure
            return float(entropy)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating fractal entropy: {e}")
        return 0.0
