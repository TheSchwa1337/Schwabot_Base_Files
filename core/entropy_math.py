#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ENTROPY MATH SYSTEM - INFORMATION THEORY FOUNDATION
=======================================================

Advanced entropy mathematical system for trading calculations.

Implements various entropy calculations including Shannon entropy,
conditional entropy, and entropy-based trading metrics.

Mathematical Foundation:
- Shannon Entropy: H = -Σ p_i * log2(p_i)
- Conditional Entropy: H(X|Y) = -Σ P(x,y) * log2(P(x|y))
- Mutual Information: I(X;Y) = Σ P(x,y) * log2(P(x,y)/(P(x)*P(y)))
- Market Entropy: Real-time market entropy calculations
- ZBE Calculations: Zero-point energy entropy analysis

CUDA Integration:
- GPU-accelerated entropy calculations with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = 'cupy (GPU)'
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = 'numpy (CPU)'
    xp = np

# Import existing Schwabot components
try:
    from .quantum_mathematical_bridge import QuantumMathematicalBridge
    from .advanced_tensor_algebra import AdvancedTensorAlgebra
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ Entropy Math using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 Entropy Math using CPU fallback: {_backend}")


@dataclass
class EntropyResult:
    """Result container for entropy calculations."""
    entropy_value: float
    calculation_type: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntropyMathSystem:
    """
    Advanced entropy mathematical system for trading calculations.

    Implements various entropy calculations including Shannon entropy,
    conditional entropy, and entropy-based trading metrics.
    """

    def __init__(self) -> None:
        """Initialize the entropy math system."""
        self.calculation_history: List[EntropyResult] = []
        self.entropy_cache: Dict[str, float] = {}
        
        # Performance tracking
        self.total_calculations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize Schwabot components if available
        if SCHWABOT_COMPONENTS_AVAILABLE:
            self.quantum_bridge = QuantumMathematicalBridge()
            self.tensor_algebra = AdvancedTensorAlgebra()

        logger.info("📊 Entropy Math System initialized")

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
            logger.error(f"Error calculating Shannon entropy: {e}")
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
            logger.error(f"Error calculating conditional entropy: {e}")
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
            logger.error(f"Error calculating mutual information: {e}")
return 0.0

    def calculate_market_entropy(self, price_changes: List[float], window_size: int = 20) -> float:
"""
Calculate market entropy from price changes.

Args:
price_changes: List of price changes
            window_size: Window size for calculation

Returns:
Market entropy value
"""
try:
            if len(price_changes) < window_size:
return 0.0

            # Use recent price changes
            recent_changes = price_changes[-window_size:]

            # Calculate probability distribution from price changes
            abs_changes = xp.abs(xp.array(recent_changes))
            total_change = xp.sum(abs_changes)

            if total_change > 0:
probabilities = abs_changes / total_change
                market_entropy = self.calculate_shannon_entropy(probabilities.tolist())
            else:
                market_entropy = 0.0

            self._log_calculation(
                "market_entropy",
                market_entropy,
                {"window_size": window_size, "price_changes_count": len(recent_changes)},
            )
            return market_entropy

except Exception as e:
            logger.error(f"Error calculating market entropy: {e}")
return 0.0

    def calculate_zbe_entropy(self, energy_levels: List[float], temperature: float = 1.0) -> float:
"""
        Calculate Zero-Point Energy (ZBE) entropy.

Args:
            energy_levels: List of energy levels
            temperature: System temperature

Returns:
ZBE entropy value
"""
try:
            if not energy_levels:
                return 0.0

            energy_array = xp.array(energy_levels)
            
            # Boltzmann distribution
            boltzmann_factors = xp.exp(-energy_array / temperature)
            partition_function = xp.sum(boltzmann_factors)
            
            if partition_function > 0:
                probabilities = boltzmann_factors / partition_function
                zbe_entropy = self.calculate_shannon_entropy(probabilities.tolist())
            else:
                zbe_entropy = 0.0

            self._log_calculation(
                "zbe_entropy",
                zbe_entropy,
                {"temperature": temperature, "energy_levels_count": len(energy_levels)},
            )
            return zbe_entropy

        except Exception as e:
            logger.error(f"Error calculating ZBE entropy: {e}")
            return 0.0

    def calculate_transition_entropy(self, sequence: List[str]) -> float:
        """
        Calculate entropy of state transitions (Markov-1).

        Args:
            sequence: Sequence of states

        Returns:
            Transition entropy value
        """
        try:
            if len(sequence) < 2:
                return 0.0

            # Count transitions
            transition_counts = {}
            state_counts = {}

            for i in range(len(sequence) - 1):
                current_state = sequence[i]
                next_state = sequence[i + 1]
                
                # Count states
                state_counts[current_state] = state_counts.get(current_state, 0) + 1
                
                # Count transitions
                if current_state not in transition_counts:
                    transition_counts[current_state] = {}
                transition_counts[current_state][next_state] = transition_counts[current_state].get(next_state, 0) + 1

            # Calculate transition probabilities
            total_entropy = 0.0
            total_transitions = len(sequence) - 1

            for current_state, transitions in transition_counts.items():
                state_count = state_counts[current_state]
                state_prob = state_count / total_transitions
                
                # Calculate conditional entropy for this state
                conditional_probs = [count / state_count for count in transitions.values()]
                conditional_entropy = self.calculate_shannon_entropy(conditional_probs)
                
                total_entropy += state_prob * conditional_entropy

            self._log_calculation(
                "transition_entropy",
                total_entropy,
                {"sequence_length": len(sequence), "unique_states": len(state_counts)},
            )
            return total_entropy

        except Exception as e:
            logger.error(f"Error calculating transition entropy: {e}")
return 0.0

    def calculate_hamming_weight(self, bits: bytes) -> int:
        """
        Calculate Hamming weight (number of 1-bits).

        Args:
            bits: Byte sequence

        Returns:
            Hamming weight
        """
        try:
            hamming_weight = 0
            for byte in bits:
                hamming_weight += bin(byte).count('1')
            
            return hamming_weight

        except Exception as e:
            logger.error(f"Error calculating Hamming weight: {e}")
            return 0

    def calculate_bit_entropy(self, digest: bytes) -> float:
        """
        Calculate Shannon entropy of 256-bit digest.

        Args:
            digest: Byte digest

        Returns:
            Bit entropy value
        """
        try:
            # Convert bytes to bit sequence
            bit_sequence = []
            for byte in digest:
                bit_sequence.extend([int(b) for b in format(byte, '08b')])

# Calculate probabilities
            total_bits = len(bit_sequence)
            if total_bits == 0:
                return 0.0

            ones_count = sum(bit_sequence)
            zeros_count = total_bits - ones_count

            probabilities = [zeros_count / total_bits, ones_count / total_bits]
            bit_entropy = self.calculate_shannon_entropy(probabilities)

            self._log_calculation(
                "bit_entropy",
                bit_entropy,
                {"digest_length": len(digest), "total_bits": total_bits},
            )
            return bit_entropy

        except Exception as e:
            logger.error(f"Error calculating bit entropy: {e}")
            return 0.0

    def calculate_normalized_entropy(self, values: List[float]) -> float:
        """
        Calculate normalized entropy (scale 0-1) for comparison.

        Args:
            values: List of values

        Returns:
            Normalized entropy value [0,1]
        """
        try:
            if not values:
                return 0.0

            # Calculate raw entropy
            value_array = xp.array(values)
            total = xp.sum(xp.abs(value_array))
            
            if total > 0:
                probabilities = xp.abs(value_array) / total
                raw_entropy = self.calculate_shannon_entropy(probabilities.tolist())
                
                # Normalize to [0,1]
                max_entropy = xp.log2(len(values))
                if max_entropy > 0:
                    normalized_entropy = raw_entropy / max_entropy
                else:
                    normalized_entropy = 0.0
            else:
                normalized_entropy = 0.0

            self._log_calculation(
                "normalized_entropy",
                normalized_entropy,
                {"values_count": len(values), "max_entropy": xp.log2(len(values))},
            )
            return float(normalized_entropy)

except Exception as e:
            logger.error(f"Error calculating normalized entropy: {e}")
            return 0.0

    def calculate_quantum_entropy(self, quantum_state: Dict[str, Any]) -> float:
        """
        Calculate quantum entropy from quantum state.

        Args:
            quantum_state: Quantum state dictionary

        Returns:
            Quantum entropy value
        """
        try:
            if SCHWABOT_COMPONENTS_AVAILABLE:
                # Use quantum bridge if available
                if hasattr(self.quantum_bridge, 'quantum_entropy_calculation'):
                    # Extract amplitudes from quantum state
                    amplitudes = []
                    for component in quantum_state.get('superposition_components', {}).values():
                        if isinstance(component, complex):
                            amplitudes.append(abs(component))
                        else:
                            amplitudes.append(float(component))
                    
                    if amplitudes:
                        # Normalize to probabilities
                        total = sum(amplitudes)
                        if total > 0:
                            probabilities = [amp / total for amp in amplitudes]
                            quantum_entropy = self.calculate_shannon_entropy(probabilities)
                        else:
                            quantum_entropy = 0.0
                    else:
                        quantum_entropy = 0.0
                else:
                    # Fallback calculation
                    quantum_entropy = self.calculate_shannon_entropy([0.5, 0.5])
            else:
                # Fallback calculation
                quantum_entropy = self.calculate_shannon_entropy([0.5, 0.5])

            self._log_calculation(
                "quantum_entropy",
                quantum_entropy,
                {"quantum_state_keys": list(quantum_state.keys())},
            )
            return quantum_entropy

        except Exception as e:
            logger.error(f"Error calculating quantum entropy: {e}")
            return 0.0

    def calculate_entropy_gradient(self, entropy_history: List[float], window_size: int = 10) -> float:
        """
        Calculate entropy gradient (rate of change).

        Args:
            entropy_history: History of entropy values
            window_size: Window size for gradient calculation

        Returns:
            Entropy gradient value
        """
        try:
            if len(entropy_history) < window_size:
                return 0.0

            recent_entropy = entropy_history[-window_size:]
            
            # Calculate gradient using linear regression
            x = xp.arange(len(recent_entropy))
            y = xp.array(recent_entropy)
            
            # Linear regression: y = mx + b
            n = len(x)
            sum_x = xp.sum(x)
            sum_y = xp.sum(y)
            sum_xy = xp.sum(x * y)
            sum_x2 = xp.sum(x * x)
            
            # Calculate slope (gradient)
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator != 0:
                gradient = (n * sum_xy - sum_x * sum_y) / denominator
            else:
                gradient = 0.0

            self._log_calculation(
                "entropy_gradient",
                gradient,
                {"window_size": window_size, "history_length": len(entropy_history)},
            )
            return float(gradient)

        except Exception as e:
            logger.error(f"Error calculating entropy gradient: {e}")
            return 0.0

    def calculate_entropy_volatility(self, entropy_history: List[float], window_size: int = 20) -> float:
        """
        Calculate entropy volatility (standard deviation).

        Args:
            entropy_history: History of entropy values
            window_size: Window size for volatility calculation

        Returns:
            Entropy volatility value
        """
        try:
            if len(entropy_history) < window_size:
                return 0.0

            recent_entropy = entropy_history[-window_size:]
            volatility = float(xp.std(xp.array(recent_entropy)))

            self._log_calculation(
                "entropy_volatility",
                volatility,
                {"window_size": window_size, "history_length": len(entropy_history)},
            )
            return volatility

        except Exception as e:
            logger.error(f"Error calculating entropy volatility: {e}")
return 0.0

    def _log_calculation(self, calculation_type: str, entropy_value: float, metadata: Dict[str, Any]) -> None:
        """Log entropy calculation result."""
        try:
            result = EntropyResult(
                entropy_value=entropy_value,
                calculation_type=calculation_type,
                timestamp=time.time(),
                metadata=metadata
            )
            
            self.calculation_history.append(result)
            self.total_calculations += 1
            
            # Keep history manageable
            if len(self.calculation_history) > 1000:
                self.calculation_history = self.calculation_history[-500:]

        except Exception as e:
            logger.error(f"Error logging calculation: {e}")

    def get_calculation_history(self, calculation_type: Optional[str] = None, limit: int = 100) -> List[EntropyResult]:
        """
        Get calculation history.

        Args:
            calculation_type: Filter by calculation type
            limit: Maximum number of results to return

        Returns:
            List of EntropyResult objects
        """
        try:
            if calculation_type:
                filtered_history = [result for result in self.calculation_history if result.calculation_type == calculation_type]
            else:
                filtered_history = self.calculation_history

            return filtered_history[-limit:]

        except Exception as e:
            logger.error(f"Error getting calculation history: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear entropy cache."""
        try:
            self.entropy_cache.clear()
            self.calculation_history.clear()
            logger.info("📊 Entropy Math cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'total_calculations': self.total_calculations,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_size': len(self.entropy_cache),
                'history_size': len(self.calculation_history),
                'backend': _backend,
                'schwabot_components_available': SCHWABOT_COMPONENTS_AVAILABLE,
                'recent_calculations': [
                    {
                        'type': result.calculation_type,
                        'value': result.entropy_value,
                        'timestamp': result.timestamp
                    }
                    for result in self.calculation_history[-10:]
                ]
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


# Global instance for easy access
entropy_math_system = EntropyMathSystem()

# Convenience functions
def shannon_entropy(probabilities: List[float]) -> float:
    """Convenience function for Shannon entropy."""
    return entropy_math_system.calculate_shannon_entropy(probabilities)

def conditional_entropy(joint_probs: np.ndarray, marginal_probs: List[float]) -> float:
    """Convenience function for conditional entropy."""
    return entropy_math_system.calculate_conditional_entropy(joint_probs, marginal_probs)

def mutual_information(joint_probs: np.ndarray, marginal_x: List[float], marginal_y: List[float]) -> float:
    """Convenience function for mutual information."""
    return entropy_math_system.calculate_mutual_information(joint_probs, marginal_x, marginal_y)

def market_entropy(price_changes: List[float], window_size: int = 20) -> float:
    """Convenience function for market entropy."""
    return entropy_math_system.calculate_market_entropy(price_changes, window_size)

def zbe_entropy(energy_levels: List[float], temperature: float = 1.0) -> float:
    """Convenience function for ZBE entropy."""
    return entropy_math_system.calculate_zbe_entropy(energy_levels, temperature)

def transition_entropy(sequence: List[str]) -> float:
    """Convenience function for transition entropy."""
    return entropy_math_system.calculate_transition_entropy(sequence)

def hamming_weight(bits: bytes) -> int:
    """Convenience function for Hamming weight."""
    return entropy_math_system.calculate_hamming_weight(bits)

def bit_entropy(digest: bytes) -> float:
    """Convenience function for bit entropy."""
    return entropy_math_system.calculate_bit_entropy(digest)

def normalized_entropy(values: List[float]) -> float:
    """Convenience function for normalized entropy."""
    return entropy_math_system.calculate_normalized_entropy(values)
