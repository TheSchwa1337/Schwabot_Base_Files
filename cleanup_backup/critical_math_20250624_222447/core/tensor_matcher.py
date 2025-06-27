from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Tensor Matcher - Schwabot UROS v1.0
==================================

Connects phase \\u2192 strategy scoring with tensor operations and matrix basket routing.
Implements the core mathematical functions for phase-weighted matrix calculations
and tensor score resolution for trade priority determination.

Core Mathematical Functions:
- Phase weight matrix: phase_weight = (bit_score * entropy) / (len(bits) + \\u03b5)
- Tensor score calculation: T = (current - entry) / entry * (phase + 1)
- Strategy mapping: strategy = f(bit_phase, entropy, volatility)
- Matrix basket routing: basket = hash_to_basket(hash, bit_phase)
"""

import hashlib
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Trading strategy types based on tensor matching."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    QUANTUM = "quantum"


class BitPhase(Enum):
    """Bit resolution phases for tensor matching."""
    FOUR_BIT = 4
    EIGHT_BIT = 8
    FORTY_TWO_BIT = 42


@dataclass
class TensorMatchResult:
    """Result of tensor matching operation."""
    phase_value: int
    bit_phase: BitPhase
    strategy_type: StrategyType
    tensor_score: float
    phase_weight: float
    basket_id: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseWeightMatrix:
    """Phase weight matrix calculation result."""
    bit_pattern: List[int]
    entropy: float
    phase_weight: float
    bit_score: float
    pattern_length: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class TensorMatcher:
    """
    Tensor Matcher for phase \\u2192 strategy scoring and matrix basket routing.

    Mathematical Foundation:
    - Phase Weight Matrix: phase_weight = (bit_score * entropy) / (len(bits) + \\u03b5)
    - Tensor Scoring: T = (current - entry) / entry * (phase + 1)
    - Strategy Mapping: strategy = f(bit_phase, entropy, volatility)
    - Matrix Basket Routing: basket = hash_to_basket(hash, bit_phase)
    """

    def __init__(self, config_path: str = "./config/tensor_matcher_config.json"):
        self.config_path = config_path

        # Strategy mappings
        self.strategy_mappings: Dict[str, Dict[str, Any]] = {
            "conservative": {
                "risk_tolerance": 0.1,
                "position_multiplier": 0.5,
                "bit_phase_range": [0, 15],
                "entropy_threshold": 3.0
            },
            "balanced": {
                "risk_tolerance": 0.3,
                "position_multiplier": 1.0,
                "bit_phase_range": [16, 255],
                "entropy_threshold": 5.0
            },
            "aggressive": {
                "risk_tolerance": 0.5,
                "position_multiplier": 1.5,
                "bit_phase_range": [256, 1023],
                "entropy_threshold": 7.0
            },
            "quantum": {
                "risk_tolerance": 0.7,
                "position_multiplier": 2.0,
                "bit_phase_range": [1024, 4398046511104],
                "entropy_threshold": 8.0
            }
        }

        # Performance tracking
        self.match_history: List[TensorMatchResult] = []
        self.phase_weight_history: List[PhaseWeightMatrix] = []

        # Integration with other components
        self.bit_phase_engine = None
        self.matrix_mapper = None
        self.profit_allocator = None

        # Load configuration
        self._load_configuration()
        logger.info("Tensor Matcher initialized")

    def _load_configuration(self) -> None:
        """Load tensor matcher configuration."""
        try:
            # Default configuration
            config = {
                "phase_weight": {
                    "epsilon": 1e-6,
                    "min_weight": 0.01,
                    "max_weight": 10.0
                },
                "tensor_scoring": {
                    "min_score": -1.0,
                    "max_score": 1.0,
                    "precision": 4
                },
                "strategy_mapping": {
                    "conservative": {"risk_tolerance": 0.1, "position_multiplier": 0.5},
                    "balanced": {"risk_tolerance": 0.3, "position_multiplier": 1.0},
                    "aggressive": {"risk_tolerance": 0.5, "position_multiplier": 1.5},
                    "quantum": {"risk_tolerance": 0.7, "position_multiplier": 2.0}
                }
            }

            logger.info("Tensor matcher configuration loaded")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def phase_weight_matrix(self, bit_pattern: List[int], entropy: float) -> float:
        """
        Calculate phase weight matrix score.

        Mathematical Formula:
        phase_weight = (bit_score * entropy) / (len(bits) + \\u03b5)

        Parameters:
        -----------
        bit_pattern : List[int]
            List of bit values
        entropy : float
            Entropy value

        Returns:
        --------
        float
            Phase weight score
        """
        try:
            if not bit_pattern:
                logger.warning("Empty bit pattern, returning 0")
                return 0.0

            # Calculate bit score
            bit_score = sum(bit_pattern)

            # Calculate phase weight using the formula: (sum(bits) * entropy) / (len(bits) + \\u03b5)
            epsilon = 1e-6  # Small constant to avoid division by zero
            phase_weight = (bit_score * entropy) / (len(bit_pattern) + epsilon)

            # Normalize to reasonable range
            phase_weight = unified_math.max(0.01, unified_math.min(10.0, phase_weight))

            # Create phase weight matrix result
            result = PhaseWeightMatrix(
                bit_pattern=bit_pattern.copy(),
                entropy=entropy,
                phase_weight=phase_weight,
                bit_score=bit_score,
                pattern_length=len(bit_pattern),
                timestamp=datetime.now()
            )

            # Store in history
            self.phase_weight_history.append(result)

            logger.debug(f"Phase weight: {phase_weight:.4f} (bit_score: {bit_score}, entropy: {entropy:.4f})")
            return phase_weight

        except Exception as e:
            logger.error(f"Error calculating phase weight matrix: {e}")
            return 0.0

    def tensor_score(self, entry_price: float, current_price: float, phase: int) -> float:
        """
        Calculate tensor score for trade priority.

        Mathematical Formula:
        T = (current - entry) / entry * (phase + 1)

        Parameters:
        -----------
        entry_price : float
            Entry price for the trade
        current_price : float
            Current market price
        phase : int
            Bit phase value

        Returns:
        --------
        float
            Tensor score for trade priority
        """
        try:
            if entry_price <= 0:
                logger.warning("Invalid entry price, returning 0")
                return 0.0

            # Calculate price delta
            delta = (current_price - entry_price) / entry_price

            # Apply phase multiplier
            tensor_score = delta * (phase + 1)

            # Normalize to reasonable range
            tensor_score = max(-1.0, unified_math.min(1.0, tensor_score))

            # Round to 4 decimal places
            result = round(tensor_score, 4)

            logger.debug(f"Tensor score: {result} (delta: {delta:.4f}, phase: {phase})")
            return result

        except Exception as e:
            logger.error(f"Error calculating tensor score: {e}")
            return 0.0

    def map_phase_to_strategy(self, phase_value: int, entropy: float, volatility: float) -> StrategyType:
        """
        Map bit phase to trading strategy.

        Parameters:
        -----------
        phase_value : int
            Bit phase value
        entropy : float
            Market entropy level
        volatility : float
            Market volatility

        Returns:
        --------
        StrategyType
            Mapped strategy type
        """
        try:
            # Determine bit phase type
            if phase_value < 16:
                bit_phase = BitPhase.FOUR_BIT
            elif phase_value < 256:
                bit_phase = BitPhase.EIGHT_BIT
            else:
                bit_phase = BitPhase.FORTY_TWO_BIT

            # Map to strategy based on bit phase and market conditions
            if bit_phase == BitPhase.FOUR_BIT:
                strategy_type = StrategyType.CONSERVATIVE
            elif bit_phase == BitPhase.EIGHT_BIT:
                if entropy > 6.0 or volatility > 0.05:
                    strategy_type = StrategyType.AGGRESSIVE
                else:
                    strategy_type = StrategyType.BALANCED
            else:  # 42-bit
                strategy_type = StrategyType.QUANTUM

            logger.debug(f"Mapped phase {phase_value} to strategy: {strategy_type.value}")
            return strategy_type

        except Exception as e:
            logger.error(f"Error mapping phase to strategy: {e}")
            return StrategyType.BALANCED

    def hash_to_basket(self, hash_value: str, bit_phase: BitPhase) -> str:
        """
        Map hash to basket ID for profit allocation.

        Parameters:
        -----------
        hash_value : str
            Hash value to map
        bit_phase : BitPhase
            Bit resolution phase

        Returns:
        --------
        str
            Basket ID for profit allocation
        """
        try:
            # Use hash to generate basket ID
            if bit_phase == BitPhase.FOUR_BIT:
                basket_id = f"basket_4bit_{int(hash_value[0:1], 16) % 16}"
            elif bit_phase == BitPhase.EIGHT_BIT:
                basket_id = f"basket_8bit_{int(hash_value[0:2], 16) % 256}"
            else:  # 42-bit
                basket_id = f"basket_42bit_{int(hash_value[0:11], 16) % 1024}"

            return basket_id

        except Exception as e:
            logger.error(f"Error mapping hash to basket: {e}")
            return f"basket_fallback_{int(time.time())}"

    def match_tensor(self, hash_value: str, entry_price: float, current_price: float,
                     market_data: Dict[str, Any]) -> TensorMatchResult:
        """
        Perform complete tensor matching operation.

        Parameters:
        -----------
        hash_value : str
            Hash value to process
        entry_price : float
            Entry price for the trade
        current_price : float
            Current market price
        market_data : Dict[str, Any]
            Market data including entropy, volatility, etc.

        Returns:
        --------
        TensorMatchResult
            Complete tensor match result
        """
        try:
            # Determine bit phase
            if self.bit_phase_engine:
                phase_value = self.bit_phase_engine.resolve_bit_phase(hash_value, "auto")
            else:
                # Fallback bit phase determination
                first_byte = int(hash_value[0:2], 16)
                if first_byte < 85:
                    phase_value = int(hash_value[0:1], 16) % 16
                elif first_byte < 170:
                    phase_value = int(hash_value[0:2], 16) % 256
                else:
                    phase_value = int(hash_value[0:11], 16) % 4398046511104

            # Determine bit phase type
            if phase_value < 16:
                bit_phase = BitPhase.FOUR_BIT
            elif phase_value < 256:
                bit_phase = BitPhase.EIGHT_BIT
            else:
                bit_phase = BitPhase.FORTY_TWO_BIT

            # Calculate phase weight matrix
            bit_pattern = [int(c, 16) for c in hash_value[:8]]  # First 8 hex chars
            entropy = market_data.get('entropy_level', 4.0)
            phase_weight = self.phase_weight_matrix(bit_pattern, entropy)

            # Calculate tensor score
            tensor_score = self.tensor_score(entry_price, current_price, phase_value)

            # Map to strategy
            volatility = market_data.get('volatility', 0.02)
            strategy_type = self.map_phase_to_strategy(phase_value, entropy, volatility)

            # Map to basket
            basket_id = self.hash_to_basket(hash_value, bit_phase)

            # Calculate confidence
            confidence = self._calculate_confidence(phase_weight, tensor_score, entropy)

            # Create result
            result = TensorMatchResult(
                phase_value=phase_value,
                bit_phase=bit_phase,
                strategy_type=strategy_type,
                tensor_score=tensor_score,
                phase_weight=phase_weight,
                basket_id=basket_id,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'hash_value': hash_value,
                    'market_data': market_data,
                    'bit_pattern': bit_pattern
                }
            )

            # Store in history
            self.match_history.append(result)

            logger.info(f"Tensor match: phase={phase_value}, strategy={strategy_type.value}, tensor={tensor_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error matching tensor: {e}")
            return None

    def _calculate_confidence(self, phase_weight: float, tensor_score: float, entropy: float) -> float:
        """Calculate confidence score for tensor match."""
        try:
            # Base confidence on phase weight stability
            weight_confidence = unified_math.min(phase_weight / 5.0, 1.0)

            # Tensor score confidence (absolute value)
            tensor_confidence = unified_math.min(unified_math.abs(tensor_score), 1.0)

            # Entropy confidence (normalized)
            entropy_confidence = unified_math.min(entropy / 8.0, 1.0)

            # Weighted combination
            confidence = (weight_confidence * 0.4 + tensor_confidence * 0.3 + entropy_confidence * 0.3)

            return round(confidence, 4)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def set_bit_phase_engine(self, bit_engine) -> None:
        """Set bit phase engine for integration."""
        self.bit_phase_engine = bit_engine
        logger.info("Bit phase engine integrated with tensor matcher")

    def set_matrix_mapper(self, matrix_mapper) -> None:
        """Set matrix mapper for integration."""
        self.matrix_mapper = matrix_mapper
        logger.info("Matrix mapper integrated with tensor matcher")

    def set_profit_allocator(self, profit_allocator) -> None:
        """Set profit allocator for integration."""
        self.profit_allocator = profit_allocator
        logger.info("Profit allocator integrated with tensor matcher")

    def get_match_statistics(self) -> Dict[str, Any]:
        """Get tensor match statistics."""
        try:
            if not self.match_history:
                return {'error': 'No match history available'}

            # Calculate statistics
            total_matches = len(self.match_history)
            strategy_counts = {strategy.value: 0 for strategy in StrategyType}
            bit_phase_counts = {phase.value: 0 for phase in BitPhase}
            tensor_scores = [r.tensor_score for r in self.match_history]
            phase_weights = [r.phase_weight for r in self.match_history]

            for result in self.match_history:
                strategy_counts[result.strategy_type.value] += 1
                bit_phase_counts[result.bit_phase.value] += 1

            return {
                'total_matches': total_matches,
                'strategy_distribution': strategy_counts,
                'bit_phase_distribution': bit_phase_counts,
                'average_tensor_score': unified_math.unified_math.mean(tensor_scores) if tensor_scores else 0.0,
                'tensor_score_std': unified_math.unified_math.std(tensor_scores) if tensor_scores else 0.0,
                'average_phase_weight': unified_math.unified_math.mean(phase_weights) if phase_weights else 0.0,
                'phase_weight_std': unified_math.unified_math.std(phase_weights) if phase_weights else 0.0
            }

        except Exception as e:
            logger.error(f"Error getting match statistics: {e}")
            return {'error': str(e)}

    def export_match_data(self, output_path: str = "tensor_match_data.json") -> None:
        """Export tensor match data to file."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'total_matches': len(self.match_history),
                'total_phase_weights': len(self.phase_weight_history),
                'recent_matches': [
                    {
                        'phase_value': result.phase_value,
                        'bit_phase': result.bit_phase.value,
                        'strategy_type': result.strategy_type.value,
                        'tensor_score': result.tensor_score,
                        'phase_weight': result.phase_weight,
                        'basket_id': result.basket_id,
                        'confidence': result.confidence,
                        'timestamp': result.timestamp.isoformat()
                    }
                    for result in self.match_history[-50:]  # Last 50 matches
                ],
                'recent_phase_weights': [
                    {
                        'bit_pattern': matrix.bit_pattern,
                        'entropy': matrix.entropy,
                        'phase_weight': matrix.phase_weight,
                        'bit_score': matrix.bit_score,
                        'timestamp': matrix.timestamp.isoformat()
                    }
                    for matrix in self.phase_weight_history[-50:]  # Last 50 phase weights
                ]
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Tensor match data exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting match data: {e}")


if __name__ == "__main__":
    # Test tensor matcher
    matcher = TensorMatcher()

    # Test phase weight matrix
    bit_pattern = [1, 0, 1, 1, 0, 1, 0, 1]
    entropy = 4.5
    phase_weight = matcher.phase_weight_matrix(bit_pattern, entropy)
    safe_print(f"Phase Weight: {phase_weight:.4f}")

    # Test tensor score
    tensor_score = matcher.tensor_score(45000.0, 46000.0, 8)
    safe_print(f"Tensor Score: {tensor_score}")

    # Test complete tensor matching
    test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
    market_data = {
        'entropy_level': 4.5,
        'volatility': 0.03,
        'market_heat': 0.6
    }

    result = matcher.match_tensor(test_hash, 45000.0, 46000.0, market_data)
    if result:
        safe_print(f"Tensor Match Result:")
        safe_print(f"  Phase: {result.phase_value}")
        safe_print(f"  Strategy: {result.strategy_type.value}")
        safe_print(f"  Tensor Score: {result.tensor_score:.4f}")
        safe_print(f"  Basket ID: {result.basket_id}")
        safe_print(f"  Confidence: {result.confidence:.4f}")

    # Get statistics
    stats = matcher.get_match_statistics()
    safe_print(f"Match Statistics: {stats}")

"""