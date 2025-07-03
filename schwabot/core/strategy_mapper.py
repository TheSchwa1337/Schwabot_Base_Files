#!/usr/bin/env python3
"""
Strategy Mapper Module
======================

Compliant + layered hash/bit strategy routing for Schwabot v0.5.
Provides strategy selection, routing, and execution coordination.
"""

import hashlib
import json
import logging
import os

# Updated import path to work from project root
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.advanced_tensor_algebra import UnifiedTensorAlgebra

# Import safe_print functions
try:
    from utils.safe_print import error as safe_error
    from utils.safe_print import info as safe_info
    from utils.safe_print import warning as safe_warning
except ImportError:

    def safe_info(message):
        print(f"[INFO] {message}")

    def safe_error(message):
        print(f"[ERROR] {message}")

    def safe_warning(message):
        print(f"[WARNING] {message}")


logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type enumeration."""

    HASH_BASED = "hash_based"
    BIT_BASED = "bit_based"
    HYBRID = "hybrid"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"


class StrategyState(Enum):
    """Strategy state enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    OPTIMIZING = "optimizing"


@dataclass
class StrategyConfig:
    """Strategy configuration."""

    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    risk_level: float  # 0.0 to 1.0
    min_confidence: float
    max_position_size: float
    stop_loss: float
    take_profit: float
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Strategy execution result."""

    strategy_id: str
    timestamp: float
    signal_type: str  # "buy", "sell", "hold"
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashStrategy:
    """Hash-based strategy definition."""

    hash_pattern: str
    strategy_type: StrategyType
    confidence_threshold: float
    execution_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyMapper:
    """
    Strategy Mapper for Schwabot v0.5.

    Provides compliant + layered hash/bit strategy routing
    with advanced pattern matching and execution coordination.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy mapper."""
        self.config = config or self._default_config()

        # Strategy registry
        self.strategies: Dict[str, StrategyConfig] = {}
        self.hash_strategies: Dict[str, HashStrategy] = {}
        self.bit_strategies: Dict[str, Dict[str, Any]] = {}

        # Execution tracking
        self.execution_history: List[StrategyResult] = []
        self.max_history_size = self.config.get("max_history_size", 1000)

        # Performance metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        # State management
        self.current_strategy = None
        self.last_update = time.time()

        # Initialize default strategies
        self._initialize_default_strategies()

        # Initialize Unified Tensor Algebra
        self.tensor_algebra = UnifiedTensorAlgebra()

        logger.info("🗺️ Strategy Mapper initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "max_history_size": 1000,
            "default_confidence_threshold": 0.7,
            "max_position_size": 0.1,  # 10% of portfolio
            "default_stop_loss": 0.2,  # 2%
            "default_take_profit": 0.5,  # 5%
            "hash_pattern_length": 64,
            "bit_pattern_length": 32,
            "strategy_rotation_enabled": True,
            "adaptive_confidence": True,
        }

    def _initialize_default_strategies(self):
        """Initialize default strategies."""
        # Hash-based strategies
        self.add_hash_strategy(
            "conservative_hash",
            "0",
            StrategyType.CONSERVATIVE,
            0.8,
            {"position_size": 0.5, "stop_loss": 0.15, "take_profit": 0.3},
        )

        self.add_hash_strategy(
            "aggressive_hash",
            "1111111111111111111111111111111111111111111111111111111111111111",
            StrategyType.AGGRESSIVE,
            0.6,
            {"position_size": 0.15, "stop_loss": 0.3, "take_profit": 0.8},
        )

        # Bit-based strategies
        self.add_bit_strategy(
            "scalping_bits",
            "10101010101010101010101010101010",
            StrategyType.SCALPING,
            0.7,
            {"position_size": 0.8, "stop_loss": 0.1, "take_profit": 0.2},
        )

        # Hybrid strategies
        self.add_hybrid_strategy(
            "swing_hybrid",
            StrategyType.SWING,
            0.75,
            {"position_size": 0.12, "stop_loss": 0.25, "take_profit": 0.6},
        )

    def add_hash_strategy(
        self,
        strategy_id: str,
        hash_pattern: str,
        strategy_type: StrategyType,
        confidence_threshold: float,
        execution_params: Dict[str, Any],
    ) -> bool:
        """Add a hash-based strategy."""
        try:
            hash_strategy = HashStrategy(
                hash_pattern=hash_pattern,
                strategy_type=strategy_type,
                confidence_threshold=confidence_threshold,
                execution_params=execution_params,
            )

            self.hash_strategies[strategy_id] = hash_strategy

            # Create corresponding strategy config
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                name=f"Hash Strategy {strategy_id}",
                description=f"Hash-based {strategy_type.value} strategy",
                risk_level=self._get_risk_level(strategy_type),
                min_confidence=confidence_threshold,
                max_position_size=execution_params.get("position_size", 0.1),
                stop_loss=execution_params.get("stop_loss", 0.2),
                take_profit=execution_params.get("take_profit", 0.5),
            )

            self.strategies[strategy_id] = strategy_config
            logger.info(f"Added hash strategy: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding hash strategy {strategy_id}: {e}")
            return False

    def add_bit_strategy(
        self,
        strategy_id: str,
        bit_pattern: str,
        strategy_type: StrategyType,
        confidence_threshold: float,
        execution_params: Dict[str, Any],
    ) -> bool:
        """Add a bit-based strategy."""
        try:
            bit_strategy = {
                "bit_pattern": bit_pattern,
                "strategy_type": strategy_type,
                "confidence_threshold": confidence_threshold,
                "execution_params": execution_params,
            }
            self.bit_strategies[strategy_id] = bit_strategy

            # Create corresponding strategy config
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                name=f"Bit Strategy {strategy_id}",
                description=f"Bit-based {strategy_type.value} strategy",
                risk_level=self._get_risk_level(strategy_type),
                min_confidence=confidence_threshold,
                max_position_size=execution_params.get("position_size", 0.1),
                stop_loss=execution_params.get("stop_loss", 0.2),
                take_profit=execution_params.get("take_profit", 0.5),
            )

            self.strategies[strategy_id] = strategy_config
            logger.info(f"Added bit strategy: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding bit strategy {strategy_id}: {e}")
            return False

    def add_hybrid_strategy(
        self,
        strategy_id: str,
        strategy_type: StrategyType,
        confidence_threshold: float,
        execution_params: Dict[str, Any],
    ) -> bool:
        """Add a hybrid strategy."""
        try:
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                name=f"Hybrid Strategy {strategy_id}",
                description=f"Hybrid {strategy_type.value} strategy",
                risk_level=self._get_risk_level(strategy_type),
                min_confidence=confidence_threshold,
                max_position_size=execution_params.get("position_size", 0.1),
                stop_loss=execution_params.get("stop_loss", 0.2),
                take_profit=execution_params.get("take_profit", 0.5),
            )
            self.strategies[strategy_id] = strategy_config
            logger.info(f"Added hybrid strategy: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding hybrid strategy {strategy_id}: {e}")
            return False

    def _get_risk_level(self, strategy_type: StrategyType) -> float:
        """Get risk level for strategy type."""
        risk_levels = {
            StrategyType.CONSERVATIVE: 0.2,
            StrategyType.SCALPING: 0.4,
            StrategyType.SWING: 0.6,
            StrategyType.HYBRID: 0.7,
            StrategyType.AGGRESSIVE: 0.8,
            StrategyType.ARBITRAGE: 0.9,
        }
        return risk_levels.get(strategy_type, 0.5)

    def select_strategy(
        self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]
    ) -> Optional[StrategyConfig]:
        """
        Select the best strategy based on market data and portfolio state.

        Args:
            market_data: Current market data
            portfolio_state: Current portfolio state

        Returns:
            Selected strategy configuration
        """
        try:
            # Generate hash from market data
            market_hash = self._generate_market_hash(market_data)

            # Find matching hash strategy
            for strategy_id, hash_strategy in self.hash_strategies.items():
                if self._hash_matches_pattern(market_hash, hash_strategy.hash_pattern):
                    strategy_config = self.strategies.get(strategy_id)
                    if strategy_config and strategy_config.enabled:
                        return strategy_config

            # Check bit patterns
            bit_sequence = self._generate_bit_sequence(market_data)
            for strategy_id, bit_strategy in self.bit_strategies.items():
                if self._bit_matches_pattern(bit_sequence, bit_strategy["bit_pattern"]):
                    strategy_config = self.strategies.get(strategy_id)
                    if strategy_config and strategy_config.enabled:
                        return strategy_config

            # Fallback to hybrid strategy
            for strategy_id, strategy_config in self.strategies.items():
                if strategy_config.strategy_type == StrategyType.HYBRID and strategy_config.enabled:
                    return strategy_config

            return None

        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return None

    def _generate_market_hash(self, market_data: Dict[str, Any]) -> str:
        """Generate hash from market data."""
        try:
            # Create hashable string from market data
            hash_data = {
                "price": market_data.get("price", 0),
                "volume": market_data.get("volume", 0),
                "timestamp": market_data.get("timestamp", time.time()),
                "volatility": market_data.get("volatility", 0),
            }
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error generating market hash: {e}")
            return "0" * 64

    def _generate_bit_sequence(self, market_data: Dict[str, Any]) -> str:
        """Generate bit sequence from market data."""
        try:
            # Convert market data to binary representation
            price = market_data.get("price", 0)
            volume = market_data.get("volume", 0)

            # Simple bit conversion (can be enhanced)
            price_bits = format(int(price * 1000) % 256, "8b")
            volume_bits = format(int(volume) % 256, "8b")

            # Combine and repeat to match pattern length
            combined = price_bits + volume_bits
            target_length = self.config["bit_pattern_length"]

            while len(combined) < target_length:
                combined += combined

            return combined[:target_length]

        except Exception as e:
            logger.error(f"Error generating bit sequence: {e}")
            return "0" * self.config["bit_pattern_length"]

    def _hash_matches_pattern(self, market_hash: str, pattern: str) -> bool:
        """Check if market hash matches pattern."""
        try:
            # Simple pattern matching (can be enhanced with fuzzy matching)
            pattern_length = len(pattern)
            if len(market_hash) < pattern_length:
                return False

            # Check for exact match or significant similarity
            similarity = sum(1 for i in range(pattern_length) if market_hash[i] == pattern[i])
            similarity_ratio = similarity / pattern_length

            return similarity_ratio > 0.8  # 80% similarity threshold

        except Exception as e:
            logger.error(f"Error checking hash pattern: {e}")
            return False

    def _bit_matches_pattern(self, bit_sequence: str, pattern: str) -> bool:
        """Check if bit sequence matches pattern."""
        try:
            if len(bit_sequence) != len(pattern):
                return False

            # Check for exact match or significant similarity
            similarity = sum(1 for i in range(len(pattern)) if bit_sequence[i] == pattern[i])
            similarity_ratio = similarity / len(pattern)

            return similarity_ratio > 0.7  # 70% similarity threshold

        except Exception as e:
            logger.error(f"Error checking bit pattern: {e}")
            return False

    def execute_strategy(
        self,
        strategy_config: StrategyConfig,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> Optional[StrategyResult]:
        """
        Execute a strategy and return the result.

        Args:
            strategy_config: Strategy configuration
            market_data: Current market data
            portfolio_state: Current portfolio state

        Returns:
            Strategy execution result
        """
        try:
            # Calculate signal based on strategy type
            signal_type, confidence = self._calculate_signal(
                strategy_config, market_data, portfolio_state
            )

            if confidence < strategy_config.min_confidence:
                return None

            # Calculate position size
            position_size = self._calculate_position_size(
                strategy_config, portfolio_state, confidence
            )

            # Create strategy result
            result = StrategyResult(
                strategy_id=strategy_config.strategy_id,
                timestamp=time.time(),
                signal_type=signal_type,
                confidence=confidence,
                position_size=position_size,
                stop_loss=strategy_config.stop_loss,
                take_profit=strategy_config.take_profit,
                metadata={
                    "strategy_type": strategy_config.strategy_type.value,
                    "risk_level": strategy_config.risk_level,
                },
            )

            # Update execution history
            self.execution_history.append(result)
            if len(self.execution_history) > self.max_history_size:
                self.execution_history.pop(0)

            self.total_executions += 1
            self.current_strategy = strategy_config.strategy_id
            self.last_update = time.time()

            logger.info(
                f"Executed strategy {strategy_config.strategy_id}: {signal_type} (confidence: {confidence:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error executing strategy {strategy_config.strategy_id}: {e}")
            self.failed_executions += 1
            return None

    def _calculate_signal(
        self,
        strategy_config: StrategyConfig,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> Tuple[str, float]:
        """Calculate trading signal based on strategy."""
        try:
            # NEW: Execute dualistic profit vectorization for pure mathematical decision-making
            consensus_result = self.tensor_algebra.execute_dualistic_profit_vectorization(
                market_data
            )

            # If mathematical consensus produces a strong signal, use it directly
            if consensus_result.consensus_confidence >= 0.7:  # High confidence threshold
                # Use pure mathematical decision
                signal_type = consensus_result.execution_signal
                if signal_type == "long":
                    signal_type = "buy"
                elif signal_type == "short":
                    signal_type = "sell"
                else:
                    signal_type = "hold"

                confidence = consensus_result.consensus_confidence

                safe_info(
                    f"🧮 Using pure mathematical decision: {signal_type} (confidence: {confidence:.3f})"
                )
                return signal_type, confidence

            # FALLBACK: Use traditional tensor algebra with strategy logic
            # Extract data for heatmap
            price_data = market_data.get("price", 0)
            volume_data = market_data.get("volume", 0)
            volatility_data = market_data.get("volatility", 0)
            liquidity_heatmap = market_data.get("liquidity_heatmap", None)

            if liquidity_heatmap is None or liquidity_heatmap.size == 0:
                # Create a dummy heatmap if not provided, for demonstration
                # In a real scenario, this would come from a data feed
                liquidity_heatmap = np.array(
                    [
                        [price_data, volume_data],
                        [volatility_data, price_data * volume_data],
                    ]
                )

            # Generate trade tensor from liquidity
            trade_tensor = self.tensor_algebra.generate_tensor_from_liquidity(liquidity_heatmap)

            # Contract strategy tensor to 1D actionable trade vector
            actionable_vector = self.tensor_algebra.contract_strategy_tensor(trade_tensor)

            # Base confidence calculation
            base_confidence = 0.5
            signal_type = "hold"

            if actionable_vector.size > 0:
                # Use the actionable vector to influence signal and confidence
                # For simplicity, if the single element is positive, it's a buy signal
                # And its magnitude can influence confidence
                trade_signal_value = actionable_vector[0]

                if trade_signal_value > 0.05:  # Example threshold for a buy signal
                    signal_type = "buy"
                    base_confidence += min(
                        0.4, trade_signal_value * 0.5
                    )  # Increase confidence based on magnitude
                elif trade_signal_value < -0.05:  # Example threshold for a sell signal
                    signal_type = "sell"
                    base_confidence += min(0.4, abs(trade_signal_value) * 0.5)
                else:
                    signal_type = "hold"

            # Combine with mathematical consensus as influence factor
            if consensus_result.consensus_confidence > 0.3:  # Medium confidence
                consensus_influence = consensus_result.consensus_confidence * 0.2  # 20% influence
                if consensus_result.execution_signal == "long" and signal_type != "sell":
                    base_confidence += consensus_influence
                elif consensus_result.execution_signal == "short" and signal_type != "buy":
                    base_confidence += consensus_influence

            # Adjust based on strategy type (original logic combined with tensor output)
            if strategy_config.strategy_type == StrategyType.CONSERVATIVE:
                if volatility_data < 0.2:  # Low volatility
                    base_confidence += 0.1

            elif strategy_config.strategy_type == StrategyType.AGGRESSIVE:
                if volatility_data > 0.5:  # High volatility
                    base_confidence += 0.2

            # Normalize confidence
            confidence = min(base_confidence, 1.0)

            safe_info(
                f"📈 Hybrid decision: {signal_type} (tensor: {confidence:.3f}, consensus influence: {consensus_result.consensus_confidence:.3f})"
            )
            return signal_type, confidence

        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return "hold", 0.0

    def _calculate_position_size(
        self,
        strategy_config: StrategyConfig,
        portfolio_state: Dict[str, Any],
        confidence: float,
    ) -> float:
        """Calculate position size based on strategy and confidence."""
        try:
            # Base position size from strategy
            base_size = (
                portfolio_state.get("available_funds", 0) * strategy_config.max_position_size
            )

            # Adjust based on confidence
            adjusted_size = base_size * confidence

            return adjusted_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy mapping."""
        return {
            "total_strategies": len(self.strategies),
            "hash_strategies": len(self.hash_strategies),
            "bit_strategies": len(self.bit_strategies),
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "current_strategy": self.current_strategy,
            "last_update": self.last_update,
            "execution_history_size": len(self.execution_history),
        }

    def get_recent_executions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent strategy executions."""
        recent_executions = self.execution_history[-count:]
        return [
            {
                "strategy_id": result.strategy_id,
                "timestamp": result.timestamp,
                "signal_type": result.signal_type,
                "confidence": result.confidence,
                "position_size": result.position_size,
                "stop_loss": result.stop_loss,
                "take_profit": result.take_profit,
                "metadata": result.metadata,
            }
            for result in recent_executions
        ]

    def get_overall_performance(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        success_rate = (
            (self.successful_executions / self.total_executions)
            if self.total_executions > 0
            else 0.0
        )
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "avg_confidence": (
                np.mean([r.confidence for r in self.execution_history])
                if self.execution_history
                else 0.0
            ),
        }

    def get_strategy_details(self, strategy_id: str) -> Optional[StrategyConfig]:
        """Get details of a specific strategy."""
        return self.strategies.get(strategy_id)

    def get_mathematical_decision_analysis(self) -> Dict[str, Any]:
        """Get analysis of mathematical decision-making vs traditional strategy logic."""
        try:
            if not hasattr(self.tensor_algebra, "consensus_history"):
                return {"error": "No consensus history available"}

            consensus_history = self.tensor_algebra.consensus_history
            if not consensus_history:
                return {"analysis": "No consensus decisions recorded yet"}

            # Analyze last 10 consensus decisions
            recent_consensus = consensus_history[-10:]

            high_confidence_decisions = [
                r for r in recent_consensus if r.consensus_confidence >= 0.7
            ]
            medium_confidence_decisions = [
                r for r in recent_consensus if 0.3 <= r.consensus_confidence < 0.7
            ]
            low_confidence_decisions = [r for r in recent_consensus if r.consensus_confidence < 0.3]

            signal_distribution = {}
            for result in recent_consensus:
                signal = result.execution_signal
                signal_distribution[signal] = signal_distribution.get(signal, 0) + 1

            # Calculate mathematical certainty trends
            certainties = [
                r.mathematical_proof.get("mathematical_certainty", 0) for r in recent_consensus
            ]
            avg_certainty = np.mean(certainties) if certainties else 0

            return {
                "total_consensus_decisions": len(consensus_history),
                "recent_decisions_analyzed": len(recent_consensus),
                "high_confidence_decisions": len(high_confidence_decisions),
                "medium_confidence_decisions": len(medium_confidence_decisions),
                "low_confidence_decisions": len(low_confidence_decisions),
                "signal_distribution": signal_distribution,
                "average_mathematical_certainty": avg_certainty,
                "pure_math_decision_rate": (
                    len(high_confidence_decisions) / len(recent_consensus)
                    if recent_consensus
                    else 0
                ),
                "last_consensus": (
                    {
                        "signal": recent_consensus[-1].execution_signal,
                        "confidence": recent_consensus[-1].consensus_confidence,
                        "mathematical_certainty": recent_consensus[-1].mathematical_proof.get(
                            "mathematical_certainty", 0
                        ),
                        "flip_transitions": len(recent_consensus[-1].flip_transitions),
                    }
                    if recent_consensus
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing mathematical decisions: {e}")
            return {"error": str(e)}

    def get_dualistic_state_summary(self) -> Dict[str, Any]:
        """Get summary of dualistic state resolution in bit-form tensor flip matrices."""
        try:
            if not hasattr(self.tensor_algebra, "active_flip_matrices"):
                return {"error": "No active flip matrices available"}

            active_matrices = self.tensor_algebra.active_flip_matrices
            if not active_matrices:
                return {"summary": "No active flip matrices"}

            # Analyze flip states
            state_distribution = {}
            for matrix in active_matrices:
                state = matrix.flip_state.value
                state_distribution[state] = state_distribution.get(state, 0) + 1

            # Calculate consensus metrics
            total_weight = sum(m.consensus_weight for m in active_matrices)
            avg_confidence = np.mean([m.confidence_score for m in active_matrices])

            # Analyze profit vectors
            profit_vectors = [m.profit_vector for m in active_matrices]
            if profit_vectors:
                avg_vector = np.mean(profit_vectors, axis=0)
                vector_std = np.std(profit_vectors, axis=0)
            else:
                avg_vector = np.array([0, 0, 0])
                vector_std = np.array([0, 0, 0])

            return {
                "active_matrices_count": len(active_matrices),
                "flip_state_distribution": state_distribution,
                "total_consensus_weight": total_weight,
                "average_matrix_confidence": avg_confidence,
                "average_profit_vector": {
                    "price_direction": float(avg_vector[0]),
                    "time_direction": float(avg_vector[1]),
                    "risk_direction": float(avg_vector[2]),
                },
                "profit_vector_volatility": {
                    "price_std": float(vector_std[0]),
                    "time_std": float(vector_std[1]),
                    "risk_std": float(vector_std[2]),
                },
                "mathematical_coherence": float(
                    avg_confidence * (1.0 / (1.0 + np.mean(vector_std)))
                ),
            }

        except Exception as e:
            logger.error(f"Error summarizing dualistic states: {e}")
            return {"error": str(e)}


def demo_strategy_mapper():
    """Demonstrates the functionality of the StrategyMapper with dualistic profit vectorization."""
    logging.basicConfig(level=logging.INFO)
    mapper = StrategyMapper()

    print("🤖 Schwabot Strategy Mapper Demo - Dualistic Profit Vectorization")
    print("=" * 80)

    # Simulate market data with enhanced parameters for bit-form matrices
    market_data = {
        "price": 50000,
        "previous_price": 49800,  # For flip state determination
        "volume": 1000,
        "timestamp": time.time(),
        "volatility": 0.25,  # Medium volatility for interesting dynamics
        "liquidity_depth": 5000,
        "liquidity_heatmap": np.random.rand(5, 5),  # Enhanced heatmap
    }
    portfolio_state = {"available_funds": 10000, "current_positions": {"BTC": 0.05}}

    print("\n📊 Market Data:")
    print(
        f"  Price: ${market_data['price']:,.2f} (Previous: ${market_data['previous_price']:,.2f})"
    )
    print(f"  Volume: {market_data['volume']:,}")
    print(f"  Volatility: {market_data['volatility']:.1%}")
    print(f"  Liquidity Depth: {market_data['liquidity_depth']:,}")

    # Test pure mathematical decision-making
    print("\n🧮 Testing Pure Mathematical Decision-Making...")
    consensus_result = mapper.tensor_algebra.execute_dualistic_profit_vectorization(market_data)
    print(f"  Mathematical Signal: {consensus_result.execution_signal}")
    print(f"  Consensus Confidence: {consensus_result.consensus_confidence:.3f}")
    print(
        f"  Mathematical Certainty: {consensus_result.mathematical_proof.get('mathematical_certainty', 0):.3f}"
    )
    print(
        f"  Profit Vector: [{consensus_result.final_profit_vector[0]:.3f}, {consensus_result.final_profit_vector[1]:.3f}, {consensus_result.final_profit_vector[2]:.3f}]"
    )

    # Test dualistic state analysis
    print("\n⚡ Analyzing Dualistic States...")
    dualistic_summary = mapper.get_dualistic_state_summary()
    if "error" not in dualistic_summary:
        print(f"  Active Matrices: {dualistic_summary['active_matrices_count']}")
        print(f"  Flip State Distribution: {dualistic_summary['flip_state_distribution']}")
        print(f"  Mathematical Coherence: {dualistic_summary['mathematical_coherence']:.3f}")
        print("  Average Profit Vector:")
        avg_vector = dualistic_summary["average_profit_vector"]
        print(f"    Price Direction: {avg_vector['price_direction']:.3f}")
        print(f"    Time Direction: {avg_vector['time_direction']:.3f}")
        print(f"    Risk Direction: {avg_vector['risk_direction']:.3f}")

    # Select and execute strategy (will use hybrid approach)
    print("\n🎯 Strategy Selection and Execution...")
    selected_strategy = mapper.select_strategy(market_data, portfolio_state)

    if selected_strategy:
        print(
            f"  Selected Strategy: {selected_strategy.name} ({selected_strategy.strategy_type.value})"
        )
        result = mapper.execute_strategy(selected_strategy, market_data, portfolio_state)

        if result:
            print(f"  Final Signal: {result.signal_type}")
            print(f"  Final Confidence: {result.confidence:.3f}")
            print(f"  Position Size: ${result.position_size:.2f}")
            print(f"  Stop Loss: {result.stop_loss:.1%}")
            print(f"  Take Profit: {result.take_profit:.1%}")
        else:
            print("  Strategy execution failed or no signal generated.")
    else:
        print("  No suitable strategy found.")

    # Analyze mathematical decision-making performance
    print("\n📈 Mathematical Decision Analysis...")
    math_analysis = mapper.get_mathematical_decision_analysis()
    if "error" not in math_analysis and "analysis" not in math_analysis:
        print(f"  Total Consensus Decisions: {math_analysis['total_consensus_decisions']}")
        print(f"  Pure Math Decision Rate: {math_analysis['pure_math_decision_rate']:.1%}")
        print(
            f"  Average Mathematical Certainty: {math_analysis['average_mathematical_certainty']:.3f}"
        )
        if math_analysis["last_consensus"]:
            last = math_analysis["last_consensus"]
            print(f"  Last Decision: {last['signal']} (confidence: {last['confidence']:.3f})")

    # Get comprehensive summary
    print("\n📋 Strategy Summary:")
    summary = mapper.get_strategy_summary()
    print(f"  Total Strategies: {summary['total_strategies']}")
    print(f"  Hash Strategies: {summary['hash_strategies']}")
    print(f"  Bit Strategies: {summary['bit_strategies']}")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Current Strategy: {summary['current_strategy']}")

    # Show recent executions
    print("\n📝 Recent Executions:")
    recent = mapper.get_recent_executions(3)
    for i, execution in enumerate(recent, 1):
        print(
            f"  {i}. {execution['signal_type']} - Confidence: {execution['confidence']:.3f}, Size: ${execution['position_size']:.2f}"
        )

    print("\n" + "=" * 80)
    print("🎉 Demo Complete! Schwabot's mathematical intelligence is operational.")
    print("💡 Key Innovation: Bit-form tensor flip matrices enable pure mathematical")
    print("   decision-making through dualistic state resolution, replacing traditional")
    print("   AI heuristics with mathematical consensus mechanisms.")


if __name__ == "__main__":
    demo_strategy_mapper()
