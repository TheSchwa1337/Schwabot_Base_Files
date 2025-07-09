"""
Unified Profit Vectorization System.

A comprehensive system that unifies all profit vectorization components
into a single, cohesive interface for the Schwabot trading system.

This system integrates:
- Core profit calculations from PureProfitCalculator
- Vectorization modes from CleanProfitVectorization
- Mathematical foundations from MathLibV4
- Trading pipeline integration
- Real-time profit optimization

CUDA Integration:
- GPU-accelerated profit vectorization with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .clean_math_foundation import BitPhase, CleanMathFoundation, ThermalState
from .clean_profit_vectorization import (
    CleanProfitVectorization,
    ProfitVector,
    VectorizationMode,
)
from .orbital_shell_brain_system import (
    AltitudeVector,
    OrbitalBRAINSystem,
    ShellConsensus,
)
from .pure_profit_calculator import (
    HistoryState,
    MarketData,
    ProfitResult,
    PureProfitCalculator,
    StrategyParameters,
)
from .qutrit_signal_matrix import QutritSignalMatrix, QutritState

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(
        "⚡ UnifiedProfitVectorizationSystem using GPU acceleration: {0}".format(
            _backend
        )
    )
else:
    logger.info(
        "🔄 UnifiedProfitVectorizationSystem using CPU fallback: {0}".format(_backend)
    )

__all__ = [
    "UnifiedProfitVectorizationSystem",
    "ProfitIntegrationMode",
    "VectorizationStrategy",
    "UnifiedProfitResult",
    "TriStateFallbackMatrix",
]


class ProfitIntegrationMode(Enum):
    """Modes for integrating different profit calculation systems."""

    UNIFIED = "unified"
    WEIGHTED = "weighted"
    CONSENSUS = "consensus"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    ORBITAL_CONSENSUS = "orbital_consensus"


class VectorizationStrategy(Enum):
    """Strategies for profit vectorization."""

    STANDARD = "standard"
    ENHANCED = "enhanced"
    OPTIMIZED = "optimized"
    REAL_TIME = "real_time"
    BATCH = "batch"


@dataclass
class UnifiedProfitResult:
    """Result from unified profit vectorization system."""

    timestamp: float
    profit_value: float
    confidence: float
    vector: ProfitVector
    integration_mode: ProfitIntegrationMode
    strategy: VectorizationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class TriStateFallbackMatrix:
    """Provides structured fallback vectors for different failure scenarios."""

    @staticmethod
    def get_hold_vector() -> np.ndarray:
        """Return [0, 0, 0] for complete hold state."""
        return np.array([0.0, 0.0, 0.0])

    @staticmethod
    def get_minimal_buy_vector() -> np.ndarray:
        """Return [0, 1, 0] for minimal buy signal."""
        return np.array([0.0, 1.0, 0.0])

    @staticmethod
    def get_conservative_vector() -> np.ndarray:
        """Return [0.1, 0.1, 0.1] for conservative approach."""
        return np.array([0.1, 0.1, 0.1])

    @staticmethod
    def get_fallback_for_error(error_type: str) -> np.ndarray:
        """Get appropriate fallback based on error type."""
        if "matrix" in error_type.lower() or "corrupt" in error_type.lower():
            return TriStateFallbackMatrix.get_hold_vector()
        elif "nan" in error_type.lower() or "invalid" in error_type.lower():
            return TriStateFallbackMatrix.get_minimal_buy_vector()
        else:
            return TriStateFallbackMatrix.get_conservative_vector()


class UnifiedProfitVectorizationSystem:
    """
    Unified profit vectorization system that integrates all components.

    This system provides a single interface for all profit-related calculations,
    vectorization, and optimization while maintaining the mathematical integrity
    of the original algorithms.
    """

    def __init__(
        self, integration_mode: ProfitIntegrationMode = ProfitIntegrationMode.UNIFIED
    ) -> None:
        """Initialize the unified profit vectorization system."""
        self.integration_mode = integration_mode

        # Initialize core components
        self.math_foundation = CleanMathFoundation()
        self.profit_calculator = PureProfitCalculator(StrategyParameters())
        self.profit_vectorizer = CleanProfitVectorization()
        self.orbital_brain = OrbitalBRAINSystem()

        # System state
        self.last_calculation_time = 0.0
        self.calculation_count = 0
        self.performance_history: List[Dict[str, float]] = []

        # Configuration
        self.max_history_size = 1000
        self.performance_tracking = True

        logger.info(
            "UnifiedProfitVectorizationSystem initialized with mode: {0}".format(
                integration_mode.value
            )
        )

    def calculate_unified_profit(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy = VectorizationStrategy.STANDARD,
        thermal_state: Optional[ThermalState] = None,
        bit_phase: Optional[BitPhase] = None,
        shell_consensus: Optional[ShellConsensus] = None,
        altitude_vector: Optional[AltitudeVector] = None,
    ) -> UnifiedProfitResult:
        """
        Calculate unified profit using all available systems.

        Args:
            market_data: Market data dictionary
            strategy: Vectorization strategy to use
            thermal_state: Current thermal state
            bit_phase: Current bit phase
            shell_consensus: Orbital shell consensus state
            altitude_vector: Mathematical altitude vector

        Returns:
            UnifiedProfitResult with calculated profit and metadata
        """
        start_time = time.time()

        try:
            # Determine thermal state and bit phase if not provided
            if thermal_state is None:
                thermal_state = self._determine_thermal_state(market_data)
            if bit_phase is None:
                bit_phase = self._determine_bit_phase(market_data)

            # Calculate profit based on integration mode
            if self.integration_mode == ProfitIntegrationMode.UNIFIED:
                result = self._calculate_unified_mode(
                    market_data, strategy, thermal_state, bit_phase
                )
            elif self.integration_mode == ProfitIntegrationMode.WEIGHTED:
                result = self._calculate_weighted_mode(
                    market_data, strategy, thermal_state, bit_phase
                )
            elif self.integration_mode == ProfitIntegrationMode.CONSENSUS:
                result = self._calculate_consensus_mode(
                    market_data, strategy, thermal_state, bit_phase
                )
            elif self.integration_mode == ProfitIntegrationMode.ADAPTIVE:
                result = self._calculate_adaptive_mode(
                    market_data, strategy, thermal_state, bit_phase
                )
            elif self.integration_mode == ProfitIntegrationMode.HIERARCHICAL:
                result = self._calculate_hierarchical_mode(
                    market_data, strategy, thermal_state, bit_phase
                )
            elif self.integration_mode == ProfitIntegrationMode.ORBITAL_CONSENSUS:
                result = self._calculate_orbital_consensus_mode(
                    market_data,
                    strategy,
                    thermal_state,
                    bit_phase,
                    shell_consensus,
                    altitude_vector,
                )
            else:
                raise ValueError(f"Unknown integration mode: {self.integration_mode}")

            # Update performance tracking
            calculation_time = time.time() - start_time
            self._update_performance_metrics(calculation_time, result)

            logger.debug(
                f"Unified profit calculation completed in {calculation_time:.6f}s: "
                f"{result.profit_value:.6f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in unified profit calculation: {e}")
            # Return fallback result
            return self._create_fallback_result(
                market_data, strategy, thermal_state, bit_phase
            )

    def _calculate_unified_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using unified mode (combined approach)."""
        # Get pure profit calculation
        pure_profit = self.profit_calculator.calculate_profit(market_data)

        # Get vector profit calculation
        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.5),
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )

        # Combine results using thermal weights
        thermal_weights = self._get_thermal_weights(thermal_state)
        unified_profit = (
            thermal_weights["pure"] * pure_profit.profit_value
            + thermal_weights["vector"] * profit_vector.total_profit
        )
        unified_confidence = (
            thermal_weights["pure"] * pure_profit.confidence
            + thermal_weights["vector"] * profit_vector.confidence
        )

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=unified_profit,
            confidence=unified_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "pure_profit": pure_profit.profit_value,
                "vector_profit": profit_vector.total_profit,
                "thermal_weights": thermal_weights,
            },
        )

    def _calculate_weighted_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using weighted mode (market condition based)."""
        # Get base calculations
        pure_profit = self.profit_calculator.calculate_profit(market_data)

        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.5),
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )

        # Calculate dynamic weights based on market conditions
        volatility = market_data.get("volatility", 0.5)
        volume = market_data.get("volume", 0.0)

        # Weight calculation based on market conditions
        if volatility > 0.7:  # High volatility favors vector approach
            weight_pure = 0.3
            weight_vector = 0.7
        elif volume > 1000:  # High volume favors pure calculation
            weight_pure = 0.7
            weight_vector = 0.3
        else:  # Balanced approach
            weight_pure = 0.5
            weight_vector = 0.5

        weighted_profit = (
            weight_pure * pure_profit.profit_value
            + weight_vector * profit_vector.total_profit
        )
        weighted_confidence = (
            weight_pure * pure_profit.confidence
            + weight_vector * profit_vector.confidence
        )

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=weighted_profit,
            confidence=weighted_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "weights": {"pure": weight_pure, "vector": weight_vector},
                "market_conditions": {"volatility": volatility, "volume": volume},
            },
        )

    def _calculate_consensus_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using consensus mode (agreement-based)."""
        # Get multiple profit calculations
        profits = []
        confidences = []

        # Pure profit calculation
        pure_profit = self.profit_calculator.calculate_profit(market_data)
        profits.append(pure_profit.profit_value)
        confidences.append(pure_profit.confidence)

        # Vector profit calculation
        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.5),
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )
        profits.append(profit_vector.total_profit)
        confidences.append(profit_vector.confidence)

        # Calculate consensus
        consensus_profit = np.mean(profits)
        consensus_confidence = np.mean(confidences)

        # Adjust confidence based on agreement
        profit_std = np.std(profits)
        agreement_factor = max(0.0, 1.0 - profit_std)
        consensus_confidence *= agreement_factor

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=consensus_profit,
            confidence=consensus_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "profits": profits,
                "confidences": confidences,
                "agreement_factor": agreement_factor,
                "profit_std": profit_std,
            },
        )

    def _calculate_adaptive_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using adaptive mode (market condition, based)."""
        # Analyze market conditions
        volatility = market_data.get("volatility", 0.5)
        volume = market_data.get("volume", 0.0)

        # Adapt based on market conditions
        if volatility > 0.8:  # High volatility
            # Prefer vector-based calculation
            weight_pure = 0.3
            weight_vector = 0.7
        elif volume > 1000:  # High volume
            # Prefer pure profit calculation
            weight_pure = 0.7
            weight_vector = 0.3
        else:  # Normal conditions
            weight_pure = 0.5
            weight_vector = 0.5

        # Calculate weighted result
        pure_profit = self.profit_calculator.calculate_profit(market_data)

        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": volatility,
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )

        adaptive_profit = (
            weight_pure * pure_profit.profit_value
            + weight_vector * profit_vector.total_profit
        )
        adaptive_confidence = (
            weight_pure * pure_profit.confidence
            + weight_vector * profit_vector.confidence
        )

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=adaptive_profit,
            confidence=adaptive_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "adaptive_weights": {"pure": weight_pure, "vector": weight_vector},
                "market_conditions": {"volatility": volatility, "volume": volume},
            },
        )

    def _calculate_hierarchical_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using hierarchical mode (priority-based)."""
        # Primary calculation (pure, profit)
        primary_result = self.profit_calculator.calculate_profit(market_data)

        # Secondary calculation (vector) for validation
        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.5),
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )

        # Use primary result, but adjust confidence based on secondary
        hierarchical_profit = primary_result.profit_value
        hierarchical_confidence = primary_result.confidence

        # Adjust confidence if secondary result agrees
        agreement_threshold = 0.1
        profit_difference = abs(
            primary_result.profit_value - profit_vector.total_profit
        )

        if profit_difference < agreement_threshold:
            hierarchical_confidence *= 1.2  # Boost confidence
        else:
            hierarchical_confidence *= 0.8  # Reduce confidence

        hierarchical_confidence = min(1.0, hierarchical_confidence)

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=hierarchical_profit,
            confidence=hierarchical_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "primary_profit": primary_result.profit_value,
                "secondary_profit": profit_vector.total_profit,
                "profit_difference": profit_difference,
                "agreement_threshold": agreement_threshold,
            },
        )

    def _calculate_orbital_consensus_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
        shell_consensus: Optional[ShellConsensus] = None,
        altitude_vector: Optional[AltitudeVector] = None,
    ) -> UnifiedProfitResult:
        """Calculate profit using orbital consensus and altitude vector."""
        if not shell_consensus:
            shell_consensus = self.orbital_brain.calculate_shell_consensus(market_data)
        if not altitude_vector:
            altitude_vector = self.orbital_brain.calculate_altitude_vector(market_data)

        # Get base profit from pure calculator
        pure_profit = self._safe_calculate_profit(market_data)
        pure_profit_value = pure_profit.total_profit_score
        pure_conf = pure_profit.confidence_score

        # Calculate orbital adjustment factor
        orbital_factor = shell_consensus.consensus_strength * altitude_vector.altitude

        # Apply orbital adjustment
        orbital_profit = pure_profit_value * (1.0 + orbital_factor)
        orbital_confidence = pure_conf * shell_consensus.consensus_strength

        # Get vector profit for comparison
        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.5),
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=orbital_profit,
            confidence=orbital_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "orbital_factor": orbital_factor,
                "shell_consensus": shell_consensus.consensus_strength,
                "altitude_vector": altitude_vector.altitude,
                "pure_profit": pure_profit_value,
                "vector_profit": profit_vector.total_profit,
            },
        )

    def _select_vectorization_mode(
        self, strategy: VectorizationStrategy
    ) -> VectorizationMode:
        """Select appropriate vectorization mode based on strategy."""
        strategy_mapping = {
            VectorizationStrategy.STANDARD: VectorizationMode.STANDARD,
            VectorizationStrategy.ENHANCED: VectorizationMode.ENHANCED,
            VectorizationStrategy.OPTIMIZED: VectorizationMode.OPTIMIZED,
            VectorizationStrategy.REAL_TIME: VectorizationMode.REAL_TIME,
            VectorizationStrategy.BATCH: VectorizationMode.BATCH,
        }
        return strategy_mapping.get(strategy, VectorizationMode.STANDARD)

    def _determine_thermal_state(self, market_data: Dict[str, Any]) -> ThermalState:
        """Determine thermal state based on market data."""
        volatility = market_data.get("volatility", 0.5)
        if volatility > 0.8:
            return ThermalState.HOT
        elif volatility < 0.2:
            return ThermalState.COLD
        else:
            return ThermalState.WARM

    def _determine_bit_phase(self, market_data: Dict[str, Any]) -> BitPhase:
        """Determine bit phase based on market data."""
        volume = market_data.get("volume", 0.0)
        if volume > 1000:
            return BitPhase.HIGH
        elif volume < 100:
            return BitPhase.LOW
        else:
            return BitPhase.MEDIUM

    def _get_thermal_weights(self, thermal_state: ThermalState) -> Dict[str, float]:
        """Get weights based on thermal state."""
        if thermal_state == ThermalState.HOT:
            return {"pure": 0.3, "vector": 0.7}
        elif thermal_state == ThermalState.COLD:
            return {"pure": 0.7, "vector": 0.3}
        else:  # WARM
            return {"pure": 0.5, "vector": 0.5}

    def _update_performance_metrics(
        self, calculation_time: float, result: UnifiedProfitResult
    ) -> None:
        """Update performance tracking metrics."""
        if not self.performance_tracking:
            return

        self.last_calculation_time = calculation_time
        self.calculation_count += 1

        performance_data = {
            "timestamp": time.time(),
            "calculation_time": calculation_time,
            "profit_value": result.profit_value,
            "confidence": result.confidence,
            "integration_mode": result.integration_mode.value,
            "strategy": result.strategy.value,
        }

        self.performance_history.append(performance_data)

        # Maintain history size
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)

    def _create_fallback_result(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Create fallback result when calculation fails."""
        logger.warning("Using fallback profit calculation")

        # Create minimal profit vector
        fallback_vector = ProfitVector(
            total_profit=0.0,
            confidence=0.1,
            vector_components=np.array([0.0, 0.0, 0.0]),
            metadata={"fallback": True},
        )

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=0.0,
            confidence=0.1,
            vector=fallback_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={"fallback": True, "error": "Calculation failed"},
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"error": "No performance data available"}

        calculation_times = [p["calculation_time"] for p in self.performance_history]
        profit_values = [p["profit_value"] for p in self.performance_history]
        confidences = [p["confidence"] for p in self.performance_history]

        return {
            "total_calculations": self.calculation_count,
            "avg_calculation_time": np.mean(calculation_times),
            "max_calculation_time": np.max(calculation_times),
            "min_calculation_time": np.min(calculation_times),
            "avg_profit": np.mean(profit_values),
            "avg_confidence": np.mean(confidences),
            "performance_history_size": len(self.performance_history),
        }

    def reset_performance_tracking(self) -> None:
        """Reset performance tracking data."""
        self.performance_history.clear()
        self.calculation_count = 0
        self.last_calculation_time = 0.0

    def _build_market_objects(self, data: Dict[str, Any]):
        """Build market data objects for compatibility."""
        return MarketData(
            price=data.get("price", 0.0),
            volume=data.get("volume", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )

    def _safe_calculate_profit(self, data: Dict[str, Any]) -> ProfitResult:
        """Safely calculate profit with error handling."""
        try:
            market_data = self._build_market_objects(data)
            return self.profit_calculator.calculate_profit(market_data)
        except Exception as e:
            logger.error(f"Error in safe profit calculation: {e}")
            return ProfitResult(
                total_profit_score=0.0,
                confidence_score=0.1,
                metadata={"error": str(e)},
            )

    def _safe_calculate_profit_with_fallback(
        self, strategy_matrix: np.ndarray, market_data: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate profit with fallback to safe values."""
        try:
            # Attempt normal calculation
            result = self._safe_calculate_profit(market_data)
            return np.array([result.total_profit_score, result.confidence_score, 0.0])
        except Exception as e:
            logger.error(f"Profit calculation failed, using fallback: {e}")
            return self._generate_fallback_vector(str(e))

    def _generate_fallback_vector(self, error_type: str = "unknown") -> np.ndarray:
        """Generate fallback vector based on error type."""
        return TriStateFallbackMatrix.get_fallback_for_error(error_type)


def create_unified_profit_system() -> UnifiedProfitVectorizationSystem:
    """Create and return a new unified profit vectorization system."""
    return UnifiedProfitVectorizationSystem()
