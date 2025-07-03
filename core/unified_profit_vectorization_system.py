#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Profit Vectorization System

A comprehensive system that unifies all profit vectorization components
into a single, cohesive interface for the Schwabot trading system.

This system integrates:
- Core profit calculations from PureProfitCalculator
- Vectorization modes from CleanProfitVectorization
- Mathematical foundations from MathLibV4
- Trading pipeline integration
- Real-time profit optimization
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clean_math_foundation import BitPhase, CleanMathFoundation, ThermalState
from .clean_profit_vectorization import CleanProfitVectorization, ProfitVector, VectorizationMode
from .pure_profit_calculator import PureProfitCalculator, StrategyParameters

logger = logging.getLogger(__name__)

__all__ = [
    "UnifiedProfitVectorizationSystem",
    "ProfitIntegrationMode",
    "VectorizationStrategy",
    "UnifiedProfitResult",
]


class ProfitIntegrationMode(Enum):
    """Modes for integrating different profit calculation systems."""

    UNIFIED = "unified"
    WEIGHTED = "weighted"
    CONSENSUS = "consensus"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


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


class UnifiedProfitVectorizationSystem:
    """
    Unified profit vectorization system that integrates all components.

    This system provides a single interface for all profit-related calculations,
    vectorization, and optimization while maintaining the mathematical integrity
    of the original algorithms.
    """

    def __init__(self, integration_mode: ProfitIntegrationMode = ProfitIntegrationMode.UNIFIED):
        """Initialize the unified profit vectorization system."""
        self.integration_mode = integration_mode

        # Initialize core components
        self.math_foundation = CleanMathFoundation()
        self.profit_calculator = PureProfitCalculator(StrategyParameters())
        self.profit_vectorizer = CleanProfitVectorization()

        # System state
        self.last_calculation_time = 0.0
        self.calculation_count = 0
        self.performance_history: List[Dict[str, float]] = []

        # Configuration
        self.max_history_size = 1000
        self.performance_tracking = True

        logger.info(
            f"UnifiedProfitVectorizationSystem initialized with mode: {
                integration_mode.value}"
        )

    def calculate_unified_profit(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy = VectorizationStrategy.STANDARD,
        thermal_state: Optional[ThermalState] = None,
        bit_phase: Optional[BitPhase] = None,
    ) -> UnifiedProfitResult:
        """
        Calculate unified profit using all available systems.

        Args:
            market_data: Market data dictionary
            strategy: Vectorization strategy to use
            thermal_state: Current thermal state
            bit_phase: Current bit phase

        Returns:
            UnifiedProfitResult with comprehensive profit analysis
        """
        start_time = time.time()

        try:
            # Determine thermal state and bit phase if not provided
            if thermal_state is None:
                thermal_state = self._determine_thermal_state(market_data)
            if bit_phase is None:
                bit_phase = self._determine_bit_phase(market_data)

            # Calculate profit using different methods based on integration mode
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
            else:  # HIERARCHICAL
                result = self._calculate_hierarchical_mode(
                    market_data, strategy, thermal_state, bit_phase
                )

            # Update performance tracking
            calculation_time = time.time() - start_time
            self._update_performance_metrics(calculation_time, result)

            # Update system state
            self.last_calculation_time = time.time()
            self.calculation_count += 1

            logger.debug(f"Unified profit calculation completed in {calculation_time:.4f}s")

            return result

        except Exception as e:
            logger.error(f"Error in unified profit calculation: {e}")
            # Return safe fallback result
            return self._create_fallback_result(market_data, strategy, thermal_state, bit_phase)

    def _calculate_unified_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using unified mode (combines all methods)."""
        # Get profit from pure profit calculator
        pure_profit = self.profit_calculator.calculate_profit(market_data)

        # Get profit vector
        vector_input = {
            "price": market_data.get("price", 0.0),
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.5),
            "signal_strength": market_data.get("signal_strength", 0.5),
            "quantity": market_data.get("quantity", 1.0),
            "thermal_state": thermal_state,
            "bit_phase": bit_phase,
        }

        # Select vectorization mode based on strategy
        vectorization_mode = self._select_vectorization_mode(strategy)
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, vectorization_mode
        )

        # Combine results
        unified_profit = (pure_profit + profit_vector.total_profit) / 2.0
        confidence = (pure_profit.confidence + profit_vector.confidence) / 2.0

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=unified_profit,
            confidence=confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "pure_profit": pure_profit,
                "vector_profit": profit_vector.total_profit,
                "thermal_state": thermal_state.value,
                "bit_phase": bit_phase.value,
            },
        )

    def _calculate_weighted_mode(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Calculate profit using weighted mode (weighted combination)."""
        # Get individual results
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

        # Apply weights based on thermal state
        weights = self._get_thermal_weights(thermal_state)
        weighted_profit = (
            weights["pure"] * pure_profit.profit_value
            + weights["vector"] * profit_vector.total_profit
        )
        weighted_confidence = (
            weights["pure"] * pure_profit.confidence + weights["vector"] * profit_vector.confidence
        )

        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=weighted_profit,
            confidence=weighted_confidence,
            vector=profit_vector,
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={
                "weights": weights,
                "pure_profit": pure_profit.profit_value,
                "vector_profit": profit_vector.total_profit,
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
        """Calculate profit using adaptive mode (market condition based)."""
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
            weight_pure * pure_profit.profit_value + weight_vector * profit_vector.total_profit
        )
        adaptive_confidence = (
            weight_pure * pure_profit.confidence + weight_vector * profit_vector.confidence
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
        # Primary calculation (pure profit)
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
        profit_difference = abs(primary_result.profit_value - profit_vector.total_profit)

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

    def _select_vectorization_mode(self, strategy: VectorizationStrategy) -> VectorizationMode:
        """Select appropriate vectorization mode based on strategy."""
        strategy_mapping = {
            VectorizationStrategy.STANDARD: VectorizationMode.STANDARD,
            VectorizationStrategy.ENHANCED: VectorizationMode.ENTROPY_WEIGHTED,
            VectorizationStrategy.OPTIMIZED: VectorizationMode.ADAPTIVE,
            VectorizationStrategy.REAL_TIME: VectorizationMode.HIGH_FREQUENCY,
            VectorizationStrategy.BATCH: VectorizationMode.CONSERVATIVE,
        }

        return strategy_mapping.get(strategy, VectorizationMode.STANDARD)

    def _determine_thermal_state(self, market_data: Dict[str, Any]) -> ThermalState:
        """Determine thermal state based on market data."""
        volatility = market_data.get("volatility", 0.5)
        volume = market_data.get("volume", 0.0)

        if volatility > 0.8 or volume > 2000:
            return ThermalState.HOT
        elif volatility > 0.5 or volume > 1000:
            return ThermalState.WARM
        else:
            return ThermalState.COOL

    def _determine_bit_phase(self, market_data: Dict[str, Any]) -> BitPhase:
        """Determine bit phase based on market data."""
        volatility = market_data.get("volatility", 0.5)

        if volatility > 0.8:
            return BitPhase.THIRTY_TWO_BIT
        elif volatility > 0.5:
            return BitPhase.SIXTEEN_BIT
        else:
            return BitPhase.EIGHT_BIT

    def _get_thermal_weights(self, thermal_state: ThermalState) -> Dict[str, float]:
        """Get weights based on thermal state."""
        weight_mappings = {
            ThermalState.COOL: {"pure": 0.6, "vector": 0.4},
            ThermalState.WARM: {"pure": 0.5, "vector": 0.5},
            ThermalState.HOT: {"pure": 0.4, "vector": 0.6},
        }

        return weight_mappings.get(thermal_state, {"pure": 0.5, "vector": 0.5})

    def _update_performance_metrics(
        self, calculation_time: float, result: UnifiedProfitResult
    ) -> None:
        """Update performance tracking metrics."""
        if not self.performance_tracking:
            return

        metrics = {
            "calculation_time": calculation_time,
            "profit_value": result.profit_value,
            "confidence": result.confidence,
            "timestamp": result.timestamp,
        }

        self.performance_history.append(metrics)

        # Keep history manageable
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size :]

    def _create_fallback_result(
        self,
        market_data: Dict[str, Any],
        strategy: VectorizationStrategy,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> UnifiedProfitResult:
        """Create a fallback result when calculation fails."""
        return UnifiedProfitResult(
            timestamp=time.time(),
            profit_value=0.0,
            confidence=0.0,
            vector=ProfitVector(
                timestamp=time.time(),
                total_profit=0.0,
                confidence=0.0,
                mode=VectorizationMode.STANDARD,
                components={},
                metadata={"fallback": True},
            ),
            integration_mode=self.integration_mode,
            strategy=strategy,
            metadata={"error": "Fallback result due to calculation failure"},
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the system."""
        if not self.performance_history:
            return {"error": "No performance data available"}

        calculation_times = [m["calculation_time"] for m in self.performance_history]
        profit_values = [m["profit_value"] for m in self.performance_history]
        confidences = [m["confidence"] for m in self.performance_history]

        return {
            "total_calculations": self.calculation_count,
            "average_calculation_time": np.mean(calculation_times),
            "max_calculation_time": np.max(calculation_times),
            "average_profit": np.mean(profit_values),
            "profit_std": np.std(profit_values),
            "average_confidence": np.mean(confidences),
            "integration_mode": self.integration_mode.value,
            "history_size": len(self.performance_history),
        }

    def reset_performance_tracking(self) -> None:
        """Reset performance tracking data."""
        self.performance_history.clear()
        self.calculation_count = 0
        logger.info("Performance tracking reset")


def create_unified_profit_system(
    integration_mode: ProfitIntegrationMode = ProfitIntegrationMode.UNIFIED,
) -> UnifiedProfitVectorizationSystem:
    """Create a new unified profit vectorization system."""
    return UnifiedProfitVectorizationSystem(integration_mode=integration_mode)
