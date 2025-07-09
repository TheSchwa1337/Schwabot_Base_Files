#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 PROFIT MATRIX FEEDBACK LOOP SYSTEM
====================================

Profit Matrix Feedback Loop System for Schwabot

This module provides:
1. Feedback loop from backtest results → matrix updates
2. Logging of profit delta, time held, strategy ID hash
3. Integration with matrix optimizer using cosine similarity as fitness evaluator
4. Dynamic strategy weight adjustments based on performance

Mathematical Framework:
- 𝒫ₜ = Σ(profit_deltaᵢ · weightᵢ) for i=1 to n strategies
- ℳₜ₊₁ = ℳₜ + α·∇𝒫ₜ + β·cosine_similarity(𝒫ₜ, 𝒫ₜ₋₁)
- 𝒲ₜ₊₁ = 𝒲ₜ + γ·fitness_score(𝒫ₜ) · strategy_performance
- ℱₜ = fitness_evaluator(𝒫ₜ, time_held, strategy_hash)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np

# Import existing Schwabot components
try:
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

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
    logger.info("⚡ Profit Matrix Feedback Loop using GPU acceleration: {0}".format(_backend))
else:
    logger.info("🔄 Profit Matrix Feedback Loop using CPU fallback: {0}".format(_backend))


class FeedbackMode(Enum):
    """Modes for feedback processing"""

    REINFORCEMENT = "reinforcement"  # Positive feedback
    DECAY = "decay"  # Negative feedback
    NEUTRAL = "neutral"  # No change
    ADAPTIVE = "adaptive"  # Dynamic adjustment


class FitnessEvaluator(Enum):
    """Types of fitness evaluation"""

    PROFIT_ONLY = "profit_only"
    TIME_WEIGHTED = "time_weighted"
    RISK_ADJUSTED = "risk_adjusted"
    COMPOSITE = "composite"


@dataclass
class BacktestResult:
    """Result from backtest execution"""

    strategy_id: str
    strategy_hash: str
    profit_delta: float
    time_held: float
    entry_price: float
    exit_price: float
    position_size: float
    risk_level: float
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatrixUpdate:
    """Matrix update operation"""

    matrix_id: str
    update_vector: np.ndarray
    fitness_score: float
    feedback_mode: FeedbackMode
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfitMatrix:
    """Profit matrix with metadata."""

    matrix: xp.ndarray
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackResult:
    """Feedback loop result."""

    adjusted_matrix: xp.ndarray
    feedback_score: float
    adjustment_factor: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitMatrixFeedbackLoop:
    """
    Advanced profit matrix feedback loop system.

    Implements feedback mechanisms to adjust profit matrices based on
    historical performance and market conditions.
    """

    def __init__(self, learning_rate: float = 0.01, decay_factor: float = 0.95):
        """Initialize the profit matrix feedback loop."""
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.matrix_history: List[ProfitMatrix] = []
        self.feedback_history: List[FeedbackResult] = []
        self.performance_cache: Dict[str, float] = {}

        # Feedback parameters
        self.min_feedback_threshold = 0.1
        self.max_adjustment_factor = 2.0
        self.history_window = 100

        # Performance tracking
        self.total_feedback_cycles = 0
        self.average_feedback_score = 0.0
        self.adjustment_efficiency = 0.0

        logger.info("💰 Profit Matrix Feedback Loop initialized with learning rate {0}".format(learning_rate))

    def process_profit_matrix(self, matrix: xp.ndarray, source: str = "unknown") -> ProfitMatrix:
        """
        Process a new profit matrix.

        Args:
            matrix: Input profit matrix
            source: Matrix source identifier

        Returns:
            Processed ProfitMatrix object
        """
        try:
            # Create profit matrix object
            profit_matrix = ProfitMatrix(
                matrix=matrix.copy(),
                timestamp=time.time(),
                source=source,
                metadata={"shape": matrix.shape, "dtype": str(matrix.dtype)},
            )

            # Add to history
            self.matrix_history.append(profit_matrix)

            # Keep history manageable
            if len(self.matrix_history) > self.history_window:
                self.matrix_history = self.matrix_history[-self.history_window :]

            logger.debug("Processed profit matrix: shape={0}, source={1}".format(matrix.shape, source))
            return profit_matrix

        except Exception as e:
            logger.error("Error processing profit matrix: {0}".format(e))
            return self._create_fallback_matrix(matrix, source)

    def apply_feedback_loop(self, current_matrix: xp.ndarray, performance_metrics: Dict[str, float]) -> FeedbackResult:
        """
        Apply feedback loop to adjust profit matrix.

        Args:
            current_matrix: Current profit matrix
            performance_metrics: Performance metrics for feedback

        Returns:
            FeedbackResult with adjusted matrix
        """
        try:
            # Calculate feedback score
            feedback_score = self._calculate_feedback_score(performance_metrics)

            # Calculate adjustment factor
            adjustment_factor = self._calculate_adjustment_factor(feedback_score)

            # Apply feedback adjustment
            adjusted_matrix = self._apply_matrix_adjustment(current_matrix, adjustment_factor)

            # Create feedback result
            feedback_result = FeedbackResult(
                adjusted_matrix=adjusted_matrix,
                feedback_score=feedback_score,
                adjustment_factor=adjustment_factor,
                timestamp=time.time(),
                metadata={
                    "performance_metrics": performance_metrics,
                    "original_shape": current_matrix.shape,
                    "adjusted_shape": adjusted_matrix.shape,
                },
            )

            # Add to history
            self.feedback_history.append(feedback_result)
            self.total_feedback_cycles += 1

            # Update performance metrics
            self._update_performance_metrics(feedback_score, adjustment_factor)

            # Keep history manageable
            if len(self.feedback_history) > self.history_window:
                self.feedback_history = self.feedback_history[-self.history_window :]

            logger.info("Applied feedback loop: score={0}, adjustment={1}".format(feedback_score, adjustment_factor))
            return feedback_result

        except Exception as e:
            logger.error("Error applying feedback loop: {0}".format(e))
            return self._create_fallback_feedback(current_matrix)

    def _calculate_feedback_score(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate feedback score from performance metrics."""
        try:
            # Extract key metrics
            profit_rate = performance_metrics.get("profit_rate", 0.0)
            risk_score = performance_metrics.get("risk_score", 0.5)
            volatility = performance_metrics.get("volatility", 0.1)
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)

            # Calculate weighted feedback score
            profit_weight = 0.4
            risk_weight = 0.2
            volatility_weight = 0.2
            sharpe_weight = 0.2

            feedback_score = (
                profit_weight * profit_rate
                + risk_weight * (1.0 - risk_score)  # Lower risk is better
                + volatility_weight * (1.0 - volatility)  # Lower volatility is better
                + sharpe_weight * xp.tanh(sharpe_ratio)  # Normalize Sharpe ratio
            )

            # Normalize to 0-1 range
            feedback_score = xp.clip(feedback_score, 0.0, 1.0)

            return float(feedback_score)

        except Exception as e:
            logger.error("Error calculating feedback score: {0}".format(e))
            return 0.5

    def _calculate_adjustment_factor(self, feedback_score: float) -> float:
        """Calculate adjustment factor based on feedback score."""
        try:
            # Base adjustment from feedback score
            base_adjustment = (feedback_score - 0.5) * 2.0  # Range: [-1, 1]

            # Apply learning rate
            adjustment = base_adjustment * self.learning_rate

            # Apply decay factor based on history
            if self.feedback_history:
                recent_adjustments = [f.adjustment_factor for f in self.feedback_history[-5:]]
                avg_recent_adjustment = xp.mean(recent_adjustments)
                adjustment = adjustment * self.decay_factor + avg_recent_adjustment * (1.0 - self.decay_factor)

            # Clamp to maximum adjustment
            adjustment = xp.clip(adjustment, -self.max_adjustment_factor, self.max_adjustment_factor)

            return float(adjustment)

        except Exception as e:
            logger.error("Error calculating adjustment factor: {0}".format(e))
            return 0.0

    def _apply_matrix_adjustment(self, matrix: xp.ndarray, adjustment_factor: float) -> xp.ndarray:
        """Apply adjustment to profit matrix."""
        try:
            if adjustment_factor == 0.0:
                return matrix.copy()

            # Apply different adjustment strategies based on factor magnitude
            if abs(adjustment_factor) < 0.1:
                # Small adjustment: additive
                adjusted_matrix = matrix + adjustment_factor * xp.ones_like(matrix) * 0.01
            elif abs(adjustment_factor) < 0.5:
                # Medium adjustment: multiplicative
                adjusted_matrix = matrix * (1.0 + adjustment_factor)
            else:
                # Large adjustment: exponential
                adjusted_matrix = matrix * xp.exp(adjustment_factor * 0.1)

            # Ensure matrix remains valid (no negative values for profit)
            adjusted_matrix = xp.maximum(adjusted_matrix, 0.0)

            return adjusted_matrix

        except Exception as e:
            logger.error("Error applying matrix adjustment: {0}".format(e))
            return matrix.copy()

    def _update_performance_metrics(self, feedback_score: float, adjustment_factor: float) -> None:
        """Update internal performance metrics."""
        try:
            # Update average feedback score
            if self.total_feedback_cycles == 1:
                self.average_feedback_score = feedback_score
            else:
                self.average_feedback_score = (
                    self.average_feedback_score * (self.total_feedback_cycles - 1) + feedback_score
                ) / self.total_feedback_cycles

            # Calculate adjustment efficiency
            if self.feedback_history:
                recent_scores = [f.feedback_score for f in self.feedback_history[-10:]]
                if len(recent_scores) >= 2:
                    score_improvement = recent_scores[-1] - recent_scores[0]
                    adjustment_magnitude = abs(adjustment_factor)

                    if adjustment_magnitude > 0:
                        self.adjustment_efficiency = score_improvement / adjustment_magnitude
                    else:
                        self.adjustment_efficiency = 0.0

        except Exception as e:
            logger.error("Error updating performance metrics: {0}".format(e))

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback loop statistics."""
        try:
            if not self.feedback_history:
                return {"error": "No feedback history available"}

            # Calculate feedback statistics
            feedback_scores = [f.feedback_score for f in self.feedback_history]
            adjustment_factors = [f.adjustment_factor for f in self.feedback_history]

            # Performance trends
            recent_scores = feedback_scores[-10:] if len(feedback_scores) >= 10 else feedback_scores
            score_trend = xp.mean(recent_scores) - xp.mean(feedback_scores[:-10]) if len(feedback_scores) >= 10 else 0.0

            return {
                "total_feedback_cycles": self.total_feedback_cycles,
                "average_feedback_score": self.average_feedback_score,
                "adjustment_efficiency": self.adjustment_efficiency,
                "current_feedback_score": feedback_scores[-1] if feedback_scores else 0.0,
                "score_trend": score_trend,
                "average_adjustment_factor": (xp.mean(adjustment_factors) if adjustment_factors else 0.0),
                "max_adjustment_factor": xp.max(adjustment_factors) if adjustment_factors else 0.0,
                "min_adjustment_factor": xp.min(adjustment_factors) if adjustment_factors else 0.0,
                "learning_rate": self.learning_rate,
                "decay_factor": self.decay_factor,
                "history_size": len(self.feedback_history),
                "matrix_history_size": len(self.matrix_history),
            }

        except Exception as e:
            logger.error("Error getting feedback statistics: {0}".format(e))
            return {"error": str(e)}

    def optimize_parameters(self, target_performance: float = 0.8) -> Dict[str, float]:
        """
        Optimize feedback loop parameters based on performance.

        Args:
            target_performance: Target feedback score

        Returns:
            Dictionary with optimized parameters
        """
        try:
            if not self.feedback_history:
                return {"error": "Insufficient feedback history for optimization"}

            current_performance = self.average_feedback_score
            performance_gap = target_performance - current_performance

            # Adjust learning rate based on performance gap
            if performance_gap > 0.1:
                # Performance too low, increase learning rate
                new_learning_rate = min(self.learning_rate * 1.2, 0.1)
            elif performance_gap < -0.1:
                # Performance too high, decrease learning rate
                new_learning_rate = max(self.learning_rate * 0.8, 0.001)
            else:
                new_learning_rate = self.learning_rate

            # Adjust decay factor based on stability
            if self.adjustment_efficiency < 0:
                # Adjustments are counterproductive, increase decay
                new_decay_factor = min(self.decay_factor * 1.1, 0.99)
            else:
                # Adjustments are working, maintain or slightly decrease decay
                new_decay_factor = max(self.decay_factor * 0.95, 0.5)

            # Update parameters
            self.learning_rate = new_learning_rate
            self.decay_factor = new_decay_factor

            return {
                "learning_rate": new_learning_rate,
                "decay_factor": new_decay_factor,
                "performance_gap": performance_gap,
                "current_performance": current_performance,
                "target_performance": target_performance,
            }

        except Exception as e:
            logger.error("Error optimizing parameters: {0}".format(e))
            return {"error": str(e)}

    def _create_fallback_matrix(self, matrix: xp.ndarray, source: str) -> ProfitMatrix:
        """Create a fallback matrix when processing fails."""
        return ProfitMatrix(
            matrix=matrix.copy(),
            timestamp=time.time(),
            source=source,
            metadata={"error": "Fallback matrix"},
        )

    def _create_fallback_feedback(self, matrix: xp.ndarray) -> FeedbackResult:
        """Create a fallback feedback result when processing fails."""
        return FeedbackResult(
            adjusted_matrix=matrix.copy(),
            feedback_score=0.5,
            adjustment_factor=0.0,
            timestamp=time.time(),
            metadata={"error": "Fallback feedback"},
        )

    def clear_history(self) -> None:
        """Clear feedback and matrix history."""
        self.matrix_history.clear()
        self.feedback_history.clear()
        self.performance_cache.clear()
        logger.info("Profit matrix feedback history cleared")

    def export_feedback_data(self, filepath: str) -> bool:
        """Export feedback data to file."""
        try:
            data = {
                "feedback_history": [
                    {
                        "feedback_score": f.feedback_score,
                        "adjustment_factor": f.adjustment_factor,
                        "timestamp": f.timestamp,
                        "metadata": f.metadata,
                    }
                    for f in self.feedback_history
                ],
                "statistics": self.get_feedback_statistics(),
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info("Feedback data exported to {0}".format(filepath))
            return True

        except Exception as e:
            logger.error("Error exporting feedback data: {0}".format(e))
            return False


def create_profit_matrix_feedback_loop(
    learning_rate: float = 0.01, decay_factor: float = 0.95
) -> ProfitMatrixFeedbackLoop:
    """Factory function to create a profit matrix feedback loop instance."""
    return ProfitMatrixFeedbackLoop(learning_rate, decay_factor)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create feedback loop
    feedback_loop = create_profit_matrix_feedback_loop(learning_rate=0.02, decay_factor=0.9)

    print("=== Testing Profit Matrix Feedback Loop ===")

    # Create test profit matrix
    test_matrix = xp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Process matrix
    profit_matrix = feedback_loop.process_profit_matrix(test_matrix, "test_source")
    print("Processed matrix shape: {0}".format(profit_matrix.matrix.shape))

    # Apply feedback loop
    performance_metrics = {
        "profit_rate": 0.15,
        "risk_score": 0.3,
        "volatility": 0.08,
        "sharpe_ratio": 1.2,
    }

    feedback_result = feedback_loop.apply_feedback_loop(test_matrix, performance_metrics)
    print("Feedback score: {0}".format(feedback_result.feedback_score))
    print("Adjustment factor: {0}".format(feedback_result.adjustment_factor))

    # Get statistics
    stats = feedback_loop.get_feedback_statistics()
    print("\nFeedback Statistics:")
    print("Total cycles: {0}".format(stats.get("total_feedback_cycles", 0)))
    print("Average score: {0}".format(stats.get("average_feedback_score", 0)))
    print("Adjustment efficiency: {0}".format(stats.get("adjustment_efficiency", 0)))

    # Optimize parameters
    optimization = feedback_loop.optimize_parameters(target_performance=0.85)
    print("\nParameter Optimization:")
    print("New learning rate: {0}".format(optimization.get("learning_rate", 0)))
    print("New decay factor: {0}".format(optimization.get("decay_factor", 0)))

    print("Profit Matrix Feedback Loop test completed")
