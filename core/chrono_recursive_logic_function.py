#!/usr/bin/env python3
"""
Chrono-Recursive Logic Function (CRLF)

A phase-resonant, time-aware logic operator that recursively evaluates profit curves,
system entropy, and strategy alignment across chronological wavefronts.

This module implements:
- Temporal resonance decay
- Recursion depth awareness
- State vector alignment
- Profit-based waveform correction
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clean_math_foundation import BitPhase, CleanMathFoundation, ThermalState
from .zpe_zbe_core import QuantumSyncStatus, ZBEBalance, ZPEVector

logger = logging.getLogger(__name__)


class CRLFTriggerState(Enum):
    """CRLF trigger states based on output thresholds."""

    HOLD = "hold"
    ESCALATE = "escalate"
    OVERRIDE = "override"
    RECURSIVE_RESET = "recursive_reset"


@dataclass
class CRLFState:
    """Current state of the Chrono-Recursive Logic Function."""

    # Core parameters
    tau: float  # Elapsed tick time since last successful strategy hash
    psi: np.ndarray  # Current strategy state vector
    delta_t: float  # Tick-phase decay offset for alignment
    entropy: float  # Entropy or error accumulation across logic pathways

    # Recursive state tracking
    recursion_depth: int = 0
    max_recursion_depth: int = 10

    # State propagation history
    psi_history: List[np.ndarray] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    crlf_output_history: List[float] = field(default_factory=list)

    # Dynamic weighting coefficients
    alpha_n: float = 0.7  # Strategy trust coefficient
    beta_n: float = 0.3  # Strategy drift coefficient
    lambda_decay: float = 0.95  # Entropy decay factor

    # Thresholds
    hold_threshold: float = 0.3
    escalate_threshold: float = 1.0
    override_threshold: float = 1.5

    # Performance tracking
    last_successful_hash: float = 0.0
    strategy_corrections: int = 0
    total_executions: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRLFResponse:
    """Response from CRLF computation."""

    crlf_output: float
    trigger_state: CRLFTriggerState
    psi_n: np.ndarray  # Current recursive state
    entropy_updated: float
    recursion_depth: int
    confidence: float
    recommendations: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChronoRecursiveLogicFunction:
    """
    Chrono-Recursive Logic Function implementation.

    Evaluates time-recursive logic in the form:
    CRLF(τ,ψ,Δ,E) = Ψₙ(τ) ⋅ ∇ψ ⋅ Δₜ ⋅ e^(-Eτ)

    Where:
    - τ: Elapsed tick time since last successful strategy hash
    - ψ: Current strategy state vector
    - ∇ψ: Spatial gradient of strategy shift (profit curve directionality)
    - Δₜ: Tick-phase decay offset for alignment
    - E: Entropy or error accumulation across logic pathways
    - Ψₙ(τ): Recursive state propagation function at time τ
    """

    def __init__(self, initial_state: Optional[CRLFState] = None):
        """Initialize the CRLF with optional initial state."""
        self.math_foundation = CleanMathFoundation()
        self.state = initial_state or self._create_default_state()

        # Performance tracking
        self.execution_history: List[CRLFResponse] = []
        self.strategy_alignment_scores: List[float] = []

        logger.info("🔮 Chrono-Recursive Logic Function initialized")

    def _create_default_state(self) -> CRLFState:
        """Create a default CRLF state."""
        return CRLFState(
            tau=0.0,
            psi=np.array([0.5, 0.5, 0.5, 0.5]),  # 4D strategy vector
            delta_t=0.0,
            entropy=0.1,
        )

    def compute_crlf(
        self,
        strategy_vector: np.ndarray,
        profit_curve: np.ndarray,
        market_entropy: float,
        time_offset: float = 0.0,
    ) -> CRLFResponse:
        """
        Compute the Chrono-Recursive Logic Function.

        Args:
            strategy_vector: Current strategy state vector
            profit_curve: Recent profit curve data
            market_entropy: Current market entropy
            time_offset: Time offset for alignment

        Returns:
            CRLFResponse with computed logic and recommendations
        """
        try:
            # Update state
            self.state.psi = strategy_vector
            self.state.delta_t = time_offset
            self.state.tau = time.time() - self.state.last_successful_hash

            # Compute recursive state function Ψₙ(τ)
            psi_n = self._compute_recursive_state_function()

            # Compute spatial gradient ∇ψ
            gradient_psi = self._compute_strategy_gradient(profit_curve)

            # Update entropy
            entropy_updated = self._update_entropy(market_entropy, gradient_psi)

            # Compute CRLF output: Ψₙ(τ) ⋅ ∇ψ ⋅ Δₜ ⋅ e^(-Eτ)
            crlf_output = self._compute_crlf_output(psi_n, gradient_psi, entropy_updated)

            # Determine trigger state
            trigger_state = self._determine_trigger_state(crlf_output)

            # Generate recommendations
            recommendations = self._generate_recommendations(crlf_output, trigger_state)

            # Update state history
            self._update_state_history(psi_n, entropy_updated, crlf_output)

            # Create response
            response = CRLFResponse(
                crlf_output=crlf_output,
                trigger_state=trigger_state,
                psi_n=psi_n,
                entropy_updated=entropy_updated,
                recursion_depth=self.state.recursion_depth,
                confidence=self._compute_confidence(crlf_output, entropy_updated),
                recommendations=recommendations,
            )

            # Store execution history
            self.execution_history.append(response)

            # Update performance metrics
            self._update_performance_metrics(response)

            logger.debug(f"CRLF computed: {crlf_output:.4f} -> {trigger_state.value}")

            return response

        except Exception as e:
            logger.error(f"Error computing CRLF: {e}")
            return self._create_fallback_response()

    def _compute_recursive_state_function(self) -> np.ndarray:
        """
        Compute recursive state function: Ψₙ(τ) = αₙ ⋅ Ψₙ₋₁(τ-1) + βₙ ⋅ Rₙ(τ)

        Where:
        - Ψₙ₋₁ is the last known strategy signal
        - Rₙ(τ) is the response function
        - αₙ, βₙ are dynamic weighting coefficients
        """
        if not self.state.psi_history:
            # First iteration - use current psi
            psi_n = self.state.psi.copy()
        else:
            # Recursive computation
            psi_prev = self.state.psi_history[-1]
            response_function = self._compute_response_function()

            psi_n = self.state.alpha_n * psi_prev + self.state.beta_n * response_function

        # Normalize to prevent divergence
        psi_n = np.clip(psi_n, 0.0, 1.0)

        return psi_n

    def _compute_response_function(self) -> np.ndarray:
        """
        Compute response function Rₙ(τ) based on current market conditions.

        This function responds to:
        - Hash triggers
        - Market anomalies
        - AI feedback
        """
        # Base response based on current entropy
        base_response = np.array([0.5] * len(self.state.psi))

        # Adjust based on entropy level
        if self.state.entropy > 0.8:
            # High entropy - conservative response
            base_response *= 0.7
        elif self.state.entropy < 0.2:
            # Low entropy - aggressive response
            base_response *= 1.3

        # Add noise for exploration
        noise = np.random.normal(0, 0.1, len(base_response))
        response = base_response + noise

        return np.clip(response, 0.0, 1.0)

    def _compute_strategy_gradient(self, profit_curve: np.ndarray) -> np.ndarray:
        """
        Compute spatial gradient ∇ψ of strategy shift.

        This represents the directionality of the profit curve.
        """
        if len(profit_curve) < 2:
            return np.array([0.0] * len(self.state.psi))

        # Compute gradient of profit curve
        profit_gradient = np.gradient(profit_curve)

        # Map profit gradient to strategy dimensions
        strategy_gradient = np.zeros(len(self.state.psi))

        # Simple mapping: use profit trend to adjust strategy weights
        avg_profit_trend = (
            np.mean(profit_gradient[-5:]) if len(profit_gradient) >= 5 else np.mean(profit_gradient)
        )

        # Adjust strategy vector based on profit trend
        if avg_profit_trend > 0:
            # Positive trend - increase aggressive strategies
            strategy_gradient[0] = 0.1  # Momentum
            strategy_gradient[1] = 0.05  # Scalping
        else:
            # Negative trend - increase conservative strategies
            strategy_gradient[2] = 0.1  # Mean reversion
            strategy_gradient[3] = 0.05  # Swing

        return strategy_gradient

    def _update_entropy(self, market_entropy: float, gradient_psi: np.ndarray) -> float:
        """
        Update entropy: E(t+1) = λ ⋅ E(t) + (1-λ) ⋅ |Δψ|

        Where Δψ is the delta shift between expected and real execution.
        """
        # Compute delta shift
        if self.state.psi_history:
            expected_psi = self.state.psi_history[-1]
            delta_psi = np.linalg.norm(self.state.psi - expected_psi)
        else:
            delta_psi = 0.0

        # Update entropy with exponential decay
        new_entropy = self.state.lambda_decay * self.state.entropy + (
            1 - self.state.lambda_decay
        ) * (market_entropy + delta_psi)

        return np.clip(new_entropy, 0.0, 1.0)

    def _compute_crlf_output(
        self, psi_n: np.ndarray, gradient_psi: np.ndarray, entropy: float
    ) -> float:
        """
        Compute CRLF output: Ψₙ(τ) ⋅ ∇ψ ⋅ Δₜ ⋅ e^(-Eτ)
        """
        # Compute dot product of recursive state and gradient
        psi_gradient_product = np.dot(psi_n, gradient_psi)

        # Apply temporal decay
        temporal_decay = math.exp(-entropy * self.state.tau)

        # Apply tick-phase offset
        phase_offset = 1.0 + self.state.delta_t

        # Compute final output
        crlf_output = psi_gradient_product * temporal_decay * phase_offset

        return crlf_output

    def _determine_trigger_state(self, crlf_output: float) -> CRLFTriggerState:
        """
        Determine trigger state based on CRLF output thresholds.

        0 < CRLF < θ → HOLD logic
        θ < CRLF < 1 → ESCALATE
        CRLF > 1.5 → OVERRIDE trigger
        CRLF < 0 → RECURSIVE RESET (fallback)
        """
        if crlf_output < 0:
            return CRLFTriggerState.RECURSIVE_RESET
        elif crlf_output < self.state.hold_threshold:
            return CRLFTriggerState.HOLD
        elif crlf_output < self.state.escalate_threshold:
            return CRLFTriggerState.ESCALATE
        elif crlf_output > self.state.override_threshold:
            return CRLFTriggerState.OVERRIDE
        else:
            return CRLFTriggerState.ESCALATE

    def _generate_recommendations(
        self, crlf_output: float, trigger_state: CRLFTriggerState
    ) -> Dict[str, Any]:
        """Generate recommendations based on CRLF output and trigger state."""
        recommendations = {
            "action": trigger_state.value,
            "confidence": self._compute_confidence(crlf_output, self.state.entropy),
            "risk_adjustment": self._compute_risk_adjustment(crlf_output),
            "strategy_weights": self._compute_strategy_weights(crlf_output),
            "temporal_urgency": self._compute_temporal_urgency(crlf_output),
        }

        # Add state-specific recommendations
        if trigger_state == CRLFTriggerState.OVERRIDE:
            recommendations.update(
                {
                    "override_matrix": "FastProfitOverrideΩ",
                    "priority": "HIGH",
                    "timeout": 300,  # 5 minutes
                }
            )
        elif trigger_state == CRLFTriggerState.RECURSIVE_RESET:
            recommendations.update(
                {
                    "reset_cycle": "Recursive_Fallback_7D",
                    "fallback_strategy": "Conservative_Mean_Reversion",
                    "recovery_time": 604800,  # 7 days
                }
            )
        elif trigger_state == CRLFTriggerState.HOLD:
            recommendations.update(
                {
                    "hold_duration": self._compute_hold_duration(crlf_output),
                    "monitoring_frequency": "HIGH",
                }
            )

        return recommendations

    def _compute_confidence(self, crlf_output: float, entropy: float) -> float:
        """Compute confidence level based on CRLF output and entropy."""
        # Higher CRLF output = higher confidence
        output_confidence = min(abs(crlf_output) / 2.0, 1.0)

        # Lower entropy = higher confidence
        entropy_confidence = 1.0 - entropy

        # Combined confidence
        confidence = (output_confidence + entropy_confidence) / 2.0

        return np.clip(confidence, 0.0, 1.0)

    def _compute_risk_adjustment(self, crlf_output: float) -> float:
        """Compute risk adjustment factor based on CRLF output."""
        if crlf_output > 1.5:
            # Override - reduce risk
            return 0.5
        elif crlf_output > 1.0:
            # Escalate - moderate risk
            return 0.8
        elif crlf_output > 0.3:
            # Normal - standard risk
            return 1.0
        else:
            # Hold - increase risk
            return 1.2

    def _compute_strategy_weights(self, crlf_output: float) -> Dict[str, float]:
        """Compute strategy weights based on CRLF output."""
        if crlf_output > 1.5:
            # Override - aggressive strategies
            return {"momentum": 0.4, "scalping": 0.3, "mean_reversion": 0.2, "swing": 0.1}
        elif crlf_output > 1.0:
            # Escalate - balanced strategies
            return {"momentum": 0.3, "scalping": 0.3, "mean_reversion": 0.2, "swing": 0.2}
        else:
            # Hold/Reset - conservative strategies
            return {"momentum": 0.1, "scalping": 0.1, "mean_reversion": 0.4, "swing": 0.4}

    def _compute_temporal_urgency(self, crlf_output: float) -> str:
        """Compute temporal urgency based on CRLF output."""
        if crlf_output > 1.5:
            return "IMMEDIATE"
        elif crlf_output > 1.0:
            return "HIGH"
        elif crlf_output > 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _compute_hold_duration(self, crlf_output: float) -> int:
        """Compute hold duration in seconds based on CRLF output."""
        # Lower CRLF output = longer hold duration
        base_duration = 300  # 5 minutes
        multiplier = max(0.1, 1.0 - abs(crlf_output))
        return int(base_duration * multiplier)

    def _update_state_history(self, psi_n: np.ndarray, entropy: float, crlf_output: float):
        """Update state history for recursive computations."""
        self.state.psi_history.append(psi_n.copy())
        self.state.entropy_history.append(entropy)
        self.state.crlf_output_history.append(crlf_output)

        # Keep history manageable
        max_history = 100
        if len(self.state.psi_history) > max_history:
            self.state.psi_history = self.state.psi_history[-max_history:]
            self.state.entropy_history = self.state.entropy_history[-max_history:]
            self.state.crlf_output_history = self.state.crlf_output_history[-max_history:]

    def _update_performance_metrics(self, response: CRLFResponse):
        """Update performance tracking metrics."""
        self.state.total_executions += 1

        # Track strategy alignment
        alignment_score = self._compute_strategy_alignment(response)
        self.strategy_alignment_scores.append(alignment_score)

        # Update recursion depth
        if response.trigger_state == CRLFTriggerState.RECURSIVE_RESET:
            self.state.recursion_depth = 0
            self.state.strategy_corrections += 1
        else:
            self.state.recursion_depth = min(
                self.state.recursion_depth + 1, self.state.max_recursion_depth
            )

    def _compute_strategy_alignment(self, response: CRLFResponse) -> float:
        """Compute strategy alignment score."""
        # Higher confidence and lower entropy = better alignment
        alignment = response.confidence * (1.0 - response.entropy_updated)
        return np.clip(alignment, 0.0, 1.0)

    def _create_fallback_response(self) -> CRLFResponse:
        """Create a fallback response when computation fails."""
        return CRLFResponse(
            crlf_output=-1.0,
            trigger_state=CRLFTriggerState.RECURSIVE_RESET,
            psi_n=self.state.psi.copy(),
            entropy_updated=1.0,
            recursion_depth=0,
            confidence=0.0,
            recommendations={
                "action": "recursive_reset",
                "fallback_strategy": "Conservative_Mean_Reversion",
                "error": "Computation failed",
            },
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.execution_history:
            return {"error": "No execution history available"}

        recent_responses = self.execution_history[-50:]  # Last 50 executions

        return {
            "total_executions": self.state.total_executions,
            "strategy_corrections": self.state.strategy_corrections,
            "current_recursion_depth": self.state.recursion_depth,
            "average_confidence": np.mean([r.confidence for r in recent_responses]),
            "average_entropy": np.mean([r.entropy_updated for r in recent_responses]),
            "trigger_state_distribution": self._get_trigger_state_distribution(),
            "strategy_alignment_trend": self._get_alignment_trend(),
            "crlf_output_statistics": self._get_crlf_statistics(),
            "recommendations": self._get_recent_recommendations(),
        }

    def _get_trigger_state_distribution(self) -> Dict[str, int]:
        """Get distribution of trigger states."""
        distribution = {}
        for response in self.execution_history:
            state = response.trigger_state.value
            distribution[state] = distribution.get(state, 0) + 1
        return distribution

    def _get_alignment_trend(self) -> List[float]:
        """Get recent strategy alignment trend."""
        return self.strategy_alignment_scores[-20:] if self.strategy_alignment_scores else []

    def _get_crlf_statistics(self) -> Dict[str, float]:
        """Get CRLF output statistics."""
        outputs = [r.crlf_output for r in self.execution_history]
        if not outputs:
            return {}

        return {
            "mean": np.mean(outputs),
            "std": np.std(outputs),
            "min": np.min(outputs),
            "max": np.max(outputs),
            "median": np.median(outputs),
        }

    def _get_recent_recommendations(self) -> List[Dict[str, Any]]:
        """Get recent recommendations."""
        recent = self.execution_history[-10:]
        return [
            {
                "action": r.recommendations.get("action", "unknown"),
                "confidence": r.confidence,
                "crlf_output": r.crlf_output,
            }
            for r in recent
        ]

    def reset_state(self):
        """Reset CRLF state for fresh computation."""
        self.state = self._create_default_state()
        self.execution_history.clear()
        self.strategy_alignment_scores.clear()
        logger.info("🔄 CRLF state reset")


def create_crlf() -> ChronoRecursiveLogicFunction:
    """Factory function to create a CRLF instance."""
    return ChronoRecursiveLogicFunction()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create CRLF instance
    crlf = create_crlf()

    # Test computation
    strategy_vector = np.array([0.6, 0.4, 0.3, 0.7])
    profit_curve = np.array([100, 105, 103, 108, 110, 107, 112])
    market_entropy = 0.3

    response = crlf.compute_crlf(strategy_vector, profit_curve, market_entropy)

    print(f"CRLF Output: {response.crlf_output:.4f}")
    print(f"Trigger State: {response.trigger_state.value}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Recommendations: {response.recommendations}")

    # Get performance summary
    summary = crlf.get_performance_summary()
    print(f"\nPerformance Summary: {summary}")
