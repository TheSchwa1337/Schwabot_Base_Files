# -*- coding: utf-8 -*-\n"""core.ghost_trigger_map
Ghost Trigger Map
== == == == == == == == =

Provides ghost-phase-aware trigger routing using the GhostPhaseStrategyLoader.
This module maps trigger signals to strategy logic while consuming
GhostPhaseDecision objects for modern, unified decision-making.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from core.ghost_phase_strategy_loader import GhostPhaseStrategyLoader, GhostPhaseDecision
from utils.safe_print import safe_print

logger = logging.getLogger(__name__)

__all__ = [
    "TriggerResult",
    "GhostTriggerMapper",
    "generate_ghost_trigger_map",
]


@dataclass
class TriggerResult:
    """Result of trigger evaluation."""
    triggered: bool
    strategy_id: str
    confidence: float
    ghost_decision: GhostPhaseDecision
    trigger_metadata: Dict[str, Any]


class GhostTriggerMapper:
    """Maps trigger signals to strategy logic using ghost phase decisions."""

    def __init__(
        self,
        overlay_json: str = "memory_stack/aleph_overlays.json",
        *,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize the ghost trigger mapper.

        Args:
            overlay_json: Path to overlay configuration
            confidence_threshold: Minimum confidence to trigger
        """
        self.ghost_loader = GhostPhaseStrategyLoader(overlay_json)
        self.confidence_threshold = confidence_threshold
        self.trigger_history: List[TriggerResult] = []

        safe_print("🎯 Ghost Trigger Mapper initialized")

    def evaluate_trigger(
        self,
        prices: Sequence[float],
        live_vector: Sequence[float],
        raw_signals: Sequence[float],
        volatility: float = 0.5,
        resonance: float = 0.7,
        threshold: float = 0.3,
    ) -> TriggerResult:
        """Evaluate trigger conditions using ghost phase logic.

        Args:
            prices: Historical price data
            live_vector: Current market state vector
            raw_signals: Strategy confidence signals
            volatility: Market volatility factor
            resonance: Signal resonance factor
            threshold: Trigger threshold

        Returns:
            TriggerResult with trigger decision and metadata
        """
        try:
            # Get ghost phase decision
            ghost_decision = self.ghost_loader.decide(prices, live_vector, raw_signals)

            # Calculate trigger confidence based on ghost decision
            confidence = self._calculate_trigger_confidence(
                ghost_decision, volatility, resonance, threshold
            )

            # Determine if trigger should fire
            triggered = confidence >= self.confidence_threshold

            # Create trigger metadata
            trigger_metadata = {
                "volatility": volatility,
                "resonance": resonance,
                "threshold": threshold,
                "phase_state": ghost_decision.phase_report.phase_state.name,
                "consensus": ghost_decision.consensus,
                "overlay_similarity": ghost_decision.overlay_match.similarity,
                "drift_weight": ghost_decision.drift_report.drift_weight,
            }

            result = TriggerResult(
                triggered=triggered,
                strategy_id=ghost_decision.strategy_id,
                confidence=confidence,
                ghost_decision=ghost_decision,
                trigger_metadata=trigger_metadata,
            )

            # Store in history
            self.trigger_history.append(result)
            if len(self.trigger_history) > 1000:  # Keep last 1000 triggers
                self.trigger_history = self.trigger_history[-1000:]

            if triggered:
                safe_print(f"🎯 Trigger fired: {ghost_decision.strategy_id} (confidence: {confidence:.3f})")

            return result

        except Exception as e:
            logger.error(f"Trigger evaluation failed: {e}")

            # Return safe fallback
            return TriggerResult(
                triggered=False,
                strategy_id="fallback_hold",
                confidence=0.0,
                ghost_decision=None,  # type: ignore
                trigger_metadata={"error": str(e)},
            )

    def _calculate_trigger_confidence(
        self,
        decision: GhostPhaseDecision,
        volatility: float,
        resonance: float,
        threshold: float,
    ) -> float:
        """Calculate trigger confidence from ghost decision and parameters."""
        # Base confidence from overlay similarity
        base_confidence = (decision.overlay_match.similarity + 1.0) / 2.0  # Normalize to [0,1]

        # Boost for consensus
        consensus_boost = 0.2 if decision.consensus else -0.1

        # Phase adjustment
        phase_multiplier = {
            "HIGH": 1.2,  # High risk can be good for certain strategies
            "MEDIUM": 1.0,
            "LOW": 0.8,   # Low phase might be less interesting
        }.get(decision.phase_report.phase_state.name, 1.0)

        # Incorporate external parameters
        volatility_factor = min(1.0, volatility * 1.5)  # Cap at 1.0
        resonance_factor = resonance
        threshold_adjustment = 1.0 - threshold  # Lower threshold = higher confidence

        # Combine all factors
        confidence = (
            base_confidence
            + consensus_boost
        ) * phase_multiplier * volatility_factor * resonance_factor * threshold_adjustment

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get trigger statistics from history."""
        if not self.trigger_history:
            return {"total_triggers": 0, "fired_triggers": 0, "fire_rate": 0.0}

        total = len(self.trigger_history)
        fired = sum(1 for t in self.trigger_history if t.triggered)

        return {
            "total_triggers": total,
            "fired_triggers": fired,
            "fire_rate": fired / total if total > 0 else 0.0,
            "avg_confidence": sum(t.confidence for t in self.trigger_history) / total,
            "recent_strategies": [t.strategy_id for t in self.trigger_history[-10:]],
        }


def generate_ghost_trigger_map(
    volatility: float,
    resonance: float,
    threshold: float,
) -> Dict[str, Any]:
    """Generate a ghost trigger map with the given parameters.

    This is a legacy compatibility function that creates a simple
    trigger configuration map for use with external systems.
    
    Args:
        volatility: Market volatility factor
        resonance: Signal resonance factor  
        threshold: Trigger threshold
        
    Returns:
        Dictionary containing trigger map configuration
    """
    return {
        "trigger_type": "ghost_phase",
        "parameters": {
            "volatility": volatility,
            "resonance": resonance,
            "threshold": threshold,
        },
        "strategy_mapping": {
            "high_risk": ["momentum_alpha_high_risk", "volatility_breakout_high_risk"],
            "medium_risk": ["trend_following_medium_risk", "swing_trading_medium_risk"],
            "low_risk": ["mean_reversion_low_risk", "hedge_protection_low_risk"],
            "hold": ["momentum_alpha_hold", "trend_following_hold"],
        },
        "confidence_thresholds": {
            "minimum": 0.3,
            "recommended": 0.5,
            "aggressive": 0.7,
        },
        "metadata": {
            "version": "ghost_phase_v1",
            "created_with": "GhostPhaseStrategyLoader",
        },
    }
