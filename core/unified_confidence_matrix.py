from core.unified_math_system import unified_math
import numpy as np
import math
# #!/usr/bin/env python3
"""Unified Confidence Matrix - Central Hub for Schwabot Confidence Calculations.

This module serves as the central hub connecting all confidence-related systems:
- Backlog logic with real-time decision making
- Ferris wheel cycles with matrix controller states
- AI consensus with internalized mathematical confidence
- Event impact with confidence calculations

Mathematical Foundation:
C_unified = α × C_backlog + β × C_ferris + γ × C_ai + δ × C_matrix

Where:
- C_backlog = Confidence from historical backlog data
- C_ferris = Confidence from Ferris wheel cycle position
- C_ai = Confidence from AI consensus
- C_matrix = Confidence from matrix controller state
- α, β, γ, δ = Weight coefficients (α + β + γ + δ = 1.0)

Flake8 compliant with comprehensive type hints and error handling.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
# from core.unified_math_system import unified_math  # F811: duplicate import

# Import core components
try:
    from core.type_defs import MatrixController, BitLevel, MatrixPhase
    from core.event_impact_mapper import EventImpact
except ImportError:
    # Fallback type definitions
    from typing import Any as MatrixController
#     from enum import Enum as BitLevel  # F811: duplicate import
#     from enum import Enum as MatrixPhase  # F811: duplicate import

logger = logging.getLogger(__name__)


class ConfidenceSource(Enum):
    """Sources of confidence data."""
BACKLOG = "backlog"
FERRIS_WHEEL = "ferris_wheel"
AI_CONSENSUS = "ai_consensus"
MATRIX_CONTROLLER = "matrix_controller"
EVENT_IMPACT = "event_impact"


@dataclass
class ConfidenceComponent:
    """Individual confidence component with metadata."""
source: ConfidenceSource
value: float
weight: float
timestamp: float
metadata: Dict[str, Any] = field(default_factory=dict)
    reliability: float = 1.0


@dataclass
class UnifiedConfidenceResult:
    """Result of unified confidence calculation."""
unified_confidence: float
components: Dict[ConfidenceSource, ConfidenceComponent]
weights: Dict[ConfidenceSource, float]
calculation_time: float
reliability_score: float
metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedConfidenceMatrix:
    """Central hub for unified confidence calculations across all systems."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified confidence matrix."""
self.config = config or self._default_config()

        # Confidence caches for each source
self.backlog_confidence_cache: Dict[str, float] = {}
self.ferris_wheel_confidence_map: Dict[int, float] = {}
self.ai_consensus_confidence_weights: Dict[str, float] = {}
self.matrix_controller_confidence_states: Dict[str, float] = {}
self.event_impact_confidence_cache: Dict[str, float] = {}

        # Weight coefficients for unified confidence
self.weight_coefficients = {
ConfidenceSource.BACKLOG: self.config.get('backlog_weight', 0.25),
            ConfidenceSource.FERRIS_WHEEL: self.config.get('ferris_weight', 0.25),
            ConfidenceSource.AI_CONSENSUS: self.config.get('ai_weight', 0.25),
            ConfidenceSource.MATRIX_CONTROLLER: self.config.get('matrix_weight', 0.25),
            ConfidenceSource.EVENT_IMPACT: self.config.get('event_weight', 0.1)
        }

        # Normalize weights to sum to 1.0
total_weight = sum(self.weight_coefficients.values())
        if total_weight > 0:
            for source in self.weight_coefficients:
self.weight_coefficients[source] /= total_weight

        # Performance tracking
self.total_calculations = 0
self.calculation_history: List[UnifiedConfidenceResult] = []
self.last_calculation_time = 0.0

logger.info("Unified Confidence Matrix initialized")

    def calculate_unified_confidence(self,
                                   backlog_state: Optional[Dict[str, Any]] = None,
ferris_wheel_position: Optional[int] = None,
ai_consensus: Optional[Dict[str, float]] = None,
matrix_controller_state: Optional[Dict[str, Any]] = None,
event_impact: Optional[EventImpact] = None) -> UnifiedConfidenceResult:
"""Calculate unified confidence across all systems.

Args:
backlog_state: Historical backlog data
ferris_wheel_position: Current Ferris wheel position
ai_consensus: AI consensus data
matrix_controller_state: Matrix controller state
event_impact: Event impact data

Returns:
UnifiedConfidenceResult with combined confidence and metadata
"""
start_time = time.time()

        try:
            # Calculate confidence for each component
components = {}

            # Backlog confidence
            if backlog_state is not None:
backlog_confidence = self._calculate_backlog_confidence(backlog_state)
                components[ConfidenceSource.BACKLOG] = ConfidenceComponent(
                    source=ConfidenceSource.BACKLOG,
value=backlog_confidence,
weight=self.weight_coefficients[ConfidenceSource.BACKLOG],
timestamp=time.time(),
                    metadata={'backlog_state': backlog_state},
reliability=self._calculate_backlog_reliability(backlog_state)


            # Ferris wheel confidence
            if ferris_wheel_position is not None:
ferris_confidence = self._calculate_ferris_wheel_confidence(ferris_wheel_position)
                components[ConfidenceSource.FERRIS_WHEEL] = ConfidenceComponent(
                    source=ConfidenceSource.FERRIS_WHEEL,
value=ferris_confidence,
weight=self.weight_coefficients[ConfidenceSource.FERRIS_WHEEL],
timestamp=time.time(),
                    metadata={'ferris_wheel_position': ferris_wheel_position},
reliability=self._calculate_ferris_reliability(ferris_wheel_position)


            # AI consensus confidence
            if ai_consensus is not None:
ai_confidence = self._calculate_ai_consensus_confidence(ai_consensus)
                components[ConfidenceSource.AI_CONSENSUS] = ConfidenceComponent(
                    source=ConfidenceSource.AI_CONSENSUS,
value=ai_confidence,
weight=self.weight_coefficients[ConfidenceSource.AI_CONSENSUS],
timestamp=time.time(),
                    metadata={'ai_consensus': ai_consensus},
reliability=self._calculate_ai_reliability(ai_consensus)


            # Matrix controller confidence
            if matrix_controller_state is not None:
matrix_confidence = self._calculate_matrix_controller_confidence(matrix_controller_state)
                components[ConfidenceSource.MATRIX_CONTROLLER] = ConfidenceComponent(
                    source=ConfidenceSource.MATRIX_CONTROLLER,
value=matrix_confidence,
weight=self.weight_coefficients[ConfidenceSource.MATRIX_CONTROLLER],
timestamp=time.time(),
                    metadata={'matrix_controller_state': matrix_controller_state},
reliability=self._calculate_matrix_reliability(matrix_controller_state)


            # Event impact confidence
            if event_impact is not None:
event_confidence = self._calculate_event_impact_confidence(event_impact)
                components[ConfidenceSource.EVENT_IMPACT] = ConfidenceComponent(
                    source=ConfidenceSource.EVENT_IMPACT,
value=event_confidence,
weight=self.weight_coefficients[ConfidenceSource.EVENT_IMPACT],
timestamp=time.time(),
                    metadata={'event_impact': event_impact.event_id},
reliability=self._calculate_event_reliability(event_impact)


            # Calculate unified confidence
unified_confidence = self._combine_confidence_components(components)

            # Calculate overall reliability
reliability_score = self._calculate_overall_reliability(components)

calculation_time = time.time() - start_time

            # Create result
result = UnifiedConfidenceResult(
                unified_confidence=unified_confidence,
components=components,
weights=self.weight_coefficients.copy(),
                calculation_time=calculation_time,
reliability_score=reliability_score,
metadata={
'total_components': len(components),
                    'calculation_id': self.total_calculations
}


            # Update performance tracking
self.total_calculations += 1
self.calculation_history.append(result)
            self.last_calculation_time = time.time()

            # Maintain history size
            if len(self.calculation_history) > self.config.get('max_history_size', 1000):
                self.calculation_history = self.calculation_history[-self.config.get('max_history_size', 1000):]

logger.debug(f"Unified confidence calculated: {unified_confidence:.3f} "
                        f"(reliability: {reliability_score:.3f})")

            return result

        except Exception as e:
logger.error(f"Error calculating unified confidence: {e}")
            # Return fallback result
            return UnifiedConfidenceResult(
                unified_confidence=0.5,  # Neutral confidence
components={},
weights=self.weight_coefficients.copy(),
                calculation_time=time.time() - start_time,
                reliability_score=0.0,
metadata={'error': str(e)}


    def _calculate_backlog_confidence(self, backlog_state: Dict[str, Any]) -> float:
        """Calculate confidence from historical backlog data."""
        try:
            # Extract relevant backlog metrics
total_trades = backlog_state.get('total_trades', 0)
            winning_trades = backlog_state.get('winning_trades', 0)
            avg_profit = backlog_state.get('avg_profit', 0.0)
            recent_performance = backlog_state.get('recent_performance', 0.5)

            # Calculate win rate
win_rate = winning_trades / unified_math.max(total_trades, 1)

            # Calculate profit factor
profit_factor = unified_math.min(avg_profit / 1000.0, 1.0)  # Normalize to [0, 1]

            # Combine factors
confidence = (win_rate * 0.4 + profit_factor * 0.3 + recent_performance * 0.3)

            # Cache result
cache_key = f"{total_trades}_{winning_trades}_{avg_profit:.2f}"
self.backlog_confidence_cache[cache_key] = confidence

            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
logger.error(f"Error calculating backlog confidence: {e}")
            return 0.5  # Neutral confidence

    def _calculate_ferris_wheel_confidence(self, ferris_wheel_position: int) -> float:
        """Calculate confidence from Ferris wheel cycle position."""
        try:
            # Normalize position to [0, 1] range (assuming 8-position wheel)
            normalized_position = (ferris_wheel_position % 8) / 8.0

            # Calculate confidence based on position
            # Higher confidence at optimal positions (0, 2, 4, 6)
            optimal_positions = [0, 2, 4, 6]
distance_to_optimal = unified_math.min(unified_math.abs(ferris_wheel_position % 8 - pos) for pos in optimal_positions)

            # Confidence decreases with distance from optimal positions
position_confidence = 1.0 - (distance_to_optimal / 4.0)

            # Add cycle momentum factor
cycle_momentum = np.unified_math.sin(2 * np.pi * normalized_position) * 0.2

confidence = position_confidence + cycle_momentum

            # Cache result
self.ferris_wheel_confidence_map[ferris_wheel_position] = confidence

            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
logger.error(f"Error calculating Ferris wheel confidence: {e}")
            return 0.5  # Neutral confidence

    def _calculate_ai_consensus_confidence(self, ai_consensus: Dict[str, float]) -> float:
        """Calculate confidence from AI consensus data."""
        try:
            # Extract AI confidence scores
chatgpt_confidence = ai_consensus.get('chatgpt', {}).get('confidence', 0.5)
            claude_confidence = ai_consensus.get('claude', {}).get('confidence', 0.5)
            gemini_confidence = ai_consensus.get('gemini', {}).get('confidence', 0.5)

            # Calculate agreement level
confidences = [chatgpt_confidence, claude_confidence, gemini_confidence]
agreement_variance = unified_math.unified_math.var(confidences)
            agreement_factor = 1.0 / (1.0 + agreement_variance)

            # Calculate average confidence
avg_confidence = unified_math.unified_math.mean(confidences)

            # Combine agreement and confidence
consensus_confidence = avg_confidence * agreement_factor

            # Cache result
cache_key = f"{chatgpt_confidence:.3f}_{claude_confidence:.3f}_{gemini_confidence:.3f}"
self.ai_consensus_confidence_weights[cache_key] = consensus_confidence

            return unified_math.max(0.0, unified_math.min(1.0, consensus_confidence))

        except Exception as e:
logger.error(f"Error calculating AI consensus confidence: {e}")
            return 0.5  # Neutral confidence

    def _calculate_matrix_controller_confidence(self, matrix_controller_state: Dict[str, Any]) -> float:
        """Calculate confidence from matrix controller state."""
        try:
            # Extract matrix state information
bit_level = matrix_controller_state.get('bit_level', '4bit')
            phase = matrix_controller_state.get('phase', 'INIT')
            confidence_score = matrix_controller_state.get('confidence_score', 0.5)
            fallback_triggered = matrix_controller_state.get('fallback_triggered', False)

            # Base confidence from controller
base_confidence = confidence_score

            # Adjust for bit level complexity
bit_level_confidence = {
'4bit': 0.8,   # Simple, reliable
'8bit': 0.7,   # Moderate complexity
'16bit': 0.6,  # High complexity
'42bit': 0.5   # Maximum complexity
}.get(bit_level, 0.5)

            # Adjust for phase stability
phase_confidence = {
'INIT': 0.6,      # Initialization
'ACCUM': 0.7,     # Accumulation
'RESON': 0.8,     # Resonance
'DISP': 0.5,      # Dispersion
'CONV': 0.9,      # Convergence
'42P': 0.4        # 42-bit phase
}.get(phase, 0.5)

            # Penalty for fallback
fallback_penalty = 0.2 if fallback_triggered else 0.0

            # Combine factors
confidence = (base_confidence * 0.4 +
                         bit_level_confidence * 0.3 +
phase_confidence * 0.3 -
fallback_penalty)

            # Cache result
cache_key = f"{bit_level}_{phase}_{confidence_score:.3f}"
self.matrix_controller_confidence_states[cache_key] = confidence

            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
logger.error(f"Error calculating matrix controller confidence: {e}")
            return 0.5  # Neutral confidence

    def _calculate_event_impact_confidence(self, event_impact: EventImpact) -> float:
        """Calculate confidence from event impact data."""
        try:
            # Base confidence from event priority
priority_confidence = event_impact.priority / 10.0

            # Sentiment confidence
sentiment_confidence = unified_math.abs(event_impact.sentiment_score)

            # Relevance confidence
relevance_confidence = event_impact.relevance_score

            # Time decay factor
time_diff = time.time() - event_impact.timestamp
            time_decay = unified_math.exp(-time_diff / 3600)  # 1-hour decay

            # Combine factors
confidence = (priority_confidence * 0.4 +
                         sentiment_confidence * 0.3 +
relevance_confidence * 0.2 +
time_decay * 0.1)

            # Cache result
cache_key = event_impact.event_id
self.event_impact_confidence_cache[cache_key] = confidence

            return unified_math.max(0.0, unified_math.min(1.0, confidence))

        except Exception as e:
logger.error(f"Error calculating event impact confidence: {e}")
            return 0.5  # Neutral confidence

    def _combine_confidence_components(self, components: Dict[ConfidenceSource, ConfidenceComponent]) -> float:
        """Combine confidence components using weighted average."""
        if not components:
            return 0.5  # Neutral confidence

total_weighted_confidence = 0.0
total_weight = 0.0

        for component in components.values():
            weighted_confidence = component.value * component.weight * component.reliability
total_weighted_confidence += weighted_confidence
total_weight += component.weight * component.reliability

        if total_weight > 0:
            return total_weighted_confidence / total_weight
        else:
            return 0.5  # Neutral confidence

    def _calculate_overall_reliability(self, components: Dict[ConfidenceSource, ConfidenceComponent]) -> float:
        """Calculate overall reliability score."""
        if not components:
            return 0.0

reliabilities = [component.reliability for component in components.values()]
        return unified_math.unified_math.mean(reliabilities)

    def _calculate_backlog_reliability(self, backlog_state: Dict[str, Any]) -> float:
        """Calculate reliability of backlog data."""
        try:
            # Factors affecting reliability
data_freshness = backlog_state.get('data_freshness', 0.5)
            data_completeness = backlog_state.get('data_completeness', 0.5)
            sample_size = backlog_state.get('total_trades', 0)

            # Sample size factor
sample_factor = unified_math.min(sample_size / 100.0, 1.0)

            # Combine factors
reliability = (data_freshness * 0.4 +
                          data_completeness * 0.3 +
sample_factor * 0.3)

            return unified_math.max(0.0, unified_math.min(1.0, reliability))

        except Exception as e:
logger.error(f"Error calculating backlog reliability: {e}")
            return 0.5

    def _calculate_ferris_reliability(self, ferris_wheel_position: int) -> float:
        """Calculate reliability of Ferris wheel data."""
        try:
            # Ferris wheel reliability is generally high
base_reliability = 0.9

            # Slight degradation for extreme positions
position_factor = 1.0 - abs((ferris_wheel_position % 8) - 4) / 8.0

            return base_reliability * position_factor

        except Exception as e:
logger.error(f"Error calculating Ferris wheel reliability: {e}")
            return 0.8

    def _calculate_ai_reliability(self, ai_consensus: Dict[str, float]) -> float:
        """Calculate reliability of AI consensus data."""
        try:
            # Check if all AI models provided data
models = ['chatgpt', 'claude', 'gemini']
available_models = sum(1 for model in models if model in ai_consensus)

            # Reliability based on model availability
availability_factor = available_models / len(models)

            # Agreement factor
confidences = [ai_consensus.get(model, {}).get('confidence', 0.5)
                          for model in models if model in ai_consensus]
agreement_variance = unified_math.unified_math.var(confidences) if confidences else 1.0
            agreement_factor = 1.0 / (1.0 + agreement_variance)

reliability = (availability_factor * 0.6 + agreement_factor * 0.4)

            return unified_math.max(0.0, unified_math.min(1.0, reliability))

        except Exception as e:
logger.error(f"Error calculating AI reliability: {e}")
            return 0.7

    def _calculate_matrix_reliability(self, matrix_controller_state: Dict[str, Any]) -> float:
        """Calculate reliability of matrix controller data."""
        try:
            # Matrix controller reliability is generally high
base_reliability = 0.85

            # Penalty for fallback mode
fallback_penalty = 0.2 if matrix_controller_state.get('fallback_triggered', False) else 0.0

            # Phase stability factor
phase = matrix_controller_state.get('phase', 'INIT')
            phase_stability = {
'INIT': 0.8, 'ACCUM': 0.9, 'RESON': 0.95,
'DISP': 0.7, 'CONV': 0.9, '42P': 0.6
}.get(phase, 0.8)

reliability = base_reliability * phase_stability - fallback_penalty

            return unified_math.max(0.0, unified_math.min(1.0, reliability))

        except Exception as e:
logger.error(f"Error calculating matrix reliability: {e}")
            return 0.8

    def _calculate_event_reliability(self, event_impact: EventImpact) -> float:
        """Calculate reliability of event impact data."""
        try:
            # Base reliability
base_reliability = 0.8

            # Source reliability
source_reliability = {
'news_api': 0.9,
'market_data': 0.95,
'social_media': 0.6,
'unknown': 0.5
}.get(event_impact.source, 0.7)

            # Time decay
time_diff = time.time() - event_impact.timestamp
            time_factor = unified_math.exp(-time_diff / 7200)  # 2-hour decay

reliability = base_reliability * source_reliability * time_factor

            return unified_math.max(0.0, unified_math.min(1.0, reliability))

        except Exception as e:
logger.error(f"Error calculating event reliability: {e}")
            return 0.7

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the confidence matrix."""
        return {
'total_calculations': self.total_calculations,
'last_calculation_time': self.last_calculation_time,
'history_size': len(self.calculation_history),
            'cache_sizes': {
'backlog': len(self.backlog_confidence_cache),
                'ferris_wheel': len(self.ferris_wheel_confidence_map),
                'ai_consensus': len(self.ai_consensus_confidence_weights),
                'matrix_controller': len(self.matrix_controller_confidence_states),
                'event_impact': len(self.event_impact_confidence_cache)
            },
'weight_coefficients': self.weight_coefficients
}

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
'backlog_weight': 0.25,
'ferris_weight': 0.25,
'ai_weight': 0.25,
'matrix_weight': 0.25,
'event_weight': 0.1,
'max_history_size': 1000,
'cache_ttl': 3600  # 1 hour
}


# Global instance for easy access
unified_confidence_matrix = UnifiedConfidenceMatrix()


def calculate_unified_confidence(backlog_state: Optional[Dict[str, Any]] = None,
                               ferris_wheel_position: Optional[int] = None,
ai_consensus: Optional[Dict[str, float]] = None,
matrix_controller_state: Optional[Dict[str, Any]] = None,
event_impact: Optional[EventImpact] = None) -> UnifiedConfidenceResult:
"""Global function to calculate unified confidence."""
    return unified_confidence_matrix.calculate_unified_confidence(
        backlog_state, ferris_wheel_position, ai_consensus,
matrix_controller_state, event_impact



def get_confidence_performance_metrics() -> Dict[str, Any]:
    """Global function to get confidence performance metrics."""
    return unified_confidence_matrix.get_performance_metrics()
