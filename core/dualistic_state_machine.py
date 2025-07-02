import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from typing import Tuple
from typing import Callable



# -*- coding: utf-8 -*-
"""Dualistic State Machine - ALEPH/ALIF State Management Engine."

Implements the core dualistic logic for Schwabot's ALEPH and ALIF states,'
managing transitions, scoring systems (nibble/rittle), and quantum coherence
across different operational phases.

Mathematical Foundation:
- State Transition: S(t+1) = f(S(t), E(t), Q(t), N(t), R(t))
- Coherence Score: C = α * N + β * R + γ * Q_phase
- Differential Profit: ΔP = P_ALEPH - P_ALIF based on market conditions

Where:
- S(t): Current state (ALEPH/ALIF)
- E(t): Entropy level
- Q(t): Quantum phase
- N(t): Nibble score
- R(t): Rittle score"
"""

logger = logging.getLogger(__name__)


class StateType(Enum):"
    """Dualistic state types."""
"
ALEPH = "ALEPH"  # Precise, analytical, structured"
ALIF = "ALIF"  # Adaptive, intuitive, flexible"
TRANSITIONING = "TRANSITIONING"


class TransitionTrigger(Enum):"
    """Triggers for state transitions."""
"
ENTROPY_THRESHOLD = "entropy_threshold""
QUANTUM_PHASE_SHIFT = "quantum_phase_shift""
PROFIT_DIFFERENTIAL = "profit_differential""
MARKET_VOLATILITY = "market_volatility""
MANUAL_OVERRIDE = "manual_override""
NIBBLE_RITTLE_IMBALANCE = "nibble_rittle_imbalance"


@dataclass
class StateMetrics:"
    """Metrics for a specific dualistic state."""

activation_count: int = 0
total_duration: float = 0.0
avg_profit_per_trade: float = 0.0
avg_nibble_score: float = 0.0
avg_rittle_score: float = 0.0
success_rate: float = 0.0
quantum_coherence_avg: float = 0.0


@dataclass
class TransitionEvent:"
    """Record of a state transition."""

timestamp: float
from_state: StateType
to_state: StateType
trigger: TransitionTrigger
trigger_values: Dict[str, float]
coherence_before: float
coherence_after: float
confidence: float


@dataclass
class DualisticSnapshot:"
    """Complete snapshot of dualistic state."""

timestamp: float
current_state: StateType
nibble_score: float
rittle_score: float
quantum_phase: float
entropy_level: float
coherence_score: float
profit_differential: float
market_volatility: float
confidence: float


class DualisticStateMachine:"
    """Advanced state machine for ALEPH/ALIF dualistic management."""

def __init__(
self,:
entropy_threshold: float = 0.6,
        quantum_phase_sensitivity: float = 0.3,
        transition_cooldown_ms: float = 1000.0,
):"
        """Initialize the dualistic state machine."

Args:
            entropy_threshold: Entropy level triggering state evaluation
quantum_phase_sensitivity: Sensitivity to quantum phase changes
transition_cooldown_ms: Minimum time between state transitions"
"""
self.entropy_threshold = entropy_threshold
self.quantum_phase_sensitivity = quantum_phase_sensitivity
self.transition_cooldown_ms = transition_cooldown_ms

# Current state
self.current_state = StateType.ALEPH  # Start with ALEPH (analytical)
self.state_activation_time = time.time()
self.last_transition_time = 0.0

# Scoring components
self.nibble_score = 0.5
self.rittle_score = 0.5
self.quantum_phase = 0.0
self.entropy_level = 0.3
self.market_volatility = 0.02

# State history and metrics
self.state_history = deque(maxlen=1000)
self.transition_history = deque(maxlen=100)
self.metrics = {StateType.ALEPH: StateMetrics(), StateType.ALIF: StateMetrics()}

# Transition rules and weights
self.transition_weights = {"
"entropy": 0.3,"
            "quantum_phase": 0.25,"
            "nibble_rittle_balance": 0.2,"
            "profit_differential": 0.15,"
            "market_volatility": 0.1,
}

# Callbacks for external integration
self.transition_callbacks: List[Callable] = []

            logger.info("
f"🎭 Dualistic State Machine initialized in {"
self.current_state.value} state""
)

def update_scores(
self,:
nibble_score: float,
rittle_score: float,
quantum_phase: float,
entropy_level: float,
market_volatility: float = None,
) -> None:"
        """Update the core scoring components."

Args:
            nibble_score: Nibble scoring component (0.0 to 1.0)
            rittle_score: Rittle scoring component (0.0 to 1.0)
            quantum_phase: Quantum phase (0.0 to 1.0)
            entropy_level: Current entropy level (0.0 to 1.0)
market_volatility: Optional market volatility override"
"""
self.nibble_score = max(0.0, min(1.0, nibble_score))
        self.rittle_score = max(0.0, min(1.0, rittle_score))
        self.quantum_phase = quantum_phase % 1.0  # Keep in [0, 1) range
        self.entropy_level = max(0.0, min(1.0, entropy_level))

if market_volatility is not None:
            self.market_volatility = max(0.0, market_volatility)

# Evaluate for potential state transition
self._evaluate_transition()

# Create snapshot
snapshot = self._create_snapshot()
self.state_history.append(snapshot)

def calculate_coherence_score(self) -> float:"
        """Calculate overall coherence score."

Mathematical formula: C = α * N + β * R + γ * Q_phase + δ * (1 - E)"
"""
alpha = 0.3  # Nibble weight
        beta = 0.3  # Rittle weight
        gamma = 0.25  # Quantum phase weight
        delta = 0.15  # Entropy weight (inverted)

coherence = (
alpha * self.nibble_score
+ beta * self.rittle_score
+ gamma * math.sin(self.quantum_phase * 2 * math.pi)
+ delta * (1.0 - self.entropy_level)
)

        return max(0.0, min(1.0, coherence))

def calculate_profit_differential(self) -> float:"
        """Calculate profit differential between ALEPH and ALIF states."

Returns:
            Positive value favors ALEPH, negative favors ALIF"
"""
# ALEPH advantages: Low entropy, structured markets, high nibble scores
aleph_advantage = (
(1.0 - self.entropy_level) * 0.4  # Low entropy favors ALEPH
            + self.nibble_score * 0.3  # High nibble favors ALEPH
            + (1.0 - self.market_volatility) * 0.3  # Low volatility favors ALEPH
)

# ALIF advantages: High entropy, volatile markets, high rittle scores
alif_advantage = (
self.entropy_level * 0.4  # High entropy favors ALIF
            + self.rittle_score * 0.3  # High rittle favors ALIF
            + self.market_volatility * 0.3  # High volatility favors ALIF
)

        return aleph_advantage - alif_advantage
"
def force_transition(self, target_state: StateType, reason: str = "manual") -> bool:"
        """Force a transition to a specific state."

Args:
            target_state: Target state to transition to
reason: Reason for forced transition

Returns:
            True if transition was successful"
"""
if target_state == self.current_state:"
            logger.info(f"🎭 Already in {target_state.value} state")
        return True

if target_state == StateType.TRANSITIONING:"
            logger.warning("Cannot force transition to TRANSITIONING state")
        return False

# Execute transition
        success = self._execute_transition("
target_state, TransitionTrigger.MANUAL_OVERRIDE, {"reason": reason}
)

if success:
            logger.info("
f"🔄 Forced transition to {"
target_state.value}: {reason}""
)

        return success

def get_current_snapshot(self) -> DualisticSnapshot:"
        """Get current state snapshot."""
        return self._create_snapshot()

def get_state_recommendations(self) -> Dict[str, Any]:"
        """Get recommendations for optimal state based on current conditions."""
profit_diff = self.calculate_profit_differential()
coherence = self.calculate_coherence_score()

# Analyze current conditions
aleph_score = self._calculate_state_suitability(StateType.ALEPH)
alif_score = self._calculate_state_suitability(StateType.ALIF)

optimal_state = StateType.ALEPH if aleph_score > alif_score else StateType.ALIF
confidence = abs(aleph_score - alif_score)

        return {"
"current_state": self.current_state.value,"
"optimal_state": optimal_state.value,"
"confidence": confidence,"
"aleph_score": aleph_score,"
"alif_score": alif_score,"
"profit_differential": profit_diff,"
"coherence_score": coherence,"
"should_transition": optimal_state != self.current_state
and confidence > 0.3,"
"transition_urgency": (
confidence if optimal_state != self.current_state else 0.0
),
}

def add_transition_callback(:
self, callback: Callable[[TransitionEvent], None]
) -> None:"
        """Add callback to be called on state transitions."""
self.transition_callbacks.append(callback)

def _evaluate_transition(self) -> None:"
        """Evaluate whether a state transition should occur."""
current_time = time.time()

# Check cooldown
if (:
current_time - self.last_transition_time
< self.transition_cooldown_ms / 1000.0
):
            return

# Calculate trigger values
trigger_values = self._calculate_trigger_values()
transition_score = self._calculate_transition_score(trigger_values)

# Determine if transition should occur
if transition_score > 0.7:  # Strong signal for transition
target_state = (
StateType.ALIF
if self.current_state == StateType.ALEPH:
else StateType.ALEPH
)

# Determine primary trigger
primary_trigger = max(trigger_values.items(), key=lambda x: x[1])[0]
trigger_enum = self._get_trigger_enum(primary_trigger)

self._execute_transition(target_state, trigger_enum, trigger_values)

def _calculate_trigger_values(self) -> Dict[str, float]:"
        """Calculate values for all transition triggers."""
profit_diff = self.calculate_profit_differential()

        return {"
"entropy": self.entropy_level,"
"quantum_phase_change": abs(math.sin(self.quantum_phase * 2 * math.pi)),"
"nibble_rittle_imbalance": abs(self.nibble_score - self.rittle_score),"
"profit_differential": abs(profit_diff),"
"market_volatility": self.market_volatility,
}

def _calculate_transition_score(self, trigger_values: Dict[str, float]) -> float:"
        """Calculate overall transition score."""
weighted_score = 0.0

for trigger, value in trigger_values.items():
            weight = self.transition_weights.get(trigger, 0.0)

# Normalize trigger values to [0, 1] and apply thresholds"
            if trigger == "entropy":
                normalized = value if value > self.entropy_threshold else 0.0"
elif trigger == "quantum_phase_change":
                normalized = value if value > self.quantum_phase_sensitivity else 0.0"
elif trigger == "nibble_rittle_imbalance":
                normalized = value if value > 0.3 else 0.0"
            elif trigger == "profit_differential":
                normalized = value if value > 0.2 else 0.0"
elif trigger == "market_volatility":
                normalized = value if value > 0.05 else 0.0
else:
                normalized = value

weighted_score += weight * normalized

        return weighted_score

def _calculate_state_suitability(self, state: StateType): -> float:"
        """Calculate how suitable a given state is for current conditions."""
if state == StateType.ALEPH:
            # ALEPH favors: low entropy, high nibble, low volatility,
# structured patterns
suitability = (
(1.0 - self.entropy_level) * 0.3
                + self.nibble_score * 0.25
                + (1.0 - self.market_volatility) * 0.2
                + (1.0 - abs(self.nibble_score - self.rittle_score)) * 0.15
                + math.cos(self.quantum_phase * 2 * math.pi) * 0.1
)
elif state == StateType.ALIF:
            # ALIF favors: high entropy, high rittle, high volatility, adaptive
# patterns
suitability = (
self.entropy_level * 0.3
                + self.rittle_score * 0.25
                + self.market_volatility * 0.2
                + abs(self.nibble_score - self.rittle_score) * 0.15
                + abs(math.sin(self.quantum_phase * 2 * math.pi)) * 0.1
)
else:
            suitability = 0.0

        return max(0.0, min(1.0, suitability))

def _execute_transition(
self,:
target_state: StateType,
trigger: TransitionTrigger,
trigger_values: Dict[str, Any],
) -> bool:"
        """Execute a state transition."""
if target_state == self.current_state:
            return False

# Calculate coherence before and after
coherence_before = self.calculate_coherence_score()

# Update state metrics for current state
self._update_state_metrics()

# Create transition event
event = TransitionEvent(
timestamp=time.time(),
from_state=self.current_state,
to_state=target_state,
trigger=trigger,
trigger_values=trigger_values,
coherence_before=coherence_before,
coherence_after=0.0,  # Will be calculated after transition
confidence=self._calculate_transition_confidence(trigger_values),
)

# Execute the transition
old_state = self.current_state
self.current_state = target_state
self.state_activation_time = time.time()
self.last_transition_time = time.time()

# Calculate coherence after transition
event.coherence_after = self.calculate_coherence_score()

# Store transition event
self.transition_history.append(event)

# Update metrics
self.metrics[target_state].activation_count += 1

# Call callbacks
for callback in self.transition_callbacks:
            try:
                callback(event)
        except Exception as e:"
                logger.error(f"Error in transition callback: {e}")

            logger.info("
f"🎭 State transition: {
old_state.value} → {"
target_state.value} """
f"(trigger: {
trigger.value}, confidence: {"
event.confidence:.3f})""
)

        return True

def _update_state_metrics(self) -> None:"
        """Update metrics for the current state."""
current_metrics = self.metrics[self.current_state]

# Update duration
duration = time.time() - self.state_activation_time
current_metrics.total_duration += duration

# Update averages (simplified)
current_metrics.avg_nibble_score = (
current_metrics.avg_nibble_score * 0.9 + self.nibble_score * 0.1
)
current_metrics.avg_rittle_score = (
current_metrics.avg_rittle_score * 0.9 + self.rittle_score * 0.1
)
current_metrics.quantum_coherence_avg = (
current_metrics.quantum_coherence_avg * 0.9
            + self.calculate_coherence_score() * 0.1
)

def _calculate_transition_confidence(self, trigger_values: Dict[str, Any]) -> float:"
        """Calculate confidence in the transition decision."""
# Base confidence on trigger strength
primary_trigger_value = max(trigger_values.values()) if trigger_values else 0.0
        base_confidence = min(1.0, primary_trigger_value * 2)

# Adjust for coherence
coherence = self.calculate_coherence_score()
coherence_adjustment = coherence * 0.2

# Adjust for state consistency
consistency_adjustment = 0.1 if len(self.transition_history) < 5 else 0.0

        return min(1.0, base_confidence + coherence_adjustment + consistency_adjustment)

def _create_snapshot(self) -> DualisticSnapshot:"
        """Create a snapshot of the current dualistic state."""
        return DualisticSnapshot(
timestamp=time.time(),
current_state=self.current_state,
nibble_score=self.nibble_score,
rittle_score=self.rittle_score,
quantum_phase=self.quantum_phase,
entropy_level=self.entropy_level,
coherence_score=self.calculate_coherence_score(),
profit_differential=self.calculate_profit_differential(),
market_volatility=self.market_volatility,
confidence=self._calculate_current_confidence(),
)

def _calculate_current_confidence(self) -> float:"
        """Calculate confidence in the current state."""
coherence = self.calculate_coherence_score()
state_duration = time.time() - self.state_activation_time

# Higher confidence for longer stable periods and higher coherence
# Max confidence after 10 seconds
time_factor = min(1.0, state_duration / 10.0)

        return coherence * 0.7 + time_factor * 0.3

def _get_trigger_enum(self, trigger_name: str): -> TransitionTrigger:"
        """Convert trigger name to enum."""
mapping = {"
"entropy": TransitionTrigger.ENTROPY_THRESHOLD,"
"quantum_phase_change": TransitionTrigger.QUANTUM_PHASE_SHIFT,"
"profit_differential": TransitionTrigger.PROFIT_DIFFERENTIAL,"
"market_volatility": TransitionTrigger.MARKET_VOLATILITY,"
"nibble_rittle_imbalance": TransitionTrigger.NIBBLE_RITTLE_IMBALANCE,
}
        return mapping.get(trigger_name, TransitionTrigger.MANUAL_OVERRIDE)

def get_performance_stats(self) -> Dict[str, Any]:"
        """Get comprehensive performance statistics."""
total_transitions = len(self.transition_history)

stats = {"
"current_state": self.current_state.value,"
"state_duration": time.time() - self.state_activation_time,"
"total_transitions": total_transitions,"
"coherence_score": self.calculate_coherence_score(),"
"profit_differential": self.calculate_profit_differential(),"
"nibble_rittle_balance": abs(self.nibble_score - self.rittle_score),"
"state_metrics": {},
}

# Add metrics for each state
for state_type, metrics in self.metrics.items():
            if metrics.activation_count > 0:
                avg_duration = metrics.total_duration / metrics.activation_count"
stats["state_metrics"][state_type.value] = {"
"activations": metrics.activation_count,"
"avg_duration": avg_duration,"
"avg_nibble": metrics.avg_nibble_score,"
"avg_rittle": metrics.avg_rittle_score,"
"avg_coherence": metrics.quantum_coherence_avg,
}

# Transition statistics
if total_transitions > 0:
            recent_transitions = list(self.transition_history)[
-10:
            ]  # Last 10 transitions
trigger_counts = {}
for transition in recent_transitions:
                trigger = transition.trigger.value
trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
"
stats["recent_transition_triggers"] = trigger_counts"
stats["avg_transition_confidence"] = sum(
t.confidence for t in recent_transitions
) / len(recent_transitions)

        return stats


def main():"
    """Demonstrate dualistic state machine functionality."""
logging.basicConfig(level=logging.INFO)
"
print("🎭 Dualistic State Machine Demo")"
print("=" * 50)

# Initialize state machine
machine = DualisticStateMachine(
entropy_threshold=0.6,
        quantum_phase_sensitivity=0.3,
        transition_cooldown_ms=500.0,
)

# Add transition callback
def on_transition(event: TransitionEvent)::
        print("
f"  🔄 Transition callback: {
event.from_state.value} → {"
event.to_state.value}""
)
print("
f"      Trigger: {
event.trigger.value}, Confidence: {"
event.confidence:.3f}""
)

machine.add_transition_callback(on_transition)

# Simulate normal ALEPH-favorable conditions"
print("\n📊 Testing ALEPH-favorable conditions...")
machine.update_scores(
nibble_score=0.8,  # High analytical score
        rittle_score=0.4,  # Low adaptive score
        quantum_phase=0.0,  # Stable phase
        entropy_level=0.2,  # Low entropy
        market_volatility=0.01,  # Low volatility
)

snapshot = machine.get_current_snapshot()"
print(f"  State: {snapshot.current_state.value}")"
print(f"  Coherence: {snapshot.coherence_score:.3f}")"
print(f"  Profit Differential: {snapshot.profit_differential:.3f}")

# Get recommendations
recommendations = machine.get_state_recommendations()'"
print(f"  Recommended: {recommendations['optimal_state']}")'"
print(f"  Confidence: {recommendations['confidence']:.3f}")

# Simulate ALIF-favorable conditions"
print("\n📊 Testing ALIF-favorable conditions...")
machine.update_scores(
nibble_score=0.3,  # Low analytical score
        rittle_score=0.9,  # High adaptive score
        quantum_phase=0.7,  # Dynamic phase
        entropy_level=0.8,  # High entropy
        market_volatility=0.08,  # High volatility
)

time.sleep(0.6)  # Wait for cooldown

snapshot = machine.get_current_snapshot()"
print(f"  State: {snapshot.current_state.value}")"
print(f"  Coherence: {snapshot.coherence_score:.3f}")"
print(f"  Profit Differential: {snapshot.profit_differential:.3f}")

# Force transition test"
print("\n🔄 Testing forced transition...")"
success = machine.force_transition(StateType.ALEPH, "testing_purposes")"
print(f"  Forced transition success: {success}")

# Performance statistics"
print("\n📊 Performance Statistics:")
stats = machine.get_performance_stats()
for key, value in stats.items():
        if isinstance(value, dict):"
            print(f"  {key}:")
for sub_key, sub_value in value.items():"
                print(f"    {sub_key}: {sub_value}")
elif isinstance(value, float):"
            print(f"  {key}: {value:.4f}")
else:"
            print(f"  {key}: {value}")
"
print("\n✅ Dualistic State Machine demo completed!")

"
if __name__ == "__main__":
    main()
"
""""
"""'"