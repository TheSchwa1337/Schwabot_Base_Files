# -*- coding: utf-8 -*-
"""Fractal Core - Grayscale Collapse and Recursive Hash Structures.

Implements the core mathematical framework for:
- Sigmoid-weighted summation collapse: C(t) = ∑ C_i / (1 + e^(-Ωt))
- State collapse probability in recursive hash structures
- Golden ratio fractal command weighting: F(n) = F(n-1) × Φ
"""

import hashlib
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, getcontext

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    # Fallback to standard math if unified_math_system is not available
    import math as unified_math

# Set high precision for financial calculations
getcontext().prec = 28

# Golden ratio constant
PHI = (1 + unified_math.sqrt(5)) / 2


@dataclass
class FractalState:
    """Represents a state in the fractal collapse system."""
    state_id: str
    weight: float
    timestamp: float
    hash_value: str
    collapse_probability: float = 0.0
    recursive_depth: int = 0


@dataclass
class GrayscaleCollapseResult:
    """Result of grayscale collapse calculation."""
    collapsed_value: float
    contributing_states: List[FractalState]
    omega_coefficient: float
    timestamp: float
    confidence_score: float


class FractalCore:
    """Core fractal mathematics for grayscale collapse and hash structures."""
    def __init__(self, omega_base: float = 1.0) -> None:
        """Initialize fractal core with omega coefficient."""
        self.omega_base = omega_base
        self.active_states: Dict[str, FractalState] = {}
        self.collapse_history: List[GrayscaleCollapseResult] = []
        self.fractal_weights: Dict[int, float] = {}

    def add_fractal_state(
        self, state_id: str, weight: float, timestamp: float, data: Optional[str] = None
    ) -> FractalState:
        """Add a new fractal state to the system."""
        hash_value = self._generate_state_hash(state_id, timestamp, data)
        state = FractalState(
            state_id=state_id,
            weight=weight,
            timestamp=timestamp,
            hash_value=hash_value,
            recursive_depth=len(self.active_states),
        )
        # Calculate collapse probability
        state.collapse_probability = self._calculate_collapse_probability(state, timestamp)
        self.active_states[state_id] = state
        return state

    def grayscale_collapse(self, target_time: float) -> GrayscaleCollapseResult:
        """Perform grayscale collapse using sigmoid-weighted summation.

        C(t) = ∑ C_i / (1 + e^(-Ωt))
        """
        if not self.active_states:
            return GrayscaleCollapseResult(
                collapsed_value=0.0,
                contributing_states=[],
                omega_coefficient=self.omega_base,
                timestamp=target_time,
                confidence_score=0.0,
            )
        collapsed_value = 0.0
        contributing_states = []
        total_weight = 0.0
        for state in self.active_states.values():
            # Calculate sigmoid weight
            omega_t = self.omega_base * (target_time - state.timestamp)
            sigmoid_denominator = 1 + unified_math.exp(-omega_t)
            # Apply sigmoid-weighted summation
            weighted_contribution = state.weight / sigmoid_denominator
            collapsed_value += weighted_contribution
            # Track contributing states
            if weighted_contribution > 0.001:  # Threshold for significance
                contributing_states.append(state)
                total_weight += state.weight
        # Calculate confidence based on state convergence
        confidence_score = unified_math.min(1.0, total_weight / len(self.active_states))
        result = GrayscaleCollapseResult(
            collapsed_value=collapsed_value,
            contributing_states=contributing_states,
            omega_coefficient=self.omega_base,
            timestamp=target_time,
            confidence_score=confidence_score,
        )
        self.collapse_history.append(result)
        return result

    def calculate_fractal_command_weight(self, depth: int) -> float:
        """Calculate fractal command weight using golden ratio.

        F(n) = F(n-1) × Φ, where Φ = golden ratio
        """
        if depth in self.fractal_weights:
            return self.fractal_weights[depth]
        if depth <= 0:
            weight = 1.0
        elif depth == 1:
            weight = PHI
        else:
            # Recursive calculation with memoization
            weight = self.calculate_fractal_command_weight(depth - 1) * PHI
        self.fractal_weights[depth] = weight
        return weight

    def recursive_hash_structure(
        self, data: str, depth: int, salt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate recursive hash structure for state tracking."""
        if depth <= 0:
            return {"hash": hashlib.sha256(data.encode()).hexdigest(), "depth": 0}
        # Generate base hash
        base_data = f"{data}_{salt or ''}"
        base_hash = hashlib.sha256(base_data.encode()).hexdigest()
        # Recursive structure
        recursive_component = self.recursive_hash_structure(base_hash, depth - 1, salt)
        # Combine with fractal weighting
        fractal_weight = self.calculate_fractal_command_weight(depth)
        return {
            "hash": base_hash,
            "depth": depth,
            "fractal_weight": fractal_weight,
            "recursive_component": recursive_component,
            "collapse_probability": self._calculate_hash_collapse_probability(
                base_hash, fractal_weight
            ),
        }

    def get_state_entropy(self) -> float:
        """Calculate entropy of current fractal states."""
        if not self.active_states:
            return 0.0
        total_weight = sum(state.weight for state in self.active_states.values())
        if total_weight == 0:
            return 0.0
        entropy = 0.0
        for state in self.active_states.values():
            probability = state.weight / total_weight
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return entropy

    def prune_collapsed_states(self, threshold: float = 0.01) -> int:
        """Remove states with collapse probability below threshold."""
        pruned_count = 0
        states_to_remove = []
        for state_id, state in self.active_states.items():
            if state.collapse_probability < threshold:
                states_to_remove.append(state_id)
        for state_id in states_to_remove:
            del self.active_states[state_id]
            pruned_count += 1
        return pruned_count

    def _generate_state_hash(
        self, state_id: str, timestamp: float, data: Optional[str]
    ) -> str:
        """Generate hash for fractal state."""
        hash_input = f"{state_id}_{timestamp}_{data or ''}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _calculate_collapse_probability(
        self, state: FractalState, current_time: float
    ) -> float:
        """Calculate collapse probability for a fractal state."""
        time_delta = current_time - state.timestamp
        omega_factor = self.omega_base * time_delta
        # Sigmoid-based probability calculation
        probability = 1 / (1 + unified_math.exp(-omega_factor))
        # Apply fractal weighting
        fractal_weight = self.calculate_fractal_command_weight(state.recursive_depth)
        # Normalize by fractal weight (higher depth = lower collapse probability)
        normalized_probability = probability / (1 + unified_math.log(fractal_weight))
        return unified_math.min(1.0, unified_math.max(0.0, normalized_probability))

    def _calculate_hash_collapse_probability(
        self, hash_value: str, fractal_weight: float
    ) -> float:
        """Calculate collapse probability based on hash and fractal weight."""
        # Use hash entropy as base probability
        hash_int = int(hash_value[:8], 16)  # First 8 hex chars
        base_probability = (hash_int % 1000000) / 1000000.0
        # Adjust by fractal weight
        adjusted_probability = base_probability / (1 + unified_math.log(fractal_weight))
        return unified_math.min(1.0, unified_math.max(0.0, adjusted_probability))


class FractalCommandDispatcher:
    """Dispatches commands based on fractal weighting and trust scores."""
    def __init__(self, fractal_core: FractalCore) -> None:
        """Initialize with fractal core reference."""
        self.fractal_core = fractal_core
        self.command_history: List[Dict[str, Any]] = []
        self.trust_scores: Dict[str, float] = {}

    def dispatch_command(
        self, command_id: str, command_data: Dict[str, Any], recursive_depth: int = 1
    ) -> Dict[str, Any]:
        """Dispatch command with fractal weighting."""
        # Calculate fractal weight for command
        fractal_weight = self.fractal_core.calculate_fractal_command_weight(recursive_depth)
        # Get or calculate trust score
        trust_score = self._calculate_trust_score(command_id)
        # Combine fractal weight with trust
        execution_priority = fractal_weight * trust_score
        command_result = {
            "command_id": command_id,
            "fractal_weight": fractal_weight,
            "trust_score": trust_score,
            "execution_priority": execution_priority,
            "recursive_depth": recursive_depth,
            "timestamp": time.time(),
            "data": command_data,
        }
        self.command_history.append(command_result)
        return command_result

    def _calculate_trust_score(self, command_id: str) -> float:
        """Calculate trust score based on historical performance."""
        if command_id not in self.trust_scores:
            # Initialize with neutral trust
            self.trust_scores[command_id] = 0.5
        # Update based on recent performance (simplified)
        recent_commands = [
            cmd for cmd in self.command_history[-10:] if cmd["command_id"] == command_id
        ]
        if recent_commands:
            # Calculate success rate (placeholder logic)
            success_rate = len(recent_commands) / 10.0
            self.trust_scores[command_id] = unified_math.min(1.0, success_rate)
        return self.trust_scores[command_id]


# Convenience functions
def create_fractal_system(
    omega_base: float = 1.0,
) -> Tuple[FractalCore, FractalCommandDispatcher]:
    """Create integrated fractal system."""
    core = FractalCore(omega_base)
    dispatcher = FractalCommandDispatcher(core)
    return core, dispatcher


def calculate_grayscale_collapse(
    states: List[Tuple[str, float, float]], target_time: float, omega_base: float = 1.0
) -> GrayscaleCollapseResult:
    """Convenience function for grayscale collapse calculation."""
    core = FractalCore(omega_base)
    for state_id, weight, timestamp in states:
        core.add_fractal_state(state_id, weight, timestamp)
    return core.grayscale_collapse(target_time)
