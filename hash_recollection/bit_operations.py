import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from core.unified_math_system import unified_math

#!/usr/bin/env python3
"""
Bit Operations Module
====================

Provides bit-level operations for trading data analysis, including
phase state management, bit manipulation, and binary pattern recognition.
Integrates with the unified math system and provides API endpoints.
"""


# Import unified math system
try:

    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False

    # Fallback math functions
    def unified_math(operation: str, *args, **kwargs):
        """Fallback unified math function."""
        if operation == "bit_phase":
            return np.random.random()  # Placeholder
        return 0.0


logger = logging.getLogger(__name__)


class PhaseState(Enum):
    """Phase state enumeration for bit operations."""

    ZERO = "zero"
    ONE = "one"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


@dataclass
class BitPhase:
    """Bit phase information."""

    timestamp: float
    phase_state: PhaseState
    bit_value: int
    confidence: float
    transition_probability: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BitSequence:
    """Bit sequence container."""

    sequence_id: str
    bits: List[int]
    phases: List[BitPhase]
    length: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BitPattern:
    """Bit pattern recognition result."""

    pattern_id: str
    pattern_type: str
    confidence: float
    start_index: int
    end_index: int
    bit_sequence: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BitOperations:
    """
    Bit Operations for trading data analysis.

    Provides bit-level analysis, phase state management, and
    pattern recognition for trading signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize bit operations."""
        self.config = config or self._default_config()

        # Bit tracking
        self.bit_history: List[BitPhase] = []
        self.max_history_size = self.config.get("max_history_size", 1000)

        # Pattern recognition
        self.pattern_history: List[BitPattern] = []
        self.max_patterns = self.config.get("max_patterns", 500)

        # Performance tracking
        self.total_bit_operations = 0
        self.total_patterns_found = 0

        # State management
        self.current_phase = PhaseState.UNCERTAIN
        self.last_update = time.time()

        logger.info("🔢 Bit Operations initialized")

    def _default_config():-> Dict[str, Any]:
        """Default configuration."""
        return {
            "max_history_size": 1000,
            "max_patterns": 500,
            "phase_thresholds": {
                "zero_threshold": 0.3,
                "one_threshold": 0.7,
                "transition_threshold": 0.1,
            },
            "pattern_confidence_threshold": 0.6,
            "sequence_length": 8,
            "pattern_detection_window": 32,
        }

    def analyze_bit_phase():-> BitPhase:
        """
        Analyze bit phase from a continuous value.

        Args:
            value: Continuous value to convert to bit phase

        Returns:
            BitPhase object with analysis results
        """
        try:
            # Normalize value to [0, 1]
            normalized_value = np.clip(value, 0.0, 1.0)

            # Determine bit value (0 or 1)
            bit_value = 1 if normalized_value > 0.5 else 0

            # Calculate phase state
            thresholds = self.config["phase_thresholds"]

            if normalized_value < thresholds["zero_threshold"]:
                phase_state = PhaseState.ZERO
            elif normalized_value > thresholds["one_threshold"]:
                phase_state = PhaseState.ONE
            elif abs(normalized_value - 0.5) < thresholds["transition_threshold"]:
                phase_state = PhaseState.TRANSITION
            else:
                phase_state = PhaseState.UNCERTAIN

            # Calculate confidence and transition probability
            confidence = self._calculate_phase_confidence(normalized_value, phase_state)
            transition_probability = self._calculate_transition_probability(
                normalized_value
            )

            bit_phase = BitPhase(
                timestamp=time.time(),
                phase_state=phase_state,
                bit_value=bit_value,
                confidence=confidence,
                transition_probability=transition_probability,
            )

            # Update history
            self._update_history(bit_phase)
            self.total_bit_operations += 1

            return bit_phase

        except Exception as e:
            logger.error(f"Error analyzing bit phase: {e}")
            return self._create_default_bit_phase()

    def _create_default_bit_phase():-> BitPhase:
        """Create default bit phase."""
        return BitPhase(
            timestamp=time.time(),
            phase_state=PhaseState.UNCERTAIN,
            bit_value=0,
            confidence=0.5,
            transition_probability=0.5,
        )

    def _calculate_phase_confidence():-> float:
        """Calculate confidence in phase state."""
        if phase_state == PhaseState.ZERO:
            return 1.0 - value
        elif phase_state == PhaseState.ONE:
            return value
        elif phase_state == PhaseState.TRANSITION:
            return 1.0 - abs(value - 0.5) * 2
        else:
            return 0.5

    def _calculate_transition_probability():-> float:
        """Calculate probability of phase transition."""
        # Higher probability near 0.5
        return 1.0 - abs(value - 0.5) * 2

    def _update_history(self, bit_phase: BitPhase):
        """Update bit history."""
        self.bit_history.append(bit_phase)
        if len(self.bit_history) > self.max_history_size:
            self.bit_history.pop(0)

        self.current_phase = bit_phase.phase_state
        self.last_update = bit_phase.timestamp

    def create_bit_sequence():-> BitSequence:
        """
        Create bit sequence from continuous values.

        Args:
            values: List of continuous values

        Returns:
            BitSequence object
        """
        try:
            bits = []
            phases = []

            for value in values:
                bit_phase = self.analyze_bit_phase(value)
                bits.append(bit_phase.bit_value)
                phases.append(bit_phase)

            sequence = BitSequence(
                sequence_id=f"seq_{int(time.time() * 1000)}",
                bits=bits,
                phases=phases,
                length=len(bits),
                timestamp=time.time(),
            )

            return sequence

        except Exception as e:
            logger.error(f"Error creating bit sequence: {e}")
            return BitSequence(
                sequence_id="error", bits=[], phases=[], length=0, timestamp=time.time()
            )

    def detect_patterns():-> List[BitPattern]:
        """
        Detect patterns in bit sequence.

        Args:
            bit_sequence: List of bits (0s and 1s)

        Returns:
            List of detected patterns
        """
        patterns = []

        try:
            if len(bit_sequence) < 4:
                return patterns

            # Detect common patterns
            patterns.extend(self._detect_repeating_patterns(bit_sequence))
            patterns.extend(self._detect_alternating_patterns(bit_sequence))
            patterns.extend(self._detect_trend_patterns(bit_sequence))

            # Update pattern history
            for pattern in patterns:
                self.pattern_history.append(pattern)
                if len(self.pattern_history) > self.max_patterns:
                    self.pattern_history.pop(0)

            self.total_patterns_found += len(patterns)

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        return patterns

    def _detect_repeating_patterns():-> List[BitPattern]:
        """Detect repeating patterns in bit sequence."""
        patterns = []

        for pattern_length in range(2, min(8, len(bits) // 2)):
            for start in range(len(bits) - pattern_length * 2):
                pattern = bits[start : start + pattern_length]
                next_pattern = bits[start + pattern_length : start + pattern_length * 2]

                if pattern == next_pattern:
                    # Check if pattern continues
                    repeat_count = 1
                    for i in range(
                        start + pattern_length * 2,
                        len(bits) - pattern_length + 1,
                        pattern_length,
                    ):
                        if bits[i : i + pattern_length] == pattern:
                            repeat_count += 1
                        else:
                            break

                    if repeat_count >= 2:
                        confidence = min(repeat_count / 4.0, 1.0)
                        patterns.append(
                            BitPattern(
                                pattern_id=f"repeat_{start}_{pattern_length}",
                                pattern_type="repeating",
                                confidence=confidence,
                                start_index=start,
                                end_index=start + repeat_count * pattern_length,
                                bit_sequence=pattern * repeat_count,
                            )
                        )

        return patterns

    def _detect_alternating_patterns():-> List[BitPattern]:
        """Detect alternating patterns in bit sequence."""
        patterns = []

        for start in range(len(bits) - 3):
            # Check for 0101 or 1010 patterns
            if len(bits) >= start + 4:
                segment = bits[start : start + 4]
                if segment in [[0, 1, 0, 1], [1, 0, 1, 0]]:
                    # Count alternating sequence
                    alt_count = 4
                    expected_next = 1 if segment[-1] == 0 else 0

                    for i in range(start + 4, len(bits)):
                        if bits[i] == expected_next:
                            alt_count += 1
                            expected_next = 1 if bits[i] == 0 else 0
                        else:
                            break

                    if alt_count >= 4:
                        confidence = min(alt_count / 8.0, 1.0)
                        patterns.append(
                            BitPattern(
                                pattern_id=f"alt_{start}",
                                pattern_type="alternating",
                                confidence=confidence,
                                start_index=start,
                                end_index=start + alt_count,
                                bit_sequence=bits[start : start + alt_count],
                            )
                        )

        return patterns

    def _detect_trend_patterns():-> List[BitPattern]:
        """Detect trend patterns in bit sequence."""
        patterns = []

        # Detect runs of 0s or 1s
        current_bit = bits[0]
        run_start = 0
        run_length = 1

        for i in range(1, len(bits)):
            if bits[i] == current_bit:
                run_length += 1
            else:
                if run_length >= 3:  # Minimum run length
                    confidence = min(run_length / 10.0, 1.0)
                    patterns.append(
                        BitPattern(
                            pattern_id=f"run_{run_start}_{current_bit}",
                            pattern_type=f"run_of_{current_bit}s",
                            confidence=confidence,
                            start_index=run_start,
                            end_index=run_start + run_length,
                            bit_sequence=bits[run_start : run_start + run_length],
                        )
                    )

                current_bit = bits[i]
                run_start = i
                run_length = 1

        # Check final run
        if run_length >= 3:
            confidence = min(run_length / 10.0, 1.0)
            patterns.append(
                BitPattern(
                    pattern_id=f"run_{run_start}_{current_bit}",
                    pattern_type=f"run_of_{current_bit}s",
                    confidence=confidence,
                    start_index=run_start,
                    end_index=run_start + run_length,
                    bit_sequence=bits[run_start : run_start + run_length],
                )
            )

        return patterns

    def get_bit_summary():-> Dict[str, Any]:
        """Get summary of bit operations."""
        if not self.bit_history:
            return {"status": "no_data"}

        recent_phases = self.bit_history[-10:]

        return {
            "current_phase": self.current_phase.value,
            "last_update": self.last_update,
            "total_operations": self.total_bit_operations,
            "total_patterns": self.total_patterns_found,
            "recent_confidence_avg": np.mean([p.confidence for p in recent_phases]),
            "recent_transition_prob_avg": np.mean(
                [p.transition_probability for p in recent_phases]
            ),
            "history_size": len(self.bit_history),
            "pattern_history_size": len(self.pattern_history),
        }

    def get_recent_patterns():-> List[Dict[str, Any]]:
        """Get recent detected patterns."""
        recent_patterns = self.pattern_history[-count:]
        return [
            {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "start_index": pattern.start_index,
                "end_index": pattern.end_index,
                "bit_sequence": pattern.bit_sequence,
            }
            for pattern in recent_patterns
        ]


# API Integration Functions
def create_bit_operations_api_endpoints(app):
    """Create FastAPI endpoints for bit operations."""
    if not hasattr(app, "bit_operations"):
        app.bit_operations = BitOperations()

    @app.post("/bit/analyze")
    async def analyze_bit_phase_endpoint(value: float):
        """Analyze bit phase from a continuous value."""
        try:
            bit_phase = app.bit_operations.analyze_bit_phase(value)
            return {
                "success": True,
                "phase_state": bit_phase.phase_state.value,
                "bit_value": bit_phase.bit_value,
                "confidence": bit_phase.confidence,
                "transition_probability": bit_phase.transition_probability,
                "timestamp": bit_phase.timestamp,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/bit/sequence")
    async def create_bit_sequence_endpoint(values: List[float]):
        """Create bit sequence from continuous values."""
        try:
            sequence = app.bit_operations.create_bit_sequence(values)
            return {
                "success": True,
                "sequence_id": sequence.sequence_id,
                "bits": sequence.bits,
                "length": sequence.length,
                "timestamp": sequence.timestamp,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/bit/patterns")
    async def detect_patterns_endpoint(bit_sequence: List[int]):
        """Detect patterns in bit sequence."""
        try:
            patterns = app.bit_operations.detect_patterns(bit_sequence)
            return {
                "success": True,
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "confidence": p.confidence,
                        "start_index": p.start_index,
                        "end_index": p.end_index,
                        "bit_sequence": p.bit_sequence,
                    }
                    for p in patterns
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/bit/summary")
    async def get_bit_summary_endpoint():
        """Get bit operations summary."""
        try:
            return {"success": True, "summary": app.bit_operations.get_bit_summary()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/bit/patterns")
    async def get_recent_patterns_endpoint(count: int = 10):
        """Get recent detected patterns."""
        try:
            return {
                "success": True,
                "patterns": app.bit_operations.get_recent_patterns(count),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return app
