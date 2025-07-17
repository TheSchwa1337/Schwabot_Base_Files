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

    def _default_config(self) -> Dict[str, Any]:
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

    def analyze_bit_phase(self, value: float) -> BitPhase:
        """Analyze bit phase from a continuous value."""
        if value < 0 or value > 1:
            return self._create_default_bit_phase()
        
        try:
            thresholds = self.config["phase_thresholds"]
            
            # Determine phase state
            if value < thresholds["zero_threshold"]:
                phase_state = PhaseState.ZERO
                bit_value = 0
            elif value > thresholds["one_threshold"]:
                phase_state = PhaseState.ONE
                bit_value = 1
            elif abs(value - 0.5) < thresholds["transition_threshold"]:
                phase_state = PhaseState.TRANSITION
                bit_value = 1 if value > 0.5 else 0
            else:
                phase_state = PhaseState.UNCERTAIN
                bit_value = 1 if value > 0.5 else 0
            
            # Calculate confidence and transition probability
            confidence = self._calculate_phase_confidence(value, phase_state)
            transition_probability = self._calculate_transition_probability(value)
            
            bit_phase = BitPhase(
                timestamp=time.time(),
                phase_state=phase_state,
                bit_value=bit_value,
                confidence=confidence,
                transition_probability=transition_probability,
                metadata={"raw_value": value}
            )
            
            self._update_history(bit_phase)
            self.total_bit_operations += 1
            
            return bit_phase
            
        except Exception as e:
            logger.error(f"Error analyzing bit phase: {e}")
            return self._create_default_bit_phase()

    def _create_default_bit_phase(self) -> BitPhase:
        """Create default bit phase."""
        return BitPhase(
            timestamp=time.time(),
            phase_state=PhaseState.UNCERTAIN,
            bit_value=0,
            confidence=0.5,
            transition_probability=0.5,
            metadata={"default": True}
        )

    def _calculate_phase_confidence(self, value: float, phase_state: PhaseState) -> float:
        """Calculate confidence in phase state."""
        try:
            thresholds = self.config["phase_thresholds"]
            
            if phase_state == PhaseState.ZERO:
                # Higher confidence when closer to 0
                confidence = 1.0 - (value / thresholds["zero_threshold"])
            elif phase_state == PhaseState.ONE:
                # Higher confidence when closer to 1
                confidence = (value - thresholds["one_threshold"]) / (1.0 - thresholds["one_threshold"])
            elif phase_state == PhaseState.TRANSITION:
                # Lower confidence during transitions
                confidence = 0.5 - abs(value - 0.5) / thresholds["transition_threshold"]
            else:  # UNCERTAIN
                confidence = 0.5
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating phase confidence: {e}")
            return 0.5

    def _calculate_transition_probability(self, value: float) -> float:
        """Calculate probability of phase transition."""
        try:
            # Higher probability near 0.5 (transition zone)
            distance_from_center = abs(value - 0.5)
            transition_prob = max(0.0, 1.0 - (distance_from_center / 0.5))
            
            # Add some randomness for realistic transitions
            transition_prob *= 0.8 + 0.2 * (time.time() % 1.0)
            
            return max(0.0, min(1.0, transition_prob))
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {e}")
            return 0.5

    def _update_history(self, bit_phase: BitPhase):
        """Update bit history."""
        self.bit_history.append(bit_phase)
        if len(self.bit_history) > self.max_history_size:
            self.bit_history.pop(0)

        self.current_phase = bit_phase.phase_state
        self.last_update = bit_phase.timestamp

    def create_bit_sequence(self, values: List[float]) -> BitSequence:
        """Create bit sequence from continuous values."""
        try:
            sequence_id = f"bit_seq_{int(time.time() * 1000)}"
            bits = []
            phases = []
            
            for value in values:
                bit_phase = self.analyze_bit_phase(value)
                bits.append(bit_phase.bit_value)
                phases.append(bit_phase)
            
            sequence = BitSequence(
                sequence_id=sequence_id,
                bits=bits,
                phases=phases,
                length=len(bits),
                timestamp=time.time(),
                metadata={"source_values_count": len(values)}
            )
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error creating bit sequence: {e}")
            return BitSequence(
                sequence_id=f"error_seq_{int(time.time() * 1000)}",
                bits=[],
                phases=[],
                length=0,
                timestamp=time.time(),
                metadata={"error": str(e)}
            )

    def detect_patterns(self, bit_sequence: List[int]) -> List[BitPattern]:
        """Detect patterns in bit sequence."""
        try:
            patterns = []
            
            # Detect different types of patterns
            repeating_patterns = self._detect_repeating_patterns(bit_sequence)
            alternating_patterns = self._detect_alternating_patterns(bit_sequence)
            trend_patterns = self._detect_trend_patterns(bit_sequence)
            
            patterns.extend(repeating_patterns)
            patterns.extend(alternating_patterns)
            patterns.extend(trend_patterns)
            
            # Filter by confidence threshold
            threshold = self.config["pattern_confidence_threshold"]
            filtered_patterns = [p for p in patterns if p.confidence >= threshold]
            
            # Update pattern history
            for pattern in filtered_patterns:
                self.pattern_history.append(pattern)
                if len(self.pattern_history) > self.max_patterns:
                    self.pattern_history.pop(0)
            
            self.total_patterns_found += len(filtered_patterns)
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    def _detect_repeating_patterns(self, bits: List[int]) -> List[BitPattern]:
        """Detect repeating patterns in bit sequence."""
        patterns = []
        
        try:
            min_pattern_length = 2
            max_pattern_length = min(8, len(bits) // 2)
            
            for pattern_length in range(min_pattern_length, max_pattern_length + 1):
                for start_idx in range(len(bits) - pattern_length * 2 + 1):
                    pattern = bits[start_idx:start_idx + pattern_length]
                    next_pattern = bits[start_idx + pattern_length:start_idx + pattern_length * 2]
                    
                    if pattern == next_pattern:
                        # Calculate confidence based on pattern repetition
                        repetitions = 1
                        for i in range(start_idx + pattern_length * 2, len(bits) - pattern_length + 1, pattern_length):
                            if bits[i:i + pattern_length] == pattern:
                                repetitions += 1
                            else:
                                break
                        
                        confidence = min(1.0, repetitions / 3.0)  # Max confidence at 3+ repetitions
                        
                        pattern_obj = BitPattern(
                            pattern_id=f"repeat_{start_idx}_{pattern_length}_{int(time.time() * 1000)}",
                            pattern_type="repeating",
                            confidence=confidence,
                            start_index=start_idx,
                            end_index=start_idx + pattern_length * repetitions,
                            bit_sequence=pattern * repetitions,
                            metadata={"pattern_length": pattern_length, "repetitions": repetitions}
                        )
                        patterns.append(pattern_obj)
                        
        except Exception as e:
            logger.error(f"Error detecting repeating patterns: {e}")
        
        return patterns

    def _detect_alternating_patterns(self, bits: List[int]) -> List[BitPattern]:
        """Detect alternating patterns in bit sequence."""
        patterns = []
        
        try:
            min_length = 4
            
            for start_idx in range(len(bits) - min_length + 1):
                for length in range(min_length, min(16, len(bits) - start_idx + 1)):
                    sequence = bits[start_idx:start_idx + length]
                    
                    # Check for alternating 0-1 pattern
                    is_alternating = True
                    for i in range(1, len(sequence)):
                        if sequence[i] == sequence[i-1]:
                            is_alternating = False
                            break
                    
                    if is_alternating and len(sequence) >= min_length:
                        confidence = min(1.0, len(sequence) / 8.0)  # Higher confidence for longer patterns
                        
                        pattern_obj = BitPattern(
                            pattern_id=f"alt_{start_idx}_{length}_{int(time.time() * 1000)}",
                            pattern_type="alternating",
                            confidence=confidence,
                            start_index=start_idx,
                            end_index=start_idx + length,
                            bit_sequence=sequence,
                            metadata={"pattern_length": length}
                        )
                        patterns.append(pattern_obj)
                        
        except Exception as e:
            logger.error(f"Error detecting alternating patterns: {e}")
        
        return patterns

    def _detect_trend_patterns(self, bits: List[int]) -> List[BitPattern]:
        """Detect trend patterns in bit sequence."""
        patterns = []
        
        try:
            min_length = 3
            
            for start_idx in range(len(bits) - min_length + 1):
                for length in range(min_length, min(12, len(bits) - start_idx + 1)):
                    sequence = bits[start_idx:start_idx + length]
                    
                    # Check for upward trend (increasing 1s)
                    ones_count = sum(sequence)
                    if ones_count > length / 2 and ones_count < length:
                        # Check if 1s are generally increasing
                        trend_score = 0
                        for i in range(1, len(sequence)):
                            if sequence[i] >= sequence[i-1]:
                                trend_score += 1
                        
                        if trend_score >= len(sequence) * 0.7:  # 70% of transitions are non-decreasing
                            confidence = min(1.0, (ones_count / length) * (trend_score / len(sequence)))
                            
                            pattern_obj = BitPattern(
                                pattern_id=f"trend_up_{start_idx}_{length}_{int(time.time() * 1000)}",
                                pattern_type="trend_up",
                                confidence=confidence,
                                start_index=start_idx,
                                end_index=start_idx + length,
                                bit_sequence=sequence,
                                metadata={"ones_count": ones_count, "trend_score": trend_score}
                            )
                            patterns.append(pattern_obj)
                    
                    # Check for downward trend (decreasing 1s)
                    zeros_count = length - ones_count
                    if zeros_count > length / 2 and zeros_count < length:
                        # Check if 0s are generally increasing
                        trend_score = 0
                        for i in range(1, len(sequence)):
                            if sequence[i] <= sequence[i-1]:
                                trend_score += 1
                        
                        if trend_score >= len(sequence) * 0.7:  # 70% of transitions are non-increasing
                            confidence = min(1.0, (zeros_count / length) * (trend_score / len(sequence)))
                            
                            pattern_obj = BitPattern(
                                pattern_id=f"trend_down_{start_idx}_{length}_{int(time.time() * 1000)}",
                                pattern_type="trend_down",
                                confidence=confidence,
                                start_index=start_idx,
                                end_index=start_idx + length,
                                bit_sequence=sequence,
                                metadata={"zeros_count": zeros_count, "trend_score": trend_score}
                            )
                            patterns.append(pattern_obj)
                        
        except Exception as e:
            logger.error(f"Error detecting trend patterns: {e}")
        
        return patterns

    def get_bit_summary(self) -> Dict[str, Any]:
        """Get summary of bit operations."""
        if not self.bit_history:
            return {"status": "no_data"}
        
        recent_phases = self.bit_history[-10:]
        
        return {
            "current_phase": self.current_phase.value,
            "last_update": self.last_update,
            "total_operations": self.total_bit_operations,
            "total_patterns": self.total_patterns_found,
            "recent_confidence_avg": sum(p.confidence for p in recent_phases) / len(recent_phases),
            "recent_transition_prob_avg": sum(p.transition_probability for p in recent_phases) / len(recent_phases),
            "history_size": len(self.bit_history),
            "pattern_history_size": len(self.pattern_history),
        }

    def get_recent_patterns(self, count: int = 10) -> List[Dict[str, Any]]:
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
                "metadata": pattern.metadata,
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
