# -*- coding: utf-8 -*-
"""
Shared Thought Types for Schwabot Core.

This module contains shared data structures and enums used across
multiple core modules to prevent circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DualisticState(Enum):
    """Represents the dualistic states of the thought engine."""

    LOGICAL = "logical"
    INTUITIVE = "intuitive"
    ALIF = "alif"  # Adaptive Learning Interference Filter


class CognitiveBias(Enum):
    """Represents different types of cognitive biases."""

    OVERCONFIDENCE = "overconfidence"
    HERDING = "herding"
    ANCHORING = "anchoring"
    CONFIRMATION = "confirmation"
    AVAILABILITY = "availability"


class AlifFeedback(Enum):
    """Represents ALIF feedback types."""

    ENHANCE_SIGNAL = "enhance_signal"
    MAINTAIN_SIGNAL = "maintain_signal"
    ATTENUATE_SIGNAL = "attenuate_signal"
    EMERGENCY_HALT = "emergency_halt"
    NO_FEEDBACK = "no_feedback"


@dataclass
class AlifFeedbackData:
    """Detailed ALIF feedback data structure."""
    
    volume_delta: float = 0.0
    resonance_delta: float = 0.0
    ai_feedback_score: float = 0.0
    error_correction: float = 0.0
    market_memory_score: float = 0.0
    adaptation_rate: float = 0.02
    feedback_confidence: float = 0.5
    routing_target: str = "cpu_2bit"
    feedback_type: AlifFeedback = AlifFeedback.NO_FEEDBACK


class BitGateType(Enum):
    """Bit gate types for lantern processing."""

    NULL_VECTOR = "NULL_VECTOR"
    LOW_TIER = "LOW_TIER"
    MID_TIER = "MID_TIER"
    PEAK_TIER = "PEAK_TIER"


class EntropyMode(Enum):
    """Entropy generation modes."""

    PROFIT_SYMBOLIC = "profit_symbolic"
    ENTROPY_RANDOM = "entropy_random"
    PATTERN_MATCH = "pattern_match"
    DUALISTIC_MAP = "dualistic_map"


class BrainTag(Enum):
    """
    Descriptive tags attached to ThoughtVectors to provide insight into
    the decision-making context.
    """

    # Thermal State Tags
    THERMAL_COOL = "thermal_cool"
    THERMAL_WARM = "thermal_warm"
    THERMAL_HOT = "thermal_hot"
    THERMAL_CRITICAL = "thermal_critical"

    # Cognitive State Tags
    LOGIC_DOMINANT = "logic_dominant"
    INTUITION_DOMINANT = "intuition_dominant"
    LOGIC_INTUITION_AGREEMENT = "logic_intuition_agreement"
    LOGIC_INTUITION_CONFLICT = "logic_intuition_conflict"

    # Confidence & Score Tags
    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    STRONG_SIGNAL_BUY = "strong_signal_buy"
    STRONG_SIGNAL_SELL = "strong_signal_sell"
    NEUTRAL_SIGNAL = "neutral_signal"

    # Historical Consultation Tags
    HISTORICAL_CONFIRMATION = "historical_confirmation"
    HISTORICAL_HESITATION = "historical_hesitation"
    NO_HISTORY_AVAILABLE = "no_history_available"

    # Bias & Mitigation Tags
    BIAS_DETECTED = "bias_detected"
    BIAS_MITIGATED = "bias_mitigated"

    # Market Condition Tags
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"

    # Whale Activity Tags
    WHALE_CONSENSUS_BUY = "whale_consensus_buy"
    WHALE_CONSENSUS_SELL = "whale_consensus_sell"
    WHALE_ACTIVITY_HIGH = "whale_activity_high"

    # New Tags from Backup Logic
    BIT_GATE_NULL = "bit_gate_null_vector"
    BIT_GATE_LOW = "bit_gate_low_tier"
    BIT_GATE_MID = "bit_gate_mid_tier"
    BIT_GATE_PEAK = "bit_gate_peak_tier"
    ENTROPY_WORD_PROFIT = "entropy_word_profit"
    ENTROPY_WORD_NAV = "entropy_word_nav"
    ENTROPY_WORD_MATH = "entropy_word_math"
    ENTROPY_WORD_DUAL = "entropy_word_dual"
    ENTROPY_WORD_CHAOS = "entropy_word_chaos"

    # ALIF State Tags
    ALIF_DOMINANT = "alif_dominant"
    ALIF_ACTIVATED = "alif_activated"
    ALIF_CORRECTION = "alif_correction"
    ALIF_ENHANCE_SIGNAL = "alif_enhance_signal"
    ALIF_MAINTAIN_SIGNAL = "alif_maintain_signal"
    ALIF_ATTENUATE_SIGNAL = "alif_attenuate_signal"
    ALIF_EMERGENCY_HALT = "alif_emergency_halt"


@dataclass
class ThoughtVector:
    """Represents a 32-bit thought vector for decision-making."""

    timestamp: float
    state: DualisticState
    thermal_state: str
    logical_score: float
    intuitive_score: float
    historical_adjustment: float
    combined_score: float
    decision: str
    confidence: float
    thought_hash_32bit: str
    bias_mitigated: Optional[CognitiveBias] = None
    historical_consultation: Dict[str, Any] = field(default_factory=dict)
    tags: List[BrainTag] = field(default_factory=list)
    # ALIF-specific fields
    alif_feedback: Optional[AlifFeedbackData] = None
    alif_score: float = 0.0
    alif_decision: Optional[str] = None


@dataclass
class BitGate:
    """Bit gate for processing states through tier navigation."""

    gate_type: BitGateType
    emoji: str
    intensity_multiplier: float = 1.0

    def process_state(self, score: float, decision: str) -> tuple[float, str]:
        """Applies the gate's intensity to a score."""
        return score * self.intensity_multiplier, decision


# Export all types
__all__ = [
    "DualisticState",
    "CognitiveBias",
    "AlifFeedback",
    "AlifFeedbackData",
    "BitGateType",
    "EntropyMode",
    "BrainTag",
    "ThoughtVector",
    "BitGate",
] 