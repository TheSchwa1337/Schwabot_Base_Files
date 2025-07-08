import asyncio
import hashlib
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from .fractal_memory_tracker import FractalMemoryTracker, FractalSnapshot, create_fractal_memory_tracker
from .unified_math_system import generate_unified_hash
from .phantom_detector import PhantomZone
from .phantom_registry import PhantomRegistry

import numpy as np

from utils.safe_print import safe_print, info, warn, error, success, debug

#!/usr/bin/env python3
"""
🧬 TWO-GRAM DETECTOR - SCHWABOT PATTERN RECOGNITION ENGINE
=========================================================

Advanced 2-gram pattern detection system that integrates with:
- Fractal memory tracking for pattern resonance
- Symbolic ASIC parity for Unicode/emoji mapping
- Entropy & T-cell health modeling for system protection
- Vector similarity matching via cosine logic
- Real-time burst detection for strategy triggers

This is Schwabot's DNA-level signal recognition layer.
"""

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """2-gram pattern classifications for trading signals."""

    VOLATILITY_BURST = "UD"  # Up-Down rapid reversal
    SWAP_PATTERN = "BE"  # BTC-ETH repetitive swap
    FLATLINE_ANOMALY = "AA"  # Low entropy repetition
    TREND_MOMENTUM = "UU"  # Sustained direction
    REVERSAL_SIGNAL = "DU"  # Down-Up reversal
    CONSOLIDATION = "CC"  # Sideways consolidation
    BREAKOUT_PULSE = "XR"  # Cross-asset breakout
    ENTROPY_SPIKE = "EE"  # High entropy event


class BurstIntensity(Enum):
    """Burst intensity levels for 2-gram frequency spikes."""

    LOW = 1  # Minimal pattern activity
    MEDIUM = 2  # Moderate pattern frequency
    HIGH = 3  # Strong pattern emergence
    CRITICAL = 4  # Extreme pattern dominance


@dataclass
class TwoGramSignal:
    """Standardized 2-gram signal packet for strategy routing."""

    pattern: str  # The 2-character pattern (e.g., "UD")
    frequency: int  # Occurrence count in window
    entropy: float  # Shannon entropy of pattern distribution
    burst_score: float  # Δ frequency intensity
    similarity_vector: List[float]  # Vector for cosine matching

    # Symbolic representation
    emoji_symbol: str  # Emoji/Unicode representation
    asic_hash: str  # ASIC-compatible hash code

    # Fractal integration
    fractal_resonance: Optional[float] = None  # Match with historical patterns
    fractal_confidence: Optional[float] = None  # Confidence in fractal match

    # Health modeling
    t_cell_activation: bool = False  # T-cell immune response triggered
    system_health_score: float = 1.0  # Overall system health (0-1)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    strategy_trigger: Optional[str] = None
    risk_level: str = "medium"
    execution_priority: int = 5  # 1-10 priority scale


@dataclass
class PatternMemory:
    """Memory structure for 2-gram pattern history."""

    pattern_frequencies: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    entropy_history: List[float] = field(default_factory=list)
    burst_events: List[Dict[str, Any]] = field(default_factory=list)
    profit_correlations: Dict[str, float] = field(default_factory=dict)
    t_cell_responses: List[Dict[str, Any]] = field(default_factory=list)


class TwoGramDetector:
    """
    Advanced 2-gram pattern detector with full Schwabot integration.

    This detector operates as Schwabot's signal DNA recognition layer,
    identifying micro-patterns that precede larger market movements.
    """

    def __init__(
        self,
        window_size: int = 100,
        burst_threshold: float = 2.0,
        similarity_threshold: float = 0.85,
        t_cell_sensitivity: float = 0.7,
        enable_fractal_memory: bool = True,
    ):
        """
        Initialize the 2-gram detector with full integration capabilities.

        Args:
            window_size: Rolling window for pattern analysis
            burst_threshold: Threshold for burst detection (sigma multiplier)
            similarity_threshold: Cosine similarity threshold for pattern matching
            t_cell_sensitivity: Sensitivity for T-cell immune responses
            enable_fractal_memory: Enable fractal memory integration
        """
        self.window_size = window_size
        self.burst_threshold = burst_threshold
        self.similarity_threshold = similarity_threshold
        self.t_cell_sensitivity = t_cell_sensitivity

        # Core pattern tracking
        self.freq_map: Dict[str, int] = defaultdict(int)
        self.pattern_history: deque = deque(maxlen=window_size)
        self.memory = PatternMemory()

        # Fractal integration
        self.fractal_memory = None
        if enable_fractal_memory:
            self.fractal_memory = create_fractal_memory_tracker(
                max_snapshots=1000, similarity_threshold=similarity_threshold
            )

        # Phantom Math integration
        self.phantom_registry = PhantomRegistry()

        # Symbolic mapping (Unicode/Emoji)
        self.symbol_map = self._initialize_symbol_map()

        # System state
        self.active_patterns: Dict[str, TwoGramSignal] = {}
        self.t_cell_active = False
        self.system_health = 1.0

        logger.info("🧬 Two-gram detector initialized with full Schwabot integration")

    def _initialize_symbol_map(self) -> Dict[str, str]:
        """Initialize symbolic representation mapping for patterns."""
        return {
            # Volatility patterns
            "UD": "⚡",  # Lightning for volatility
            "DU": "🔄",  # Cycle for reversal
            "UU": "📈",  # Chart up for trend
            "DD": "📉",  # Chart down for decline
            # Asset swap patterns
            "BE": "🔁",  # Exchange for BTC-ETH swaps
            "EB": "🔀",  # Shuffle for ETH-BTC swaps
            "BU": "🚀",  # Rocket for BTC surge
            "EU": "🌟",  # Star for ETH surge
            # Anomaly patterns
            "AA": "🧊",  # Ice for flatline
            "ZZ": "😴",  # Sleep for dead market
            "XX": "⚠️",  # Warning for unknown
            "EE": "🌪️",  # Tornado for entropy spike
            # System health patterns
            "OK": "✅",  # Check for healthy
            "ER": "❌",  # X for error
            "WN": "⚠️",  # Warning
            "CC": "🔒",  # Lock for consolidation
        }

    async def analyze_sequence(self, sequence: str, context: Optional[Dict[str, Any]] = None) -> List[TwoGramSignal]:
        """
        Analyze a character sequence for 2-gram patterns with full integration.

        Args:
            sequence: Input character sequence (e.g., market direction symbols)
            context: Additional context (market data, timestamps, etc.)

        Returns:
            List of detected 2-gram signals with full metadata
        """
        try:
            if len(sequence) < 2:
                return []

            # Reset frequency map for this analysis
            current_freq = defaultdict(int)
            signals = []

            # Extract all 2-grams from sequence
            for i in range(len(sequence) - 1):
                pattern = sequence[i : i + 2]
                current_freq[pattern] += 1
                self.pattern_history.append(pattern)

            # Calculate entropy for this sequence
            entropy = self._calculate_shannon_entropy(current_freq)

            # Detect bursts and generate signals
            for pattern, frequency in current_freq.items():
                # Calculate burst score
                burst_score = self._calculate_burst_score(pattern, frequency)

                # Generate similarity vector
                similarity_vector = self._generate_similarity_vector(pattern, frequency, entropy)

                # Check fractal resonance
                fractal_resonance, fractal_confidence = await self._check_fractal_resonance(
                    pattern, similarity_vector, context
                )

                # Assess T-cell response
                t_cell_activation, health_score = self._assess_t_cell_response(pattern, frequency, entropy, burst_score)

                # Create signal
                signal = TwoGramSignal(
                    pattern=pattern,
                    frequency=frequency,
                    entropy=entropy,
                    burst_score=burst_score,
                    similarity_vector=similarity_vector,
                    emoji_symbol=self.symbol_map.get(pattern, "🔍"),
                    asic_hash=self._generate_asic_hash(pattern, frequency),
                    fractal_resonance=fractal_resonance,
                    fractal_confidence=fractal_confidence,
                    t_cell_activation=t_cell_activation,
                    system_health_score=health_score,
                    strategy_trigger=self._determine_strategy_trigger(pattern, burst_score),
                    risk_level=self._assess_risk_level(pattern, entropy, burst_score),
                    execution_priority=self._calculate_execution_priority(burst_score, fractal_confidence),
                )

                signals.append(signal)
                self.active_patterns[pattern] = signal

            # Update memory structures
            await self._update_pattern_memory(signals, context)

            # Log significant signals
            significant_signals = [s for s in signals if s.burst_score > self.burst_threshold]
            if significant_signals:
                info("🧬 Detected {0} significant 2-gram patterns".format(len(significant_signals)))
                for signal in significant_signals:
                    debug(
                        "  {0} {1}: burst={2}, ".format(signal.emoji_symbol, signal.pattern, signal.burst_score:.2f)
                        "entropy={0}, health={1}".format(signal.entropy:.3f, signal.system_health_score:.2f)
                    )

            return signals

        except Exception as e:
            error("Error in 2-gram sequence analysis: {0}".format(e))
            return []

    def _calculate_shannon_entropy(self, freq_map: Dict[str, int]) -> float:
        """Calculate Shannon entropy of 2-gram distribution."""
        try:
            if not freq_map:
                return 0.0

            total = sum(freq_map.values())
            entropy = 0.0

            for frequency in freq_map.values():
                if frequency > 0:
                    probability = frequency / total
                    entropy -= probability * math.log2(probability)

            return entropy

        except Exception as e:
            logger.error("Error calculating Shannon entropy: {0}".format(e))
            return 0.0

    def _calculate_burst_score(self, pattern: str, current_frequency: int) -> float:
        """Calculate burst intensity score for a pattern."""
        try:
            # Get historical frequencies for this pattern
            historical_freqs = self.memory.pattern_frequencies.get(pattern, [])

            if len(historical_freqs) < 2:
                return 0.0  # Need history for burst detection

            # Calculate mean and standard deviation
            mean_freq = np.mean(historical_freqs)
            std_freq = np.std(historical_freqs)

            if std_freq == 0:
                return 0.0

            # Calculate Z-score (burst intensity)
            burst_score = (current_frequency - mean_freq) / std_freq

            return max(0.0, burst_score)  # Only positive bursts

        except Exception as e:
            logger.error("Error calculating burst score for {0}: {1}".format(pattern, e))
            return 0.0

    def _generate_similarity_vector(self, pattern: str, frequency: int, entropy: float) -> List[float]:
        """Generate similarity vector for cosine matching."""
        try:
            # Create multidimensional vector representation
            vector = []

            # ASCII values normalized
            ascii_vals = [ord(c) / 128.0 for c in pattern]
            vector.extend(ascii_vals)

            # Frequency component (log-normalized)
            freq_component = math.log(frequency + 1) / 10.0
            vector.append(freq_component)

            # Entropy component
            vector.append(entropy / 10.0)

            # Pattern type encoding
            pattern_type_encoding = self._encode_pattern_type(pattern)
            vector.extend(pattern_type_encoding)

            # Temporal component (based on recent history)
            temporal_component = self._calculate_temporal_component(pattern)
            vector.append(temporal_component)

            return vector

        except Exception as e:
            logger.error("Error generating similarity vector: {0}".format(e))
            return [0.0] * 8  # Fallback vector

    def _encode_pattern_type(self, pattern: str) -> List[float]:
        """Encode pattern type as numerical vector."""
        # One-hot-like encoding for different pattern types
        encodings = {
            "UD": [1.0, 0.0, 0.0],  # Volatility
            "DU": [1.0, 0.0, 0.0],  # Volatility
            "BE": [0.0, 1.0, 0.0],  # Swap
            "EB": [0.0, 1.0, 0.0],  # Swap
            "AA": [0.0, 0.0, 1.0],  # Anomaly
            "ZZ": [0.0, 0.0, 1.0],  # Anomaly
        }

        return encodings.get(pattern, [0.5, 0.5, 0.5])  # Default mixed encoding

    def _calculate_temporal_component(self, pattern: str) -> float:
        """Calculate temporal component based on recent pattern activity."""
        try:
            recent_patterns = list(self.pattern_history)[-20:]  # Last 20 patterns
            pattern_count = recent_patterns.count(pattern)
            return pattern_count / 20.0

        except Exception:
            return 0.0

    async def _check_fractal_resonance(
        self, pattern: str, similarity_vector: List[float], context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Check for fractal resonance with historical patterns."""
        try:
            if not self.fractal_memory:
                return None, None

            # Convert similarity vector to matrix for fractal matching
            vector_matrix = np.array(similarity_vector).reshape(2, -1)
            if vector_matrix.shape[1] < 4:
                # Pad to minimum required size
                pad_width = 4 - vector_matrix.shape[1]
                vector_matrix = np.pad(vector_matrix, ((0, 0), (0, pad_width)), 'constant')

            # Check for fractal match
            fractal_match = self.fractal_memory.match_fractal(
                current_matrix=vector_matrix[:2, :4],  # Use first 2x4 for consistency
                strategy_id="2gram_{0}".format(pattern),
                market_context=context,
            )

            if fractal_match:
                return fractal_match.similarity_score, fractal_match.confidence
            else:
                return None, None

        except Exception as e:
            logger.error("Error checking fractal resonance: {0}".format(e))
            return None, None

    def _assess_t_cell_response(
        self, pattern: str, frequency: int, entropy: float, burst_score: float
    ) -> Tuple[bool, float]:
        """Assess T-cell immune response for system protection."""
        try:
            # T-cell activation conditions
            activation_triggers = []

            # 1. Extremely low entropy (flatline anomaly)
            if entropy < 0.2:
                activation_triggers.append("flatline_anomaly")

            # 2. Extremely high burst (potential manipulation)
            if burst_score > 5.0:
                activation_triggers.append("burst_anomaly")

            # 3. Suspicious pattern repetition
            if frequency > 50:  # Very high frequency in window
                activation_triggers.append("repetition_anomaly")

            # 4. Known dangerous patterns
            dangerous_patterns = ["XX", "ZZ", "ER"]
            if pattern in dangerous_patterns:
                activation_triggers.append("dangerous_pattern")

            # Calculate health score
            base_health = 1.0

            # Reduce health for anomalies
            health_reduction = 0.0
            if entropy < 0.1:
                health_reduction += 0.3
            if burst_score > 3.0:
                health_reduction += 0.2
            if frequency > 30:
                health_reduction += 0.1

            health_score = max(0.0, base_health - health_reduction)

            # T-cell activation if health score drops below sensitivity threshold
            t_cell_activation = health_score < self.t_cell_sensitivity

            if t_cell_activation:
                warn("🛡️ T-cell activation for pattern {0}: triggers={1}".format(pattern, activation_triggers))

                # Log T-cell response
                self.memory.t_cell_responses.append(
                    {
                        "timestamp": time.time(),
                        "pattern": pattern,
                        "triggers": activation_triggers,
                        "health_score": health_score,
                        "frequency": frequency,
                        "entropy": entropy,
                        "burst_score": burst_score,
                    }
                )

            # Update system state
            self.t_cell_active = t_cell_activation or self.t_cell_active
            self.system_health = min(self.system_health, health_score)

            return t_cell_activation, health_score

        except Exception as e:
            logger.error("Error in T-cell assessment: {0}".format(e))
            return False, 1.0

    def _generate_asic_hash(self, pattern: str, frequency: int) -> str:
        """Generate ASIC-compatible hash for pattern representation."""
        try:
            # Create hash input from pattern and frequency
            hash_input = "{0}:{1}:{2}".format(pattern, frequency, time.time():.0f)

            # Generate hash using unified hash system
            unified_hash = generate_unified_hash(hash_input)

            # Create ASIC-compatible representation (hex format)
            asic_hash = "0x{0}".format(unified_hash[:8])

            return asic_hash

        except Exception as e:
            logger.error("Error generating ASIC hash: {0}".format(e))
            return "0x00000000"

    def _determine_strategy_trigger(self, pattern: str, burst_score: float) -> Optional[str]:
        """Determine which strategy should be triggered by this pattern."""
        strategy_mappings = {
            "UD": "volatility_reversal_entry",
            "DU": "reversal_momentum_entry",
            "BE": "swap_arbitrage_trigger",
            "EB": "swap_reversal_trigger",
            "UU": "trend_momentum_entry",
            "DD": "downtrend_reversal_watch",
            "AA": "flatline_caution_mode",
            "EE": "entropy_spike_response",
        }

        # Only trigger if burst score is significant
        if burst_score > self.burst_threshold:
            return strategy_mappings.get(pattern)

        return None

    def _assess_risk_level(self, pattern: str, entropy: float, burst_score: float) -> str:
        """Assess risk level for the detected pattern."""
        # High risk conditions
        if entropy < 0.1 or burst_score > 4.0:
            return "high"
        elif pattern in ["XX", "ZZ", "ER"]:
            return "high"
        # Medium risk conditions
        elif entropy < 0.5 or burst_score > 2.0:
            return "medium"
        # Low risk
        else:
            return "low"

    def _calculate_execution_priority(self, burst_score: float, fractal_confidence: Optional[float]) -> int:
        """Calculate execution priority (1-10 scale)."""
        try:
            base_priority = 5

            # Burst score component
            burst_component = min(3, int(burst_score))

            # Fractal confidence component
            fractal_component = 0
            if fractal_confidence:
                fractal_component = min(2, int(fractal_confidence * 2))

            total_priority = base_priority + burst_component + fractal_component
            return min(10, max(1, total_priority))

        except Exception:
            return 5

    async def _update_pattern_memory(self, signals: List[TwoGramSignal], context: Optional[Dict[str, Any]]):
        """Update pattern memory structures with new signal data."""
        try:
            # Update frequency history
            for signal in signals:
                self.memory.pattern_frequencies[signal.pattern].append(signal.frequency)

                # Limit history size
                if len(self.memory.pattern_frequencies[signal.pattern]) > 100:
                    self.memory.pattern_frequencies[signal.pattern] = self.memory.pattern_frequencies[signal.pattern][
                        -100:
                    ]

            # Update entropy history
            if signals:
                avg_entropy = np.mean([s.entropy for s in signals])
                self.memory.entropy_history.append(avg_entropy)

                if len(self.memory.entropy_history) > 1000:
                    self.memory.entropy_history = self.memory.entropy_history[-1000:]

            # Record burst events
            burst_signals = [s for s in signals if s.burst_score > self.burst_threshold]
            for signal in burst_signals:
                self.memory.burst_events.append(
                    {
                        "timestamp": signal.timestamp,
                        "pattern": signal.pattern,
                        "burst_score": signal.burst_score,
                        "frequency": signal.frequency,
                        "entropy": signal.entropy,
                        "strategy_trigger": signal.strategy_trigger,
                        "context": context,
                    }
                )

            # Save fractal snapshots for significant patterns
            if self.fractal_memory and burst_signals:
                for signal in burst_signals:
                    vector_matrix = np.array(signal.similarity_vector).reshape(2, -1)
                    if vector_matrix.shape[1] >= 4:
                        self.fractal_memory.save_snapshot(
                            q_matrix=vector_matrix[:2, :4],
                            strategy_id="2gram_{0}".format(signal.pattern),
                            profit_result=None,  # Will be updated later
                            market_context=context,
                        )

        except Exception as e:
            logger.error("Error updating pattern memory: {0}".format(e))

    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about detected patterns."""
        try:
            stats = {
                "active_patterns": len(self.active_patterns),
                "total_patterns_tracked": len(self.memory.pattern_frequencies),
                "total_burst_events": len(self.memory.burst_events),
                "t_cell_responses": len(self.memory.t_cell_responses),
                "system_health": self.system_health,
                "t_cell_active": self.t_cell_active,
                "window_size": self.window_size,
                "burst_threshold": self.burst_threshold,
            }

            # Pattern frequency distribution
            pattern_distribution = {}
            for pattern, frequencies in self.memory.pattern_frequencies.items():
                pattern_distribution[pattern] = {
                    "total_occurrences": len(frequencies),
                    "avg_frequency": np.mean(frequencies) if frequencies else 0,
                    "max_frequency": max(frequencies) if frequencies else 0,
                    "recent_trend": frequencies[-5:] if len(frequencies) >= 5 else frequencies,
                }

            stats["pattern_distribution"] = pattern_distribution

            # Recent burst activity
            recent_bursts = [b for b in self.memory.burst_events if time.time() - b["timestamp"] < 3600]
            stats["recent_burst_count"] = len(recent_bursts)

            # Entropy statistics
            if self.memory.entropy_history:
                stats["entropy_stats"] = {
                    "current_entropy": self.memory.entropy_history[-1],
                    "avg_entropy": np.mean(self.memory.entropy_history),
                    "entropy_trend": (
                        self.memory.entropy_history[-10:]
                        if len(self.memory.entropy_history) >= 10
                        else self.memory.entropy_history
                    ),
                }

            # Fractal memory statistics
            if self.fractal_memory:
                fractal_stats = self.fractal_memory.get_pattern_statistics()
                stats["fractal_memory"] = fractal_stats

            return stats

        except Exception as e:
            logger.error("Error getting pattern statistics: {0}".format(e))
            return {}

    async def trigger_strategy_from_pattern(self, pattern: str) -> Optional[Dict[str, Any]]:
        """Trigger a strategy based on a detected 2-gram pattern."""
        try:
            if pattern not in self.active_patterns:
                return None

            signal = self.active_patterns[pattern]

            if not signal.strategy_trigger:
                return None

            # Create strategy trigger packet
            strategy_packet = {
                "trigger_type": "2gram_pattern",
                "strategy_name": signal.strategy_trigger,
                "pattern": signal.pattern,
                "emoji_symbol": signal.emoji_symbol,
                "burst_score": signal.burst_score,
                "frequency": signal.frequency,
                "entropy": signal.entropy,
                "fractal_resonance": signal.fractal_resonance,
                "fractal_confidence": signal.fractal_confidence,
                "execution_priority": signal.execution_priority,
                "risk_level": signal.risk_level,
                "asic_hash": signal.asic_hash,
                "timestamp": signal.timestamp,
                "system_health": signal.system_health_score,
                "t_cell_active": signal.t_cell_activation,
            }

            info("🎯 Strategy triggered: {0} from pattern {1}{2}".format(signal.strategy_trigger, signal.emoji_symbol, signal.pattern))

            return strategy_packet

        except Exception as e:
            logger.error("Error triggering strategy from pattern {0}: {1}".format(pattern, e))
            return None

    def cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vector_a)
            b = np.array(vector_b)

            if len(a) != len(b):
                # Pad shorter vector
                max_len = max(len(a), len(b))
                a = np.pad(a, (0, max_len - len(a)), 'constant')
                b = np.pad(b, (0, max_len - len(b)), 'constant')

            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)

        except Exception as e:
            logger.error("Error calculating cosine similarity: {0}".format(e))
            return 0.0

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the 2-gram detector."""
        try:
            health_report = {
                "detector_status": "healthy" if self.system_health > 0.7 else "degraded",
                "system_health_score": self.system_health,
                "t_cell_active": self.t_cell_active,
                "active_pattern_count": len(self.active_patterns),
                "memory_utilization": {
                    "pattern_frequencies": len(self.memory.pattern_frequencies),
                    "entropy_history": len(self.memory.entropy_history),
                    "burst_events": len(self.memory.burst_events),
                    "t_cell_responses": len(self.memory.t_cell_responses),
                },
            }

            # Check for anomalies
            anomalies = []

            if self.system_health < 0.5:
                anomalies.append("critically_low_health")

            if self.t_cell_active:
                anomalies.append("immune_response_active")

            if len(self.memory.burst_events) > 100:
                anomalies.append("excessive_burst_activity")

            health_report["anomalies"] = anomalies
            health_report["overall_status"] = "critical" if anomalies else "healthy"

            return health_report

        except Exception as e:
            logger.error("Error in health check: {0}".format(e))
            return {"detector_status": "error", "error": str(e)}

    async def get_recent_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent patterns for visualization and analysis."""
        try:
            recent_patterns = []

            # Get recent active patterns
            for pattern, signal in list(self.active_patterns.items())[-limit:]:
                pattern_data = {
                    "pattern": signal.pattern,
                    "emoji_symbol": signal.emoji_symbol,
                    "frequency": signal.frequency,
                    "burst_score": signal.burst_score,
                    "entropy": signal.entropy,
                    "timestamp": signal.timestamp,
                    "fractal_resonance": signal.fractal_resonance,
                    "fractal_confidence": signal.fractal_confidence,
                    "t_cell_activation": signal.t_cell_activation,
                    "system_health_score": signal.system_health_score,
                    "strategy_trigger": signal.strategy_trigger,
                    "risk_level": signal.risk_level,
                    "execution_priority": signal.execution_priority,
                    "asic_hash": signal.asic_hash,
                }
                recent_patterns.append(pattern_data)

            # If we don't have enough active patterns, add from memory
            if len(recent_patterns) < limit:
                # Get recent burst events
                recent_bursts = sorted(self.memory.burst_events, key=lambda x: x["timestamp"], reverse=True)[
                    : limit - len(recent_patterns)
                ]

                for burst in recent_bursts:
                    pattern_data = {
                        "pattern": burst["pattern"],
                        "emoji_symbol": self.symbol_map.get(burst["pattern"], "🔍"),
                        "frequency": burst["frequency"],
                        "burst_score": burst["burst_score"],
                        "entropy": burst["entropy"],
                        "timestamp": burst["timestamp"],
                        "fractal_resonance": None,
                        "fractal_confidence": None,
                        "t_cell_activation": False,
                        "system_health_score": 1.0,
                        "strategy_trigger": burst.get("strategy_trigger"),
                        "risk_level": "medium",
                        "execution_priority": 5,
                        "asic_hash": self._generate_asic_hash(burst["pattern"], burst["frequency"]),
                    }
                    recent_patterns.append(pattern_data)

            return recent_patterns[:limit]

        except Exception as e:
            logger.error("Error getting recent patterns: {0}".format(e))
            return []


# Factory function for easy integration
def create_two_gram_detector(config: Optional[Dict[str, Any]] = None) -> TwoGramDetector:
    """Create a two-gram detector instance with optional configuration."""
    if config is None:
        config = {}

    return TwoGramDetector(
        window_size=config.get("window_size", 100),
        burst_threshold=config.get("burst_threshold", 2.0),
        similarity_threshold=config.get("similarity_threshold", 0.85),
        t_cell_sensitivity=config.get("t_cell_sensitivity", 0.7),
        enable_fractal_memory=config.get("enable_fractal_memory", True),
    )


# Integration test function
async def test_two_gram_integration():
    """Test the 2-gram detector with sample market data."""
    print("🧬 Testing Two-Gram Detector Integration")
    print("=" * 50)

    detector = create_two_gram_detector()

    # Simulate market direction sequence
    market_sequence = "UUDDUDUDBEEBBEAAAZZXREEUUDDBEUUDE"
    context = {"market_data": {"btc_price": 50000, "eth_price": 3000, "volume": 1000000}, "timestamp": time.time()}

    # Analyze sequence
    signals = await detector.analyze_sequence(market_sequence, context)

    print("Detected {0} 2-gram signals:".format(len(signals)))
    for signal in signals:
        print(
            "  {0} {1}: burst={2}, ".format(signal.emoji_symbol, signal.pattern, signal.burst_score:.2f)
            "freq={0}, entropy={1}".format(signal.frequency, signal.entropy:.3f)
        )

        if signal.strategy_trigger:
            print("    → Strategy: {0} (priority: {1})".format(signal.strategy_trigger, signal.execution_priority))

    # Get statistics
    stats = await detector.get_pattern_statistics()
    print(f"\nDetector Statistics:")
    print("  Active patterns: {0}".format(stats['active_patterns']))
    print("  System health: {0}".format(stats['system_health']:.2f))
    print("  T-cell active: {0}".format(stats['t_cell_active']))

    # Health check
    health = await detector.health_check()
    print("\nHealth Status: {0}".format(health['overall_status']))

    print("✅ Two-gram detector test completed")


if __name__ == "__main__":
    asyncio.run(test_two_gram_integration())
