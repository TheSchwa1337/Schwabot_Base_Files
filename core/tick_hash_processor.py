from __future__ import annotations
import numpy as np

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Tick Hash Processor - Hash-Based Tick Analysis & Pattern Detection.

This module processes tick-based hash signatures for pattern recognition,
frequency analysis, and entropy-based anomaly detection in real-time trading.

Mathematical Foundation:
- Tick variance entropy: E_tick = -Σ(p_i * unified_math.log(p_i))
- Levenshtein drift correction: δ_hash = L(h_1, h_2) * e^(-γt)
- Recursive trigger gate: ψ_tick = Θ(Δ_volume) * χ(η_momentum)
- Hash frequency analysis: f_hash = FFT(hash_sequence)

Windows CLI compatible with comprehensive error handling.
"""


import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# from core.unified_math_system import unified_math  # F811: duplicate import

logger = logging.getLogger(__name__)


@dataclass
class TickHashMetrics:
    """Tick hash analysis metrics."""

    hash_value: str                    # Generated hash
    frequency_score: float             # Frequency analysis score
    pattern_score: float               # Pattern recognition score
    entropy_level: float               # Hash entropy level
    drift_correction: float            # Levenshtein drift correction
    trigger_gate_status: bool          # Recursive trigger gate status
    confidence_level: float            # Overall confidence level


@dataclass
class HashPatternAnalysis:
    """Hash pattern analysis results."""

    pattern_strength: float            # Pattern strength [0, 1]
    recurring_sequences: List[str]     # Detected recurring sequences
    anomaly_score: float               # Anomaly detection score
    stability_index: float             # Pattern stability index


class TickHashProcessor:
    """Processes tick hashes for pattern detection and anomaly analysis."""

    def __init__(self):
        """Initialize tick hash processor."""
        self.hash_history: deque = deque(maxlen=1000)
        self.frequency_map: Dict[str, int] = defaultdict(int)
        self.pattern_cache: Dict[str, float] = {}
        self.entropy_window = 50
        self.drift_decay_rate = 0.1

        # Trigger gate parameters
        self.volume_threshold = 0.5
        self.momentum_threshold = 0.3

        # Pattern detection parameters
        self.min_pattern_length = 4
        self.max_pattern_length = 16
        self.pattern_confidence_threshold = 0.7

    def generate_tick_hash(
        self,
        price: float,
        volume: float,
        timestamp: float,
        additional_data: Optional[Dict] = None,
    ) -> str:
        """Generate hash signature for tick data.

        Parameters
        ----------
        price : float
            Current price
        volume : float
            Current volume
        timestamp : float
            Tick timestamp
        additional_data : Dict, optional
            Additional data to include in hash

        Returns
        -------
        str
            Generated hash signature
        """
        try:
            # Create hash input string
            hash_input = f"{price:.8f}|{volume:.6f}|{timestamp:.3f}"

            if additional_data:
                for key, value in sorted(additional_data.items()):
                    hash_input += f"|{key}:{value}"

            # Generate SHA-256 hash
            hash_object = hashlib.sha256(hash_input.encode())
            tick_hash = hash_object.hexdigest()[:16]  # Use first 16 characters

            # Update frequency map
            self.frequency_map[tick_hash] += 1

            # Add to history
            self.hash_history.append({
                'hash': tick_hash,
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            })

            return tick_hash

        except Exception as e:
            logger.error(f"Error generating tick hash: {e}")
            return "error_hash_" + str(int(time.time()))

    def calculate_tick_variance_entropy(self, hash_sequence: List[str]) -> float:
        """Calculate tick variance entropy from hash sequence.

        Mathematical Formula:
        E_tick = -Σ(p_i * unified_math.log(p_i))

        Parameters
        ----------
        hash_sequence : List[str]
            Sequence of hash values

        Returns
        -------
        float
            Entropy level [0, 1]
        """
        try:
            if not hash_sequence:
                return 0.5

            # Calculate hash character frequency
            char_counts = defaultdict(int)
            total_chars = 0

            for hash_val in hash_sequence:
                for char in hash_val:
                    char_counts[char] += 1
                    total_chars += 1

            if total_chars == 0:
                return 0.5

            # Calculate entropy
            entropy = 0.0
            for count in char_counts.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * np.log2(probability)

            # Normalize to [0, 1] range (max entropy for hex is log2(16) = 4)
            normalized_entropy = entropy / 4.0

            return unified_math.max(0.0, unified_math.min(1.0, normalized_entropy))

        except Exception as e:
            logger.error(f"Error calculating tick variance entropy: {e}")
            return 0.5

    def calculate_levenshtein_drift_correction(
        self,
        hash1: str,
        hash2: str,
        time_delta: float,
    ) -> float:
        """Calculate Levenshtein drift correction.

        Mathematical Formula:
        δ_hash = L(h_1, h_2) * e^(-γt)

        Parameters
        ----------
        hash1 : str
            First hash
        hash2 : str
            Second hash
        time_delta : float
            Time difference between hashes

        Returns
        -------
        float
            Drift correction value
        """
        try:
            # Calculate Levenshtein distance
            len1, len2 = len(hash1), len(hash2)

            # Create distance matrix
            matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

            # Initialize matrix
            for i in range(len1 + 1):
                matrix[i][0] = i
            for j in range(len2 + 1):
                matrix[0][j] = j

            # Fill matrix
            for i in range(1, len1 + 1):
                for j in range(1, len2 + 1):
                    if hash1[i-1] == hash2[j-1]:
                        cost = 0
                    else:
                        cost = 1

                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + cost  # substitution
                    )

            levenshtein_distance = matrix[len1][len2]

            # Apply time decay
            drift_correction = levenshtein_distance * unified_math.exp(-self.drift_decay_rate * time_delta)

            # Normalize to reasonable range
            max_distance = unified_math.max(len1, len2)
            if max_distance > 0:
                drift_correction = drift_correction / max_distance

            return drift_correction

        except Exception as e:
            logger.error(f"Error calculating Levenshtein drift correction: {e}")
            return 0.0

    def evaluate_recursive_trigger_gate(
        self,
        volume_delta: float,
        momentum_eta: float,
    ) -> bool:
        """Evaluate recursive trigger gate status.

        Mathematical Formula:
        ψ_tick = Θ(Δ_volume) * χ(η_momentum)

        Parameters
        ----------
        volume_delta : float
            Volume change indicator
        momentum_eta : float
            Momentum indicator

        Returns
        -------
        bool
            Trigger gate status
        """
        try:
            # Heaviside step function for volume
            theta_volume = 1.0 if volume_delta > self.volume_threshold else 0.0

            # Chi function for momentum (sigmoid-like)
            chi_momentum = 1.0 / (1.0 + unified_math.exp(-5 * (momentum_eta - self.momentum_threshold)))

            # Gate trigger logic
            gate_value = theta_volume * chi_momentum

            return gate_value > 0.5

        except Exception as e:
            logger.error(f"Error evaluating recursive trigger gate: {e}")
            return False

    def analyze_hash_frequency(self, target_hash: str) -> float:
        """Analyze hash frequency for pattern detection.

        Parameters
        ----------
        target_hash : str
            Hash to analyze

        Returns
        -------
        float
            Frequency analysis score [0, 1]
        """
        try:
            if not self.hash_history:
                return 0.0

            # Get frequency of target hash
            hash_frequency = self.frequency_map.get(target_hash, 0)

            # Calculate relative frequency
            total_hashes = len(self.hash_history)
            relative_frequency = hash_frequency / total_hashes if total_hashes > 0 else 0.0

            # Apply frequency scoring (rare hashes get higher scores)
            if relative_frequency == 0:
                frequency_score = 0.0
            elif relative_frequency < 0.01:  # Very rare
                frequency_score = 0.9
            elif relative_frequency < 0.05:  # Rare
                frequency_score = 0.7
            elif relative_frequency < 0.1:   # Uncommon
                frequency_score = 0.5
            else:  # Common
                frequency_score = 0.2

            return frequency_score

        except Exception as e:
            logger.error(f"Error analyzing hash frequency: {e}")
            return 0.0

    def detect_hash_patterns(self, window_size: int = 20) -> HashPatternAnalysis:
        """Detect patterns in recent hash sequence.

        Parameters
        ----------
        window_size : int
            Size of analysis window

        Returns
        -------
        HashPatternAnalysis
            Pattern analysis results
        """
        try:
            if len(self.hash_history) < window_size:
                return HashPatternAnalysis(0.0, [], 0.0, 0.0)

            # Get recent hash sequence
            recent_hashes = [entry['hash'] for entry in list(self.hash_history)[-window_size:]]

            # Find recurring sequences
            recurring_sequences = []
            pattern_scores = []

            for length in range(self.min_pattern_length, unified_math.min(self.max_pattern_length, len(recent_hashes))):
                for start in range(len(recent_hashes) - length + 1):
                    pattern = ''.join(recent_hashes[start:start + length])

                    # Count occurrences
                    occurrences = 0
                    for i in range(len(recent_hashes) - length + 1):
                        if ''.join(recent_hashes[i:i + length]) == pattern:
                            occurrences += 1

                    if occurrences > 1:
                        pattern_strength = occurrences / (len(recent_hashes) - length + 1)
                        if pattern_strength > 0.2:  # Significant pattern
                            recurring_sequences.append(pattern)
                            pattern_scores.append(pattern_strength)

            # Calculate overall pattern strength
            if pattern_scores:
                overall_pattern_strength = unified_math.max(pattern_scores)
            else:
                overall_pattern_strength = 0.0

            # Calculate anomaly score (high when few patterns detected)
            anomaly_score = 1.0 - overall_pattern_strength

            # Calculate stability index
            hash_entropy = self.calculate_tick_variance_entropy(recent_hashes)
            stability_index = 1.0 - hash_entropy  # High entropy = low stability

            return HashPatternAnalysis(
                pattern_strength=overall_pattern_strength,
                recurring_sequences=recurring_sequences[:5],  # Top 5 patterns
                anomaly_score=anomaly_score,
                stability_index=stability_index,
            )

        except Exception as e:
            logger.error(f"Error detecting hash patterns: {e}")
            return HashPatternAnalysis(0.0, [], 0.5, 0.5)

    def analyze_tick_hash(
        self,
        tick_hash: str,
        volume_delta: float = 0.0,
        momentum_eta: float = 0.0,
    ) -> TickHashMetrics:
        """Perform comprehensive tick hash analysis.

        Parameters
        ----------
        tick_hash : str
            Hash to analyze
        volume_delta : float
            Volume change indicator
        momentum_eta : float
            Momentum indicator

        Returns
        -------
        TickHashMetrics
            Complete hash analysis metrics
        """
        try:
            # Frequency analysis
            frequency_score = self.analyze_hash_frequency(tick_hash)

            # Pattern analysis
            pattern_analysis = self.detect_hash_patterns()
            pattern_score = pattern_analysis.pattern_strength

            # Entropy calculation
            recent_hashes = [entry['hash'] for entry in list(self.hash_history)[-self.entropy_window:]]
            entropy_level = self.calculate_tick_variance_entropy(recent_hashes)

            # Drift correction (compare with previous hash)
            drift_correction = 0.0
            if len(self.hash_history) >= 2:
                prev_hash = self.hash_history[-2]['hash']
                prev_timestamp = self.hash_history[-2]['timestamp']
                current_timestamp = time.time()
                time_delta = current_timestamp - prev_timestamp

                drift_correction = self.calculate_levenshtein_drift_correction(
                    prev_hash, tick_hash, time_delta
                )

            # Trigger gate evaluation
            trigger_gate_status = self.evaluate_recursive_trigger_gate(
                volume_delta, momentum_eta
            )

            # Calculate overall confidence
            confidence_components = [
                frequency_score * 0.25,
                pattern_score * 0.25,
                (1.0 - entropy_level) * 0.25,  # Lower entropy = higher confidence
                (1.0 - drift_correction) * 0.25,  # Lower drift = higher confidence
            ]
            confidence_level = sum(confidence_components)

            return TickHashMetrics(
                hash_value=tick_hash,
                frequency_score=frequency_score,
                pattern_score=pattern_score,
                entropy_level=entropy_level,
                drift_correction=drift_correction,
                trigger_gate_status=trigger_gate_status,
                confidence_level=confidence_level,
            )

        except Exception as e:
            logger.error(f"Error analyzing tick hash: {e}")
            return self._create_safe_metrics(tick_hash)

    def get_frequency(self, tick_hash: str) -> float:
        """Get frequency score for a tick hash (compatibility method)."""
        return self.analyze_hash_frequency(tick_hash)

    def analyze_pattern(self, tick_hash: str) -> float:
        """Analyze pattern for a tick hash (compatibility method)."""
        pattern_analysis = self.detect_hash_patterns()
        return pattern_analysis.pattern_strength

    def _create_safe_metrics(self, tick_hash: str) -> TickHashMetrics:
        """Create safe fallback metrics."""
        return TickHashMetrics(
            hash_value=tick_hash,
            frequency_score=0.0,
            pattern_score=0.0,
            entropy_level=0.5,
            drift_correction=0.0,
            trigger_gate_status=False,
            confidence_level=0.0,
        )

    def get_processor_summary(self) -> Dict:
        """Get tick hash processor summary."""
        return {
            "hash_history_size": len(self.hash_history),
            "unique_hashes": len(self.frequency_map),
            "most_frequent_hash": unified_math.max(self.frequency_map.items(), key=lambda x: x[1])[0] if self.frequency_map else None,
            "max_frequency": unified_math.max(self.frequency_map.values()) if self.frequency_map else 0,
            "entropy_window": self.entropy_window,
            "pattern_cache_size": len(self.pattern_cache),
        }


def main() -> None:
    """Demo function for testing tick hash processor."""
    safe_print("Tick Hash Processor Demo")
    safe_print("=" * 30)

    processor = TickHashProcessor()

    # Generate test tick hashes
    test_data = [
        (50000, 1.5, time.time()),
        (50050, 2.1, time.time() + 1),
        (49980, 1.8, time.time() + 2),
        (50100, 2.5, time.time() + 3),
        (50075, 1.9, time.time() + 4),
        (50200, 3.2, time.time() + 5),
    ]

    safe_print("Generating tick hashes:")
    for price, volume, timestamp in test_data:
        tick_hash = processor.generate_tick_hash(price, volume, timestamp)
        safe_print(f"  Price: ${price:,.0f}, Volume: {volume:.1f} -> Hash: {tick_hash}")

    # Analyze latest hash
    if processor.hash_history:
        latest_hash = processor.hash_history[-1]['hash']

        safe_print(f"\nAnalyzing hash: {latest_hash}")
        metrics = processor.analyze_tick_hash(
            latest_hash,
            volume_delta=0.3,
            momentum_eta=0.6
        )

        safe_print(f"  Frequency Score: {metrics.frequency_score:.3f}")
        safe_print(f"  Pattern Score: {metrics.pattern_score:.3f}")
        safe_print(f"  Entropy Level: {metrics.entropy_level:.3f}")
        safe_print(f"  Drift Correction: {metrics.drift_correction:.3f}")
        safe_print(f"  Trigger Gate: {metrics.trigger_gate_status}")
        safe_print(f"  Confidence Level: {metrics.confidence_level:.3f}")

    # Pattern detection
    safe_print("\nPattern Detection:")
    pattern_analysis = processor.detect_hash_patterns()
    safe_print(f"  Pattern Strength: {pattern_analysis.pattern_strength:.3f}")
    safe_print(f"  Recurring Sequences: {len(pattern_analysis.recurring_sequences)}")
    safe_print(f"  Anomaly Score: {pattern_analysis.anomaly_score:.3f}")
    safe_print(f"  Stability Index: {pattern_analysis.stability_index:.3f}")

    # Processor summary
    summary = processor.get_processor_summary()
    safe_print(f"\nProcessor Summary: {summary}")


if __name__ == "__main__":
    main()
