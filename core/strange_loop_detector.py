"""
Strange Loop Detector
Handles detection of volatile continuous hash loops, echo patterns, and self-referential states.
"""

import logging
import time
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from .entropy_tracker import EntropyState

logger = logging.getLogger(__name__)


@dataclass
class EchoPattern:
    """Represents a detected echo or strange loop pattern"""
    hash_value: int
    first_occurrence: float  # timestamp
    last_occurrence: float   # timestamp
    frequency: int
    entropy_drift: float     # How much entropy has drifted between occurrences
    confidence: float        # Confidence this is a genuine echo vs noise
    pattern_type: str        # 'echo', 'strange_loop', 'volatility_spike'


class StrangeLoopDetector:
    """
    Detects and handles strange loops, echo patterns, and volatile hash cycles.
    
    Uses a Bloom filter approximation and entropy drift analysis to identify:
    1. Volatile continuous hash loops
    2. Echo patterns (same hash, different entropy context)
    3. Strange loops (self-referential patterns)
    4. Hash collision storms during high volatility
    """
    
    def __init__(self, history_length: int = 10000, echo_threshold: float = 0.1):
        """
        Initialize the strange loop detector.
        
        Args:
            history_length: How many recent hashes to track
            echo_threshold: Entropy drift threshold for echo detection
        """
        # Recent hash history (acts as a Bloom filter approximation)
        self.recent_hashes: deque = deque(maxlen=history_length)
        self.hash_timestamps: Dict[int, List[float]] = defaultdict(list)
        self.hash_entropies: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # Echo detection parameters
        self.echo_threshold = echo_threshold
        self.min_echo_interval = 0.1  # Minimum time between occurrences to consider echo
        self.max_echo_interval = 30.0  # Maximum time interval for echo detection
        
        # Strange loop tracking
        self.active_loops: Dict[int, EchoPattern] = {}
        self.loop_breakers: Set[int] = set()  # Hashes flagged to break loops
        
        # Volatility spike detection
        self.hash_frequency_window = deque(maxlen=1000)  # Recent hash frequencies
        self.volatility_threshold = 5.0  # Hashes per second threshold
        
        # Performance metrics
        self.total_detections = 0
        self.echo_detections = 0
        self.loop_breaks = 0
        
        logger.info("Strange loop detector initialized")

    def process_hash(self, hash_value: int, entropy_state: EntropyState) -> Optional[EchoPattern]:
        """
        Process a new hash and check for strange loops or echo patterns.
        
        Args:
            hash_value: The computed hash value
            entropy_state: The associated entropy state
            
        Returns:
            EchoPattern if an echo/loop is detected, None otherwise
        """
        current_time = time.time()
        entropy_vector = np.array([
            entropy_state.price_entropy,
            entropy_state.volume_entropy,
            entropy_state.time_entropy
        ])
        
        # Add to recent history
        self.recent_hashes.append((hash_value, current_time))
        self.hash_timestamps[hash_value].append(current_time)
        self.hash_entropies[hash_value].append(entropy_vector)
        
        # Clean old entries (keep only recent data)
        self._cleanup_old_entries(current_time)
        
        # Check for echo patterns
        echo_pattern = self._detect_echo_pattern(hash_value, entropy_vector, current_time)
        if echo_pattern:
            self.echo_detections += 1
            self.total_detections += 1
            return echo_pattern
        
        # Check for strange loops
        loop_pattern = self._detect_strange_loop(hash_value, entropy_vector, current_time)
        if loop_pattern:
            self.total_detections += 1
            return loop_pattern
        
        # Check for volatility spikes
        self._update_volatility_tracking(current_time)
        
        return None

    def _detect_echo_pattern(self, hash_value: int, entropy_vector: np.ndarray, 
                           current_time: float) -> Optional[EchoPattern]:
        """
        Detect if this hash represents an echo pattern.
        
        An echo occurs when:
        1. Hash has been seen before within the echo interval
        2. Entropy has drifted significantly from previous occurrence
        3. Pattern doesn't match expected market behavior
        """
        timestamps = self.hash_timestamps[hash_value]
        if len(timestamps) < 2:
            return None
        
        # Check timing - must be within echo interval
        time_since_last = current_time - timestamps[-2]
        if time_since_last < self.min_echo_interval or time_since_last > self.max_echo_interval:
            return None
        
        # Calculate entropy drift
        previous_entropies = self.hash_entropies[hash_value]
        if len(previous_entropies) < 2:
            return None
        
        entropy_drift = np.linalg.norm(entropy_vector - previous_entropies[-2])
        
        # If entropy has drifted significantly, this is likely an echo
        if entropy_drift > self.echo_threshold:
            confidence = min(entropy_drift / self.echo_threshold, 1.0)
            
            pattern = EchoPattern(
                hash_value=hash_value,
                first_occurrence=timestamps[0],
                last_occurrence=current_time,
                frequency=len(timestamps),
                entropy_drift=entropy_drift,
                confidence=confidence,
                pattern_type='echo'
            )
            
            logger.info(f"Echo pattern detected: hash={hash_value}, drift={entropy_drift:.4f}")
            return pattern
        
        return None

    def _detect_strange_loop(self, hash_value: int, entropy_vector: np.ndarray,
                           current_time: float) -> Optional[EchoPattern]:
        """
        Detect strange loops - self-referential patterns that may indicate
        the system is recursively triggering itself.
        """
        timestamps = self.hash_timestamps[hash_value]
        if len(timestamps) < 3:  # Need at least 3 occurrences for loop detection
            return None
        
        # Check for rapid repetition pattern
        recent_intervals = np.diff(timestamps[-3:])
        
        # Strange loop indicators:
        # 1. Very regular intervals (low variance)
        # 2. High frequency
        # 3. Low entropy drift (same context repeating)
        interval_variance = np.var(recent_intervals)
        mean_interval = np.mean(recent_intervals)
        
        if (interval_variance < 0.01 and  # Very regular
            mean_interval < 1.0 and       # High frequency 
            len(timestamps) > 5):         # Sustained pattern
            
            # Calculate entropy stability
            recent_entropies = self.hash_entropies[hash_value][-3:]
            entropy_stability = 1.0 - np.mean([
                np.linalg.norm(recent_entropies[i] - recent_entropies[i-1])
                for i in range(1, len(recent_entropies))
            ])
            
            if entropy_stability > 0.8:  # Very stable entropy = strange loop
                confidence = entropy_stability
                
                pattern = EchoPattern(
                    hash_value=hash_value,
                    first_occurrence=timestamps[0],
                    last_occurrence=current_time,
                    frequency=len(timestamps),
                    entropy_drift=1.0 - entropy_stability,
                    confidence=confidence,
                    pattern_type='strange_loop'
                )
                
                # Flag this hash for loop breaking
                self.loop_breakers.add(hash_value)
                self.loop_breaks += 1
                
                logger.warning(f"Strange loop detected: hash={hash_value}, "
                             f"stability={entropy_stability:.4f}")
                return pattern
        
        return None

    def _update_volatility_tracking(self, current_time: float):
        """Update volatility tracking for spike detection."""
        self.hash_frequency_window.append(current_time)
        
        # Calculate current hash rate
        if len(self.hash_frequency_window) > 10:
            recent_window = list(self.hash_frequency_window)[-100:]  # Last 100 hashes
            time_span = recent_window[-1] - recent_window[0]
            if time_span > 0:
                hash_rate = len(recent_window) / time_span
                
                # Flag volatility spike if rate exceeds threshold
                if hash_rate > self.volatility_threshold:
                    logger.info(f"Hash volatility spike detected: {hash_rate:.2f} hashes/sec")

    def _cleanup_old_entries(self, current_time: float):
        """Clean up old entries to prevent memory bloat."""
        cutoff_time = current_time - self.max_echo_interval * 2
        
        # Clean hash timestamps
        for hash_val in list(self.hash_timestamps.keys()):
            timestamps = self.hash_timestamps[hash_val]
            # Keep only recent timestamps
            recent_timestamps = [t for t in timestamps if t > cutoff_time]
            
            if recent_timestamps:
                self.hash_timestamps[hash_val] = recent_timestamps
                # Also clean corresponding entropies
                cutoff_idx = len(timestamps) - len(recent_timestamps)
                self.hash_entropies[hash_val] = self.hash_entropies[hash_val][cutoff_idx:]
            else:
                # No recent activity, remove entirely
                del self.hash_timestamps[hash_val]
                if hash_val in self.hash_entropies:
                    del self.hash_entropies[hash_val]

    def should_break_loop(self, hash_value: int) -> bool:
        """
        Check if this hash should trigger a loop break.
        
        Returns True if the hash is flagged as a loop breaker.
        """
        return hash_value in self.loop_breakers

    def clear_loop_breaker(self, hash_value: int):
        """Clear a hash from the loop breaker set after handling."""
        self.loop_breakers.discard(hash_value)

    def get_metrics(self) -> Dict:
        """Get strange loop detection metrics."""
        current_time = time.time()
        
        # Calculate current hash rate
        hash_rate = 0.0
        if len(self.hash_frequency_window) > 1:
            recent_hashes = list(self.hash_frequency_window)[-100:]
            if len(recent_hashes) > 1:
                time_span = recent_hashes[-1] - recent_hashes[0]
                if time_span > 0:
                    hash_rate = len(recent_hashes) / time_span
        
        return {
            'total_detections': self.total_detections,
            'echo_detections': self.echo_detections,
            'loop_breaks': self.loop_breaks,
            'active_loop_breakers': len(self.loop_breakers),
            'tracked_hashes': len(self.hash_timestamps),
            'current_hash_rate': hash_rate,
            'volatility_spike_active': hash_rate > self.volatility_threshold
        }

    def reset_metrics(self):
        """Reset detection metrics."""
        self.total_detections = 0
        self.echo_detections = 0
        self.loop_breaks = 0
        logger.info("Strange loop detector metrics reset") 