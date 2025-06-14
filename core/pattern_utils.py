"""
Pattern Utils Module
Handles pattern matching, entry/exit rules, confidence calculations,
and hash similarity scoring for trading decisions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


# Entry keys for 4-bit patterns (0-15)
# These represent bullish micro-patterns
ENTRY_KEYS = {0x3, 0x5, 0x6, 0x9, 0xA, 0xC}  # Binary patterns with positive bias

# Exit keys for 4-bit patterns
EXIT_KEYS = {0x0, 0x1, 0x8, 0xE, 0xF}  # Binary patterns with negative bias


@dataclass
class PatternMatch:
    """Represents a pattern matching result"""
    hash_value: int
    confidence: float
    similarity_score: float
    phase_state: 'PhaseState'
    action: str  # 'entry', 'exit', 'hold', 'wait'
    reasons: List[str]


class PatternUtils:
    """
    Utilities for pattern matching, confidence calculation,
    and trading signal generation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pattern utilities.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Trading thresholds
        self.thresholds = {
            'density_entry': self.config.get('density_entry', 0.57),
            'density_exit': self.config.get('density_exit', 0.42),
            'variance_entry': self.config.get('variance_entry', 0.002),
            'variance_exit': self.config.get('variance_exit', 0.007),
            'confidence_min': self.config.get('confidence_min', 0.7),
            'pattern_strength_min': self.config.get('pattern_strength_min', 0.7)
        }
        
        # Latency compensation
        self.measured_latency_ms = 0.0
        self.latency_history = []
        
    def is_entry_phase(self, phase_state: 'PhaseState', pattern_analysis: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Determine if current phase state indicates entry (Rule #5).
        
        Entry if:
        - b4 ∈ ENTRY_KEYS
        - density > 0.57
        - variance_short < 0.002  
        - pattern_strength > 0.7
        
        Args:
            phase_state: Current phase state
            pattern_analysis: Pattern analysis metrics
            
        Returns:
            Tuple of (is_entry, reasons)
        """
        reasons = []
        checks = []
        
        # Check 4-bit pattern
        b4_check = phase_state.b4 in ENTRY_KEYS
        checks.append(b4_check)
        if not b4_check:
            reasons.append(f"b4={phase_state.b4} not in entry keys")
        
        # Check density
        density_check = phase_state.density > self.thresholds['density_entry']
        checks.append(density_check)
        if not density_check:
            reasons.append(f"density={phase_state.density:.3f} <= {self.thresholds['density_entry']}")
        
        # Check short-term variance
        variance_check = phase_state.variance_short < self.thresholds['variance_entry']
        checks.append(variance_check)
        if not variance_check:
            reasons.append(f"variance_short={phase_state.variance_short:.4f} >= {self.thresholds['variance_entry']}")
        
        # Check pattern strength
        pattern_strength = pattern_analysis.get('pattern_strength', 0.0)
        strength_check = pattern_strength > self.thresholds['pattern_strength_min']
        checks.append(strength_check)
        if not strength_check:
            reasons.append(f"pattern_strength={pattern_strength:.3f} <= {self.thresholds['pattern_strength_min']}")
        
        is_entry = all(checks)
        
        if is_entry:
            reasons = [f"Entry signal: b4={phase_state.b4}, density={phase_state.density:.3f}, "
                      f"var={phase_state.variance_short:.4f}, strength={pattern_strength:.3f}"]
        
        return is_entry, reasons
    
    def is_exit_phase(self, phase_state: 'PhaseState', pattern_analysis: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Determine if current phase state indicates exit (Rule #5).
        
        Exit if:
        - b4 ∉ ENTRY_KEYS OR
        - density < 0.42 OR
        - variance_short > 0.007
        
        Args:
            phase_state: Current phase state
            pattern_analysis: Pattern analysis metrics
            
        Returns:
            Tuple of (is_exit, reasons)
        """
        reasons = []
        
        # Check 4-bit pattern
        if phase_state.b4 not in ENTRY_KEYS:
            reasons.append(f"b4={phase_state.b4} not in entry keys")
        
        # Check density threshold
        if phase_state.density < self.thresholds['density_exit']:
            reasons.append(f"density={phase_state.density:.3f} < {self.thresholds['density_exit']}")
        
        # Check variance threshold
        if phase_state.variance_short > self.thresholds['variance_exit']:
            reasons.append(f"variance_short={phase_state.variance_short:.4f} > {self.thresholds['variance_exit']}")
        
        # Additional exit conditions
        if phase_state.tier <= 1:
            reasons.append(f"tier={phase_state.tier} <= 1 (exit zone)")
        
        is_exit = len(reasons) > 0
        
        if is_exit:
            reasons.insert(0, "Exit signal triggered:")
        
        return is_exit, reasons
    
    def compare_hashes(self, h1: int, h2: int, e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Calculate similarity score between two hashes considering both
        hash distance and entropy similarity (Rule #6).
        
        S(h1,h2) = 0.5 * [1 - Ham(h1,h2)/64] + 0.5 * [1 - ||E1-E2||₂/max||E||]
        
        Args:
            h1: First hash value
            h2: Second hash value
            e1: First entropy vector [price, volume, time]
            e2: Second entropy vector [price, volume, time]
            
        Returns:
            Similarity score 0-1
        """
        # Calculate Hamming distance for hash similarity
        xor_result = h1 ^ h2
        hamming_dist = bin(xor_result).count('1')
        hash_similarity = 1.0 - (hamming_dist / 64.0)  # Assuming 64-bit hash portion
        
        # Calculate entropy vector similarity
        if e1 is not None and e2 is not None:
            # Euclidean distance between entropy vectors
            entropy_dist = np.linalg.norm(e1 - e2)
            
            # Normalize by maximum possible distance
            # Max entropy is roughly log2(50) ≈ 5.64 per dimension
            max_dist = np.sqrt(3 * (5.64 ** 2))  # 3D vector max distance
            entropy_similarity = 1.0 - min(entropy_dist / max_dist, 1.0)
        else:
            entropy_similarity = hash_similarity  # Fallback to hash only
        
        # Combined similarity score
        combined_similarity = 0.5 * hash_similarity + 0.5 * entropy_similarity
        
        return combined_similarity
    
    def find_similar_hashes(self, target_hash: int, hash_database: Dict, 
                           threshold: float = 0.85) -> List[Tuple[int, float]]:
        """
        Find hashes similar to target using Hamming distance.
        
        Args:
            target_hash: Hash to compare against
            hash_database: Dictionary of hash entries
            threshold: Minimum similarity threshold
            
        Returns:
            List of (hash, similarity) tuples
        """
        similar = []
        
        for hash_value, entry in hash_database.items():
            if hash_value == target_hash:
                continue
            
            # Get entropy vectors
            target_entropy = entry.get_entropy_vector() if hasattr(entry, 'get_entropy_vector') else None
            other_entropy = entry.get_entropy_vector() if hasattr(entry, 'get_entropy_vector') else None
            
            # Calculate similarity
            similarity = self.compare_hashes(
                target_hash, hash_value,
                target_entropy, other_entropy
            )
            
            if similarity >= threshold:
                similar.append((hash_value, similarity))
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar
    
    def calculate_confidence(self, hash_value: int, similar_hashes: List[Tuple[int, float]], 
                           hash_database: Dict, pattern_analysis: Dict) -> float:
        """
        Calculate pattern confidence using cluster analysis (Rule #7).
        
        C(h) = min(ln(Σf_j + 1)/3, 1) * (1 + pattern_strength) * (1 - σ²_mid)
        
        Args:
            hash_value: Target hash
            similar_hashes: List of (hash, similarity) tuples
            hash_database: Hash database
            pattern_analysis: Pattern analysis metrics
            
        Returns:
            Confidence score 0-1
        """
        if not similar_hashes:
            return 0.0
        
        # Sum frequencies of similar hashes
        total_frequency = sum(
            hash_database[h].frequency 
            for h, _ in similar_hashes 
            if h in hash_database and hasattr(hash_database[h], 'frequency')
        )
        
        # Add target hash frequency
        if hash_value in hash_database and hasattr(hash_database[hash_value], 'frequency'):
            total_frequency += hash_database[hash_value].frequency
        
        # Base confidence from frequency
        base_confidence = min(np.log(total_frequency + 1) / 3.0, 1.0)
        
        # Pattern strength multiplier
        pattern_strength = pattern_analysis.get('pattern_strength', 0.5)
        strength_multiplier = 1.0 + pattern_strength
        
        # Variance penalty (lower variance = higher confidence)
        variance_mid = pattern_analysis.get('variance_mid', 0.0)
        variance_penalty = 1.0 - min(variance_mid, 1.0)
        
        # Combined confidence
        confidence = base_confidence * strength_multiplier * variance_penalty
        
        # Normalize to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def adjust_for_latency(self, base_threshold: float, threshold_type: str = 'exit') -> float:
        """
        Adjust threshold based on measured latency (Rule #10).
        
        d_exit = d_exit_nominal * (1 + L/1000)
        
        Args:
            base_threshold: Base threshold value
            threshold_type: 'exit' or 'entry'
            
        Returns:
            Adjusted threshold
        """
        # Update latency measurement
        self._update_latency_measurement()
        
        # Calculate adjustment factor
        latency_factor = 1.0 + (self.measured_latency_ms / 1000.0)
        
        # Apply adjustment (inflate exit thresholds, deflate entry thresholds)
        if threshold_type == 'exit':
            adjusted = base_threshold * latency_factor
        else:  # entry
            adjusted = base_threshold / latency_factor
        
        return adjusted
    
    def _update_latency_measurement(self):
        """Update rolling latency measurement."""
        # In production, this would measure actual round-trip times
        # For now, we'll simulate with a placeholder
        current_latency = time.time() % 10  # Simulated 0-10ms latency
        
        self.latency_history.append(current_latency)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
        
        self.measured_latency_ms = np.mean(self.latency_history) if self.latency_history else 0.0
    
    def check_pattern_match(self, hash_value: int, phase_state: 'PhaseState',
                           pattern_analysis: Dict, hash_database: Dict,
                           entropy_vector: Optional[np.ndarray] = None) -> PatternMatch:
        """
        Comprehensive pattern matching and signal generation.
        
        Args:
            hash_value: Current hash
            phase_state: Current phase state  
            pattern_analysis: Pattern analysis metrics
            hash_database: Hash database
            entropy_vector: Current entropy vector
            
        Returns:
            PatternMatch object with action and confidence
        """
        # Find similar hashes
        similar_hashes = self.find_similar_hashes(hash_value, hash_database)
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            hash_value, similar_hashes, hash_database, pattern_analysis
        )
        
        # Check entry conditions
        is_entry, entry_reasons = self.is_entry_phase(phase_state, pattern_analysis)
        
        # Check exit conditions
        is_exit, exit_reasons = self.is_exit_phase(phase_state, pattern_analysis)
        
        # Determine action
        if is_exit:
            action = 'exit'
            reasons = exit_reasons
        elif is_entry and confidence >= self.thresholds['confidence_min']:
            action = 'entry'
            reasons = entry_reasons + [f"Confidence={confidence:.3f}"]
        elif is_entry and confidence < self.thresholds['confidence_min']:
            action = 'wait'
            reasons = [f"Entry conditions met but confidence={confidence:.3f} < {self.thresholds['confidence_min']}"]
        else:
            action = 'hold'
            reasons = ["No clear signal"]
        
        # Calculate average similarity
        avg_similarity = np.mean([sim for _, sim in similar_hashes[:5]]) if similar_hashes else 0.0
        
        return PatternMatch(
            hash_value=hash_value,
            confidence=confidence,
            similarity_score=avg_similarity,
            phase_state=phase_state,
            action=action,
            reasons=reasons
        )
    
    def update_thresholds(self, market_volatility: float):
        """
        Dynamically update thresholds based on market conditions.
        
        Args:
            market_volatility: Current market volatility measure
        """
        # Adjust thresholds based on volatility
        # Higher volatility = more conservative thresholds
        vol_factor = 1.0 + (market_volatility - 0.5) * 0.2
        
        self.thresholds['density_entry'] = min(0.65, 0.57 * vol_factor)
        self.thresholds['density_exit'] = max(0.35, 0.42 / vol_factor)
        self.thresholds['variance_entry'] = 0.002 / vol_factor
        self.thresholds['variance_exit'] = 0.007 * vol_factor
        
        logger.info(f"Updated thresholds for volatility={market_volatility:.3f}: {self.thresholds}") 