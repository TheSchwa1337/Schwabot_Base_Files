"""
Behavior Pattern Tracker
=======================

Implements pattern tracking and analysis for the Forever Fractal system.
Handles temporal frequency decay and n-gram behavior stitching.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib
from collections import defaultdict
from .spectral_state import SpectralState

class BehaviorPatternTracker:
    """Tracks and analyzes behavior patterns with temporal decay"""
    
    def __init__(self, decay_rate: float = 1/86400):
        """
        Initialize pattern tracker
        
        Args:
            decay_rate: Rate of temporal decay (λ)
        """
        self.decay_rate = decay_rate
        self.pattern_frequencies = defaultdict(float)
        self.pattern_timestamps = defaultdict(float)
        self.ngram_sequences = defaultdict(list)
        self.pattern_hashes = set()
        self.last_update = datetime.now().timestamp()
        
    def track_pattern(self, 
                     triplet: str,
                     context: str,
                     action: str,
                     fractal_depth: int,
                     timestamp: Optional[float] = None) -> str:
        """
        Track a new behavior pattern
        
        Args:
            triplet: Triplet sequence
            context: Context information
            action: Action taken
            fractal_depth: Current fractal depth
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Pattern hash
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # Calculate pattern hash
        pattern_data = f"{triplet}|{context}|{action}|{fractal_depth}"
        pattern_hash = hashlib.md5(pattern_data.encode()).hexdigest()
        
        # Update pattern frequency with decay
        self._apply_temporal_decay(timestamp)
        self.pattern_frequencies[pattern_hash] += 1
        self.pattern_timestamps[pattern_hash] = timestamp
        
        # Track n-gram sequence
        self.ngram_sequences[pattern_hash].append({
            'triplet': triplet,
            'context': context,
            'action': action,
            'timestamp': timestamp
        })
        
        # Store hash
        self.pattern_hashes.add(pattern_hash)
        
        return pattern_hash
    
    def _apply_temporal_decay(self, current_time: float) -> None:
        """
        Apply temporal decay to all pattern frequencies
        
        Args:
            current_time: Current timestamp
        """
        dt = current_time - self.last_update
        decay_factor = np.exp(-self.decay_rate * dt)
        
        for pattern_hash in self.pattern_frequencies:
            self.pattern_frequencies[pattern_hash] *= decay_factor
            
        self.last_update = current_time
    
    def get_pattern_frequency(self, pattern_hash: str) -> float:
        """
        Get current frequency of a pattern
        
        Args:
            pattern_hash: Pattern hash to look up
            
        Returns:
            Current frequency with decay applied
        """
        self._apply_temporal_decay(datetime.now().timestamp())
        return self.pattern_frequencies.get(pattern_hash, 0.0)
    
    def stitch_ngram_sequence(self, 
                            pattern_hash: str,
                            n: int = 3) -> List[Dict]:
        """
        Get n-gram sequence for a pattern
        
        Args:
            pattern_hash: Pattern hash to look up
            n: Length of n-gram sequence
            
        Returns:
            List of n-gram sequences
        """
        sequence = self.ngram_sequences.get(pattern_hash, [])
        if len(sequence) < n:
            return []
            
        return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
    
    def calculate_pattern_similarity(self, 
                                  hash1: str,
                                  hash2: str) -> float:
        """
        Calculate similarity between two patterns
        
        Args:
            hash1: First pattern hash
            hash2: Second pattern hash
            
        Returns:
            Similarity score [0,1]
        """
        seq1 = self.ngram_sequences.get(hash1, [])
        seq2 = self.ngram_sequences.get(hash2, [])
        
        if not seq1 or not seq2:
            return 0.0
            
        # Calculate Jaccard similarity of n-grams
        ngrams1 = set(tuple(s['triplet'] for s in seq1[i:i+3])
                     for i in range(len(seq1)-2))
        ngrams2 = set(tuple(s['triplet'] for s in seq2[i:i+3])
                     for i in range(len(seq2)-2))
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_most_frequent_patterns(self, 
                                 min_frequency: float = 0.1,
                                 max_patterns: int = 10) -> List[Tuple[str, float]]:
        """
        Get most frequent patterns
        
        Args:
            min_frequency: Minimum frequency threshold
            max_patterns: Maximum number of patterns to return
            
        Returns:
            List of (pattern_hash, frequency) tuples
        """
        self._apply_temporal_decay(datetime.now().timestamp())
        
        # Filter and sort patterns
        patterns = [(h, f) for h, f in self.pattern_frequencies.items()
                   if f >= min_frequency]
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return patterns[:max_patterns]
    
    def update_spectral_state(self, state: SpectralState) -> None:
        """
        Update spectral state with pattern information
        
        Args:
            state: Spectral state to update
        """
        # Get pattern frequency
        frequency = self.get_pattern_frequency(state.pattern_hash)
        
        # Update recursive awareness based on pattern frequency
        state.recursive_awareness = min(1.0, frequency * 2)
        
        # Update memory weight
        state.memory_weight = np.exp(-self.decay_rate * 
                                   (datetime.now().timestamp() - state.timestamp))
    
    def save_state(self, filepath: str) -> None:
        """
        Save tracker state to file
        
        Args:
            filepath: Path to save state
        """
        state = {
            'decay_rate': self.decay_rate,
            'pattern_frequencies': dict(self.pattern_frequencies),
            'pattern_timestamps': dict(self.pattern_timestamps),
            'ngram_sequences': dict(self.ngram_sequences),
            'pattern_hashes': list(self.pattern_hashes),
            'last_update': self.last_update
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'BehaviorPatternTracker':
        """
        Load tracker state from file
        
        Args:
            filepath: Path to load state from
            
        Returns:
            Loaded BehaviorPatternTracker instance
        """
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        tracker = cls(decay_rate=state['decay_rate'])
        tracker.pattern_frequencies = defaultdict(float, state['pattern_frequencies'])
        tracker.pattern_timestamps = defaultdict(float, state['pattern_timestamps'])
        tracker.ngram_sequences = defaultdict(list, state['ngram_sequences'])
        tracker.pattern_hashes = set(state['pattern_hashes'])
        tracker.last_update = state['last_update']
        
        return tracker 