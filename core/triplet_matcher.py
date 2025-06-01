"""
Triplet Matching System
=====================

Handles fractal echo detection and triplet matching for matrix fault resolution.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import numpy as np
from .fractal_core import ForeverFractalCore, FractalState

@dataclass
class TripletMatch:
    """Represents a matched triplet with metadata"""
    states: List[FractalState]
    coherence: float
    timestamp: float
    is_mirror: bool = False

class TripletMatcher:
    """Handles triplet matching and echo detection"""
    
    def __init__(self, fractal_core: ForeverFractalCore, 
                 epsilon: float = 0.1,
                 min_coherence: float = 0.7):
        self.fractal_core = fractal_core
        self.epsilon = epsilon
        self.min_coherence = min_coherence
        self.match_history: List[TripletMatch] = []
        self.collapse_threshold = 3  # Number of matches to trigger collapse
        
    def find_matching_triplet(self, current_states: List[FractalState]) -> Optional[TripletMatch]:
        """
        Find a matching triplet in history.
        
        Args:
            current_states: Current triplet states
            
        Returns:
            Matching triplet if found
        """
        if len(current_states) < 3:
            return None
            
        # Compute coherence
        coherence = self.fractal_core.compute_coherence(current_states)
        if coherence < self.min_coherence:
            return None
            
        # Quantize states
        quantized = [
            self.fractal_core.quantize_vector(state.vector)
            for state in current_states
        ]
        
        # Check for mirror match
        if self.fractal_core.check_mirror(quantized[0]):
            return TripletMatch(
                states=current_states,
                coherence=coherence,
                timestamp=time.time(),
                is_mirror=True
            )
            
        # Check history for matches
        for match in reversed(self.match_history):
            if time.time() - match.timestamp > 3600:  # Skip old matches
                continue
                
            # Compare coherence scores
            if abs(match.coherence - coherence) < self.epsilon:
                return match
                
        return None
    
    def register_triplet(self, states: List[FractalState]) -> TripletMatch:
        """
        Register a new triplet match.
        
        Args:
            states: States forming the triplet
            
        Returns:
            Registered triplet match
        """
        coherence = self.fractal_core.compute_coherence(states)
        
        match = TripletMatch(
            states=states,
            coherence=coherence,
            timestamp=time.time()
        )
        
        self.match_history.append(match)
        
        # Register mirror if coherence is high
        if coherence > 0.9:
            quantized = self.fractal_core.quantize_vector(states[0].vector)
            self.fractal_core.register_mirror(quantized)
            
        return match
    
    def check_collapse(self) -> bool:
        """
        Check if enough matches indicate a recursive collapse.
        
        Returns:
            True if collapse detected
        """
        recent_matches = [
            m for m in self.match_history
            if time.time() - m.timestamp < 300  # Last 5 minutes
        ]
        
        return len(recent_matches) >= self.collapse_threshold
    
    def get_match_stats(self) -> Dict[str, float]:
        """
        Get statistics about triplet matches.
        
        Returns:
            Dictionary of statistics
        """
        if not self.match_history:
            return {}
            
        total_matches = len(self.match_history)
        mirror_matches = sum(1 for m in self.match_history if m.is_mirror)
        
        # Calculate average coherence
        avg_coherence = sum(m.coherence for m in self.match_history) / total_matches
        
        return {
            'total_matches': total_matches,
            'mirror_matches': mirror_matches,
            'mirror_ratio': mirror_matches / total_matches,
            'avg_coherence': avg_coherence
        } 