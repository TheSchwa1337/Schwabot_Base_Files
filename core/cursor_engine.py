"""
Cursor Engine
============

Implements the Quantum Triplet Walker cursor that synchronizes triplet data
through tick-deltas, aligns market structures, and maintains logs for various
patterns and triggers.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .braid_pattern_engine import BraidPatternEngine, BraidPattern

@dataclass
class CursorState:
    """Represents the current state of the cursor"""
    triplet: Tuple[float, float, float]
    delta_idx: int
    braid_angle: float
    timestamp: float

class Cursor:
    """Quantum Triplet Walker cursor implementation"""
    
    def __init__(self):
        self.state: Optional[CursorState] = None
        self.history: List[CursorState] = []
        self.pattern_engine = BraidPatternEngine()
        
    def tick(self, triplet: Tuple[float, float, float], timestamp: float) -> Optional[BraidPattern]:
        """Process a new tick with triplet data"""
        if self.state is None:
            self.state = CursorState(triplet, 0, 0.0, timestamp)
            return None
            
        # Calculate delta and braid angle
        delta_idx = self._calculate_delta(triplet, self.state.triplet)
        braid_angle = self._calculate_braid_angle(triplet, self.state.triplet)
        
        # Update state
        self.state = CursorState(triplet, delta_idx, braid_angle, timestamp)
        self.history.append(self.state)
        
        # Update pattern engine
        self.pattern_engine.add_strand(delta_idx, braid_angle)
        
        # Check for patterns
        return self.pattern_engine.classify()
        
    def _calculate_delta(self, new: Tuple[float, float, float], 
                        old: Tuple[float, float, float]) -> int:
        """Calculate the delta index between triplets"""
        deltas = [abs(n - o) for n, o in zip(new, old)]
        return np.argmax(deltas)
        
    def _calculate_braid_angle(self, new: Tuple[float, float, float], 
                              old: Tuple[float, float, float]) -> float:
        """Calculate the braid angle between triplets"""
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 360.0
        
    def get_current_pattern(self) -> Optional[BraidPattern]:
        """Get the current braid pattern"""
        return self.pattern_engine.classify()
        
    def get_pattern_frequency(self, pattern_name: str, window: int = 100) -> float:
        """Calculate frequency of a specific pattern in recent history"""
        if not self.history:
            return 0.0
            
        recent = self.history[-window:]
        pattern_count = 0
        
        for i in range(len(recent) - 2):
            pattern = self.pattern_engine.classify()
            if pattern and pattern.name == pattern_name:
                pattern_count += 1
                
        return pattern_count / len(recent)
        
    def get_history(self, limit: Optional[int] = None) -> List[CursorState]:
        """Get cursor history, optionally limited to last N states"""
        if limit is None:
            return self.history
        return self.history[-limit:]
        
    def clear_history(self):
        """Clear cursor history"""
        self.history.clear()
        self.pattern_engine.clear_history()
        
    def add_custom_pattern(self, name: str, pattern: List[Tuple[int, float]]):
        """Add a custom pattern to the pattern library"""
        self.pattern_engine.add_custom_pattern(name, pattern) 