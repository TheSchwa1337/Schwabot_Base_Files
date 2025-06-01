"""
Truth Collapse Engine
===================

Implements the truth collapse mapping system that detects closed loops
in the braid pattern sequence and generates mirror trades or reverse
ghost entries when appropriate.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .cursor_engine import Cursor, CursorState
from .braid_pattern_engine import BraidPattern, BraidPatternEngine

@dataclass
class CollapseState:
    """Represents the state of a collapse detection"""
    is_collapsed: bool
    loop_sum: float
    mirror_trades: List[Dict]
    ghost_entries: List[Dict]
    confidence: float
    metadata: Dict

class CollapseEngine:
    """Manages truth collapse detection and trade generation"""
    
    def __init__(self, loop_threshold: float = 0.01):
        self.loop_threshold = loop_threshold
        self.collapse_history: List[CollapseState] = []
        self.mirror_trade_history: List[Dict] = []
        self.ghost_entry_history: List[Dict] = []
        
    def check_collapse(self, cursor_states: List[CursorState], 
                      pattern: Optional[BraidPattern] = None) -> CollapseState:
        """
        Check for truth collapse in the sequence
        
        Args:
            cursor_states: List of recent cursor states
            pattern: Current braid pattern (optional)
            
        Returns:
            CollapseState indicating if collapse occurred
        """
        # Calculate loop sum
        loop_sum = self._calculate_loop_sum(cursor_states)
        
        # Check if loop is closed
        is_collapsed = abs(loop_sum) < self.loop_threshold
        
        # Generate mirror trades and ghost entries if collapsed
        mirror_trades = []
        ghost_entries = []
        confidence = 0.0
        
        if is_collapsed:
            mirror_trades = self._generate_mirror_trades(cursor_states)
            ghost_entries = self._generate_ghost_entries(cursor_states)
            confidence = self._calculate_collapse_confidence(cursor_states)
        
        # Create collapse state
        state = CollapseState(
            is_collapsed=is_collapsed,
            loop_sum=loop_sum,
            mirror_trades=mirror_trades,
            ghost_entries=ghost_entries,
            confidence=confidence,
            metadata={
                "pattern": pattern.name if pattern else None,
                "timestamp": cursor_states[-1].timestamp if cursor_states else None
            }
        )
        
        # Update history
        self.collapse_history.append(state)
        if is_collapsed:
            self.mirror_trade_history.extend(mirror_trades)
            self.ghost_entry_history.extend(ghost_entries)
            
        return state
    
    def _calculate_loop_sum(self, states: List[CursorState]) -> float:
        """Calculate the sum of deltas in the sequence"""
        if not states:
            return 0.0
            
        # Sum the delta indices
        return sum(state.delta_idx for state in states)
    
    def _generate_mirror_trades(self, states: List[CursorState]) -> List[Dict]:
        """Generate mirror trades based on collapsed sequence"""
        trades = []
        
        if not states:
            return trades
            
        # Analyze the sequence for mirror opportunities
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            
            # Check for potential mirror point
            if abs(current.delta_idx) == abs(next_state.delta_idx):
                trade = {
                    "entry_price": current.triplet[0],
                    "exit_price": next_state.triplet[0],
                    "direction": "long" if current.delta_idx > 0 else "short",
                    "confidence": self._calculate_trade_confidence(current, next_state),
                    "timestamp": current.timestamp
                }
                trades.append(trade)
                
        return trades
    
    def _generate_ghost_entries(self, states: List[CursorState]) -> List[Dict]:
        """Generate ghost entries based on collapsed sequence"""
        entries = []
        
        if not states:
            return entries
            
        # Find potential ghost entry points
        for i in range(len(states) - 2):
            current = states[i]
            next_state = states[i + 1]
            future = states[i + 2]
            
            # Check for ghost pattern
            if (current.delta_idx == -next_state.delta_idx and 
                abs(future.delta_idx) > abs(current.delta_idx)):
                entry = {
                    "price": current.triplet[0],
                    "direction": "long" if future.delta_idx > 0 else "short",
                    "confidence": self._calculate_ghost_confidence(current, next_state, future),
                    "timestamp": current.timestamp
                }
                entries.append(entry)
                
        return entries
    
    def _calculate_collapse_confidence(self, states: List[CursorState]) -> float:
        """Calculate confidence in collapse detection"""
        if not states:
            return 0.0
            
        # Base confidence on sequence length and loop tightness
        length_factor = min(len(states) / 10.0, 1.0)
        tightness = 1.0 - (abs(self._calculate_loop_sum(states)) / self.loop_threshold)
        
        return length_factor * tightness
    
    def _calculate_trade_confidence(self, state1: CursorState, 
                                  state2: CursorState) -> float:
        """Calculate confidence in mirror trade"""
        # Base confidence on angle alignment
        angle_diff = abs(state1.braid_angle - state2.braid_angle) % 360.0
        angle_factor = 1.0 - (angle_diff / 180.0)
        
        # Consider delta magnitude
        delta_factor = min(abs(state1.delta_idx) / 3.0, 1.0)
        
        return angle_factor * delta_factor
    
    def _calculate_ghost_confidence(self, state1: CursorState, 
                                  state2: CursorState,
                                  state3: CursorState) -> float:
        """Calculate confidence in ghost entry"""
        # Check for alternating pattern
        if state1.delta_idx != -state2.delta_idx:
            return 0.0
            
        # Base confidence on future delta magnitude
        magnitude_factor = min(abs(state3.delta_idx) / 3.0, 1.0)
        
        # Consider angle progression
        angle_progression = (state3.braid_angle - state1.braid_angle) % 360.0
        angle_factor = 1.0 - (abs(angle_progression - 180.0) / 180.0)
        
        return magnitude_factor * angle_factor
    
    def get_collapse_history(self, limit: Optional[int] = None) -> List[CollapseState]:
        """Get collapse detection history"""
        if limit is None:
            return self.collapse_history
        return self.collapse_history[-limit:]
    
    def get_mirror_trade_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get mirror trade history"""
        if limit is None:
            return self.mirror_trade_history
        return self.mirror_trade_history[-limit:]
    
    def get_ghost_entry_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get ghost entry history"""
        if limit is None:
            return self.ghost_entry_history
        return self.ghost_entry_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear all history"""
        self.collapse_history.clear()
        self.mirror_trade_history.clear()
        self.ghost_entry_history.clear()

# Example usage
if __name__ == "__main__":
    # Initialize collapse engine
    engine = CollapseEngine()
    
    # Test collapse detection
    test_states = [
        CursorState(triplet=(0.1, 0.2, 0.3), delta_idx=1, braid_angle=45.0, timestamp=1.0),
        CursorState(triplet=(0.2, 0.3, 0.4), delta_idx=-1, braid_angle=225.0, timestamp=2.0),
        CursorState(triplet=(0.3, 0.4, 0.5), delta_idx=1, braid_angle=45.0, timestamp=3.0)
    ]
    
    collapse_state = engine.check_collapse(test_states)
    print(f"Collapse detected: {collapse_state.is_collapsed}")
    print(f"Loop sum: {collapse_state.loop_sum}")
    print(f"Mirror trades: {collapse_state.mirror_trades}")
    print(f"Ghost entries: {collapse_state.ghost_entries}") 