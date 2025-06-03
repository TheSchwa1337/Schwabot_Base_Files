"""
Time Entropy Edge Case Handler
Manages edge cases in time-based entropy calculations.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class EntropyState:
    """Represents the current entropy state"""
    entropy: float
    timestamp: float
    confidence: float
    edge_case_detected: bool

class TimeEntropyEdgeCase:
    """Handles edge cases in time-based entropy calculations"""
    
    def __init__(self):
        self.entropy_history: Dict[str, EntropyState] = {}
        self.edge_case_threshold = 0.8
        self.min_confidence = 0.6
        
    def evaluate(self, timestamp: float, entropy: float, 
                confidence: float = 1.0) -> bool:
        """
        Evaluate if current state represents an edge case.
        
        Args:
            timestamp: Current timestamp
            entropy: Current entropy value
            confidence: Confidence in the entropy calculation
            
        Returns:
            bool: True if edge case is detected
        """
        # Store current state
        state = EntropyState(
            entropy=entropy,
            timestamp=timestamp,
            confidence=confidence,
            edge_case_detected=False
        )
        
        # Check for edge cases
        if confidence < self.min_confidence:
            state.edge_case_detected = True
            return True
            
        if entropy > self.edge_case_threshold:
            state.edge_case_detected = True
            return True
            
        # Store state
        self.entropy_history[str(timestamp)] = state
        
        return False
        
    def get_entropy_state(self, timestamp: float) -> Optional[EntropyState]:
        """Get entropy state for a specific timestamp"""
        return self.entropy_history.get(str(timestamp))
        
    def clear_history(self):
        """Clear entropy history"""
        self.entropy_history.clear() 