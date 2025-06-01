"""
Braid Pattern Engine
===================

Implements braid group pattern analysis and classification,
inspired by Braid Group Bₙ topology. Each cursor movement is a "strand",
with triplet delta representing strand permutation order and
braid angle representing mod-360 twist for rotational memory encoding.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class BraidPattern:
    """Represents a detected braid pattern"""
    name: str
    confidence: float
    strands: List[Tuple[int, float]]
    metadata: Dict

class BraidPatternEngine:
    """Analyzes and classifies braid patterns from cursor movements"""
    
    def __init__(self):
        self.braids: List[Tuple[int, float]] = []
        self.patterns: Dict[str, List[Tuple[int, float]]] = {
            "Golden Entry Braid": [(+2, 64), (-1, 289), (+3, 81)],
            "Spiral Exit Braid": [(-2, 196), (+1, 121), (-3, 169)],
            "Ghost Echo Braid": [(+1, 45), (+1, 90), (+1, 135)],
            "Quantum Flip Braid": [(-3, 270), (+3, 90), (-3, 270)]
        }
        
    def add_strand(self, delta_idx: int, angle: float):
        """Add a new strand to the braid sequence"""
        self.braids.append((delta_idx, angle))
        if len(self.braids) > 16:  # Keep last 16 strands
            self.braids.pop(0)
            
    def classify(self, length: int = 3) -> Optional[BraidPattern]:
        """Classify the current braid pattern"""
        if len(self.braids) < length:
            return None
            
        current = self.braids[-length:]
        
        # Check against known patterns
        for name, pattern in self.patterns.items():
            if self._pattern_matches(current, pattern):
                return BraidPattern(
                    name=name,
                    confidence=self._calculate_confidence(current, pattern),
                    strands=current,
                    metadata={"length": length}
                )
                
        return None
    
    def _pattern_matches(self, current: List[Tuple[int, float]], 
                        pattern: List[Tuple[int, float]], 
                        tolerance: float = 0.1) -> bool:
        """Check if current braid matches a known pattern"""
        if len(current) != len(pattern):
            return False
            
        for (c_delta, c_angle), (p_delta, p_angle) in zip(current, pattern):
            if c_delta != p_delta:
                return False
            if abs(c_angle - p_angle) > tolerance:
                return False
        return True
    
    def _calculate_confidence(self, current: List[Tuple[int, float]], 
                            pattern: List[Tuple[int, float]]) -> float:
        """Calculate confidence score for pattern match"""
        angle_diffs = [abs(c[1] - p[1]) for c, p in zip(current, pattern)]
        return 1.0 - (sum(angle_diffs) / (360.0 * len(pattern)))
    
    def get_braid_history(self, limit: Optional[int] = None) -> List[Tuple[int, float]]:
        """Get braid history, optionally limited to last N strands"""
        if limit is None:
            return self.braids
        return self.braids[-limit:]
    
    def clear_history(self):
        """Clear braid history"""
        self.braids.clear()
        
    def add_custom_pattern(self, name: str, pattern: List[Tuple[int, float]]):
        """Add a custom pattern to the pattern library"""
        self.patterns[name] = pattern 