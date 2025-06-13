"""
Pattern Match Visualizer
======================

Visualizes pattern matches and confidence scores for braid patterns,
providing human-readable output for analysis and debugging.
"""

from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class PatternMatch:
    name: str
    confidence: float
    strands: List[Tuple[int, float]]
    metadata: Dict

class PatternVisualizer:
    def __init__(self, bar_width: int = 20):
        self.bar_width = bar_width
        
    def _format_confidence_bar(self, confidence: float) -> str:
        """Format confidence score as ASCII bar"""
        filled = int(self.bar_width * confidence)
        return f"{'█' * filled}{'.' * (self.bar_width - filled)}"
        
    def visualize_match(self, match: PatternMatch) -> str:
        """Create visual representation of a pattern match"""
        confidence_bar = self._format_confidence_bar(match.confidence)
        
        # Format strands
        strand_str = "\n".join(
            f"  Strand {i+1}: {delta:+d} @ {angle:.1f}°"
            for i, (delta, angle) in enumerate(match.strands)
        )
        
        # Build visualization
        lines = [
            f"[{match.name}] Match Confidence: {match.confidence:.2f} {confidence_bar}",
            "Strands:",
            strand_str
        ]
        
        # Add metadata if present
        if match.metadata:
            meta_lines = ["Metadata:"]
            for key, value in match.metadata.items():
                meta_lines.append(f"  {key}: {value}")
            lines.extend(meta_lines)
            
        return "\n".join(lines)
        
    def visualize_matches(self, matches: List[PatternMatch]) -> str:
        """Visualize multiple pattern matches"""
        if not matches:
            return "No pattern matches found"
            
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Visualize each match
        lines = ["Pattern Matches:"]
        for i, match in enumerate(matches, 1):
            lines.append(f"\nMatch {i}:")
            lines.append(self.visualize_match(match))
            
        return "\n".join(lines)
        
    def visualize_confidence_matrix(self, matches: List[PatternMatch]) -> str:
        """Create ASCII matrix of confidence scores"""
        if not matches:
            return "No pattern matches found"
            
        # Get unique pattern names
        names = sorted(set(m.name for m in matches))
        
        # Create matrix
        matrix = np.zeros((len(names), len(matches)))
        for i, name in enumerate(names):
            for j, match in enumerate(matches):
                if match.name == name:
                    matrix[i, j] = match.confidence
                    
        # Format matrix
        lines = ["Confidence Matrix:"]
        lines.append(" " * 20 + "".join(f"{i+1:4d}" for i in range(len(matches))))
        
        for i, name in enumerate(names):
            row = f"{name:20s}"
            for j in range(len(matches)):
                conf = matrix[i, j]
                if conf > 0:
                    bar = self._format_confidence_bar(conf)
                    row += f"{bar:4s}"
                else:
                    row += "    "
            lines.append(row)
            
        return "\n".join(lines) 