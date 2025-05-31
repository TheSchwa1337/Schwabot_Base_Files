"""
Tesseract Pattern Processor - 8D pattern analysis.
"""

import numpy as np
from typing import List, Dict, Tuple

class TesseractProcessor:
    """8-dimensional pattern analysis engine."""
    
    def __init__(self):
        # Dimensional interpretation weights
        self.dimension_labels = [
            "Market_Sentiment", "Price_Momentum", 
            "Volume_Pressure", "Volatility_Index",
            "Temporal_Phase", "Liquidity_Depth",
            "Whale_Activity", "Technical_Signal"
        ]
        
        # Prime-based weighting for dimensional separation
        self.prime_weights = [2, 3, 5, 7, 11, 13, 17, 19]
        
    def extract_tesseract_pattern(self, full_hash: str) -> List[int]:
        """
        Extract 8D tesseract pattern from SHA-256 hash.
        """
        segments = [full_hash[i:i+4] for i in range(0, 64, 4)]
        base_pattern = []
        
        for i in range(8):
            if i < len(segments):
                segment = segments[i]
                ordinal_sum = sum(ord(c) for c in segment)
                weighted_value = (ordinal_sum * self.prime_weights[i]) % 16
                base_pattern.append(weighted_value)
            else:
                base_pattern.append(0)
        
        return base_pattern
    
    def calculate_tesseract_metrics(self, pattern: List[int]) -> Dict[str, float]:
        """
        Calculate geometric metrics in tesseract space.
        """
        if len(pattern) != 8:
            raise ValueError("Pattern must be 8-dimensional")
        
        # Basic geometric properties
        magnitude = np.sqrt(sum(x**2 for x in pattern))
        centroid_distance = np.sqrt(sum((x - 7.5)**2 for x in pattern))
        
        # Dimensional analysis
        primary_axis = pattern[:4]  # First 4 dimensions
        secondary_axis = pattern[4:]  # Last 4 dimensions
        
        axis_correlation = np.corrcoef(primary_axis, secondary_axis)[0, 1]
        
        # Pattern stability (variance from center)
        stability = 1.0 / (1.0 + np.var(pattern))
        
        # Harmonic analysis
        odd_sum = sum(pattern[1::2])  # Odd indices
        even_sum = sum(pattern[::2])  # Even indices
        harmonic_ratio = odd_sum / even_sum if even_sum != 0 else float('inf')
        
        return {
            'magnitude': magnitude,
            'centroid_distance': centroid_distance,
            'axis_correlation': axis_correlation,
            'stability': stability,
            'harmonic_ratio': harmonic_ratio,
            'primary_dominance': sum(primary_axis) / sum(pattern) if sum(pattern) > 0 else 0,
            'dimensional_spread': max(pattern) - min(pattern)
        } 