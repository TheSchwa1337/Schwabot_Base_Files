"""
Tesseract Pattern Processor - 8D pattern analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, NamedTuple
import hashlib
import scipy.stats as stats

class TesseractMetrics(NamedTuple):
    magnitude: float
    centroid_distance: float
    axis_correlation: float
    stability: float
    harmonic_ratio: float
    primary_dominance: float
    dimensional_spread: float
    normalized_entropy: float
    
    def __repr__(self):
        return (
            f"TesseractMetrics(magnitude={self.magnitude:.4f}, "
            f"centroid_distance={self.centroid_distance:.4f}, "
            f"axis_correlation={self.axis_correlation:.4f}, "
            f"stability={self.stability:.4f}, "
            f"harmonic_ratio={self.harmonic_ratio:.4f}, "
            f"primary_dominance={self.primary_dominance:.4f}, "
            f"dimensional_spread={self.dimensional_spread:.4f}, "
            f"normalized_entropy={self.normalized_entropy:.4f})"
        )

class TesseractProcessor:
    """8-dimensional pattern analysis engine."""
    
    def __init__(self):
        """
        Initializes the dimensional labels and prime-based weights
        used to interpret SHA-256 hashes into 8D market patterns.
        """
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
        
        # Normalize the pattern to a discrete distribution
        normalized = np.array(base_pattern) / sum(base_pattern) if sum(base_pattern) > 0 else np.zeros_like(base_pattern)
        
        # Calculate entropy of the normalized pattern
        entropy = stats.entropy(normalized)
        
        return base_pattern, entropy
    
    def calculate_tesseract_metrics(self, pattern: List[int], entropy: float) -> TesseractMetrics:
        """
        Calculate geometric metrics in tesseract space.
        """
        if len(pattern) != 8:
            raise ValueError("Pattern must be 8-dimensional")
        
        # Basic geometric properties
        magnitude = np.sqrt(sum(x**2 for x in pattern))
        centroid_distance = np.sqrt(sum((x - 7.5)**2 for x in pattern))
        
        # Dimensional analysis
        primary_axis = pattern[:4]
        secondary_axis = pattern[4:]
        
        axis_correlation = np.corrcoef(primary_axis, secondary_axis)[0, 1]
        
        # Harmonic analysis
        odd_sum = sum(pattern[1::2])
        even_sum = sum(pattern[::2])
        harmonic_ratio = odd_sum / even_sum if even_sum != 0 else float('inf')
        
        # Pattern stability (variance from center)
        stability = 1.0 / (1.0 + np.var(pattern))
        
        primary_dominance = sum(primary_axis) / sum(pattern) if sum(pattern) > 0 else 0
        dimensional_spread = max(pattern) - min(pattern)
        
        return TesseractMetrics(
            magnitude, centroid_distance, axis_correlation,
            stability, harmonic_ratio, primary_dominance, dimensional_spread,
            normalized_entropy=entropy
        )

    def describe_pattern(self, metrics: TesseractMetrics) -> str:
        """
        Give a qualitative description based on metrics.
        """
        if metrics.harmonic_ratio > 1.5:
            return "Bullish-biased harmonic structure"
        elif metrics.harmonic_ratio < 0.7:
            return "Bearish-biased harmonic structure"
        else:
            return "Neutral harmonic field with balanced resonance"

    def pattern_to_json(self, pattern: List[int], metrics: TesseractMetrics) -> Dict:
        """
        Allow metrics to be saved and piped directly into Schwabot memory or ColdBase archives.
        """
        return {
            "pattern": pattern,
            "metrics": metrics._asdict(),
            "labels": dict(zip(self.dimension_labels, pattern))
        }

if __name__ == "__main__":
    processor = TesseractProcessor()
    test_hash = "e3b0c44298fc1c149afbf4c8996fb924" * 2
    base_pattern, entropy = processor.extract_tesseract_pattern(test_hash)
    metrics = processor.calculate_tesseract_metrics(base_pattern, entropy)

    print("\nTesseract 8D Pattern:", base_pattern)
    print("\nMetric Breakdown:")
    for field in metrics._fields:
        print(f"  {field}: {getattr(metrics, field):.4f}")
    
    description = processor.describe_pattern(metrics)
    print("\nPattern Description:", description)

    json_result = processor.pattern_to_json(base_pattern, metrics)
    print("\nJSON Result:")
    print(json_result) 