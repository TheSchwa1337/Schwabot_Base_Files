"""
Entropy Analysis - Statistical analysis of hash distributions.
"""

import numpy as np
from collections import Counter
from typing import Dict, List

class EntropyAnalyzer:
    """Statistical analysis of entropy distributions and patterns."""
    
    def __init__(self):
        self.entropy_history = []
        self.pattern_history = []
        
    def analyze_entropy_distribution(self, entropy_values: List[int]) -> Dict:
        """
        Comprehensive statistical analysis of entropy tag distribution.
        
        Args:
            entropy_values (List[int]): A list of entropy values.
            
        Returns:
            Dict: A dictionary containing statistical analysis results.
        """
        if not entropy_values:
            raise ValueError("Entropy values cannot be empty.")
        
        counter = Counter(entropy_values)
        unique_count = len(counter)
        total_count = len(entropy_values)
        
        # Basic statistics
        mean_entropy = np.mean(entropy_values)
        std_entropy = np.std(entropy_values, ddof=1)  # Use sample standard deviation
        min_entropy = min(entropy_values)
        max_entropy = max(entropy_values)
        
        # Distribution uniformity analysis
        expected_frequency = total_count / 144  # Perfect uniform distribution
        chi_square = sum((observed - expected_frequency)**2 / expected_frequency 
                        for observed in counter.values())
        
        # Entropy of the entropy (information theory)
        probabilities = [count/total_count for count in counter.values()]
        information_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Gap analysis
        all_possible = set(range(144))
        observed_values = set(entropy_values)
        gaps = sorted(all_possible - observed_values)
        
        return {
            'total_samples': total_count,
            'unique_entropies': unique_count,
            'coverage_percentage': (unique_count / 144) * 100,
            'mean': mean_entropy,
            'std': std_entropy,
            'min': min_entropy,
            'max': max_entropy,
            'chi_square': chi_square,
            'information_entropy': information_entropy,
            'uniformity_score': 1.0 / (1.0 + chi_square/total_count) if total_count else 0.0,
            'gaps': gaps
        } 