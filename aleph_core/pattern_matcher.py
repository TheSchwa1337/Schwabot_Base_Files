"""
Pattern Matching Engine - Similarity analysis and contextual matching.
"""

from typing import List, Dict, Tuple
import numpy as np

class PatternMatcher:
    """Advanced pattern similarity analysis."""
    
    def __init__(self):
        self.similarity_cache = {}
        self.match_history = []
        
    def hamming_distance(self, pattern1: List[int], pattern2: List[int]) -> int:
        """Calculate Hamming distance between patterns."""
        return sum(1 for i in range(len(pattern1)) if pattern1[i] != pattern2[i])
    
    def manhattan_distance(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate Manhattan distance between patterns."""
        return sum(abs(pattern1[i] - pattern2[i]) for i in range(len(pattern1)))
    
    def euclidean_distance(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate Euclidean distance between patterns."""
        return np.sqrt(sum((pattern1[i] - pattern2[i])**2 for i in range(len(pattern1))))
    
    def cosine_similarity(self, pattern1: List[int], pattern2: List[int]) -> float:
        """Calculate cosine similarity between pattern vectors."""
        dot_product = sum(pattern1[i] * pattern2[i] for i in range(len(pattern1)))
        norm1 = np.sqrt(sum(x**2 for x in pattern1))
        norm2 = np.sqrt(sum(x**2 for x in pattern2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def comprehensive_similarity(self, pattern1: List[int], pattern2: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive similarity metrics.
        """
        hamming_dist = self.hamming_distance(pattern1, pattern2)
        manhattan_dist = self.manhattan_distance(pattern1, pattern2)
        euclidean_dist = self.euclidean_distance(pattern1, pattern2)
        cosine_sim = self.cosine_similarity(pattern1, pattern2)
        
        # Normalize distances to [0,1] similarity scores
        hamming_sim = 1.0 - (hamming_dist / len(pattern1))
        manhattan_sim = 1.0 - (manhattan_dist / (len(pattern1) * 15))  # Max possible
        euclidean_sim = 1.0 - (euclidean_dist / np.sqrt(len(pattern1) * 15**2))
        
        # Weighted composite similarity
        composite_score = (
            0.3 * hamming_sim +
            0.25 * manhattan_sim + 
            0.25 * euclidean_sim +
            0.2 * cosine_sim
        )
        
        return {
            'hamming_similarity': hamming_sim,
            'manhattan_similarity': manhattan_sim,
            'euclidean_similarity': euclidean_sim,
            'cosine_similarity': cosine_sim,
            'composite_similarity': composite_score
        }
    
    def find_pattern_clusters(self, patterns: List[List[int]], threshold: float = 0.7) -> List[List[int]]:
        """
        Group similar patterns into clusters using similarity threshold.
        """
        clusters = []
        used_indices = set()
        
        for i, pattern in enumerate(patterns):
            if i in used_indices:
                continue
            
            cluster = [i]
            used_indices.add(i)
            
            for j, other_pattern in enumerate(patterns[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self.comprehensive_similarity(pattern, other_pattern)
                if similarity['composite_similarity'] >= threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters 