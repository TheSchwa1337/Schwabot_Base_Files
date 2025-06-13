"""
Bitmap Engine
Handles bitmap generation and SHA key management for market patterns.
"""

from typing import Dict, List, Optional
import hashlib
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class BitmapPattern:
    """Represents a bitmap pattern with metadata"""
    pattern: List[int]
    market_signature: Dict
    timestamp: float
    entropy: float
    confidence: float

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

class BitmapEngine:
    """Manages bitmap generation and pattern matching"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.last_update = {}
        
    def infer_current_bitmap(self, market_signature: Dict) -> List[int]:
        """
        Infer current bitmap from market signature.
        
        Args:
            market_signature: Dictionary containing market data
            
        Returns:
            List[int]: Current bitmap pattern
        """
        pattern = []
        for key, value in sorted(market_signature.items()):
            h = hashlib.sha256(str(value).encode()).digest()
            pattern.extend([int(b) % 2 for b in h[:6]])  # 48 bits per field
        return pattern
        
    def compute_entropy(self, bitmap: List[int]) -> float:
        counts = np.bincount(bitmap, minlength=2)
        probs = counts / np.sum(counts)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def compute_confidence(self, bitmap: List[int]) -> float:
        return 1.0 - 2 * abs(np.mean(bitmap) - 0.5)

    def similarity_score(self, pattern_a: List[int], pattern_b: List[int]) -> float:
        if len(pattern_a) != len(pattern_b):
            raise ValueError("Patterns must be same length")
        diff = sum(a != b for a, b in zip(pattern_a, pattern_b))
        return 1.0 - (diff / len(pattern_a))

    def find_similar_patterns(self, query_bitmap: List[int], threshold: float = 0.85) -> List[BitmapPattern]:
        matches = []
        for pattern in self.pattern_cache.values():
            score = self.similarity_score(query_bitmap, pattern.pattern)
            if score >= threshold:
                matches.append(pattern)
        return matches

    def store_pattern(self, bitmap: List[int], market_signature: Dict, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        # Compute SHA-256 hash of the current bitmap
        sha_hash = hashlib.sha256(np.array(bitmap).tobytes()).hexdigest()
        
        # Store pattern with SHA and timestamp
        self.pattern_cache[sha_hash] = BitmapPattern(
            pattern=bitmap,
            market_signature=market_signature,
            timestamp=timestamp,
            entropy=self.compute_entropy(bitmap),
            confidence=self.compute_confidence(bitmap)
        )

    def retrieve_pattern(self, sha_hash: str) -> BitmapPattern:
        if sha_hash in self.pattern_cache:
            return self.pattern_cache[sha_hash]
        else:
            raise ValueError(f"Pattern with SHA {sha_hash} not found.")
        
    def clear_cache(self):
        """Clear pattern cache"""
        self.pattern_cache.clear()
        self.last_update.clear() 