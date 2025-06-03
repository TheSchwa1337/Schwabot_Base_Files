"""
Bitmap Engine
Handles bitmap generation and SHA key management for market patterns.
"""

from typing import Dict, List, Optional
import hashlib
import numpy as np
from dataclasses import dataclass

@dataclass
class BitmapPattern:
    """Represents a bitmap pattern with metadata"""
    pattern: List[int]
    timestamp: float
    entropy: float
    confidence: float

class BitmapEngine:
    """Manages bitmap generation and pattern matching"""
    
    def __init__(self):
        self.pattern_cache: Dict[str, BitmapPattern] = {}
        self.last_update: Dict[str, float] = {}
        
    def infer_current_bitmap(self, market_signature: Dict) -> List[int]:
        """
        Infer current bitmap from market signature.
        
        Args:
            market_signature: Dictionary containing market data
            
        Returns:
            List[int]: Current bitmap pattern
        """
        # Generate a deterministic pattern based on market signature
        pattern = []
        for key, value in sorted(market_signature.items()):
            # Convert value to integer and take modulo 2 for binary pattern
            pattern.append(int(hash(str(value))) % 2)
            
        return pattern
        
    def generate_sha_key(self, bitmap: List[int]) -> str:
        """
        Generate SHA key from bitmap pattern.
        
        Args:
            bitmap: List of binary values representing the pattern
            
        Returns:
            str: SHA-256 hash of the pattern
        """
        # Convert bitmap to bytes
        pattern_bytes = bytes(bitmap)
        
        # Generate SHA-256 hash
        sha_key = hashlib.sha256(pattern_bytes).hexdigest()
        
        return sha_key
        
    def store_pattern(self, sha_key: str, pattern: List[int], 
                     entropy: float, confidence: float) -> None:
        """Store a bitmap pattern with metadata"""
        self.pattern_cache[sha_key] = BitmapPattern(
            pattern=pattern,
            timestamp=0.0,  # Will be set by caller
            entropy=entropy,
            confidence=confidence
        )
        
    def get_pattern(self, sha_key: str) -> Optional[BitmapPattern]:
        """Retrieve a stored pattern by SHA key"""
        return self.pattern_cache.get(sha_key)
        
    def clear_cache(self):
        """Clear pattern cache"""
        self.pattern_cache.clear()
        self.last_update.clear() 