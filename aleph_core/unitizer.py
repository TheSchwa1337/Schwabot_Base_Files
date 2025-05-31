"""
Core Aleph Unitizer - Hash processing and signature generation.
"""

import hashlib
import time
from typing import Dict, List, Tuple

class AlephUnitizer:
    """Core hash unitization engine."""
    
    def __init__(self):
        self.processed_count = 0
        self.collision_tracker = {}
        
    def generate_hash_signature(self, data: str) -> Dict[str, any]:
        """
        Generate complete hash signature with multiple representations.
        """
        full_hash = hashlib.sha256(data.encode()).hexdigest()
        short_tag = full_hash[:8]
        
        # Track collisions
        if short_tag in self.collision_tracker:
            self.collision_tracker[short_tag] += 1
        else:
            self.collision_tracker[short_tag] = 1
        
        # Calculate entropy with cyclic properties
        entropy_sum = sum(ord(c) for c in short_tag)
        entropy_tag = entropy_sum % 144
        
        # Additional metrics
        hex_variance = sum((int(c, 16) - 7.5)**2 for c in short_tag) / 8
        
        signature = {
            'full_hash': full_hash,
            'short_tag': short_tag,
            'entropy_tag': entropy_tag,
            'entropy_sum': entropy_sum,
            'hex_variance': hex_variance,
            'collision_count': self.collision_tracker[short_tag],
            'timestamp': time.time(),
            'sequence_id': self.processed_count
        }
        
        self.processed_count += 1
        return signature 