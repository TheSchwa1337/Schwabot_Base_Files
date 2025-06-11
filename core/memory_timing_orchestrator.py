"""
Memory Timing Orchestrator
=========================

Coordinates dynamic timing, memory management, and hash function selection
across the Schwabot system. Integrates with UMPipeline, TimingManager, and
TripletMatcher for optimal memory-key timing and thermal-aware execution.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
import numpy as np
import hashlib
import time
from datetime import datetime
import logging
from pathlib import Path
import yaml
import threading
from queue import Queue
import psutil

from .timing_manager import TimingManager, TimingState
from .ump_pipeline import UMPipeline, MemoryEntry
from .triplet_matcher import TripletMatcher, TripletMatch
from .zbe_temperature_tensor import ZBETemperatureTensor

logger = logging.getLogger(__name__)

@dataclass
class MemoryKey:
    """Memory key with timing information"""
    key: int
    timestamp: float
    access_count: int
    last_access: float
    memory_usage: int
    thermal_state: float

@dataclass
class HashFunctionProfile:
    """Profile for hash function performance"""
    name: str
    bit_depth: int
    thermal_efficiency: float
    memory_efficiency: float
    success_rate: float
    last_used: float
    usage_count: int = 0

class MemoryTimingOrchestrator:
    """Orchestrates memory timing and hash function selection"""
    
    def __init__(self):
        self.timing_manager = TimingManager()
        self.ump_pipeline = UMPipeline()
        self.triplet_matcher = TripletMatcher()
        self.zbe_tensor = ZBETemperatureTensor()
        
        # Memory key storage
        self.memory_keys: Dict[int, MemoryKey] = {}
        
        # Hash function profiles
        self.hash_profiles = {
            'sha256': HashFunctionProfile('sha256', 256, 1.0, 1.0, 0.0, 0.0),
            'sha512': HashFunctionProfile('sha512', 512, 1.5, 1.2, 0.0, 0.0),
            'blake2b': HashFunctionProfile('blake2b', 256, 0.8, 0.9, 0.0, 0.0),
            'keccak': HashFunctionProfile('keccak', 256, 1.2, 1.1, 0.0, 0.0)
        }

    def select_hash_function(self, bit_depth: int, thermal_state: float) -> str:
        """
        Select optimal hash function based on bit depth and thermal state.
        
        Args:
            bit_depth: Current bit depth (4, 8, 16, 42, 81)
            thermal_state: Current CPU temperature
            
        Returns:
            Selected hash function name
        """
        # Filter functions by bit depth compatibility
        compatible = {
            name: profile for name, profile in self.hash_profiles.items()
            if profile.bit_depth >= bit_depth
        }
        
        if not compatible:
            return 'sha256'  # Default fallback
            
        # Calculate selection scores
        scores = {}
        for name, profile in compatible.items():
            # Base score from success rate
            score = profile.success_rate
            
            # Adjust for thermal impact
            thermal_penalty = (profile.thermal_efficiency * thermal_state) / 100.0
            score -= thermal_penalty
            
            # Adjust for memory impact
            memory_penalty = profile.memory_efficiency * 0.1
            score -= memory_penalty
            
            # Add recency bonus
            time_since_last = time.time() - profile.last_used
            recency_bonus = min(1.0, time_since_last / 3600.0)  # 1 hour max
            score += recency_bonus * 0.2
            
            scores[name] = score
            
        # Select best scoring function
        return max(scores.items(), key=lambda x: x[1])[0]

    def generate_memory_key(self, data: bytes, bit_depth: int) -> MemoryKey:
        """
        Generate a new memory key with timing metadata.
        
        Args:
            data: Data to hash
            bit_depth: Current bit depth
            
        Returns:
            New memory key
        """
        # Get current thermal state
        thermal_state = self.zbe_tensor.read_cpu_temperature()
        
        # Select hash function
        hash_func = self.select_hash_function(bit_depth, thermal_state)
        
        # Update hash function profile
        profile = self.hash_profiles[hash_func]
        profile.usage_count += 1
        profile.last_used = time.time()
        
        # Generate hash
        hash_value = hash(data)
        
        # Create memory key
        key = MemoryKey(
            key=hash_value,
            timestamp=time.time(),
            access_count=0,
            last_access=time.time(),
            memory_usage=0,
            thermal_state=thermal_state
        )
        
        # Store key
        self.memory_keys[hash_value] = key
        
        return key

    def access_memory_key(self, hash_value: int) -> Optional[MemoryKey]:
        """
        Access a memory key and update its metadata.
        
        Args:
            hash_value: Hash value to access
            
        Returns:
            Memory key if found
        """
        if hash_value not in self.memory_keys:
            return None
            
        key = self.memory_keys[hash_value]
        key.access_count += 1
        key.last_access = time.time()
        
        # Update memory weight based on access pattern
        time_since_creation = time.time() - key.timestamp
        access_frequency = key.access_count / (time_since_creation + 1)
        key.memory_usage = min(1.0, access_frequency * 0.1)
        
        return key

    def update_success_score(self, hash_value: int, success_score: float):
        """
        Update success score for a memory key and its hash function.
        
        Args:
            hash_value: Hash value to update
            success_score: New success score
        """
        if hash_value not in self.memory_keys:
            return
            
        key = self.memory_keys[hash_value]
        key.success_rate = success_score
        
        # Update hash function profile
        for profile in self.hash_profiles.values():
            if profile.bit_depth >= key.memory_usage:
                # Update success rate with exponential moving average
                profile.success_rate = (0.9 * profile.success_rate + 
                                     0.1 * success_score)

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage and timing"""
        return {
            'total_keys': len(self.memory_keys),
            'active_keys': sum(1 for k in self.memory_keys.values() 
                             if time.time() - k.last_access < 3600),
            'hash_function_stats': {
                name: {
                    'usage_count': profile.usage_count,
                    'success_rate': profile.success_rate,
                    'thermal_efficiency': profile.thermal_efficiency
                }
                for name, profile in self.hash_profiles.items()
            },
            'memory_weight_distribution': {
                'high': sum(1 for k in self.memory_keys.values() 
                          if k.memory_usage > 0.7),
                'medium': sum(1 for k in self.memory_keys.values() 
                            if 0.3 <= k.memory_usage <= 0.7),
                'low': sum(1 for k in self.memory_keys.values() 
                          if k.memory_usage < 0.3)
            }
        }

    def cleanup_old_keys(self, max_age_hours: int = 24):
        """
        Remove old memory keys.
        
        Args:
            max_age_hours: Maximum age in hours
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_keys = [
            hash_value for hash_value, key in self.memory_keys.items()
            if current_time - key.timestamp > max_age_seconds
        ]
        
        for hash_value in old_keys:
            del self.memory_keys[hash_value]

# Example usage
if __name__ == "__main__":
    orchestrator = MemoryTimingOrchestrator()
    
    # Generate test memory key
    test_data = b"test_data"
    key = orchestrator.generate_memory_key(test_data, bit_depth=16)
    print(f"Generated key: {key}")
    
    # Access key
    accessed_key = orchestrator.access_memory_key(key.key)
    print(f"Accessed key: {accessed_key}")
    
    # Update success score
    orchestrator.update_success_score(key.key, 0.8)
    
    # Get stats
    stats = orchestrator.get_memory_stats()
    print("\nMemory Stats:")
    print(stats) 