#!/usr/bin/env python3
"""
Centralized Math Results Cache
Caches mathematical operation results to avoid redundant calculations.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np


class MathResultsCache:
    """Cache for mathematical operation results."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_key(self, operation: str, params: Any) -> str:
        """Generate cache key from operation and parameters."""
        # Convert params to a hashable format
        if isinstance(params, (list, tuple)):
            param_str = str([self._hash_param(p) for p in params])
        elif isinstance(params, dict):
            param_str = str({k: self._hash_param(v) for k, v in sorted(params.items())})
        else:
            param_str = str(self._hash_param(params))
        
        key_data = f"{operation}:{param_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _hash_param(self, param: Any) -> str:
        """Hash a parameter for cache key generation."""
        if isinstance(param, np.ndarray):
            return hashlib.sha256(param.tobytes()).hexdigest()
        elif isinstance(param, (list, tuple)):
            return str([self._hash_param(p) for p in param])
        elif isinstance(param, dict):
            return str({k: self._hash_param(v) for k, v in sorted(param.items())})
        else:
            return str(param)
    
    def get(self, operation: str, params: Any) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(operation, params)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return entry["result"]
            else:
                # Expired, remove
                del self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, operation: str, params: Any, result: Any):
        """Cache a result."""
        key = self._generate_key(operation, params)
        
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
            "operation": operation
        }
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        
        # Evict if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }
    
    def get_or_compute(self, operation: str, params: Any, compute_func) -> Any:
        """Get cached result or compute and cache it."""
        cached_result = self.get(operation, params)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        result = compute_func()
        self.set(operation, params, result)
        return result

# Global cache instance
math_cache = MathResultsCache()

def get_math_cache() -> MathResultsCache:
    """Get the global math cache instance."""
    return math_cache
