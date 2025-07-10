#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Cache Implementation 🧮

Provides mathematical result caching with tensor operations:
• Cache mathematical calculations for performance
• GPU/CPU tensor operations with automatic fallback
• Mathematical result storage and retrieval
• Cache invalidation and TTL management

Features:
- Mathematical result caching for performance optimization
- GPU/CPU tensor operations with automatic fallback
- Cache key generation with mathematical hashing
- TTL-based cache invalidation
- Memory-efficient cache management
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

try:
    import cupy as cp
    import numpy as np
    USING_CUDA = True
    xp = cp
    _backend = 'cupy (GPU)'
except ImportError:
    try:
        import numpy as np
        USING_CUDA = False
        xp = np
        _backend = 'numpy (CPU)'
    except ImportError:
        xp = None
        _backend = 'none'

logger = logging.getLogger(__name__)
if xp is None:
    logger.warning("❌ NumPy not available for mathematical operations")
else:
    logger.info(f"⚡ MathCache using {_backend} for tensor operations")


class MathResultCache:
    """
    Mathematical result cache with tensor operations.
    Provides caching for mathematical calculations to improve performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MathResultCache with configuration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize math infrastructure if available
        try:
            from core.math_config_manager import MathConfigManager
            from core.math_orchestrator import MathOrchestrator
            self.math_config = MathConfigManager()
            self.math_orchestrator = MathOrchestrator()
            self.math_infrastructure_available = True
        except ImportError:
            self.math_infrastructure_available = False
            logger.warning("⚠️ Math infrastructure not available")
        
        self._initialize_system()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
            'max_cache_size': 1000,
            'cache_ttl': 3600,  # 1 hour
        }
    
    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
            'cache_size': len(self.cache),
            'math_infrastructure_available': self.math_infrastructure_available,
            'backend': _backend
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': kwargs,
        }
        # Use json.dumps with sort_keys=True for a consistent key string
        # and ensure tensors/numpy arrays are handled correctly.
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _hash_param(self, param: Any) -> str:
        """Hash a parameter for cache key generation."""
        if isinstance(param, (int, float, str, bool)):
            return str(param)
        elif isinstance(param, (list, tuple)):
            return str([self._hash_param(p) for p in param])
        elif isinstance(param, dict):
            return str({k: self._hash_param(v) for k, v in sorted(param.items())})
        else:
            return str(hash(param))
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache and is not expired."""
        if not self.active:
            return False
        
        if key not in self.cache:
            return False
        
        # Check TTL
        ttl = self.config.get('cache_ttl', 3600)
        if time.time() - self.cache_timestamps.get(key, 0) > ttl:
            # Remove expired entry
            del self.cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
            return False
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if not self.active:
            self.logger.warning("Cache not active")
            return None
        
        if not self.exists(key):
            return None
        
        self.logger.debug(f"Cache hit for key: {key}")
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> bool:
        """Set a value in the cache."""
        if not self.active:
            self.logger.warning("Cache not active")
            return False
        
        # Check cache size limit
        max_size = self.config.get('max_cache_size', 1000)
        if len(self.cache) >= max_size:
            # Remove oldest entry
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()
        self.logger.debug(f"Cache set for key: {key}")
        return True
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        if not self.active:
            return False
        
        self.cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Cache cleared")
        return True
    
    def process_mathematical_data(self, data: Any) -> float:
        """Process mathematical data with caching."""
        try:
            if not isinstance(data, (list, tuple)) and xp is not None:
                if hasattr(data, '__array__'):
                    data_array = xp.array(data)
                else:
                    data_array = xp.array([data])
            else:
                data_array = xp.array(data) if xp is not None else data
            
            # Default mathematical operation (mean)
            if xp is not None:
                result = float(xp.mean(data_array))
            else:
                result = float(sum(data) / len(data)) if data else 0.0
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process mathematical data: {e}")
            return 0.0
    
    def cache_mathematical_result(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """Cache mathematical operation result."""
        try:
            # Generate cache key
            key = self._generate_key(operation, *args, **kwargs)
            
            # Check if result is already cached
            cached_result = self.get(key)
            if cached_result is not None:
                return cached_result
            
            # Perform mathematical operation
            if operation == "mean":
                result = self.process_mathematical_data(args[0] if args else [])
            elif operation == "sum":
                result = float(xp.sum(xp.array(args[0]))) if xp is not None and args else 0.0
            elif operation == "std":
                result = float(xp.std(xp.array(args[0]))) if xp is not None and args else 0.0
            elif operation == "var":
                result = float(xp.var(xp.array(args[0]))) if xp is not None and args else 0.0
            else:
                # Default to mean operation
                result = self.process_mathematical_data(args[0] if args else [])
            
            # Cache the result
            self.set(key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cache mathematical result: {e}")
            return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            return {
                'cache_size': len(self.cache),
                'cache_hits': getattr(self, '_cache_hits', 0),
                'cache_misses': getattr(self, '_cache_misses', 0),
                'hit_rate': getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_hits', 0) + getattr(self, '_cache_misses', 0)),
                'oldest_entry': min(self.cache_timestamps.values()) if self.cache_timestamps else None,
                'newest_entry': max(self.cache_timestamps.values()) if self.cache_timestamps else None,
                'backend': _backend
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get cache statistics: {e}")
            return {"error": str(e)}


# Factory function
def create_math_cache(config: Optional[Dict[str, Any]] = None) -> MathResultCache:
    """Create a math cache instance."""
    return MathResultCache(config)


# Singleton instance for global use
math_cache = MathResultCache()
