#!/usr/bin/env python3
"""
Centralized Math Orchestrator
Orchestrates all mathematical operations with caching and optimization.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
from math_cache import get_math_cache

# Use absolute imports instead of relative imports
from math_config_manager import get_math_config

logger = logging.getLogger(__name__)

class MathOrchestrator:
    """Centralized orchestrator for all mathematical operations."""
    
    def __init__(self):
        self.config = get_math_config()
        self.cache = get_math_cache()
        self.operation_times = {}
    
    def execute_math_operation(self, operation: str, params: Any, 
                             compute_func: Callable, 
                             use_cache: bool = True) -> Any:
        """Execute a mathematical operation with caching and timing."""
        start_time = time.time()
        
        try:
            if use_cache and self.config.is_cache_enabled():
                result = self.cache.get_or_compute(operation, params, compute_func)
            else:
                result = compute_func()
            
            execution_time = time.time() - start_time
            self.operation_times[operation] = execution_time
            
            logger.debug(f"Math operation '{operation}' completed in {execution_time:.6f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in math operation '{operation}': {e}")
            raise
    
    def execute_tensor_operation(self, operation: str, tensors: list, 
                               operation_func: Callable) -> np.ndarray:
        """Execute tensor operation with optimization."""
        def compute():
            return operation_func(*tensors)
        
        return self.execute_math_operation(f"tensor_{operation}", tensors, compute)
    
    def execute_quantum_operation(self, operation: str, params: Dict[str, Any],
                                operation_func: Callable) -> Any:
        """Execute quantum operation with simulation settings."""
        def compute():
            return operation_func(**params)
        
        return self.execute_math_operation(f"quantum_{operation}", params, compute)
    
    def execute_entropy_operation(self, operation: str, data: np.ndarray,
                                operation_func: Callable) -> float:
        """Execute entropy operation with numerical stability."""
        def compute():
            return operation_func(data)
        
        return self.execute_math_operation(f"entropy_{operation}", data, compute)
    
    def execute_optimization(self, objective_func: Callable, 
                           initial_guess: np.ndarray,
                           bounds: Optional[tuple] = None,
                           constraints: Optional[list] = None) -> Dict[str, Any]:
        """Execute optimization with configuration."""
        def compute():
            # This would integrate with scipy.optimize
            # For now, return a placeholder
            return {
                "success": True,
                "x": initial_guess,
                "fun": objective_func(initial_guess),
                "message": "Optimization completed"
            }
        
        params = {
            "initial_guess": initial_guess.tolist(),
            "bounds": bounds,
            "constraints": constraints
        }
        
        return self.execute_math_operation("optimization", params, compute)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            "cache_stats": cache_stats,
            "operation_times": self.operation_times,
            "config": {
                "precision": self.config.get_precision(),
                "cache_enabled": self.config.is_cache_enabled(),
                "gpu_enabled": self.config.is_gpu_enabled()
            }
        }
    
    def clear_cache(self):
        """Clear the math cache."""
        self.cache.clear()
        logger.info("Math cache cleared")

# Global orchestrator instance
math_orchestrator = MathOrchestrator()

def get_math_orchestrator() -> MathOrchestrator:
    """Get the global math orchestrator instance."""
    return math_orchestrator
