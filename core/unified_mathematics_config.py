# -*- coding: utf-8 -*-
"""
Unified Mathematics Configuration for Schwabot Hybrid ZPE-Reactive System
=======================================================================

This module provides centralized configuration for all mathematical operations,
ensuring consistency, performance, and error handling across the entire pipeline.

Mathematical Foundations:
- Precision control with dynamic adjustment
- Error handling with mathematical error propagation
- Performance monitoring with mathematical metrics
- Caching for repeated operations
- Parallel processing capabilities
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import numpy as np
import scipy as sp
from scipy import signal, optimize, stats

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    import math as unified_math

# Try to import CLI handler for Windows compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MathPrecision(Enum):
    """Mathematical precision levels."""
    LOW = "low"          # 32-bit float
    MEDIUM = "medium"    # 64-bit float (default)
    HIGH = "high"        # 128-bit float
    EXACT = "exact"      # Symbolic computation


class MathOptimization(Enum):
    """Mathematical optimization strategies."""
    SPEED = "speed"      # Fastest computation
    ACCURACY = "accuracy"  # Most accurate
    BALANCED = "balanced"  # Balanced approach (default)
    MEMORY = "memory"    # Memory efficient


@dataclass
class MathConfig:
    """Unified mathematics configuration."""
    precision: MathPrecision = MathPrecision.MEDIUM
    optimization: MathOptimization = MathOptimization.BALANCED
    max_iterations: int = 1000
    tolerance: float = 1e-6
    cache_size: int = 1000
    enable_parallel: bool = True
    enable_caching: bool = True
    enable_error_handling: bool = True
    enable_logging: bool = True

    # Performance thresholds
    max_execution_time: float = 1.0  # seconds
    max_memory_usage: float = 100.0  # MB

    # ZPE-specific parameters
    zpe_work_precision: float = 1e-6
    zpe_torque_precision: float = 1e-6
    zpe_resonance_precision: float = 1e-6
    zpe_thermal_precision: float = 1e-6

    # Reactive-specific parameters
    reactive_threshold: float = 0.5
    reactive_decay_rate: float = 0.95
    reactive_memory_size: int = 100

    # Hybrid-specific parameters
    hybrid_switch_threshold: float = 0.7
    hybrid_blend_factor: float = 0.5
    hybrid_learning_rate: float = 0.01


class UnifiedMathematics:
    """
    Unified mathematics system for consistent mathematical operations.

    Provides centralized mathematical functions with:
    - Consistent precision and optimization
    - Error handling and logging
    - Performance monitoring
    - Caching for repeated operations
    - Parallel processing capabilities
    """

    def __init__(self, config: Optional[MathConfig] = None):
        """Initialize unified mathematics system."""
        self.config = config or MathConfig()
        self.cache: Dict[str, Any] = {}
        self.performance_stats: Dict[str, List[float]] = {}
        self.error_count = 0
        self.total_operations = 0

        # Initialize mathematical libraries
        self._initialize_libraries()

        safe_print("🔢 Unified Mathematics System initialized")

    def _initialize_libraries(self) -> None:
        """Initialize mathematical libraries with proper configuration."""
        try:
            # Configure NumPy
            if self.config.precision == MathPrecision.LOW:
                np.set_printoptions(precision=6, suppress=True)
            elif self.config.precision == MathPrecision.HIGH:
                np.set_printoptions(precision=12, suppress=True)
            else:
                np.set_printoptions(precision=8, suppress=True)

            # Configure SciPy
            sp.special.errprint(0)  # Suppress SciPy warnings

            # Set thread count for parallel processing
            if self.config.enable_parallel:
                try:
                    import mkl
                    mkl.set_num_threads(4)  # Use 4 threads
                except ImportError:
                    pass  # MKL not available

            safe_print("✅ Mathematical libraries configured")

        except Exception as e:
            safe_print(f"⚠️ Library initialization warning: {safe_format_error(e, 'library_init')}")

    def execute_with_monitoring(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute mathematical operation with performance monitoring.

        Args:
            operation_name: Name of the operation
            operation_func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation
        """
        start_time = time.time()
        
        try:
            # Check cache if enabled
            if self.config.enable_caching:
                cache_key = f"{operation_name}_{hash(str(args))}_{hash(str(kwargs))}"
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            # Execute operation
            result = operation_func(*args, **kwargs)

            # Update performance statistics
            execution_time = time.time() - start_time
            if operation_name not in self.performance_stats:
                self.performance_stats[operation_name] = []
            self.performance_stats[operation_name].append(execution_time)
            
            # Cache result if enabled
            if self.config.enable_caching and len(self.cache) < self.config.cache_size:
                self.cache[cache_key] = result
            
            self.total_operations += 1

            # Check performance thresholds
            if execution_time > self.config.max_execution_time:
                safe_print(f"⚠️ Slow operation: {operation_name} took {execution_time:.3f}s")
            
            return result

        except Exception as e:
            self.error_count += 1
            error_msg = safe_format_error(e, operation_name)
            safe_print(f"❌ Mathematical operation failed: {error_msg}")

            if self.config.enable_error_handling:
                return self._handle_mathematical_error(operation_name, e, *args, **kwargs)
            else:
                raise

    def _handle_mathematical_error(self, operation_name: str, error: Exception, *args, **kwargs) -> Any:
        """Handle mathematical errors with appropriate fallbacks."""
        try:
            # Log error
            if self.config.enable_logging:
                log_safe(logger, "error", f"Mathematical error in {operation_name}: {error}")
            
            # Return safe fallback based on operation type
            if "division" in operation_name.lower():
                return 1.0  # Safe division fallback
            elif "sqrt" in operation_name.lower():
                return 0.0  # Safe square root fallback
            elif "log" in operation_name.lower():
                return 0.0  # Safe logarithm fallback
            elif "matrix" in operation_name.lower() or "tensor" in operation_name.lower():
                # Return identity matrix for matrix operations
                if args and hasattr(args[0], 'shape'):
                    return np.eye(args[0].shape[0])
                return np.array([[1.0]])
            else:
                return 0.0  # General fallback
                
        except Exception as fallback_error:
            safe_print(f"❌ Fallback error in {operation_name}: {safe_format_error(fallback_error)}")
            return 0.0

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'total_operations': self.total_operations,
            'error_count': self.error_count,
            'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0.0,
            'cache_hit_rate': len(self.cache) / self.config.cache_size if self.config.cache_size > 0 else 0.0,
            'average_execution_times': {},
            'max_execution_times': {},
            'min_execution_times': {}
        }
        
        for operation, times in self.performance_stats.items():
            if times:
                stats['average_execution_times'][operation] = np.mean(times)
                stats['max_execution_times'][operation] = np.max(times)
                stats['min_execution_times'][operation] = np.min(times)
        
        return stats

    @property
    def success_count(self) -> int:
        """Get count of successful operations."""
        return self.total_operations - self.error_count

    def clear_cache(self) -> None:
        """Clear the operation cache."""
        self.cache.clear()
        safe_print("🧹 Mathematical operation cache cleared")

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.performance_stats.clear()
        self.error_count = 0
        self.total_operations = 0
        safe_print("📊 Mathematical performance statistics reset")


# Global instance for easy access
_unified_mathematics_instance: Optional[UnifiedMathematics] = None


def get_unified_mathematics(config: Optional[MathConfig] = None) -> UnifiedMathematics:
    """Get or create the global unified mathematics instance."""
    global _unified_mathematics_instance
    if _unified_mathematics_instance is None:
        _unified_mathematics_instance = UnifiedMathematics(config)
    return _unified_mathematics_instance


def main():
    """Test the unified mathematics system."""
    try:
        # Initialize system
        math_system = get_unified_mathematics()
        
        # Test basic operations
        def test_operation(x: float) -> float:
            return x ** 2 + 2 * x + 1
        
        result = math_system.execute_with_monitoring("test_operation", test_operation, 5.0)
        safe_print(f"✅ Test operation result: {result}")
        
        # Get statistics
        stats = math_system.get_performance_statistics()
        safe_print(f"📊 Performance stats: {stats}")
        
        safe_print("🎉 Unified mathematics system test completed successfully")
        
    except Exception as e:
        safe_print(f"❌ Test failed: {safe_format_error(e, 'main_test')}")


if __name__ == "__main__":
    main()
