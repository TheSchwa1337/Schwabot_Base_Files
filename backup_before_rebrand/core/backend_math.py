#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Math Module - GPU/CPU Acceleration Support
=================================================

Provides backend support for mathematical operations with GPU acceleration
when available, falling back to CPU (NumPy) when needed.
"""

import os
import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Force override if explicitly set
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() in ("true", "1", "yes")

try:
    if FORCE_CPU:
        raise ImportError("Forced CPU fallback triggered.")
    import cupy as xp
    GPU_ENABLED = True
except ImportError:
    import numpy as xp
    GPU_ENABLED = False

@dataclass
class MathResult:
    """Result of a mathematical operation."""
    value: Any
    operation: str
    timestamp: float
    metadata: Dict[str, Any]

class BackendMath:
    """Backend mathematical operations for Schwabot."""
    
    def __init__(self):
        """Initialize the backend math system."""
        self.operation_history: List[MathResult] = []
        self.cache: Dict[str, Any] = {}
        
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self._log_operation("add", result, {"a": a, "b": b})
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        result = a - b
        self._log_operation("subtract", result, {"a": a, "b": b})
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self._log_operation("multiply", result, {"a": a, "b": b})
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        self._log_operation("divide", result, {"a": a, "b": b})
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Raise a number to a power."""
        result = math.pow(base, exponent)
        self._log_operation("power", result, {"base": base, "exponent": exponent})
        return result
    
    def sqrt(self, value: float) -> float:
        """Calculate square root."""
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(value)
        self._log_operation("sqrt", result, {"value": value})
        return result
    
    def log(self, value: float, base: float = math.e) -> float:
        """Calculate logarithm."""
        if value <= 0:
            raise ValueError("Cannot calculate logarithm of non-positive number")
        result = math.log(value, base)
        self._log_operation("log", result, {"value": value, "base": base})
        return result
    
    def exp(self, value: float) -> float:
        """Calculate exponential."""
        result = math.exp(value)
        self._log_operation("exp", result, {"value": value})
        return result
    
    def sin(self, value: float) -> float:
        """Calculate sine."""
        result = math.sin(value)
        self._log_operation("sin", result, {"value": value})
        return result
    
    def cos(self, value: float) -> float:
        """Calculate cosine."""
        result = math.cos(value)
        self._log_operation("cos", result, {"value": value})
        return result
    
    def tan(self, value: float) -> float:
        """Calculate tangent."""
        result = math.tan(value)
        self._log_operation("tan", result, {"value": value})
        return result
    
    def mean(self, values: List[float]) -> float:
        """Calculate mean of a list of values."""
        if not values:
            raise ValueError("Cannot calculate mean of empty list")
        result = sum(values) / len(values)
        self._log_operation("mean", result, {"values": values})
        return result
    
    def std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            raise ValueError("Need at least 2 values for standard deviation")
        mean_val = self.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        result = math.sqrt(variance)
        self._log_operation("std", result, {"values": values})
        return result
    
    def _log_operation(self, operation: str, result: Any, metadata: Dict[str, Any]):
        """Log a mathematical operation."""
        import time
        math_result = MathResult(
            value=result,
            operation=operation,
            timestamp=time.time(),
            metadata=metadata
        )
        self.operation_history.append(math_result)
        
        # Keep only last 1000 operations
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]

def get_backend():
    """Get the current backend (CuPy or NumPy)."""
    return xp

def is_gpu():
    """Check if GPU acceleration is enabled."""
    return GPU_ENABLED

def backend_info():
    """Get information about the current backend."""
    return {
        "backend": "CuPy" if GPU_ENABLED else "NumPy",
        "accelerated": GPU_ENABLED,
        "force_cpu": FORCE_CPU,
    }

# Global instance
backend_math = BackendMath()

def get_backend_math() -> BackendMath:
    """Get the global backend math instance."""
    return backend_math
