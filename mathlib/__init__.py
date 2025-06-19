#!/usr/bin/env python3
"""
Mathematical Library Package - Unified Mathematical Framework
===========================================================

Comprehensive mathematical library for Schwabot framework providing
multi-tier mathematical capabilities from basic operations to AI-enhanced
automatic differentiation and profit optimization.

Package Structure:
- MathLib (V1): Core mathematical functions and utilities
- MathLibV2: Enhanced mathematical operations with advanced algorithms  
- MathLibV3: AI-infused mathematical library with automatic differentiation
- Mathematical constants and utility functions

Exports:
- All mathematical classes and functions
- Dual number class for automatic differentiation
- Utility functions (kelly_fraction, cvar, gradient computation)
- Mathematical constants

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add core directory to Python path
_core_path = Path(__file__).parent.parent / "core"
if str(_core_path) not in sys.path:
    sys.path.insert(0, str(_core_path))

# Import all mathematical components
try:
    # Core mathematical libraries
    from mathlib import MathLib, mathematical_constants
    from mathlib_v2 import MathLibV2
    from mathlib_v3 import (
        MathLibV3, 
        Dual,
        grad,
        jacobian,
        kelly_fraction,
        cvar
    )
    
    # Additional mathematical components
    from advanced_mathematical_core import (
        QuantumThermalCoupler,
        FractalIndexCalculator,
        VoidWellProcessor
    )
    
    # Spectral and filtering components
    from spectral_transform import SpectralAnalyzer, DLTWaveformEngine
    from filters import KalmanFilter, ParticleFilter, TimeAwareEMA
    
    # Mathematical constants
    from constants import MathematicalConstants
    
    # Core mathematical library alias for compatibility
    CoreMathLib = MathLib
    CoreMathLibV2 = MathLibV2
    CoreMathLibV3 = MathLibV3
    
except ImportError as e:
    # Fallback imports for graceful degradation
    import warnings
    warnings.warn(f"Some mathematical components could not be imported: {e}")
    
    # Minimal fallback classes
    class MathLib:
        def __init__(self):
            self.version = "1.0.0-fallback"
    
    class MathLibV2:
        def __init__(self):
            self.version = "2.0.0-fallback"
    
    class MathLibV3:
        def __init__(self):
            self.version = "3.0.0-fallback"
    
    class Dual:
        def __init__(self, val: float, eps: float = 0.0):
            self.val = val
            self.eps = eps
    
    # Stub functions
    def grad(func, x): return 0.0
    def jacobian(func, x): return []
    def kelly_fraction(mu, sigma_sq): return 0.0
    def cvar(returns, alpha=0.95): return 0.0
    def mathematical_constants(): return {}
    
    # Aliases
    CoreMathLib = MathLib
    CoreMathLibV2 = MathLibV2
    CoreMathLibV3 = MathLibV3


# Create a GradedProfitVector class for backward compatibility
class GradedProfitVector:
    """Graded profit vector for mathematical trading analysis"""
    
    def __init__(self, profits: list, weights: list = None, grades: list = None):
        self.profits = profits
        self.weights = weights or [1.0] * len(profits)
        self.grades = grades or ['A'] * len(profits)
        self.size = len(profits)
    
    def total_profit(self) -> float:
        """Calculate total weighted profit"""
        return sum(p * w for p, w in zip(self.profits, self.weights))
    
    def average_grade(self) -> str:
        """Calculate average grade (simplified)"""
        grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        avg_val = sum(grade_values.get(g, 0) for g in self.grades) / len(self.grades)
        
        if avg_val >= 3.5: return 'A'
        elif avg_val >= 2.5: return 'B'
        elif avg_val >= 1.5: return 'C'
        elif avg_val >= 0.5: return 'D'
        else: return 'F'


# Enhanced mathematical functions for compatibility
def add(a, b):
    """Addition function for backward compatibility"""
    return a + b

def subtract(a, b):
    """Subtraction function"""
    return a - b

def multiply(a, b):
    """Multiplication function"""
    return a * b

def divide(a, b):
    """Division function with zero check"""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


# Package metadata
__version__ = "3.0.0"
__author__ = "Schwabot Mathematical Framework"
__description__ = "Unified mathematical library with AI-enhanced capabilities"

# All exports for easy importing
__all__ = [
    # Main mathematical classes
    'MathLib', 'MathLibV2', 'MathLibV3',
    'CoreMathLib', 'CoreMathLibV2', 'CoreMathLibV3',
    
    # Automatic differentiation
    'Dual', 'grad', 'jacobian',
    
    # Financial mathematics
    'kelly_fraction', 'cvar', 'GradedProfitVector',
    
    # Basic operations
    'add', 'subtract', 'multiply', 'divide',
    
    # Constants and utilities
    'mathematical_constants',
    
    # Package metadata
    '__version__', '__author__', '__description__'
]


def main() -> None:
    """Main function for testing mathematical library integration"""
    try:
        print(f"🧮 Mathematical Library Package v{__version__} - Integration Test")
        
        # Test MathLib V1
        math_v1 = MathLib()
        print(f"✅ MathLib V1: {math_v1.version}")
        
        # Test MathLib V2  
        math_v2 = MathLibV2()
        print(f"✅ MathLib V2: {math_v2.version}")
        
        # Test MathLib V3
        math_v3 = MathLibV3()
        print(f"✅ MathLib V3: {math_v3.version}")
        
        # Test Dual numbers
        x = Dual(2.0, 1.0)
        y = x * x + 3 * x + 1  # f(x) = x² + 3x + 1, f'(x) = 2x + 3
        print(f"✅ Dual numbers: f(2) = {y.val}, f'(2) = {y.eps}")
        
        # Test GradedProfitVector
        profits = [100, 150, -50, 200]
        grades = ['A', 'B', 'C', 'A']
        vector = GradedProfitVector(profits, grades=grades)
        print(f"✅ Profit vector: Total={vector.total_profit()}, Grade={vector.average_grade()}")
        
        # Test basic operations
        print(f"✅ Basic ops: 5 + 3 = {add(5, 3)}, 10 / 2 = {divide(10, 2)}")
        
        print("🎉 Mathematical library integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
