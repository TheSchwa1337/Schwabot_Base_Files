"""
Mathematical Library Package for Schwabot Trading System
===================================================

This package provides the mathematical foundation for Schwabot's trading system,
including core mathematical operations, tensor calculations, and advanced
mathematical models for market analysis.
"""

from core.mathlib import CoreMathLib, GradedProfitVector
from core.mathlib_v2 import SmartStop
from core.math_core import BaseAnalyzer, AnalysisResult

# Import mathematical functions directly
try:
    import sys
    from pathlib import Path
    # Add parent directory to path to import mathlib.py
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from mathlib import entropy, klein_bottle, recursive_operation, add, subtract
    MATHLIB_FUNCTIONS_AVAILABLE = True
except ImportError:
    # Fallback implementations
    def entropy(data):
        """Fallback entropy function"""
        import numpy as np
        if len(data) == 0:
            return 0.0
        unique, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0.0
    
    def klein_bottle(point):
        """Fallback Klein bottle function"""
        import math
        u, v = point
        x = (2 + math.cos(v/2) * math.sin(u)) * math.cos(v)
        y = (2 + math.cos(v/2) * math.sin(u)) * math.sin(v)
        z = math.sin(v/2) * math.sin(u)
        return (x, y, z, 0)
    
    def recursive_operation(depth_limit, current_depth=0, operation_type='fibonacci'):
        """Fallback recursive operation"""
        if operation_type == 'fibonacci':
            # Standard fibonacci: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)
            if depth_limit <= 0:
                return 0.0
            elif depth_limit == 1:
                return 1.0
            else:
                return (recursive_operation(depth_limit - 1, 0, operation_type) + 
                       recursive_operation(depth_limit - 2, 0, operation_type))
        elif operation_type == 'factorial':
            if depth_limit <= 1:
                return 1.0
            else:
                return depth_limit * recursive_operation(depth_limit - 1, 0, operation_type)
        else:
            return 1.0
    
    def add(a, b):
        """Fallback add function"""
        return a + b
    
    def subtract(a, b):
        """Fallback subtract function"""
        return a - b
    
    MATHLIB_FUNCTIONS_AVAILABLE = False

# Optional imports with fallbacks
try:
    from core.enhanced_fractal_core import QuantizationProfile
except ModuleNotFoundError:
    QuantizationProfile = None  # Logged later; not fatal

try:
    from core.cyclic_core import CyclicPattern
except ModuleNotFoundError:
    CyclicPattern = None  # Logged later; not fatal

# Export mathematical components (only available ones)
__all__ = [
    'CoreMathLib',
    'GradedProfitVector',
    'SmartStop',
    'BaseAnalyzer',
    'AnalysisResult',
    'entropy',
    'klein_bottle',
    'recursive_operation',
    'add',
    'subtract'
]

# Add optional components if available
if QuantizationProfile is not None:
    __all__.append('QuantizationProfile')
if CyclicPattern is not None:
    __all__.append('CyclicPattern') 