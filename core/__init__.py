"""
Core Package for Schwabot Trading System
======================================

This package contains the core mathematical and trading components of the Schwabot system.
It provides the foundation for quantitative trading, pattern recognition, and state management.
"""

# Core Mathematical Components
from .mathlib import CoreMathLib, GradedProfitVector
from .math_core import BaseAnalyzer, AnalysisResult

# Trading Logic Components
from .basket_swapper import BasketSwapper, SwapCriteria
from .basket_swap_logic import BasketSwapLogic
from .profit_tensor import ProfitTensorStore
from .cooldown_manager import CooldownManager
from .profit_protection import ProfitProtection

# Advanced Mathematical Components
from .enhanced_fractal_core import QuantizationProfile
from .cyclic_core import CyclicPattern
from .dormant_engine import DormantState

# Export all public components
__all__ = [
    # Core Math
    'CoreMathLib',
    'GradedProfitVector',
    'BaseAnalyzer',
    'AnalysisResult',
    
    # Trading Logic
    'BasketSwapper',
    'SwapCriteria',
    'BasketSwapLogic',
    'ProfitTensorStore',
    'CooldownManager',
    'ProfitProtection',
    
    # Advanced Math
    'QuantizationProfile',
    'CyclicPattern',
    'DormantState'
] 