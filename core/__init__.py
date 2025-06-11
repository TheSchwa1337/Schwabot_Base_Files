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

# Logging and Monitoring Components
from .basket_log_controller import BasketLogController, LogEntry

# Entropy and Allocation Components
from .basket_entropy_allocator import BasketEntropyAllocator, EntropyBand, DataProvider

# Matrix and Rendering Components
from .matrix_fault_resolver import MatrixFaultResolver
from .line_render_engine import LineRenderEngine

# Adaptive Profit Chain Components
from .adaptive_profit_chain import APCFSystem

# Drift Shell Engine
from .drift_shell_engine import DriftShellEngine

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
    'DormantState',
    
    # Logging and Monitoring
    'BasketLogController',
    'LogEntry',
    
    # Entropy and Allocation
    'BasketEntropyAllocator',
    'EntropyBand',
    'DataProvider',
    
    # Matrix and Rendering
    'MatrixFaultResolver',
    'LineRenderEngine',
    
    # Adaptive Profit Chain
    'APCFSystem',

    # Drift Shell Engine
    'DriftShellEngine',

    # New components
    'NCCOManager',
    'SFSSSRouter'
] 