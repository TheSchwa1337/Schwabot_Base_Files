# -*- coding: utf-8 -*-
"""
Core module for the Schwabot Trading System.

This module provides access to the clean implementations that are
fully functional and error-free.
"""

# Clean implementations - fully functional and error-free
try:
    from .clean_math_foundation import (
        BitPhase,
        CleanMathFoundation,
        MathOperation,
        ThermalState,
        create_math_foundation,
        quick_calculation,
    )

    CLEAN_MATH_AVAILABLE = True
except ImportError:
    CLEAN_MATH_AVAILABLE = False

try:
    from .clean_profit_vectorization import (
        CleanProfitVectorization,
        ProfitVector,
        VectorizationMode,
        create_profit_vectorizer,
    )

    CLEAN_PROFIT_AVAILABLE = True
except ImportError:
    CLEAN_PROFIT_AVAILABLE = False

try:
    from .clean_trading_pipeline import (
        CleanTradingPipeline,
        MarketData,
        StrategyBranch,
        TradingDecision,
        create_trading_pipeline,
        run_trading_simulation,
    )

    CLEAN_PIPELINE_AVAILABLE = True
except ImportError:
    CLEAN_PIPELINE_AVAILABLE = False

# Core exports - only clean implementations
__all__ = [
    # Clean implementations (recommended)
    "CleanMathFoundation",
    "MathOperation",
    "ThermalState",
    "BitPhase",
    "create_math_foundation",
    "quick_calculation",
    "CleanProfitVectorization",
    "VectorizationMode",
    "ProfitVector",
    "create_profit_vectorizer",
    "CleanTradingPipeline",
    "MarketData",
    "TradingDecision",
    "StrategyBranch",
    "create_trading_pipeline",
    "run_trading_simulation",
    # Availability flags
    "CLEAN_MATH_AVAILABLE",
    "CLEAN_PROFIT_AVAILABLE",
    "CLEAN_PIPELINE_AVAILABLE",
    # Utility functions
    "get_system_status",
    "create_clean_trading_system",
]


def get_system_status():
    """Get the status of all system components."""
    return {
        "clean_implementations": {
            "math_foundation": CLEAN_MATH_AVAILABLE,
            "profit_vectorization": CLEAN_PROFIT_AVAILABLE,
            "trading_pipeline": CLEAN_PIPELINE_AVAILABLE,
        },
        "system_operational": (
            CLEAN_MATH_AVAILABLE and CLEAN_PROFIT_AVAILABLE and CLEAN_PIPELINE_AVAILABLE
        ),
    }


def create_clean_trading_system(initial_capital=100000.0):
    """
    Create a complete clean trading system with all components.

    Returns:
        Dictionary with all initialized components
    """
    if not (CLEAN_MATH_AVAILABLE and CLEAN_PROFIT_AVAILABLE and CLEAN_PIPELINE_AVAILABLE):
        raise ImportError("Clean implementations not available")

    return {
        "math_foundation": create_math_foundation(),
        "profit_vectorizer": create_profit_vectorizer(),
        "trading_pipeline": create_trading_pipeline(initial_capital=initial_capital),
    }
