"""
Core module for Schwabot trading system.

This module provides clean, error-free implementations of the core
mathematical and trading components.
"""

# -*- coding: utf-8 -*-

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
    from .orbital_shell_brain_system import OrbitalBRAINSystem, OrbitalShell
    ORBITAL_BRAIN_AVAILABLE = True
except ImportError:
    ORBITAL_BRAIN_AVAILABLE = False

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

try:
    from .algorithmic_portfolio_balancer import (
        AlgorithmicPortfolioBalancer,
        RebalancingStrategy,
        AssetAllocation,
        create_portfolio_balancer,
    )
    PORTFOLIO_BALANCER_AVAILABLE = True
except ImportError:
    PORTFOLIO_BALANCER_AVAILABLE = False

try:
    from .btc_usdc_trading_integration import (
        BTCUSDCTradingIntegration,
        BTCUSDCTradingConfig,
        create_btc_usdc_integration,
    )
    BTC_USDC_INTEGRATION_AVAILABLE = True
except ImportError:
    BTC_USDC_INTEGRATION_AVAILABLE = False

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
    # New orbital brain components
    "OrbitalBRAINSystem",
    "OrbitalShell",
    # Portfolio balancing components
    "AlgorithmicPortfolioBalancer",
    "RebalancingStrategy", 
    "AssetAllocation",
    "create_portfolio_balancer",
    # BTC/USDC integration components
    "BTCUSDCTradingIntegration",
    "BTCUSDCTradingConfig",
    "create_btc_usdc_integration",
    # Availability flags
    "CLEAN_MATH_AVAILABLE",
    "CLEAN_PROFIT_AVAILABLE",
    "CLEAN_PIPELINE_AVAILABLE",
    "ORBITAL_BRAIN_AVAILABLE",
    "PORTFOLIO_BALANCER_AVAILABLE",
    "BTC_USDC_INTEGRATION_AVAILABLE",
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
            "orbital_brain_system": ORBITAL_BRAIN_AVAILABLE,
            "portfolio_balancer": PORTFOLIO_BALANCER_AVAILABLE,
            "btc_usdc_integration": BTC_USDC_INTEGRATION_AVAILABLE,
        },
        "system_operational": (
            CLEAN_MATH_AVAILABLE and 
            CLEAN_PROFIT_AVAILABLE and 
            CLEAN_PIPELINE_AVAILABLE and
            PORTFOLIO_BALANCER_AVAILABLE and
            BTC_USDC_INTEGRATION_AVAILABLE
        ),
    }


def create_clean_trading_system(initial_capital=100000.0):
    """
    Create a complete clean trading system with all components.

    Args:
        initial_capital: Initial capital for the trading system

    Returns:
        Dictionary with all initialized components
    """
    if not (CLEAN_MATH_AVAILABLE and CLEAN_PROFIT_AVAILABLE and CLEAN_PIPELINE_AVAILABLE):
        raise ImportError("Clean implementations not available")

    # Base configuration
    config = {
        "portfolio_config": {
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.05,
            "max_rebalance_frequency": 3600,
        },
        "btc_usdc_config": {
            "symbol": "BTC/USDC",
            "base_order_size": 0.001,
            "max_order_size": 0.01,
            "enable_portfolio_balancing": True,
        },
        "exchange_config": {
            "exchange": "binance",
            "sandbox": True,
        }
    }

    system = {
        "math_foundation": create_math_foundation(),
        "profit_vectorizer": create_profit_vectorizer(),
        "trading_pipeline": create_trading_pipeline(initial_capital=initial_capital),
    }

    if ORBITAL_BRAIN_AVAILABLE:
        system["orbital_brain"] = OrbitalBRAINSystem()

    if PORTFOLIO_BALANCER_AVAILABLE:
        system["portfolio_balancer"] = create_portfolio_balancer(config)

    if BTC_USDC_INTEGRATION_AVAILABLE:
        system["btc_usdc_integration"] = create_btc_usdc_integration(config)

    return system
