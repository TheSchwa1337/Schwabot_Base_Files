#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot - Advanced AI Trading System
=====================================

Core trading intelligence system for cryptocurrency markets.
Integrates mathematical analysis, risk management, and automated trading.

Main Components:
- Core mathematical framework (core module)
- Risk management and portfolio optimization
- Market data processing and analysis
- Trading strategy execution
- Real-time monitoring and alerting
"""

from typing import Dict, Any

__version__ = "0.5.0"
__author__ = "Schwabot Development Team"
__license__ = "Proprietary"

# Core imports for basic functionality
try:
    from core.mathlib_v4 import MathLibV4
    from core.matrix_math_utils import analyze_price_matrix, risk_parity_weights
    from core.profit_optimization_engine import ProfitOptimizationEngine
    from core.pure_profit_calculator import PureProfitCalculator

    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    import warnings

    warnings.warn(f"Core mathematical modules not available: {e}", stacklevel=2)

# Configuration management
try:
    from config.schwabot_config import SchwabotConfig

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Utility functions
try:
    from utils.logging_setup import setup_logging
    from utils.safe_print import safe_print

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# System status


def get_system_status() -> Dict[str, Any]:
    """Get the current system status and available components."""
    return {
        'version': __version__,
        'core_available': CORE_AVAILABLE,
        'config_available': CONFIG_AVAILABLE,
        'utils_available': UTILS_AVAILABLE,
        'ready': CORE_AVAILABLE and CONFIG_AVAILABLE and UTILS_AVAILABLE,
    }


# Main system class for compatibility


class SchwabotSystem:
    """Main Schwabot system interface."""

    def __init__(self) -> None:
        """Initialize the Schwabot system."""
        self.version = __version__
        self.status = get_system_status()

        if not self.status['ready']:
            raise RuntimeError("Schwabot system is not ready - missing dependencies")

    def get_version(self) -> str:
        """Get the system version."""
        return self.version

    def get_status(self) -> Dict[str, Any]:
        """Get the system status."""
        return self.status


# Export main classes and functions
__all__ = [
    'SchwabotSystem',
    'get_system_status',
    '__version__',
    '__author__',
    '__license__',
]

# Add conditional exports based on availability
if CORE_AVAILABLE:
    __all__.extend(
        [
            'MathLibV4',
            'PureProfitCalculator',
            'ProfitOptimizationEngine',
            'analyze_price_matrix',
            'risk_parity_weights',
        ]
    )

if CONFIG_AVAILABLE:
    __all__.append('SchwabotConfig')

if UTILS_AVAILABLE:
    __all__.extend(['setup_logging', 'safe_print'])
