"""
Engine Package for Schwabot Trading System
=======================================

This package contains the execution layer that combines core trading logic
with configuration and runtime state. It orchestrates the entire trading
system's operation and manages the trade execution lifecycle.
"""

from core import (
    BasketSwapper,
    ProfitTensorStore,
    CooldownManager,
    BasketSwapLogic,
    ProfitProtection
)

from config import load_config

# Export execution components
__all__ = [
    'BasketSwapper',
    'ProfitTensorStore',
    'CooldownManager',
    'BasketSwapLogic',
    'ProfitProtection',
    'load_config'
] 