"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Core API Package

This package contains all modules related to live exchange API integration.
It exposes the primary classes for easy access from other parts of the
Schwabot system.
"""

from .exchange_connection import ExchangeConnection, ExchangeManager
from .integration_manager import ApiIntegrationManager

__all__ = [
# Enums
"ExchangeType",
"OrderType",
"OrderSide",
"ConnectionStatus",
"DataType",
"TradingMode",
"RiskLevel",
"MarketRegime",
# Data Models
"APICredentials",
"MarketData",
"OrderRequest",
"OrderResponse",
"PortfolioPosition",
"TradingSignal",
"RiskMetrics",
"Config",
"Result",
# Core Classes
"ExchangeConnection",
"ExchangeManager",
"ApiIntegrationManager",
]
