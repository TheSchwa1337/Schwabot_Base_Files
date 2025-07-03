# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from .integration_manager import ApiIntegrationManager
from .exchange_connection import ExchangeConnection
from .enums import ExchangeType, OrderType, OrderSide, ConnectionStatus
from .data_models import APICredentials, MarketData, OrderRequest, OrderResponse, PortfolioPosition

"""Schwabot Core API Package

This package contains all modules related to live exchange API integration.

It exposes the primary classes for easy access from other parts of the
Schwabot system.
"""

__all__ = [
    # Enums
    "ExchangeType",
    "OrderType",
    "OrderSide",
    "ConnectionStatus",
    # Data Models
    "APICredentials",
    "MarketData",
    "OrderRequest",
    "OrderResponse",
    "PortfolioPosition",
    # Core Classes
    "ExchangeConnection",
    "ApiIntegrationManager",
]
