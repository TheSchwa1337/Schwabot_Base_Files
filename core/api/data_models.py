import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .enums import ExchangeType, OrderSide, OrderType


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API System Data Models
======================

Contains all data models (dataclasses) for the Schwabot live API
integration system.
"""


@dataclass
class APICredentials:
    """API credentials for exchanges."""
    exchange: ExchangeType
    api_key: str
    secret: str
    passphrase: str = ""
    sandbox: bool = True
    testnet: bool = True


@dataclass
class MarketData:
    """Real-time market data."""
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float
    change_24h: float
    timestamp: float
    exchange: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRequest:
    """Order request structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResponse:
    """Order response structure."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    amount: float
    price: float
    filled: float
    remaining: float
    cost: float
    status: str
    timestamp: float
    fee: Optional[Dict[str, Any]] = None
    info: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PortfolioPosition:
    """Portfolio position."""
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    value_usd: float
    pnl: float
    pnl_percentage: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)