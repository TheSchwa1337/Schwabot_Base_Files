import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import ExchangeType, OrderSide, OrderType

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

API System Data Models ======================

Contains all data models (dataclasses) for the Schwabot live API
integration system.


@dataclass
class APICredentials:API credentials for exchanges.exchange: ExchangeType
    api_key: str
    secret: str
    passphrase: str = sandbox: bool = True
    testnet: bool = True


@dataclass
class MarketData:Real-time market data.symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float
    change_24h: float
    timestamp: float
    exchange: str
    metadata: Dict[str, Any] = field(default_factory = dict)


@dataclass
class OrderRequest:Order request structure.symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory = dict)


@dataclass
class OrderResponse:Order response structure.order_id: str
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
    info: Dict[str, Any] = field(default_factory = dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PortfolioPosition:Portfolio position.symbol: str
    amount: float
    entry_price: float
    current_price: float
    value_usd: float
    pnl: float
    pnl_percentage: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory = dict)Data models for API responses.# =====================================================================
#  Core Data Structures for API Payloads
# =====================================================================


@dataclass
class APIPricePoint:Represents a single price point in a time series.timestamp: int
    price: float
    volume: Optional[float] = None


@dataclass
class APIMarketDepth:Represents the market depth for an asset.last_update_id: int
    bids: list[tuple[float, float]]  # (price, quantity)
    asks: list[tuple[float, float]]  # (price, quantity)


@dataclass
class APITrade:Represents a single executed trade.id: int
    price: float
    qty: float
    quote_qty: float
    timestamp: int
    is_buyer_maker: bool


@dataclass
class APINewsArticle:Represents a single news article.id: str
    source: str
    headline: str
    summary: str
    url: str
    timestamp: int
    sentiment: Optional[float] = None  # e.g., -1.0 to 1.0


@dataclass
class APIFearAndGreedIndex:Represents a Fear and Greed Index value.value: int  # 0-100
    value_classification: str  # e.g., Extreme Feartimestamp: int


@dataclass
class APIGenericData:A generic container for other data types.source: str
    data_type: str
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: int
