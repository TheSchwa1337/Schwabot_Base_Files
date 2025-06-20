#!/usr/bin/env python3
"""Trading enums for Schwabot BTC integration.

This module contains all enumeration types used for trading operations,
order management, and exchange communication.
"""

from enum import Enum


class ExchangeType(Enum):
    """Exchange type enumeration."""
    COINBASE = "coinbase"
    BINANCE = "binance"
    KRAKEN = "kraken"
    GEMINI = "gemini"
    BITFINEX = "bitfinex"
    CUSTOM = "custom"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    POST_ONLY = "post_only"
    FILL_OR_KILL = "fill_or_kill"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DataType(Enum):
    """Data type enumeration."""
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    CANDLES = "candles"
    BALANCE = "balance"
    ORDERS = "orders"
    POSITIONS = "positions" 