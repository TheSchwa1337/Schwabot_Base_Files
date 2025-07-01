#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API System Enums
================

Contains all enumerations for the Schwabot live API integration system.
"""

from enum import Enum


class ExchangeType(Enum):
    """Supported exchange types."""
    COINBASE = "coinbase"
    BINANCE = "binance"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    OKX = "okx"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class ConnectionStatus(Enum):
    """Connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting" 