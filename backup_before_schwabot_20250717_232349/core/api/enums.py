"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API enumeration types for Schwabot trading system.
Provides comprehensive enums for exchange types, order management, and system status.
"""


from enum import Enum
class ExchangeType(Enum):
"""Class for Schwabot trading functionality."""
BINANCE = "binance"
COINBASE = "coinbase"
KRAKEN = "kraken"
KUCOIN = "kucoin"
OKX = "okx"
BYBIT = "bybit"
GATE = "gate"
MEXC = "mexc"

class OrderSide(Enum):
"""Class for Schwabot trading functionality."""
BUY = "buy"
SELL = "sell"

class OrderType(Enum):
"""Class for Schwabot trading functionality."""
MARKET = "market"
LIMIT = "limit"
STOP = "stop"
STOP_LIMIT = "stop_limit"
TAKE_PROFIT = "take_profit"
TAKE_PROFIT_LIMIT = "take_profit_limit"

class DataType(Enum):
"""Class for Schwabot trading functionality."""
TRADE = "trade"
ORDER_BOOK = "order_book"
NEWS = "news"
MARKET_DATA = "market_data"
PORTFOLIO = "portfolio"
ANALYTICS = "analytics"

class ConnectionStatus(Enum):
"""Class for Schwabot trading functionality."""
DISCONNECTED = "disconnected"
CONNECTING = "connecting"
CONNECTED = "connected"
ERROR = "error"
RECONNECTING = "reconnecting"
MAINTENANCE = "maintenance"

class TradingMode(Enum):
"""Class for Schwabot trading functionality."""
DEMO = "demo"
LIVE = "live"
BACKTEST = "backtest"
PAPER = "paper"
SIMULATION = "simulation"

class RiskLevel(Enum):
"""Class for Schwabot trading functionality."""
CONSERVATIVE = "conservative"
MODERATE = "moderate"
AGGRESSIVE = "aggressive"
CUSTOM = "custom"

class MarketRegime(Enum):
"""Class for Schwabot trading functionality."""
TRENDING_UP = "trending_up"
TRENDING_DOWN = "trending_down"
SIDEWAYS = "sideways"
VOLATILE = "volatile"
CALM = "calm"
BREAKOUT = "breakout"
CONSOLIDATION = "consolidation"
