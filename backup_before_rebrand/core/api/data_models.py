"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models Module
===================
Provides comprehensive data models functionality for the Schwabot trading system.

Main Classes:
- APICredentials: Core apicredentials functionality
- MarketData: Enhanced market data with comprehensive fields
- OrderRequest: Advanced order request structure
- OrderResponse: Complete order response tracking
- PortfolioPosition: Portfolio position management
- TradingSignal: Mathematical trading signal structure
- RiskMetrics: Risk management metrics
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Import enums
from .enums import ExchangeType, OrderSide, OrderType, ConnectionStatus, TradingMode, RiskLevel, MarketRegime

class Status(Enum):
"""Class for Schwabot trading functionality."""
ACTIVE = "active"
INACTIVE = "inactive"
ERROR = "error"
PROCESSING = "processing"
MAINTENANCE = "maintenance"

class Mode(Enum):
"""Class for Schwabot trading functionality."""
NORMAL = "normal"
DEBUG = "debug"
TEST = "test"
PRODUCTION = "production"
EMERGENCY = "emergency"

@dataclass
class Config:
"""Class for Schwabot trading functionality."""
enabled: bool = True
timeout: float = 30.0
retries: int = 3
debug: bool = False
log_level: str = "INFO"

@dataclass
class Result:
"""Class for Schwabot trading functionality."""
success: bool = False
data: Optional[Dict[str, Any]] = None
error: Optional[str] = None
timestamp: float = field(default_factory=time.time)
metadata: Dict[str, Any] = field(default_factory=dict)

class APICredentials:
"""Class for Schwabot trading functionality."""

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)
self.active = False
self.initialized = False
self._initialize_system()

def _default_config(self) -> Dict[str, Any]:
return {
'enabled': True,
'timeout': 30.0,
'retries': 3,
'debug': False,
'log_level': 'INFO',
}

def _initialize_system(self) -> None:
try:
self.logger.info(f"Initializing {self.__class__.__name__}")
self.initialized = True
self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
except Exception as e:
self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
self.initialized = False

def activate(self) -> bool:
if not self.initialized:
self.logger.error("System not initialized")
return False
try:
self.active = True
self.logger.info(f"✅ {self.__class__.__name__} activated")
return True
except Exception as e:
self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
return False

def deactivate(self) -> bool:
try:
self.active = False
self.logger.info(f"✅ {self.__class__.__name__} deactivated")
return True
except Exception as e:
self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
return False

def get_status(self) -> Dict[str, Any]:
return {
'active': self.active,
'initialized': self.initialized,
'config': self.config,
}

@dataclass
class MarketData:
"""Class for Schwabot trading functionality."""
symbol: str
timestamp: float
exchange: Optional[str] = None
price: float = 0.0
bid: Optional[float] = None
ask: Optional[float] = None
last_price: Optional[float] = None
volume: float = 0.0
volume_24h: Optional[float] = None
volume_1h: Optional[float] = None
high_24h: Optional[float] = None
low_24h: Optional[float] = None
change_24h: Optional[float] = None
change_percent_24h: Optional[float] = None
spread: Optional[float] = None
market_cap: Optional[float] = None
circulating_supply: Optional[float] = None
volatility: Optional[float] = None
momentum: Optional[float] = None
volume_profile: Optional[float] = None
trend_strength: Optional[float] = None
entropy_level: Optional[float] = None
btc_price: Optional[float] = None
eth_price: Optional[float] = None
usdc_volume: Optional[float] = None
on_chain_signals: Dict[str, float] = field(default_factory=dict)
market_regime: Optional[MarketRegime] = None
sentiment_score: Optional[float] = None
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderRequest:
"""Class for Schwabot trading functionality."""
symbol: str
side: OrderSide
order_type: OrderType
amount: float
price: Optional[float] = None
stop_loss: Optional[float] = None
take_profit: Optional[float] = None
client_order_id: Optional[str] = None
time_in_force: str = "GTC"
post_only: bool = False
reduce_only: bool = False
iceberg_qty: Optional[float] = None
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderResponse:
"""Class for Schwabot trading functionality."""
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
execution_time: Optional[float] = None

@dataclass
class PortfolioPosition:
"""Class for Schwabot trading functionality."""
symbol: str
amount: float
entry_price: float
current_price: float
value_usd: float
pnl: float
pnl_percentage: float
timestamp: float
unrealized_pnl: float = 0.0
realized_pnl: float = 0.0
avg_entry_price: Optional[float] = None
leverage: float = 1.0
margin_used: float = 0.0
risk_score: Optional[float] = None
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingSignal:
"""Class for Schwabot trading functionality."""
symbol: str
timestamp: float
signal_type: str
confidence: float
strength: float
price_target: Optional[float] = None
stop_loss: Optional[float] = None
take_profit: Optional[float] = None
risk_reward_ratio: Optional[float] = None
strategy_name: Optional[str] = None
mathematical_indicators: Dict[str, float] = field(default_factory=dict)
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskMetrics:
"""Class for Schwabot trading functionality."""
portfolio_value: float
total_exposure: float
max_drawdown: float
sharpe_ratio: Optional[float] = None
sortino_ratio: Optional[float] = None
var_95: Optional[float] = None
var_99: Optional[float] = None
volatility: float = 0.0
beta: Optional[float] = None
correlation_matrix: Optional[np.ndarray] = None
timestamp: float = field(default_factory=time.time)
metadata: Dict[str, Any] = field(default_factory=dict)

def create_data_models(config: Optional[Dict[str, Any]] = None):
return APICredentials(config)
