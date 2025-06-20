#!/usr/bin/env python3
"""Trading data containers for Schwabot BTC integration.

This module contains all dataclass containers used for trading operations,
order management, and exchange communication.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .enums import ExchangeType, OrderSide, OrderType, OrderStatus, DataType


@dataclass
class ExchangeConfig:
    """Exchange configuration container."""
    
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    sandbox: bool = True
    base_url: str = ""
    timeout: int = 30
    rate_limit: int = 100  # requests per minute
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class OrderRequest:
    """Order request container."""
    
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResponse:
    """Order response container."""
    
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketData:
    """Market data container."""
    
    symbol: str
    data_type: DataType
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Balance:
    """Balance container."""
    
    currency: str
    available: float
    total: float
    locked: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Performance metrics for BTC integration."""
    
    total_orders: int
    successful_orders: int
    failed_orders: int
    average_execution_time: float
    total_execution_time: float
    average_slippage: float
    total_volume: float
    api_calls: int
    api_errors: int
    cache_hits: int
    cache_misses: int 