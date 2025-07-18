"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exchange Connection Module
==========================
Manages connection to cryptocurrency exchanges via CCXT.
Provides real-time market data, order placement, and balance management.
"""

from typing import Any, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
import time
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

# Try to import CCXT
try:
import ccxt
import ccxt.async_support as ccxt_async
CCXT_AVAILABLE = True
except ImportError:
CCXT_AVAILABLE = False
logger.warning("CCXT library not available. Exchange functionality will be limited.")

class ExchangeType(Enum):
"""Class for Schwabot trading functionality."""
"""Supported exchange types."""
BINANCE = "binance"
COINBASE = "coinbase"
KRAKEN = "kraken"
KUCOIN = "kucoin"
OKX = "okx"

@dataclass
class ExchangeCredentials:
"""Class for Schwabot trading functionality."""
"""Exchange credentials container."""
api_key: str
secret: str
passphrase: Optional[str] = None
sandbox: bool = True


class ExchangeConnection:
"""Class for Schwabot trading functionality."""
"""Manages connection to cryptocurrency exchanges via CCXT."""

def __init__(self, exchange_type: ExchangeType, credentials: ExchangeCredentials, config: Dict[str, Any]) -> None:
self.exchange_type = exchange_type
self.credentials = credentials
self.config = config
self.status = "DISCONNECTED"
self.exchange = None
self.async_exchange = None
self.last_heartbeat = 0
self.last_error = None
self.reconnect_attempts = 0
self.successful_requests = 0
self.failed_requests = 0
self.market_data_cache = {}
self.cache_expiry = config.get("cache_expiry", 5)  # 5 seconds
self.max_reconnect_attempts = config.get("max_reconnect_attempts", 5)
self.reconnect_delay = config.get("reconnect_delay", 5)
self.rate_limit_delay = config.get("rate_limit_delay", 1)  # 1 second between requests

logger.info(f"Exchange connection initialized for {exchange_type.value}")

async def connect(self) -> bool:
"""Establishes connection to the exchange."""
if not CCXT_AVAILABLE:
logger.error("CCXT library not available. Cannot connect to exchange.")
self.status = "ERROR"
self.last_error = "CCXT library not installed."
return False

self.status = "CONNECTING"
logger.info(f"Connecting to {self.exchange_type.value}...")

try:
# Initialize CCXT exchange
exchange_class = getattr(ccxt, self.exchange_type.value)

# Prepare exchange configuration
exchange_config = {
'apiKey': self.credentials.api_key,
'secret': self.credentials.secret,
'enableRateLimit': True,
'rateLimit': self.rate_limit_delay * 1000,  # Convert to milliseconds
'timeout': 30000,   # 30 second timeout
'options': {
'defaultType': 'spot',
'adjustForTimeDifference': True,
'recvWindow': 60000,  # 60 second receive window
}
}

# Add passphrase for exchanges that require it (Coinbase, Kraken)
if self.credentials.passphrase:
exchange_config['passphrase'] = self.credentials.passphrase

# Handle sandbox mode - Coinbase doesn't support sandbox
if self.credentials.sandbox and self.exchange_type != ExchangeType.COINBASE:
exchange_config['sandbox'] = True

# Add testnet configuration for Binance
if self.exchange_type == ExchangeType.BINANCE:
exchange_config['urls'] = {
'api': {
'public': 'https://testnet.binance.vision/api',
'private': 'https://testnet.binance.vision/api',
}
}
else:
# For live trading or Coinbase, don't use sandbox
exchange_config['sandbox'] = False

# Initialize both sync and async versions
self.exchange = exchange_class(exchange_config)
self.async_exchange = getattr(ccxt_async, self.exchange_type.value)(exchange_config)

# Test connection by loading markets
await self.async_exchange.load_markets()

# Test API access by fetching account info (only if API keys are provided)
if self.credentials.api_key and self.credentials.secret:
try:
await self.async_exchange.fetch_balance()
logger.info(f"API access verified for {self.exchange_type.value}")
except Exception as e:
logger.warning(f"API access test failed for {self.exchange_type.value}: {e}")
# Don't fail the connection if API test fails - we can still get public data
else:
logger.info(f"Public-only mode for {self.exchange_type.value}")

self.status = "CONNECTED"
self.last_heartbeat = time.time()
self.reconnect_attempts = 0
logger.info(f"Successfully connected to {self.exchange_type.value}")
return True

except Exception as e:
self.status = "ERROR"
self.last_error = str(e)
self.reconnect_attempts += 1
logger.error(f"Failed to connect to {self.exchange_type.value}: {e}")
return False

async def disconnect(self):
"""Closes the connection to the exchange."""
if self.status == "DISCONNECTED":
return

logger.info(f"Disconnecting from {self.exchange_type.value}...")
try:
if self.async_exchange:
await self.async_exchange.close()
self.status = "DISCONNECTED"
logger.info(f"Disconnected from {self.exchange_type.value}")
except Exception as e:
logger.error(f"Error during disconnection: {e}")

async def reconnect(self) -> bool:
"""Attempt to reconnect to the exchange."""
if self.reconnect_attempts >= self.max_reconnect_attempts:
logger.error(f"Max reconnection attempts reached for {self.exchange_type.value}")
return False

logger.info(f"Attempting to reconnect to {self.exchange_type.value} (attempt {self.reconnect_attempts + 1})")
await asyncio.sleep(self.reconnect_delay)

success = await self.connect()
if success:
logger.info(f"Successfully reconnected to {self.exchange_type.value}")
return success

async def get_market_data(self, symbol: str):
"""Fetches market data for a given symbol, using a cache."""
if self.status != "CONNECTED":
if not await self.reconnect():
return None

# Check cache first
cached_data = self.market_data_cache.get(symbol)
if cached_data and (time.time() - cached_data.get('timestamp', 0) < self.cache_expiry):
return cached_data

try:
# Fetch ticker data
ticker = await self.async_exchange.fetch_ticker(symbol)

market_data = {
'symbol': symbol,
'price': ticker['last'],
'bid': ticker['bid'],
'ask': ticker['ask'],
'volume': ticker['baseVolume'],
'high': ticker['high'],
'low': ticker['low'],
'change': ticker['change'],
'change_percent': ticker['percentage'],
'timestamp': time.time()
}

self.market_data_cache[symbol] = market_data
self.successful_requests += 1
return market_data

except Exception as e:
self.failed_requests += 1
logger.error(f"Error fetching market data for {symbol}: {e}")
return None

async def get_balance(self):
"""Fetches account balance."""
if self.status != "CONNECTED":
if not await self.reconnect():
return None

try:
balance = await self.async_exchange.fetch_balance()
self.successful_requests += 1
return balance
except Exception as e:
self.failed_requests += 1
logger.error(f"Error fetching balance: {e}")
return None

async def place_order(self, order_request: Dict[str, Any]):
"""Places an order on the exchange."""
if self.status != "CONNECTED":
if not await self.reconnect():
return None

try:
# Validate required fields
required_fields = ['symbol', 'type', 'side', 'amount']
for field in required_fields:
if field not in order_request:
raise ValueError(f"Missing required field: {field}")

# Place the order
order = await self.async_exchange.create_order(
symbol=order_request['symbol'],
type=order_request['type'],
side=order_request['side'],
amount=order_request['amount'],
price=order_request.get('price'),
params=order_request.get('params', {})
)

self.successful_requests += 1
logger.info(f"Order placed successfully: {order['id']}")
return order

except Exception as e:
self.failed_requests += 1
logger.error(f"Error placing order: {e}")
return None

async def get_order_status(self, order_id: str, symbol: str = None):
"""Fetches the status of an order."""
if self.status != "CONNECTED":
if not await self.reconnect():
return None

try:
order = await self.async_exchange.fetch_order(order_id, symbol)
self.successful_requests += 1
return order
except Exception as e:
self.failed_requests += 1
logger.error(f"Error fetching order status: {e}")
return None

async def cancel_order(self, order_id: str, symbol: str = None):
"""Cancels an order."""
if self.status != "CONNECTED":
if not await self.reconnect():
return None

try:
result = await self.async_exchange.cancel_order(order_id, symbol)
self.successful_requests += 1
logger.info(f"Order cancelled successfully: {order_id}")
return result
except Exception as e:
self.failed_requests += 1
logger.error(f"Error cancelling order: {e}")
return None

def get_status(self) -> Dict[str, Any]:
"""Returns the current status of the connection."""
return {
'exchange': self.exchange_type.value,
'status': self.status,
'last_heartbeat': self.last_heartbeat,
'last_error': self.last_error,
'reconnect_attempts': self.reconnect_attempts,
'successful_requests': self.successful_requests,
'failed_requests': self.failed_requests,
'cache_size': len(self.market_data_cache)
}

async def health_check(self) -> bool:
"""Performs a health check on the connection."""
try:
if self.status != "CONNECTED":
return False

# Try to fetch a simple market data to test connection
test_symbol = "BTC/USDT" if self.exchange_type == ExchangeType.BINANCE else "BTC-USD"
market_data = await self.get_market_data(test_symbol)

if market_data:
self.last_heartbeat = time.time()
return True
else:
return False

except Exception as e:
logger.error(f"Health check failed: {e}")
return False


class ExchangeManager:
"""Class for Schwabot trading functionality."""
"""Manages multiple exchange connections."""

def __init__(self, config: Dict[str, Any]) -> None:
self.config = config
self.connections = {}
self._load_api_keys()

def _load_api_keys(self) -> Dict[str, Dict[str, str]]:
"""Load API keys from environment variables."""
api_keys = {}

# Load from environment variables
exchanges = ['binance', 'coinbase', 'kraken', 'kucoin', 'okx']

for exchange in exchanges:
api_key = os.getenv(f'{exchange.upper()}_API_KEY')
secret = os.getenv(f'{exchange.upper()}_SECRET')
passphrase = os.getenv(f'{exchange.upper()}_PASSPHRASE')

if api_key and secret:
api_keys[exchange] = {
'api_key': api_key,
'secret': secret,
'passphrase': passphrase
}
logger.info(f"Loaded API keys for {exchange}")
else:
logger.info(f"No API keys found for {exchange} - will use public data only")

return api_keys

def initialize_connections(self) -> None:
"""Initialize all configured exchange connections."""
exchange_configs = self.config.get('exchanges', {})
api_keys = self._load_api_keys()

for exchange_name, config in exchange_configs.items():
if config.get('enabled', False):
try:
# Get credentials
credentials_data = api_keys.get(exchange_name, {})

# Create credentials object
credentials = ExchangeCredentials(
api_key=credentials_data.get('api_key', ''),
secret=credentials_data.get('secret', ''),
passphrase=credentials_data.get('passphrase'),
sandbox=config.get('sandbox', True)
)

# Create connection
exchange_type = ExchangeType(exchange_name)
connection = ExchangeConnection(exchange_type, credentials, config)
self.connections[exchange_name] = connection

logger.info(f"Initialized connection for {exchange_name}")

except Exception as e:
logger.error(f"Failed to initialize {exchange_name}: {e}")

async def connect_all(self):
"""Connect to all initialized exchanges."""
for exchange_name, connection in self.connections.items():
try:
success = await connection.connect()
if success:
logger.info(f"Connected to {exchange_name}")
else:
logger.error(f"Failed to connect to {exchange_name}")
except Exception as e:
logger.error(f"Error connecting to {exchange_name}: {e}")

async def disconnect_all(self):
"""Disconnect from all exchanges."""
for exchange_name, connection in self.connections.items():
try:
await connection.disconnect()
except Exception as e:
logger.error(f"Error disconnecting from {exchange_name}: {e}")

def get_connection(self, exchange_name: str) -> Optional[ExchangeConnection]:
"""Get a specific exchange connection."""
return self.connections.get(exchange_name)

def get_all_status(self) -> Dict[str, Any]:
"""Get status of all connections."""
return {
exchange_name: connection.get_status()
for exchange_name, connection in self.connections.items()
}

async def health_check_all(self) -> Dict[str, bool]:
"""Perform health check on all connections."""
results = {}
for exchange_name, connection in self.connections.items():
try:
results[exchange_name] = await connection.health_check()
except Exception as e:
logger.error(f"Health check failed for {exchange_name}: {e}")
results[exchange_name] = False
return results