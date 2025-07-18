"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Market Data Integration
=================================
Provides real-time market data from multiple exchanges via WebSocket connections.
Supports Binance, Coinbase, and Kraken with automatic reconnection and error handling.
"""

import aiohttp
import websockets
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import time
import logging
import asyncio
import json

logger = logging.getLogger(__name__)


@dataclass
class PriceUpdate:
"""Class for Schwabot trading functionality."""
"""Real-time price update from exchange."""
symbol: str
price: float
volume: float
bid: float
ask: float
timestamp: float
exchange: str
high_24h: Optional[float] = None
low_24h: Optional[float] = None
change_24h: Optional[float] = None
change_percent_24h: Optional[float] = None

class RealTimeMarketDataIntegration:
"""Class for Schwabot trading functionality."""
"""Real-time market data integration with WebSocket connections."""


def __init__(self, config: Dict[str, Any]) -> None:
self.config = config
self.websocket_connections = {}
self.price_cache = {}
self.callbacks = []
self.is_running = False
self.reconnect_delay = config.get('reconnect_delay', 5)
self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
self.price_cache_ttl = config.get('price_cache_ttl', 60)  # 60 seconds

# Statistics
self.messages_received = 0
self.connection_errors = 0
self.last_cleanup = time.time()

# Connection status
self.connection_status = {}

logger.info("Real-time market data integration initialized")


def add_price_callback(self, callback: Callable[[PriceUpdate], None]) -> None:
"""Add a callback function to be called on price updates."""
self.callbacks.append(callback)
logger.info(f"Added price callback, total callbacks: {len(self.callbacks)}")


def remove_price_callback(self, callback: Callable[[PriceUpdate], None]) -> None:
"""Remove a price callback function."""
if callback in self.callbacks:
self.callbacks.remove(callback)
logger.info(f"Removed price callback, total callbacks: {len(self.callbacks)}")

async def start(self):
"""Start all WebSocket connections."""
if self.is_running:
logger.warning("Market data integration already running")
return

self.is_running = True
logger.info("Starting real-time market data integration...")

# Initialize WebSocket connections
exchanges = self.config.get('exchanges', {})

for exchange_name, exchange_config in exchanges.items():
if exchange_config.get('websocket_enabled', False):
await self.setup_websocket(exchange_name, exchange_config)

# Start cleanup task
asyncio.create_task(self._cleanup_old_prices())

logger.info("Real-time market data integration started")

async def stop(self):
"""Stop all WebSocket connections."""
self.is_running = False
logger.info("Stopping real-time market data integration...")

# Close all WebSocket connections
for exchange_name, websocket in self.websocket_connections.items():
try:
await websocket.close()
logger.info(f"Closed {exchange_name} WebSocket connection")
except Exception as e:
logger.error(f"Error closing {exchange_name} WebSocket: {e}")

self.websocket_connections.clear()
logger.info("Real-time market data integration stopped")

async def setup_websocket(self, exchange_name: str, config: Dict[str, Any]):
"""Setup WebSocket connection for exchange."""
try:
if exchange_name == 'binance':
await self._setup_binance_websocket(config)
elif exchange_name == 'coinbase':
await self._setup_coinbase_websocket(config)
elif exchange_name == 'kraken':
await self._setup_kraken_websocket(config)
else:
logger.warning(f"Unsupported exchange for WebSocket: {exchange_name}")

except Exception as e:
logger.error(f"WebSocket setup error for {exchange_name}: {e}")

async def _setup_binance_websocket(self, config: Dict[str, Any]):
"""Setup Binance WebSocket connection."""
symbols = config.get('symbols', ['btcusdt', 'ethusdt', 'solusdt'])
streams = [f"{symbol}@ticker" for symbol in symbols]

uri = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"

async def binance_handler():
reconnect_attempts = 0

while self.is_running and reconnect_attempts < self.max_reconnect_attempts:
try:
async with websockets.connect(uri, ping_interval=30, ping_timeout=10) as websocket:
self.websocket_connections['binance'] = websocket
self.connection_status['binance'] = 'connected'
logger.info("✅ Binance WebSocket connected")
reconnect_attempts = 0

async for message in websocket:
if not self.is_running:
break

try:
data = json.loads(message)
await self._process_binance_message(data)
self.messages_received += 1
except json.JSONDecodeError as e:
logger.error(f"Binance JSON decode error: {e}")
except Exception as e:
logger.error(f"Binance message processing error: {e}")

except Exception as e:
reconnect_attempts += 1
self.connection_errors += 1
self.connection_status['binance'] = 'disconnected'
logger.error(f"Binance WebSocket error (attempt {reconnect_attempts}): {e}")

if self.is_running:
await asyncio.sleep(self.reconnect_delay)

asyncio.create_task(binance_handler())

async def _setup_coinbase_websocket(self, config: Dict[str, Any]):
"""Setup Coinbase WebSocket connection."""
symbols = config.get('symbols', ['BTC-USD', 'ETH-USD', 'SOL-USD'])

# Coinbase WebSocket subscription message
subscribe_message = {
"type": "subscribe",
"product_ids": symbols,
"channels": ["ticker"]
}

uri = "wss://ws-feed.exchange.coinbase.com"

async def coinbase_handler():
reconnect_attempts = 0

while self.is_running and reconnect_attempts < self.max_reconnect_attempts:
try:
async with websockets.connect(uri, ping_interval=30, ping_timeout=10) as websocket:
self.websocket_connections['coinbase'] = websocket
self.connection_status['coinbase'] = 'connected'

# Send subscription message
await websocket.send(json.dumps(subscribe_message))
logger.info("✅ Coinbase WebSocket connected")
reconnect_attempts = 0

async for message in websocket:
if not self.is_running:
break

try:
data = json.loads(message)
await self._process_coinbase_message(data)
self.messages_received += 1
except json.JSONDecodeError as e:
logger.error(f"Coinbase JSON decode error: {e}")
except Exception as e:
logger.error(f"Coinbase message processing error: {e}")

except Exception as e:
reconnect_attempts += 1
self.connection_errors += 1
self.connection_status['coinbase'] = 'disconnected'
logger.error(f"Coinbase WebSocket error (attempt {reconnect_attempts}): {e}")

if self.is_running:
await asyncio.sleep(self.reconnect_delay)

asyncio.create_task(coinbase_handler())

async def _setup_kraken_websocket(self, config: Dict[str, Any]):
"""Setup Kraken WebSocket connection."""
symbols = config.get('symbols', ['XBT/USD', 'ETH/USD', 'SOL/USD'])

# Kraken WebSocket subscription message
subscribe_message = {
"event": "subscribe",
"pair": symbols,
"subscription": {
"name": "ticker"
}
}

uri = "wss://ws.kraken.com"

async def kraken_handler():
reconnect_attempts = 0

while self.is_running and reconnect_attempts < self.max_reconnect_attempts:
try:
async with websockets.connect(uri, ping_interval=30, ping_timeout=10) as websocket:
self.websocket_connections['kraken'] = websocket
self.connection_status['kraken'] = 'connected'

# Send subscription message
await websocket.send(json.dumps(subscribe_message))
logger.info("✅ Kraken WebSocket connected")
reconnect_attempts = 0

async for message in websocket:
if not self.is_running:
break

try:
data = json.loads(message)
await self._process_kraken_message(data)
self.messages_received += 1
except json.JSONDecodeError as e:
logger.error(f"Kraken JSON decode error: {e}")
except Exception as e:
logger.error(f"Kraken message processing error: {e}")

except Exception as e:
reconnect_attempts += 1
self.connection_errors += 1
self.connection_status['kraken'] = 'disconnected'
logger.error(f"Kraken WebSocket error (attempt {reconnect_attempts}): {e}")

if self.is_running:
await asyncio.sleep(self.reconnect_delay)

asyncio.create_task(kraken_handler())

async def _process_binance_message(self, data: Dict[str, Any]):
"""Process Binance WebSocket message."""
try:
if 'data' in data:
ticker_data = data['data']
symbol = ticker_data['s'].replace('USDT', '/USD')
price = float(ticker_data['c'])

price_update = PriceUpdate(
symbol=symbol,
price=price,
volume=float(ticker_data['v']),
bid=float(ticker_data['b']),
ask=float(ticker_data['a']),
timestamp=time.time(),
exchange='binance',
high_24h=float(ticker_data['h']),
low_24h=float(ticker_data['l']),
change_24h=float(ticker_data['P']),
change_percent_24h=float(ticker_data['P'])
)

await self._update_price_cache(price_update)
await self._notify_callbacks(price_update)

except Exception as e:
logger.error(f"Binance message processing error: {e}")

async def _process_coinbase_message(self, data: Dict[str, Any]):
"""Process Coinbase WebSocket message."""
try:
if data.get('type') == 'ticker':
symbol = data['product_id'].replace('-', '/')
price = float(data['price'])

price_update = PriceUpdate(
symbol=symbol,
price=price,
volume=float(data.get('volume_24h', 0)),
bid=float(data.get('best_bid', price)),
ask=float(data.get('best_ask', price)),
timestamp=time.time(),
exchange='coinbase',
high_24h=float(data.get('high_24h', price)),
low_24h=float(data.get('low_24h', price)),
change_24h=float(data.get('price_24h', 0)),
change_percent_24h=float(data.get('price_change_percent_24h', 0))
)

await self._update_price_cache(price_update)
await self._notify_callbacks(price_update)

except Exception as e:
logger.error(f"Coinbase message processing error: {e}")

async def _process_kraken_message(self, data: Dict[str, Any]):
"""Process Kraken WebSocket message."""
try:
if isinstance(data, list) and len(data) > 1:
ticker_data = data[1]
if isinstance(ticker_data, dict) and 'c' in ticker_data:
symbol = data[3].replace('/', '/')  # Kraken format
price = float(ticker_data['c'][0])

price_update = PriceUpdate(
symbol=symbol,
price=price,
volume=float(ticker_data.get('v', [0])[1]),
bid=float(ticker_data.get('b', [price])[0]),
ask=float(ticker_data.get('a', [price])[0]),
timestamp=time.time(),
exchange='kraken',
high_24h=float(ticker_data.get('h', [price])[1]),
low_24h=float(ticker_data.get('l', [price])[1]),
change_24h=float(ticker_data.get('p', [0])[1]),
change_percent_24h=0.0  # Kraken doesn't provide this directly
)

await self._update_price_cache(price_update)
await self._notify_callbacks(price_update)

except Exception as e:
logger.error(f"Kraken message processing error: {e}")

async def _update_price_cache(self, price_update: PriceUpdate):
"""Update price cache with new price data."""
try:
self.price_cache[price_update.symbol] = {
'price': price_update.price,
'volume': price_update.volume,
'bid': price_update.bid,
'ask': price_update.ask,
'timestamp': price_update.timestamp,
'exchange': price_update.exchange,
'high_24h': price_update.high_24h,
'low_24h': price_update.low_24h,
'change_24h': price_update.change_24h,
'change_percent_24h': price_update.change_percent_24h
}
except Exception as e:
logger.error(f"Price cache update error: {e}")

async def _notify_callbacks(self, price_update: PriceUpdate):
"""Notify all registered callbacks of price update."""
for callback in self.callbacks:
try:
if asyncio.iscoroutinefunction(callback):
await callback(price_update)
else:
callback(price_update)
except Exception as e:
logger.error(f"Callback error: {e}")

async def _cleanup_old_prices(self):
"""Clean up old price data from cache."""
while self.is_running:
try:
current_time = time.time()
expired_symbols = []

for symbol, price_data in self.price_cache.items():
if current_time - price_data['timestamp'] > self.price_cache_ttl:
expired_symbols.append(symbol)

for symbol in expired_symbols:
del self.price_cache[symbol]

if expired_symbols:
logger.debug(f"Cleaned up {len(expired_symbols)} expired price entries")

self.last_cleanup = current_time
await asyncio.sleep(60)  # Clean up every minute

except Exception as e:
logger.error(f"Price cleanup error: {e}")
await asyncio.sleep(60)

def get_price(self, symbol: str, exchange: str = None) -> Optional[Dict[str, Any]]:
"""Get current price for a symbol."""
try:
price_data = self.price_cache.get(symbol)
if price_data and (time.time() - price_data['timestamp']) < self.price_cache_ttl:
if exchange is None or price_data['exchange'] == exchange:
return price_data
return None
except Exception as e:
logger.error(f"Error getting price for {symbol}: {e}")
return None

def get_all_prices(self) -> Dict[str, Dict[str, Any]]:
"""Get all current prices."""
return self.price_cache.copy()

def get_statistics(self) -> Dict[str, Any]:
"""Get integration statistics."""
return {
'messages_received': self.messages_received,
'connection_errors': self.connection_errors,
'price_cache_size': len(self.price_cache),
'active_callbacks': len(self.callbacks),
'last_cleanup': self.last_cleanup,
'is_running': self.is_running
}

def get_connection_status(self) -> Dict[str, str]:
"""Get connection status for all exchanges."""
return self.connection_status.copy()


async def test_market_data_integration():
"""Test the market data integration."""
config = {
'exchanges': {
'binance': {
'websocket_enabled': True,
'symbols': ['btcusdt', 'ethusdt']
},
'coinbase': {
'websocket_enabled': True,
'symbols': ['BTC-USD', 'ETH-USD']
}
},
'reconnect_delay': 5,
'max_reconnect_attempts': 3,
'price_cache_ttl': 60
}

integration = RealTimeMarketDataIntegration(config)

def test_callback(price_update: PriceUpdate):
print(f"Price update: {price_update.symbol} = ${price_update.price} from {price_update.exchange}")

integration.add_price_callback(test_callback)

await integration.start()

try:
# Run for 30 seconds
await asyncio.sleep(30)
finally:
await integration.stop()
print("Test completed")


if __name__ == "__main__":
asyncio.run(test_market_data_integration())