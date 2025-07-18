"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 REAL MARKET DATA FEED - SCHWABOT LIVE DATA INTEGRATION
========================================================

Real-time market data feed from multiple exchanges.
NO SIMULATED DATA - ONLY LIVE API FEEDS.

This module provides real market data to the Math-to-Trade Signal Router:
- Live BTC/USD, ETH/USD, SOL/USD prices
- Real volume data from exchanges
- Order book depth and liquidity
- Real-time trade execution data

Data Sources:
- Coinbase Pro API
- Binance API
- Kraken API
- CoinGecko API (backup)

Author: Schwabot Team
Date: 2025-01-02
"""

import asyncio
import logging
import time
import websockets
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import ccxt.async_support as ccxt_async
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
"""Class for Schwabot trading functionality."""
"""Real market data point"""
symbol: str
price: float
volume_24h: float
volume_1h: float
bid: float
ask: float
spread: float
timestamp: float
exchange: str
high_24h: float
low_24h: float
change_24h: float
change_percent_24h: float
market_cap: Optional[float] = None
circulating_supply: Optional[float] = None


@dataclass
class OrderBookData:
"""Class for Schwabot trading functionality."""
"""Real order book data"""
symbol: str
bids: List[tuple]  # [(price, volume), ...]
asks: List[tuple]  # [(price, volume), ...]
timestamp: float
exchange: str
best_bid: float
best_ask: float
spread: float
total_bid_volume: float
total_ask_volume: float


@dataclass
class TradeData:
"""Class for Schwabot trading functionality."""
"""Real trade execution data"""
symbol: str
price: float
volume: float
side: str  # 'buy' or 'sell'
timestamp: float
exchange: str
trade_id: str


class RealMarketDataFeed:
"""Class for Schwabot trading functionality."""
"""Real-time market data feed from multiple exchanges"""

def __init__(self, config: Dict[str, Any]) -> None:
self.config = config
self.exchanges = {}
self.websocket_connections = {}
self.data_callbacks = []
self.order_book_callbacks = []
self.trade_callbacks = []

# Data storage
self.latest_data = {}
self.order_books = {}
self.recent_trades = {}

# Connection status
self.connected_exchanges = set()
self.last_data_update = {}

# Symbols to track
self.symbols = config.get('symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD'])

async def initialize(self):
"""Initialize exchange connections and websockets"""
try:
# Initialize REST API connections
await self._initialize_rest_apis()

# Initialize WebSocket connections
await self._initialize_websockets()

logger.info("✅ Real Market Data Feed initialized")

except Exception as e:
logger.error(f"❌ Failed to initialize market data feed: {e}")
raise

async def _initialize_rest_apis(self):
"""Initialize REST API connections"""
try:
# Initialize Coinbase Pro
if self.config.get('coinbase', {}).get('enabled', True):
self.exchanges['coinbase'] = ccxt_async.coinbase({
'enableRateLimit': True,
'sandbox': False  # Use live data
})
logger.info("✅ Coinbase REST API initialized")

# Initialize Binance
if self.config.get('binance', {}).get('enabled', True):
self.exchanges['binance'] = ccxt_async.binance({
'enableRateLimit': True,
'sandbox': False  # Use live data
})
logger.info("✅ Binance REST API initialized")

# Initialize Kraken
if self.config.get('kraken', {}).get('enabled', True):
self.exchanges['kraken'] = ccxt_async.kraken({
'enableRateLimit': True,
})
logger.info("✅ Kraken REST API initialized")

# Load markets for all exchanges
for name, exchange in self.exchanges.items():
await exchange.load_markets()
self.connected_exchanges.add(name)
logger.info(f"📊 {name} markets loaded")

except Exception as e:
logger.error(f"❌ Failed to initialize REST APIs: {e}")
raise

async def _initialize_websockets(self):
"""Initialize WebSocket connections for real-time data"""
try:
# Start WebSocket tasks for each exchange
if 'coinbase' in self.exchanges:
asyncio.create_task(self._coinbase_websocket())

if 'binance' in self.exchanges:
asyncio.create_task(self._binance_websocket())

if 'kraken' in self.exchanges:
asyncio.create_task(self._kraken_websocket())

logger.info("✅ WebSocket connections initiated")

except Exception as e:
logger.error(f"❌ Failed to initialize WebSockets: {e}")

async def _coinbase_websocket(self):
"""Coinbase Pro WebSocket connection"""
uri = "wss://ws-feed.pro.coinbase.com"

# Subscribe message
subscribe_msg = {
"type": "subscribe",
"product_ids": ["BTC-USD", "ETH-USD", "SOL-USD"],
"channels": ["ticker", "level2", "matches"]
}

while True:
try:
async with websockets.connect(uri) as websocket:
await websocket.send(json.dumps(subscribe_msg))
logger.info("✅ Coinbase WebSocket connected")

async for message in websocket:
try:
data = json.loads(message)
await self._process_coinbase_message(data)
except Exception as e:
logger.error(f"❌ Coinbase message processing error: {e}")

except Exception as e:
logger.error(f"❌ Coinbase WebSocket error: {e}")
await asyncio.sleep(5)  # Reconnect after 5 seconds

async def _binance_websocket(self):
"""Binance WebSocket connection"""
streams = [
"btcusdt@ticker",
"ethusdt@ticker",
"solusdt@ticker",
"btcusdt@depth20@1000ms",
"ethusdt@depth20@1000ms",
"solusdt@depth20@1000ms"
]

uri = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"

while True:
try:
async with websockets.connect(uri) as websocket:
logger.info("✅ Binance WebSocket connected")

async for message in websocket:
try:
data = json.loads(message)
await self._process_binance_message(data)
except Exception as e:
logger.error(f"❌ Binance message processing error: {e}")

except Exception as e:
logger.error(f"❌ Binance WebSocket error: {e}")
await asyncio.sleep(5)

async def _kraken_websocket(self):
"""Kraken WebSocket connection"""
uri = "wss://ws.kraken.com"

subscribe_msg = {
"event": "subscribe",
"pair": ["XBT/USD", "ETH/USD", "SOL/USD"],
"subscription": {"name": "ticker"}
}

while True:
try:
async with websockets.connect(uri) as websocket:
await websocket.send(json.dumps(subscribe_msg))
logger.info("✅ Kraken WebSocket connected")

async for message in websocket:
try:
data = json.loads(message)
await self._process_kraken_message(data)
except Exception as e:
logger.error(f"❌ Kraken message processing error: {e}")

except Exception as e:
logger.error(f"❌ Kraken WebSocket error: {e}")
await asyncio.sleep(5)

async def _process_coinbase_message(self, data: Dict[str, Any]):
"""Process Coinbase WebSocket message"""
try:
if data.get('type') == 'ticker':
symbol = data.get('product_id', '').replace('-', '/')

market_data = MarketDataPoint(
symbol=symbol,
price=float(data.get('price', 0)),
volume_24h=float(data.get('volume_24h', 0)),
volume_1h=0.0,  # Not provided by Coinbase ticker
bid=float(data.get('best_bid', 0)),
ask=float(data.get('best_ask', 0)),
spread=float(data.get('best_ask', 0)) - float(data.get('best_bid', 0)),
timestamp=time.time(),
exchange='coinbase',
high_24h=float(data.get('high_24h', 0)),
low_24h=float(data.get('low_24h', 0)),
change_24h=0.0,  # Calculate from open
change_percent_24h=0.0
)

await self._update_market_data(market_data)

elif data.get('type') == 'l2update':
# Order book update
symbol = data.get('product_id', '').replace('-', '/')
await self._update_order_book_coinbase(symbol, data)

elif data.get('type') == 'match':
# Trade data
symbol = data.get('product_id', '').replace('-', '/')
trade_data = TradeData(
symbol=symbol,
price=float(data.get('price', 0)),
volume=float(data.get('size', 0)),
side=data.get('side', ''),
timestamp=time.time(),
exchange='coinbase',
trade_id=data.get('trade_id', '')
)
await self._update_trade_data(trade_data)

except Exception as e:
logger.error(f"❌ Coinbase message processing failed: {e}")

async def _process_binance_message(self, data: Dict[str, Any]):
"""Process Binance WebSocket message"""
try:
if 'stream' in data and data['stream'].endswith('@ticker'):
ticker_data = data.get('data', {})
symbol = ticker_data.get('s', '').replace('USDT', '/USD')

market_data = MarketDataPoint(
symbol=symbol,
price=float(ticker_data.get('c', 0)),  # Current price
volume_24h=float(ticker_data.get('v', 0)),
volume_1h=0.0,  # Calculate from data
bid=float(ticker_data.get('b', 0)),
ask=float(ticker_data.get('a', 0)),
spread=float(ticker_data.get('a', 0)) - float(ticker_data.get('b', 0)),
timestamp=time.time(),
exchange='binance',
high_24h=float(ticker_data.get('h', 0)),
low_24h=float(ticker_data.get('l', 0)),
change_24h=float(ticker_data.get('P', 0)),
change_percent_24h=float(ticker_data.get('P', 0))
)

await self._update_market_data(market_data)

elif 'stream' in data and '@depth' in data['stream']:
# Order book depth
depth_data = data.get('data', {})
symbol = data['stream'].split('@')[0].upper().replace('USDT', '/USD')
await self._update_order_book_binance(symbol, depth_data)

except Exception as e:
logger.error(f"❌ Binance message processing failed: {e}")

async def _process_kraken_message(self, data: List[Any]):
"""Process Kraken WebSocket message"""
try:
if isinstance(data, list) and len(data) >= 3:
if data[2] == 'ticker':
ticker_data = data[1]
symbol = data[3].replace('XBT', 'BTC')

# Kraken ticker format: [ask, bid, last_trade, volume, etc.]
ask_data = ticker_data.get('a', ['0', '0', '0'])
bid_data = ticker_data.get('b', ['0', '0', '0'])
last_trade = ticker_data.get('c', ['0', '0'])
volume = ticker_data.get('v', ['0', '0'])
high_low = ticker_data.get('h', ['0', '0'])

market_data = MarketDataPoint(
symbol=symbol,
price=float(last_trade[0]),
volume_24h=float(volume[1]),  # 24h volume
volume_1h=float(volume[0]),   # Today volume
bid=float(bid_data[0]),
ask=float(ask_data[0]),
spread=float(ask_data[0]) - float(bid_data[0]),
timestamp=time.time(),
exchange='kraken',
high_24h=float(high_low[1]),  # 24h high
low_24h=float(high_low[0]),   # 24h low (actually today's low)
change_24h=0.0,
change_percent_24h=0.0
)

await self._update_market_data(market_data)

except Exception as e:
logger.error(f"❌ Kraken message processing failed: {e}")

async def _update_market_data(self, data: MarketDataPoint):
"""Update latest market data and notify callbacks"""
self.latest_data[f"{data.symbol}_{data.exchange}"] = data
self.last_data_update[data.exchange] = time.time()

# Notify all registered callbacks
for callback in self.data_callbacks:
try:
await callback(data)
except Exception as e:
logger.error(f"❌ Data callback error: {e}")

async def _update_order_book_coinbase(self, symbol: str, data: Dict[str, Any]):
"""Update order book from Coinbase data"""
try:
# Process L2 update
changes = data.get('changes', [])

if symbol not in self.order_books:
self.order_books[symbol] = {'bids': [], 'asks': []}

# Apply changes to order book
for change in changes:
side, price_str, size_str = change
price = float(price_str)
size = float(size_str)

if side == 'buy':
# Update bids
self.order_books[symbol]['bids'] = [
(p, s) for p, s in self.order_books[symbol]['bids']
if p != price
]
if size > 0:
self.order_books[symbol]['bids'].append((price, size))
self.order_books[symbol]['bids'].sort(reverse=True)

elif side == 'sell':
# Update asks
self.order_books[symbol]['asks'] = [
(p, s) for p, s in self.order_books[symbol]['asks']
if p != price
]
if size > 0:
self.order_books[symbol]['asks'].append((price, size))
self.order_books[symbol]['asks'].sort()

# Create order book data
bids = self.order_books[symbol]['bids'][:20]  # Top 20
asks = self.order_books[symbol]['asks'][:20]  # Top 20

order_book = OrderBookData(
symbol=symbol,
bids=bids,
asks=asks,
timestamp=time.time(),
exchange='coinbase',
best_bid=bids[0][0] if bids else 0.0,
best_ask=asks[0][0] if asks else 0.0,
spread=(asks[0][0] - bids[0][0]) if bids and asks else 0.0,
total_bid_volume=sum(s for p, s in bids),
total_ask_volume=sum(s for p, s in asks)
)

# Notify callbacks
for callback in self.order_book_callbacks:
try:
await callback(order_book)
except Exception as e:
logger.error(f"❌ Order book callback error: {e}")

except Exception as e:
logger.error(f"❌ Coinbase order book update failed: {e}")

async def _update_order_book_binance(self, symbol: str, data: Dict[str, Any]):
"""Update order book from Binance data"""
try:
bids = [(float(p), float(s)) for p, s in data.get('bids', [])]
asks = [(float(p), float(s)) for p, s in data.get('asks', [])]

order_book = OrderBookData(
symbol=symbol,
bids=bids,
asks=asks,
timestamp=time.time(),
exchange='binance',
best_bid=bids[0][0] if bids else 0.0,
best_ask=asks[0][0] if asks else 0.0,
spread=(asks[0][0] - bids[0][0]) if bids and asks else 0.0,
total_bid_volume=sum(s for p, s in bids),
total_ask_volume=sum(s for p, s in asks)
)

# Notify callbacks
for callback in self.order_book_callbacks:
try:
await callback(order_book)
except Exception as e:
logger.error(f"❌ Order book callback error: {e}")

except Exception as e:
logger.error(f"❌ Binance order book update failed: {e}")

async def _update_trade_data(self, trade: TradeData):
"""Update trade data and notify callbacks"""
symbol_key = f"{trade.symbol}_{trade.exchange}"

if symbol_key not in self.recent_trades:
self.recent_trades[symbol_key] = []

# Add trade and keep only last 100
self.recent_trades[symbol_key].append(trade)
if len(self.recent_trades[symbol_key]) > 100:
self.recent_trades[symbol_key] = self.recent_trades[symbol_key][-100:]

# Notify callbacks
for callback in self.trade_callbacks:
try:
await callback(trade)
except Exception as e:
logger.error(f"❌ Trade callback error: {e}")

def register_data_callback(self, callback: Callable[[MarketDataPoint], None]) -> None:
"""Register callback for market data updates"""
self.data_callbacks.append(callback)
logger.info(f"✅ Market data callback registered: {callback.__name__}")

def register_order_book_callback(self, callback: Callable[[OrderBookData], None]) -> None:
"""Register callback for order book updates"""
self.order_book_callbacks.append(callback)
logger.info(f"✅ Order book callback registered: {callback.__name__}")

def register_trade_callback(self, callback: Callable[[TradeData], None]) -> None:
"""Register callback for trade updates"""
self.trade_callbacks.append(callback)
logger.info(f"✅ Trade callback registered: {callback.__name__}")

def get_latest_price(self, symbol: str, exchange: str = None) -> Optional[float]:
"""Get latest price for symbol"""
if exchange:
key = f"{symbol}_{exchange}"
data = self.latest_data.get(key)
return data.price if data else None
else:
# Return average across all exchanges
prices = []
for key, data in self.latest_data.items():
if data.symbol == symbol:
prices.append(data.price)
return sum(prices) / len(prices) if prices else None

def get_latest_volume(self, symbol: str, exchange: str = None) -> Optional[float]:
"""Get latest 24h volume for symbol"""
if exchange:
key = f"{symbol}_{exchange}"
data = self.latest_data.get(key)
return data.volume_24h if data else None
else:
# Return sum across all exchanges
volumes = []
for key, data in self.latest_data.items():
if data.symbol == symbol:
volumes.append(data.volume_24h)
return sum(volumes) if volumes else None

def get_aggregated_market_data(self, symbol: str) -> Optional[MarketDataPoint]:
"""Get aggregated market data across all exchanges"""
exchange_data = []

for key, data in self.latest_data.items():
if data.symbol == symbol:
exchange_data.append(data)

if not exchange_data:
return None

# Calculate weighted average price and aggregate volume
total_volume = sum(d.volume_24h for d in exchange_data)
if total_volume == 0:
return exchange_data[0]  # Return first if no volume

weighted_price = sum(d.price * d.volume_24h for d in exchange_data) / total_volume

return MarketDataPoint(
symbol=symbol,
price=weighted_price,
volume_24h=total_volume,
volume_1h=sum(d.volume_1h for d in exchange_data),
bid=max(d.bid for d in exchange_data),
ask=min(d.ask for d in exchange_data),
spread=min(d.ask for d in exchange_data) - max(d.bid for d in exchange_data),
timestamp=time.time(),
exchange='aggregated',
high_24h=max(d.high_24h for d in exchange_data),
low_24h=min(d.low_24h for d in exchange_data),
change_24h=sum(d.change_24h * d.volume_24h for d in exchange_data) / total_volume,
change_percent_24h=sum(d.change_percent_24h * d.volume_24h for d in exchange_data) / total_volume
)

async def get_coingecko_data(self, symbol: str) -> Optional[MarketDataPoint]:
"""Get backup data from CoinGecko API"""
try:
# Map symbols to CoinGecko IDs
symbol_map = {
'BTC/USD': 'bitcoin',
'ETH/USD': 'ethereum',
'SOL/USD': 'solana'
}

coin_id = symbol_map.get(symbol)
if not coin_id:
return None

url = f"https://api.coingecko.com/api/v3/simple/price"
params = {
'ids': coin_id,
'vs_currencies': 'usd',
'include_24hr_vol': 'true',
'include_24hr_change': 'true',
'include_market_cap': 'true'
}

async with aiohttp.ClientSession() as session:
async with session.get(url, params=params) as response:
if response.status == 200:
data = await response.json()
coin_data = data.get(coin_id, {})

return MarketDataPoint(
symbol=symbol,
price=float(coin_data.get('usd', 0)),
volume_24h=float(coin_data.get('usd_24h_vol', 0)),
volume_1h=0.0,
bid=0.0,  # Not provided
ask=0.0,  # Not provided
spread=0.0,
timestamp=time.time(),
exchange='coingecko',
high_24h=0.0,  # Not provided
low_24h=0.0,   # Not provided
change_24h=float(coin_data.get('usd_24h_change', 0)),
change_percent_24h=float(coin_data.get('usd_24h_change', 0)),
market_cap=float(coin_data.get('usd_market_cap', 0))
)

except Exception as e:
logger.error(f"❌ CoinGecko API error: {e}")
return None

def get_connection_status(self) -> Dict[str, Any]:
"""Get connection status for all exchanges"""
status = {
'connected_exchanges': list(self.connected_exchanges),
'total_exchanges': len(self.exchanges),
'last_updates': self.last_data_update,
'data_points': len(self.latest_data),
'websocket_status': {
exchange: exchange in self.connected_exchanges
for exchange in self.exchanges
}
}

return status


# Example usage and integration
async def main_market_data_example():
"""Example of how to use the Real Market Data Feed"""

config = {
'coinbase': {'enabled': True},
'binance': {'enabled': True},
'kraken': {'enabled': True},
'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD']
}

# Initialize market data feed
feed = RealMarketDataFeed(config)
await feed.initialize()

# Register callback for market data
async def on_market_data(data: MarketDataPoint):
logger.info(f"📊 {data.exchange}: {data.symbol} = ${data.price:.2f} (Vol: {data.volume_24h:.2f})")

# Register callback for order book
async def on_order_book(order_book: OrderBookData):
logger.info(f"📖 {order_book.exchange}: {order_book.symbol} - Spread: ${order_book.spread:.2f}")

# Register callbacks
feed.register_data_callback(on_market_data)
feed.register_order_book_callback(on_order_book)

logger.info("🚀 Real Market Data Feed started")

# Keep running
try:
while True:
# Get aggregated data
btc_data = feed.get_aggregated_market_data('BTC/USD')
if btc_data:
logger.info(f"🟡 BTC Aggregated: ${btc_data.price:.2f} (Vol: {btc_data.volume_24h:.2f})")

# Check connection status
status = feed.get_connection_status()
logger.info(f"🔗 Connected: {status['connected_exchanges']}")

await asyncio.sleep(30)  # Update every 30 seconds

except KeyboardInterrupt:
logger.info("🛑 Market data feed stopped")


if __name__ == "__main__":
# Configure logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

# Run market data feed
asyncio.run(main_market_data_example())