"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Coinbase API Integration
===============================
Direct integration with Coinbase API for live trading, bypassing CCXT.
Provides real-time market data, order placement, and account management.
"""

import aiohttp
from typing import Any, Dict, List, Optional
import time
import logging
import json
import hmac
import hashlib
import asyncio
import base64

logger = logging.getLogger(__name__)

class CoinbaseDirectAPI:
"""Class for Schwabot trading functionality."""
"""Direct Coinbase API integration for live trading."""


def __init__(self, api_key: str, secret: str, passphrase: str, sandbox: bool = False) -> None:
self.api_key = api_key
self.secret = secret
self.passphrase = passphrase
self.sandbox = sandbox

# API endpoints
if sandbox:
self.base_url = "https://api-public.sandbox.exchange.coinbase.com"
else:
self.base_url = "https://api.exchange.coinbase.com"

# WebSocket endpoints
if sandbox:
self.ws_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
else:
self.ws_url = "wss://ws-feed.exchange.coinbase.com"

self.session = None
self.status = "DISCONNECTED"
self.last_heartbeat = 0

logger.info(f"Coinbase Direct API initialized (sandbox: {sandbox})")

async def _get_session(self) -> aiohttp.ClientSession:
"""Get or create HTTP session."""
if self.session is None or self.session.closed:
self.session = aiohttp.ClientSession(
timeout=aiohttp.ClientTimeout(total=30),
headers={
'Content-Type': 'application/json',
'User-Agent': 'Schwabot-Trading-System/1.0'
}
)
return self.session

def _generate_signature(
self,
timestamp: str,
method: str,
request_path: str,
body: str = '') -> str:
"""Generate Coinbase API signature."""
message = timestamp + method + request_path + body
signature = hmac.new(
self.secret.encode('utf-8'),
message.encode('utf-8'),
hashlib.sha256
)
return base64.b64encode(signature.digest()).decode('utf-8')

def _get_auth_headers(self, method: str, request_path: str,
body: str = '') -> Dict[str, str]:
"""Get authenticated headers for API requests."""
timestamp = str(int(time.time()))
signature = self._generate_signature(timestamp, method, request_path, body)

return {
'CB-ACCESS-KEY': self.api_key,
'CB-ACCESS-SIGN': signature,
'CB-ACCESS-TIMESTAMP': timestamp,
'CB-ACCESS-PASSPHRASE': self.passphrase
}

async def connect(self) -> bool:
"""Connect to Coinbase API."""
try:
self.status = "CONNECTING"
logger.info("Connecting to Coinbase API...")

# Test connection by getting server time
session = await self._get_session()
async with session.get(f"{self.base_url}/time") as response:
if response.status == 200:
data = await response.json()
logger.info(f"Coinbase server time: {data.get('iso')}")
self.status = "CONNECTED"
self.last_heartbeat = time.time()
logger.info("Successfully connected to Coinbase API")
return True
else:
logger.error(f"Failed to connect to Coinbase API: {response.status}")
self.status = "ERROR"
return False

except Exception as e:
logger.error(f"Error connecting to Coinbase API: {e}")
self.status = "ERROR"
return False

async def disconnect(self):
"""Disconnect from Coinbase API."""
if self.session and not self.session.closed:
await self.session.close()
self.status = "DISCONNECTED"
logger.info("Disconnected from Coinbase API")

async def get_accounts(self) -> Optional[List[Dict[str, Any]]]:
"""Get account information."""
try:
session = await self._get_session()
headers = self._get_auth_headers('GET', '/accounts')

async with session.get(f"{self.base_url}/accounts", headers=headers) as response:
if response.status == 200:
accounts = await response.json()
return accounts
else:
logger.error(
f"Failed to get accounts: {response.status}")
return None

except Exception as e:
logger.error(f"Error getting accounts: {e}")
return None

async def get_product_ticker(
self, product_id: str) -> Optional[Dict[str, Any]]:
"""Get product ticker information."""
try:
session = await self._get_session()
path = f"/products/{product_id}/ticker"

async with session.get(f"{self.base_url}{path}") as response:
if response.status == 200:
ticker = await response.json()
return ticker
else:
logger.error(
f"Failed to get ticker for {product_id}: {response.status}")
return None

except Exception as e:
logger.error(
f"Error getting ticker for {product_id}: {e}")
return None

async def place_order(self, product_id: str, side: str, order_type: str,
size: Optional[str] = None, price: Optional[str] = None,
client_order_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
"""Place an order on Coinbase."""
try:
session = await self._get_session()

# Prepare order data
order_data = {
'product_id': product_id,
'side': side,
'type': order_type
}

if size:
order_data['size'] = size
if price:
order_data['price'] = price
if client_order_id:
order_data['client_order_id'] = client_order_id

body = json.dumps(
order_data)
path = '/orders'
headers = self._get_auth_headers(
'POST', path, body)

async with session.post(f"{self.base_url}{path}", headers=headers, data=body) as response:
if response.status == 200:
order = await response.json()
logger.info(
f"Order placed successfully: {order.get('id')}")
return order
else:
error_data = await response.text()
logger.error(
f"Failed to place order: {response.status} - {error_data}")
return None

except Exception as e:
logger.error(
f"Error placing order: {e}")
return None

async def get_order(
self, order_id: str) -> Optional[Dict[str, Any]]:
"""Get order information."""
try:
session = await self._get_session()
path = f'/orders/{order_id}'
headers = self._get_auth_headers(
'GET', path)

async with session.get(f"{self.base_url}{path}", headers=headers) as response:
if response.status == 200:
order = await response.json()
return order
else:
logger.error(
f"Failed to get order {order_id}: {response.status}")
return None

except Exception as e:
logger.error(
f"Error getting order {order_id}: {e}")
return None

async def cancel_order(
self, order_id: str) -> Optional[Dict[str, Any]]:
"""Cancel an order."""
try:
session = await self._get_session()
path = f'/orders/{order_id}'
headers = self._get_auth_headers(
'DELETE', path)

async with session.delete(f"{self.base_url}{path}", headers=headers) as response:
if response.status == 200:
result = await response.json()
logger.info(
f"Order {order_id} cancelled successfully")
return result
else:
logger.error(
f"Failed to cancel order {order_id}: {response.status}")
return None

except Exception as e:
logger.error(
f"Error cancelling order {order_id}: {e}")
return None

async def get_open_orders(
self, product_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
"""Get open orders."""
try:
session = await self._get_session()
path = '/orders?status=open'
if product_id:
path += f'&product_id={product_id}'

headers = self._get_auth_headers(
'GET', path)

async with session.get(f"{self.base_url}{path}", headers=headers) as response:
if response.status == 200:
orders = await response.json()
return orders
else:
logger.error(
f"Failed to get open orders: {response.status}")
return None

except Exception as e:
logger.error(
f"Error getting open orders: {e}")
return None

async def get_products(
self) -> Optional[List[Dict[str, Any]]]:
"""Get available products."""
try:
session = await self._get_session()

async with session.get(f"{self.base_url}/products") as response:
if response.status == 200:
products = await response.json()
return products
else:
logger.error(
f"Failed to get products: {response.status}")
return None

except Exception as e:
logger.error(
f"Error getting products: {e}")
return None

async def get_product_candles(self, product_id: str, start: str, end: str,
granularity: int = 60) -> Optional[List[List[Any]]]:
"""Get product candles (OHLCV data)."""
try:
session = await self._get_session()
path = f'/products/{product_id}/candles?start={start}&end={end}&granularity={granularity}'

async with session.get(f"{self.base_url}{path}") as response:
if response.status == 200:
candles = await response.json()
return candles
else:
logger.error(
f"Failed to get candles for {product_id}: {response.status}")
return None

except Exception as e:
logger.error(
f"Error getting candles for {product_id}: {e}")
return None

async def health_check(
self) -> bool:
"""Perform health check."""
try:
session = await self._get_session()
async with session.get(f"{self.base_url}/time") as response:
if response.status == 200:
self.last_heartbeat = time.time()
return True
else:
return False
except Exception as e:
logger.error(
f"Health check failed: {e}")
return False

def get_status(
self) -> Dict[str, Any]:
"""Get API status."""
return {
'status': self.status,
'last_heartbeat': self.last_heartbeat,
'sandbox': self.sandbox,
'base_url': self.base_url
}

class CoinbaseWebSocket:
"""Class for Schwabot trading functionality."""
"""WebSocket connection for real-time Coinbase data."""

def __init__(self, sandbox: bool = False) -> None:
self.sandbox = sandbox
if sandbox:
self.ws_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
else:
self.ws_url = "wss://ws-feed.exchange.coinbase.com"

self.websocket = None
self.is_connected = False
self.callbacks = []

logger.info(f"Coinbase WebSocket initialized (sandbox: {sandbox})")

def add_callback(self, callback) -> None:
"""Add callback for price updates."""
self.callbacks.append(callback)

async def connect(self, product_ids: List[str]):
"""Connect to WebSocket and subscribe to products."""
try:
import websockets

# Prepare subscription message
subscribe_message = {
"type": "subscribe",
"product_ids": product_ids,
"channels": ["ticker", "level2", "heartbeat"]
}

async with websockets.connect(self.ws_url) as websocket:
self.websocket = websocket
self.is_connected = True

# Send subscription
await websocket.send(json.dumps(subscribe_message))
logger.info(f"Subscribed to {product_ids}")

# Listen for messages
async for message in websocket:
if not self.is_connected:
break

try:
data = json.loads(message)
await self._handle_message(data)
except json.JSONDecodeError as e:
logger.error(f"JSON decode error: {e}")
except Exception as e:
logger.error(f"Message handling error: {e}")

except Exception as e:
logger.error(f"WebSocket connection error: {e}")
self.is_connected = False

async def _handle_message(self, data: Dict[str, Any]):
"""Handle incoming WebSocket messages."""
message_type = data.get('type')

if message_type == 'ticker':
# Price update
for callback in self.callbacks:
try:
await callback(data)
except Exception as e:
logger.error(f"Callback error: {e}")

elif message_type == 'heartbeat':
# Heartbeat message
pass

elif message_type == 'error':
logger.error(f"WebSocket error: {data}")

async def disconnect(self):
"""Disconnect from WebSocket."""
self.is_connected = False
if self.websocket:
await self.websocket.close()
logger.info("Disconnected from Coinbase WebSocket")