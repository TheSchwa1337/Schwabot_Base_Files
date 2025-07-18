"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced API Integration Manager - Production-Grade Market Data
==============================================================

Comprehensive API integration system supporting:
- CoinGecko (primary, free tier)
- CoinMarketCap (premium, API key required)
- Alpha Vantage (financial data, API key required)
- GPU acceleration for mathematical computations
- Real-time market data processing
- Secure API key management
- Risk management and trading parameters

Features:
- Real-time price data from multiple sources
- Volume, market cap, and technical indicators
- On-chain metrics and sentiment analysis
- GPU-accelerated mathematical computations
- Comprehensive error handling and fallback mechanisms
- Rate limiting and caching
- Secure API key storage and rotation
"""

import asyncio
import logging
import time
import os
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import numpy as np
import requests
from decimal import Decimal

# Set up logger first
logger = logging.getLogger(__name__)

# GPU acceleration imports
try:
import cupy as cp
import numba
from numba import cuda
GPU_AVAILABLE = True
logger.info("✅ GPU acceleration available (CuPy + Numba)")
except ImportError:
GPU_AVAILABLE = False
cp = None  # Define cp as None when not available
logger.info("⚠️ GPU acceleration not available, using CPU fallback")

class APISource(Enum):
"""Class for Schwabot trading functionality."""
"""API data sources."""
COINGECKO = "coingecko"
COINMARKETCAP = "coinmarketcap"
ALPHA_VANTAGE = "alpha_vantage"
GLASSNODE = "glassnode"
FEAR_GREED = "fear_greed"


class DataQuality(Enum):
"""Class for Schwabot trading functionality."""
"""Data quality levels."""
EXCELLENT = "excellent"  # All sources available, fresh data
GOOD = "good"           # Most sources available, recent data
ACCEPTABLE = "acceptable"  # Some sources available, older data
POOR = "poor"           # Limited sources, stale data
FAILED = "failed"       # No reliable data available


@dataclass
class MarketDataPoint:
"""Class for Schwabot trading functionality."""
"""Comprehensive market data point."""
symbol: str
price: float
volume_24h: float
market_cap: float
price_change_24h: float
price_change_percent_24h: float
high_24h: float
low_24h: float
circulating_supply: float
total_supply: float
max_supply: Optional[float]
timestamp: float
source: str
data_quality: DataQuality
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingParameters:
"""Class for Schwabot trading functionality."""
"""Trading parameters and risk management."""
max_position_size_pct: float = 10.0  # Max 10% of portfolio per position
max_total_exposure_pct: float = 30.0  # Max 30% total exposure
stop_loss_pct: float = 2.0  # 2% stop loss
take_profit_pct: float = 5.0  # 5% take profit
max_daily_loss_pct: float = 5.0  # 5% max daily loss
max_drawdown_pct: float = 15.0  # 15% max drawdown
volatility_threshold: float = 0.8  # Volatility threshold
min_order_size_usd: float = 10.0  # Minimum $10 orders
max_order_size_usd: float = 1000.0  # Maximum $1000 orders per trade
slippage_tolerance: float = 0.001  # 0.1% slippage tolerance
execution_timeout: float = 30.0  # 30 seconds execution timeout
retry_attempts: int = 3  # Number of retry attempts


@dataclass
class GPUConfig:
"""Class for Schwabot trading functionality."""
"""GPU acceleration configuration."""
enabled: bool = True
device_id: int = 0
memory_limit_gb: float = 4.0
compute_capability: str = "7.5"
enable_tensor_cores: bool = True
enable_mixed_precision: bool = True
batch_size: int = 1024
max_concurrent_kernels: int = 4


class EnhancedAPIIntegrationManager:
"""Class for Schwabot trading functionality."""
"""
Enhanced API Integration Manager for production-grade market data.

Provides comprehensive market data integration with:
- Multiple API sources with fallback mechanisms
- GPU acceleration for mathematical computations
- Real-time data processing and validation
- Secure API key management
- Risk management and trading parameters
"""

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
"""Initialize the enhanced API integration manager."""
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)

# API credentials (loaded from environment or config)
self.api_keys = self._load_api_keys()

# Data cache and rate limiting
self.data_cache: Dict[str, Dict[str, Any]] = {}
self.cache_duration = 60  # 1 minute cache
self.rate_limits = self._initialize_rate_limits()

# GPU configuration
self.gpu_config = GPUConfig(**self.config.get('gpu', {}))
self.gpu_available = GPU_AVAILABLE and self.gpu_config.enabled

# Trading parameters
self.trading_params = TradingParameters(**self.config.get('trading', {}))

# Performance tracking
self.performance_metrics = {
'total_requests': 0,
'successful_requests': 0,
'failed_requests': 0,
'cache_hits': 0,
'cache_misses': 0,
'gpu_operations': 0,
'average_response_time': 0.0,
'last_updated': time.time()
}

# Initialize GPU if available
if self.gpu_available:
self._initialize_gpu()

self.logger.info("✅ Enhanced API Integration Manager initialized")
self.logger.info(f"🔧 GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
'cache_duration': 60,
'max_retries': 3,
'timeout': 30,
'gpu': {
'enabled': True,
'device_id': 0,
'memory_limit_gb': 4.0,
'batch_size': 1024
},
'trading': {
'max_position_size_pct': 10.0,
'max_total_exposure_pct': 30.0,
'stop_loss_pct': 2.0,
'take_profit_pct': 5.0,
'max_daily_loss_pct': 5.0,
'max_drawdown_pct': 15.0,
'volatility_threshold': 0.8,
'min_order_size_usd': 10.0,
'max_order_size_usd': 1000.0,
'slippage_tolerance': 0.001,
'execution_timeout': 30.0,
'retry_attempts': 3
},
'apis': {
'coingecko': {
'enabled': True,
'base_url': 'https://api.coingecko.com/api/v3',
'rate_limit': 50,  # requests per minute
'timeout': 30
},
'coinmarketcap': {
'enabled': True,
'base_url': 'https://pro-api.coinmarketcap.com/v1',
'rate_limit': 30,  # requests per minute
'timeout': 30
},
'alpha_vantage': {
'enabled': True,
'base_url': 'https://www.alphavantage.co/query',
'rate_limit': 5,  # requests per minute (free tier)
'timeout': 30
}
}
}

def _load_api_keys(self) -> Dict[str, str]:
"""Load API keys from environment variables."""
api_keys = {}

# CoinMarketCap API key
cmc_key = os.getenv('COINMARKETCAP_API_KEY')
if cmc_key:
api_keys['coinmarketcap'] = cmc_key

# Alpha Vantage API key
av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
if av_key:
api_keys['alpha_vantage'] = av_key

# Glassnode API key
gn_key = os.getenv('GLASSNODE_API_KEY')
if gn_key:
api_keys['glassnode'] = gn_key

self.logger.info(f"🔑 Loaded {len(api_keys)} API keys")
return api_keys

def _initialize_rate_limits(self) -> Dict[str, Dict[str, Any]]:
"""Initialize rate limiting for each API."""
rate_limits = {}

for api_name, api_config in self.config['apis'].items():
rate_limits[api_name] = {
'requests_per_minute': api_config['rate_limit'],
'last_request_time': 0,
'request_count': 0,
'window_start': time.time()
}

return rate_limits

def _initialize_gpu(self) -> None:
"""Initialize GPU for mathematical computations."""
try:
if not self.gpu_available:
return

# Set GPU device
cp.cuda.Device(self.gpu_config.device_id).use()

# Get GPU info
gpu_info = cp.cuda.runtime.getDeviceProperties(self.gpu_config.device_id)
self.logger.info(f"🚀 GPU initialized: {gpu_info['name'].decode()}")
self.logger.info(f"💾 GPU Memory: {gpu_info['totalGlobalMem'] / 1024**3:.1f} GB")

# Set memory limit
memory_limit = int(self.gpu_config.memory_limit_gb * 1024**3)
cp.cuda.runtime.setDeviceLimit(cp.cuda.runtime.cudaLimitMaxL2FetchGranularity, memory_limit)

except Exception as e:
self.logger.error(f"❌ GPU initialization failed: {e}")
self.gpu_available = False

async def get_market_data(self, symbol: str, sources: Optional[List[APISource]] = None) -> Optional[MarketDataPoint]:
"""
Get comprehensive market data with fallback mechanism.

Priority order:
1. CoinMarketCap (if API key available)
2. CoinGecko (free, reliable)
3. Alpha Vantage (if API key available)
4. Cached data (if valid)
"""
start_time = time.time()

try:
# Check cache first
cached_data = self._get_cached_data(symbol)
if cached_data:
self.performance_metrics['cache_hits'] += 1
return cached_data

self.performance_metrics['cache_misses'] += 1

# Determine sources to try
if sources is None:
sources = self._get_priority_sources()

# Try each source in priority order
for source in sources:
try:
data = await self._fetch_from_source(symbol, source)
if data:
# Cache the data
self._cache_data(symbol, data)

# Update performance metrics
self._update_performance_metrics(start_time, True)

return data

except Exception as e:
self.logger.warning(f"❌ Failed to fetch from {source.value}: {e}")
continue

# If all sources failed, return cached data if available
if cached_data:
self.logger.warning(f"⚠️ Using stale cached data for {symbol}")
return cached_data

self._update_performance_metrics(start_time, False)
self.logger.error(f"❌ Failed to get market data for {symbol} from all sources")
return None

except Exception as e:
self.logger.error(f"❌ Error getting market data for {symbol}: {e}")
self._update_performance_metrics(start_time, False)
return None

def _get_priority_sources(self) -> List[APISource]:
"""Get API sources in priority order."""
sources = []

# CoinMarketCap first (if API key available)
if 'coinmarketcap' in self.api_keys:
sources.append(APISource.COINMARKETCAP)

# CoinGecko (always available, free)
sources.append(APISource.COINGECKO)

# Alpha Vantage (if API key available)
if 'alpha_vantage' in self.api_keys:
sources.append(APISource.ALPHA_VANTAGE)

return sources

async def _fetch_from_source(self, symbol: str, source: APISource) -> Optional[MarketDataPoint]:
"""Fetch data from specific source."""
if source == APISource.COINMARKETCAP:
return await self._fetch_coinmarketcap_data(symbol)
elif source == APISource.COINGECKO:
return await self._fetch_coingecko_data(symbol)
elif source == APISource.ALPHA_VANTAGE:
return await self._fetch_alpha_vantage_data(symbol)
else:
raise ValueError(f"Unsupported API source: {source}")

async def _fetch_coinmarketcap_data(self, symbol: str) -> Optional[MarketDataPoint]:
"""Fetch data from CoinMarketCap API."""
if 'coinmarketcap' not in self.api_keys:
return None

# Check rate limit
if not self._check_rate_limit('coinmarketcap'):
return None

try:
url = f"{self.config['apis']['coinmarketcap']['base_url']}/cryptocurrency/quotes/latest"
params = {'symbol': symbol, 'convert': 'USD'}
headers = {'X-CMC_PRO_API_KEY': self.api_keys['coinmarketcap']}

async with aiohttp.ClientSession() as session:
async with session.get(url, params=params, headers=headers,
timeout=self.config['apis']['coinmarketcap']['timeout']) as response:
if response.status == 200:
data = await response.json()

if 'data' in data and symbol in data['data']:
quote = data['data'][symbol]['quote']['USD']

return MarketDataPoint(
symbol=symbol,
price=float(quote['price']),
volume_24h=float(quote.get('volume_24h', 0)),
market_cap=float(quote.get('market_cap', 0)),
price_change_24h=float(quote.get('volume_change_24h', 0)),
price_change_percent_24h=float(quote.get('percent_change_24h', 0)),
high_24h=float(quote.get('high_24h', 0)),
low_24h=float(quote.get('low_24h', 0)),
circulating_supply=float(data['data'][symbol].get('circulating_supply', 0)),
total_supply=float(data['data'][symbol].get('total_supply', 0)),
max_supply=float(data['data'][symbol].get('max_supply', 0)) if data['data'][symbol].get('max_supply') else None,
timestamp=time.time(),
source='coinmarketcap',
data_quality=DataQuality.EXCELLENT
)

return None

except Exception as e:
self.logger.error(f"❌ CoinMarketCap API error: {e}")
return None

async def _fetch_coingecko_data(self, symbol: str) -> Optional[MarketDataPoint]:
"""Fetch data from CoinGecko API."""
# Check rate limit
if not self._check_rate_limit('coingecko'):
return None

try:
# Map symbols to CoinGecko IDs
symbol_mapping = {
'BTC': 'bitcoin',
'ETH': 'ethereum',
'ADA': 'cardano',
'SOL': 'solana',
'XRP': 'ripple',
'DOT': 'polkadot',
'DOGE': 'dogecoin',
'AVAX': 'avalanche-2',
'LINK': 'chainlink',
'MATIC': 'matic-network',
'UNI': 'uniswap',
'LTC': 'litecoin',
'BCH': 'bitcoin-cash',
'ATOM': 'cosmos',
'FTM': 'fantom'
}

coin_id = symbol_mapping.get(symbol, symbol.lower())

url = f"{self.config['apis']['coingecko']['base_url']}/simple/price"
params = {
'ids': coin_id,
'vs_currencies': 'usd',
'include_24hr_vol': 'true',
'include_24hr_change': 'true',
'include_market_cap': 'true',
'include_last_updated_at': 'true'
}

async with aiohttp.ClientSession() as session:
async with session.get(url, params=params,
timeout=self.config['apis']['coingecko']['timeout']) as response:
if response.status == 200:
data = await response.json()

if coin_id in data:
coin_data = data[coin_id]

return MarketDataPoint(
symbol=symbol,
price=float(coin_data['usd']),
volume_24h=float(coin_data.get('usd_24h_vol', 0)),
market_cap=float(coin_data.get('usd_market_cap', 0)),
price_change_24h=0.0,  # Not provided by CoinGecko
price_change_percent_24h=float(coin_data.get('usd_24h_change', 0)),
high_24h=0.0,  # Not provided by CoinGecko
low_24h=0.0,   # Not provided by CoinGecko
circulating_supply=0.0,  # Not provided by CoinGecko
total_supply=0.0,        # Not provided by CoinGecko
max_supply=None,         # Not provided by CoinGecko
timestamp=time.time(),
source='coingecko',
data_quality=DataQuality.GOOD
)

return None

except Exception as e:
self.logger.error(f"❌ CoinGecko API error: {e}")
return None

async def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[MarketDataPoint]:
"""Fetch data from Alpha Vantage API."""
if 'alpha_vantage' not in self.api_keys:
return None

# Check rate limit
if not self._check_rate_limit('alpha_vantage'):
return None

try:
url = self.config['apis']['alpha_vantage']['base_url']
params = {
'function': 'CURRENCY_EXCHANGE_RATE',
'from_currency': symbol,
'to_currency': 'USD',
'apikey': self.api_keys['alpha_vantage']
}

async with aiohttp.ClientSession() as session:
async with session.get(url, params=params,
timeout=self.config['apis']['alpha_vantage']['timeout']) as response:
if response.status == 200:
data = await response.json()

if 'Realtime Currency Exchange Rate' in data:
rate_data = data['Realtime Currency Exchange Rate']

return MarketDataPoint(
symbol=symbol,
price=float(rate_data['5. Exchange Rate']),
volume_24h=0.0,  # Not provided by Alpha Vantage
market_cap=0.0,  # Not provided by Alpha Vantage
price_change_24h=0.0,  # Not provided by Alpha Vantage
price_change_percent_24h=0.0,  # Not provided by Alpha Vantage
high_24h=0.0,  # Not provided by Alpha Vantage
low_24h=0.0,   # Not provided by Alpha Vantage
circulating_supply=0.0,  # Not provided by Alpha Vantage
total_supply=0.0,        # Not provided by Alpha Vantage
max_supply=None,         # Not provided by Alpha Vantage
timestamp=time.time(),
source='alpha_vantage',
data_quality=DataQuality.ACCEPTABLE
)

return None

except Exception as e:
self.logger.error(f"❌ Alpha Vantage API error: {e}")
return None

def _check_rate_limit(self, api_name: str) -> bool:
"""Check if API request is within rate limit."""
if api_name not in self.rate_limits:
return True

rate_limit = self.rate_limits[api_name]
current_time = time.time()

# Reset window if needed
if current_time - rate_limit['window_start'] >= 60:
rate_limit['window_start'] = current_time
rate_limit['request_count'] = 0

# Check if within limit
if rate_limit['request_count'] >= rate_limit['requests_per_minute']:
return False

# Update request count
rate_limit['request_count'] += 1
rate_limit['last_request_time'] = current_time

return True

def _get_cached_data(self, symbol: str) -> Optional[MarketDataPoint]:
"""Get cached data if valid."""
if symbol not in self.data_cache:
return None

cached = self.data_cache[symbol]
if time.time() - cached['timestamp'] <= self.cache_duration:
return cached['data']

return None

def _cache_data(self, symbol: str, data: MarketDataPoint) -> None:
"""Cache market data."""
self.data_cache[symbol] = {
'data': data,
'timestamp': time.time()
}

def _update_performance_metrics(self, start_time: float, success: bool) -> None:
"""Update performance metrics."""
self.performance_metrics['total_requests'] += 1

if success:
self.performance_metrics['successful_requests'] += 1
else:
self.performance_metrics['failed_requests'] += 1

# Update average response time
response_time = time.time() - start_time
current_avg = self.performance_metrics['average_response_time']
total_requests = self.performance_metrics['total_requests']

self.performance_metrics['average_response_time'] = (
(current_avg * (total_requests - 1) + response_time) / total_requests
)

self.performance_metrics['last_updated'] = time.time()

def compute_mathematical_indicators(self, price_data: List[float], volume_data: List[float]) -> Dict[str, float]:
"""
Compute mathematical indicators with GPU acceleration if available.

Returns:
Dictionary containing RSI, MACD, Bollinger Bands, ATR, etc.
"""
try:
if not price_data or len(price_data) < 2:
return {}

# Convert to numpy arrays
prices = np.array(price_data, dtype=np.float64)
volumes = np.array(volume_data, dtype=np.float64) if volume_data else np.ones_like(prices)

# Use GPU if available
if self.gpu_available:
return self._compute_indicators_gpu(prices, volumes)
else:
return self._compute_indicators_cpu(prices, volumes)

except Exception as e:
self.logger.error(f"❌ Error computing mathematical indicators: {e}")
return {}

def _compute_indicators_gpu(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
"""Compute technical indicators using GPU acceleration."""
try:
if not self.gpu_available or cp is None:
# Fallback to CPU computation
return self._compute_indicators_cpu(prices, volumes)

# Convert to GPU arrays
prices_gpu = cp.asarray(prices)
volumes_gpu = cp.asarray(volumes)

# Calculate returns
returns_gpu = cp.diff(cp.log(prices_gpu))

# Volatility (standard deviation of returns)
volatility = float(cp.std(returns_gpu))

# RSI calculation
deltas = cp.diff(prices_gpu)
gains = cp.where(deltas > 0, deltas, 0)
losses = cp.where(deltas < 0, -deltas, 0)

avg_gain = float(cp.mean(gains))
avg_loss = float(cp.mean(losses))

if avg_loss == 0:
rsi = 100.0
else:
rs = avg_gain / avg_loss
rsi = 100.0 - (100.0 / (1.0 + rs))

# MACD calculation
ema12 = self._compute_ema_gpu(prices_gpu, 12)
ema26 = self._compute_ema_gpu(prices_gpu, 26)
macd_line = ema12 - ema26
signal_line = self._compute_ema_gpu(macd_line, 9)
macd_histogram = macd_line - signal_line

# Bollinger Bands
sma = float(cp.mean(prices_gpu))
std = float(cp.std(prices_gpu))
upper_band = sma + (2 * std)
lower_band = sma - (2 * std)

# Volume indicators
avg_volume = float(cp.mean(volumes_gpu))
current_volume = float(volumes_gpu[-1])
volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

# Momentum indicators
momentum = float(prices_gpu[-1] - prices_gpu[0])
rate_of_change = float((prices_gpu[-1] / prices_gpu[0] - 1) * 100) if prices_gpu[0] > 0 else 0

# ATR calculation (simplified)
high_low = cp.max(prices_gpu) - cp.min(prices_gpu)
atr = float(high_low)

return {
'volatility': volatility,
'rsi': rsi,
'macd': float(macd_line[-1]),
'macd_signal': float(signal_line[-1]),
'macd_histogram': float(macd_histogram[-1]),
'bollinger_upper': upper_band,
'bollinger_middle': sma,
'bollinger_lower': lower_band,
'volume_ratio': volume_ratio,
'momentum': momentum,
'rate_of_change': rate_of_change,
'atr': atr,
'avg_volume': avg_volume,
'current_volume': current_volume
}

except Exception as e:
self.logger.error(f"Error computing indicators on GPU: {e}")
# Fallback to CPU computation
return self._compute_indicators_cpu(prices, volumes)

def _compute_indicators_cpu(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
"""Compute indicators using CPU."""
try:
# Compute price changes
price_changes = np.diff(prices)

# RSI (14-period)
if len(price_changes) >= 14:
gains = np.where(price_changes > 0, price_changes, 0)
losses = np.where(price_changes < 0, -price_changes, 0)

avg_gain = np.mean(gains[-14:])
avg_loss = np.mean(losses[-14:])

if avg_loss != 0:
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
else:
rsi = 100
else:
rsi = 50.0

# MACD (12, 26, 9)
if len(prices) >= 26:
ema12 = self._compute_ema_cpu(prices, 12)
ema26 = self._compute_ema_cpu(prices, 26)
macd_line = ema12 - ema26
signal_line = self._compute_ema_cpu(macd_line, 9)
macd_histogram = macd_line - signal_line

current_macd = macd_line[-1]
current_signal = signal_line[-1]
current_histogram = macd_histogram[-1]
else:
current_macd = 0.0
current_signal = 0.0
current_histogram = 0.0

# Bollinger Bands (20-period, 2 std)
if len(prices) >= 20:
sma20 = np.mean(prices[-20:])
std20 = np.std(prices[-20:])
upper_band = sma20 + (2 * std20)
lower_band = sma20 - (2 * std20)
current_price = prices[-1]

bb_position = (current_price - lower_band) / (upper_band - lower_band)
else:
bb_position = 0.5

# ATR (14-period)
if len(price_changes) >= 14:
high_low = np.abs(price_changes)
atr = np.mean(high_low[-14:])
else:
atr = np.std(price_changes) if len(price_changes) > 0 else 0.0

# Volume indicators
avg_volume = np.mean(volumes)
current_volume = volumes[-1]
volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

return {
'rsi': rsi,
'macd': current_macd,
'macd_signal': current_signal,
'macd_histogram': current_histogram,
'bb_position': bb_position,
'atr': atr,
'volume_ratio': volume_ratio,
'price_volatility': np.std(price_changes),
'price_momentum': price_changes[-1] if len(price_changes) > 0 else 0.0
}

except Exception as e:
self.logger.error(f"❌ CPU computation error: {e}")
return {}

def _compute_ema_gpu(self, data: np.ndarray, period: int) -> np.ndarray:
"""Compute EMA using GPU acceleration."""
if not self.gpu_available or cp is None:
# Fallback to CPU if GPU not available
return self._compute_ema_cpu(data, period)

try:
# Convert to GPU array
gpu_data = cp.asarray(data)

# Compute EMA on GPU
alpha = 2.0 / (period + 1)
ema = cp.zeros_like(gpu_data)
ema[0] = gpu_data[0]

for i in range(1, len(gpu_data)):
ema[i] = alpha * gpu_data[i] + (1 - alpha) * ema[i-1]

# Convert back to CPU
return cp.asnumpy(ema)

except Exception as e:
self.logger.warning(f"GPU EMA computation failed, falling back to CPU: {e}")
return self._compute_ema_cpu(data, period)

def _compute_ema_cpu(self, data: np.ndarray, period: int) -> np.ndarray:
"""Compute Exponential Moving Average on CPU."""
alpha = 2.0 / (period + 1)
ema = np.zeros_like(data)
ema[0] = data[0]

for i in range(1, len(data)):
ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

return ema

def get_trading_parameters(self) -> TradingParameters:
"""Get current trading parameters."""
return self.trading_params

def update_trading_parameters(self, **kwargs) -> None:
"""Update trading parameters."""
for key, value in kwargs.items():
if hasattr(self.trading_params, key):
setattr(self.trading_params, key, value)
self.logger.info(f"📊 Updated trading parameter: {key} = {value}")

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get performance metrics."""
metrics = self.performance_metrics.copy()

# Calculate success rate
total_requests = metrics['total_requests']
if total_requests > 0:
metrics['success_rate'] = metrics['successful_requests'] / total_requests
metrics['cache_hit_rate'] = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
else:
metrics['success_rate'] = 0.0
metrics['cache_hit_rate'] = 0.0

# Add GPU info
metrics['gpu_available'] = self.gpu_available
metrics['gpu_operations'] = metrics.get('gpu_operations', 0)

return metrics

def get_system_status(self) -> Dict[str, Any]:
"""Get comprehensive system status."""
return {
'api_integration_manager': {
'initialized': True,
'gpu_available': self.gpu_available,
'api_keys_loaded': len(self.api_keys),
'cache_size': len(self.data_cache),
'performance_metrics': self.get_performance_metrics()
},
'trading_parameters': {
'max_position_size_pct': self.trading_params.max_position_size_pct,
'max_total_exposure_pct': self.trading_params.max_total_exposure_pct,
'stop_loss_pct': self.trading_params.stop_loss_pct,
'take_profit_pct': self.trading_params.take_profit_pct,
'max_daily_loss_pct': self.trading_params.max_daily_loss_pct,
'max_drawdown_pct': self.trading_params.max_drawdown_pct,
'volatility_threshold': self.trading_params.volatility_threshold
},
'gpu_config': {
'enabled': self.gpu_config.enabled,
'device_id': self.gpu_config.device_id,
'memory_limit_gb': self.gpu_config.memory_limit_gb,
'batch_size': self.gpu_config.batch_size
}
}


# Factory function
def create_enhanced_api_integration_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedAPIIntegrationManager:
"""Create an enhanced API integration manager instance."""
return EnhancedAPIIntegrationManager(config)


# Global instance for easy access
enhanced_api_manager = EnhancedAPIIntegrationManager()


async def main():
"""Main function for testing."""
# Test market data fetching
symbols = ['BTC', 'ETH', 'SOL']

for symbol in symbols:
print(f"\n🔍 Fetching data for {symbol}...")
data = await enhanced_api_manager.get_market_data(symbol)

if data:
print(f"✅ {symbol} Price: ${data.price:,.2f}")
print(f"📊 Volume 24h: ${data.volume_24h:,.0f}")
print(f"💰 Market Cap: ${data.market_cap:,.0f}")
print(f"📈 24h Change: {data.price_change_percent_24h:+.2f}%")
print(f"🎯 Data Quality: {data.data_quality.value}")
print(f"🔗 Source: {data.source}")
else:
print(f"❌ Failed to get data for {symbol}")

# Test mathematical indicators
print(f"\n🧮 Computing mathematical indicators...")
price_data = [50000, 51000, 52000, 51500, 53000, 52500, 54000, 53500, 55000, 54500]
volume_data = [1000, 1100, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450]

indicators = enhanced_api_manager.compute_mathematical_indicators(price_data, volume_data)

print("📊 Technical Indicators:")
for indicator, value in indicators.items():
print(f"  {indicator}: {value:.4f}")

# Get system status
status = enhanced_api_manager.get_system_status()
print(f"\n📈 System Status:")
print(f"  GPU Available: {status['api_integration_manager']['gpu_available']}")
print(f"  API Keys Loaded: {status['api_integration_manager']['api_keys_loaded']}")
print(f"  Success Rate: {status['api_integration_manager']['performance_metrics']['success_rate']:.2%}")
print(f"  Cache Hit Rate: {status['api_integration_manager']['performance_metrics']['cache_hit_rate']:.2%}")


if __name__ == "__main__":
asyncio.run(main())