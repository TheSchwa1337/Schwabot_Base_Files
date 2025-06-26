# -*- coding: utf-8 -*-\n# Import safe print for Windows compatibility
try:
from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    pass
    pass
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
    pass
    pass
def safe_print(message):


    pass
    pass
    print(message)
def info(message):


    pass
    pass
    print(f"[INFO] {message}")
def warn(message):


    pass
    pass
    print(f"[WARN] {message}")
def error(message):


    pass
    pass
    print(f"[ERROR] {message}")
def success(message):


    pass
    pass
    print(f"[SUCCESS] {message}")
def debug(message):


    pass
    pass
    print(f"[DEBUG] {message}")
# #!/usr/bin/env python3
"""
API Bridge Manager - Multi-Source Crypto API Integration for Schwabot
=====================================================================

This module implements a comprehensive, asynchronous API bridge manager for Schwabot,
providing unified access to multiple crypto data sources. It handles rate
limiting, error recovery, data normalization, and caching.

Core Functionality:
- Fully asynchronous multi-source API integration (CoinMarketCap, CoinGecko)
- Graceful rate limiting and request management using an async-first approach
- Robust data normalization and validation into a unified `CryptoData` format
- Error handling with exponential backoff for retries
- In-memory caching with configurable TTL for performance optimization
- Publishes new data onto the central FaultBus.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .fault_bus import FaultBus

logger = logging.getLogger(__name__)


# --- Enums and Data-Classes ---

class APISource(Enum):


    """Enumeration of supported API sources."""
COINMARKETCAP = "coinmarketcap"
COINGECKO = "coingecko"


@dataclass
class CryptoData:


    """A unified data structure for cryptocurrency information."""
symbol: str
name: str
price: float
market_cap: float
volume_24h: float
source: APISource
timestamp: datetime
price_change_percentage_24h: Optional[float] = None
rank: Optional[int] = None
circulating_supply: Optional[float] = None
total_supply: Optional[float] = None
max_supply: Optional[float] = None
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:


    """Represents an entry in the data cache."""
data: CryptoData
timestamp: datetime


# --- API Adapter Base Class ---

class ApiAdapter(ABC):


    """Abstract base class for API adapters."""

def __init__(self, session: aiohttp.ClientSession, api_key: Optional[str] = None):


    pass
    pass
        self.session = session
self.api_key = api_key

@property
@abstractmethod
def base_url(self) -> str:


    pass
    pass
        """The base URL for the API."""
        pass

@abstractmethod
async def get_crypto_data(self, symbols: List[str]) -> List[CryptoData]:
        """Fetch data for given cryptocurrency symbols."""
        pass

def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:


    pass
    pass
        """Safely retrieve a value from a nested dictionary."""
keys = key.split('.')
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k)
            else:
                return default
        return data if data is not None else default


# --- Concrete API Adapters ---

class CoinGeckoAdapter(ApiAdapter):


    """API adapter for CoinGecko."""

@property
def base_url(self) -> str:


    pass
    pass
        return "https://api.coingecko.com/api/v3"

async def get_crypto_data(self, symbols: List[str]) -> List[CryptoData]:
        # CoinGecko uses 'ids' which are often full names (e.g., 'bitcoin', 'ethereum')
        # A robust solution might need a symbol-to-id mapping service.
endpoint = "/coins/markets"
params = {
"vs_currency": "usd",
"ids": ",".join(s.lower() for s in symbols),
            "order": "market_cap_desc",
"per_page": len(symbols),
            "page": 1,
"sparkline": "false",
}
        try:
async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return self._parse_data(data)
        except aiohttp.ClientError as e:
logger.error(f"CoinGecko API error: {e}")
            return []

def _parse_data(self, data: List[Dict]) -> List[CryptoData]:


    pass
    pass
        parsed_data = []
        for item in data:
            try:
crypto = CryptoData(
                    symbol=self._safe_get(item, 'symbol', '').upper(),
                    name=self._safe_get(item, 'name', 'Unknown'),
                    price=float(self._safe_get(item, 'current_price', 0.0)),
                    market_cap=float(self._safe_get(item, 'market_cap', 0.0)),
                    volume_24h=float(self._safe_get(item, 'total_volume', 0.0)),
                    rank=int(self._safe_get(item, 'market_cap_rank', 0)),
                    price_change_percentage_24h=float(self._safe_get(item, 'price_change_percentage_24h', 0.0)),
                    circulating_supply=float(self._safe_get(item, 'circulating_supply', 0.0)),
                    total_supply=float(self._safe_get(item, 'total_supply', 0.0)),
                    max_supply=self._safe_get(item, 'max_supply'),
                    source=APISource.COINGECKO,
timestamp=datetime.now(),

parsed_data.append(crypto)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse CoinGecko item: {item}. Error: {e}")
        return parsed_data


class CoinMarketCapAdapter(ApiAdapter):


    """API adapter for CoinMarketCap."""

@property
def base_url(self) -> str:


    pass
    pass
        return "https://pro-api.coinmarketcap.com/v1"

async def get_crypto_data(self, symbols: List[str]) -> List[CryptoData]:
        if not self.api_key:
logger.warning("CoinMarketCap API key is not configured.")
            return []
endpoint = "/cryptocurrency/quotes/latest"
headers = {"X-CMC_PRO_API_KEY": self.api_key}
params = {"symbol": ",".join(s.upper() for s in symbols)}
        try:
async with self.session.get(f"{self.base_url}{endpoint}", headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return self._parse_data(data)
        except aiohttp.ClientError as e:
logger.error(f"CoinMarketCap API error: {e}")
            return []

def _parse_data(self, data: Dict) -> List[CryptoData]:


    pass
    pass
        parsed_data = []
        if 'data' not in data:
            return []
        for symbol_upper in data['data']:
            # API can return a list for a symbol, take the first
item_data = data['data'][symbol_upper]
            if isinstance(item_data, list):
                item_data = item_data[0]

            try:
quote = self._safe_get(item_data, 'quote.USD', {})
                crypto = CryptoData(
                    symbol=self._safe_get(item_data, 'symbol', '').upper(),
                    name=self._safe_get(item_data, 'name', 'Unknown'),
                    price=float(self._safe_get(quote, 'price', 0.0)),
                    market_cap=float(self._safe_get(quote, 'market_cap', 0.0)),
                    volume_24h=float(self._safe_get(quote, 'volume_24h', 0.0)),
                    rank=int(self._safe_get(item_data, 'cmc_rank', 0)),
                    price_change_percentage_24h=float(self._safe_get(quote, 'percent_change_24h', 0.0)),
                    circulating_supply=float(self._safe_get(item_data, 'circulating_supply', 0.0)),
                    total_supply=float(self._safe_get(item_data, 'total_supply', 0.0)),
                    max_supply=self._safe_get(item_data, 'max_supply'),
                    source=APISource.COINMARKETCAP,
timestamp=datetime.now(),

parsed_data.append(crypto)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse CoinMarketCap item for {symbol_upper}. Error: {e}")
        return parsed_data


# --- API Bridge Manager ---

class APIBridgeManager:


    """Manages access to multiple crypto data APIs asynchronously."""

def __init__(


        self,
fault_bus: Optional[FaultBus] = None,
config_path: str = "./config/api_config.json",
cache_ttl_seconds: int = 300
):
self.bus = fault_bus
self.config_path = config_path
self._api_keys: Dict[APISource, str] = {}
self._session: Optional[aiohttp.ClientSession] = None
self._adapters: Dict[APISource, ApiAdapter] = {}
self._cache: Dict[str, CacheEntry] = {}
self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._request_stats = defaultdict(lambda: defaultdict(int))
        self._is_initialized = False

async def initialize(self):
        """Asynchronously initializes the session and adapters."""
        if self._is_initialized:
return
self._session = aiohttp.ClientSession()
        self._load_configuration()
        self._initialize_adapters()
        self._is_initialized = True
logger.info("APIBridgeManager initialized")

def _load_configuration(self):


    pass
    pass
        """Loads API configuration from a JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                api_keys = config.get("api_keys", {})
                self._api_keys = {
APISource(source): key for source, key in api_keys.items() if key
                }
logger.info(f"Loaded configuration for {len(self._api_keys)} API sources.")
            else:
logger.warning(f"Configuration file not found at {self.config_path}.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading configuration: {e}.")

def _initialize_adapters(self):


    pass
    pass
        """Initializes API adapters based on loaded configuration."""
        if not self._session:
return
adapter_map = {
APISource.COINMARKETCAP: CoinMarketCapAdapter,
APISource.COINGECKO: CoinGeckoAdapter,
}
        for source, adapter_class in adapter_map.items():
            api_key = self._api_keys.get(source)
            if source == APISource.COINGECKO or api_key:
self._adapters[source] = adapter_class(self._session, api_key)
        logger.info(f"Initialized {len(self._adapters)} API adapters.")

async def close(self):
        """Closes the underlying aiohttp session."""
        if self._session and not self._session.closed:
await self._session.close()
            logger.info("APIBridgeManager session closed.")

async def get_crypto_data(
        self,
symbols: List[str],
sources: Optional[List[APISource]] = None,
use_cache: bool = True,
) -> Dict[str, CryptoData]:
"""
Retrieves cryptocurrency data from specified sources, using a cache
and a fallback strategy.

Args:
symbols: List of cryptocurrency symbols (e.g., ["BTC", "ETH"]).
            sources: List of APISources to query. If None, uses all configured.
use_cache: Whether to use the internal cache.

Returns:
A dictionary mapping symbol to its CryptoData.
"""
        if not self._is_initialized:
await self.initialize()

combined_results: Dict[str, CryptoData] = {}
symbols_to_fetch_set = set(s.upper() for s in symbols)

        if use_cache:
cached_data, symbols_to_fetch_set = self._check_cache(list(symbols_to_fetch_set))
            combined_results.update(cached_data)

        if not symbols_to_fetch_set:
            return combined_results

adapters_to_use = [self._adapters[s] for s in (sources or self._adapters.keys()) if s in self._adapters]

fetched_data_this_run: List[CryptoData] = []
tasks = [self._fetch_with_retry(adapter, list(symbols_to_fetch_set)) for adapter in adapters_to_use]
        results = await asyncio.gather(*tasks)

        for result_list in results:
fetched_data_this_run.extend(result_list)

        for item in fetched_data_this_run:
            if item.symbol not in combined_results:
combined_results[item.symbol] = item
self._cache[item.symbol] = CacheEntry(data=item, timestamp=datetime.now())

        if self.bus and fetched_data_this_run:
await self._publish_to_bus(fetched_data_this_run)

        return combined_results

async def _publish_to_bus(self, data: List[CryptoData]):
        """Publishes a list of crypto data to the fault bus."""
publish_tasks = [
self.bus.publish(
                "new_market_price",
price=item.price,
timestamp=item.timestamp.timestamp(),
                symbol=item.symbol,
source=item.source.value
) for item in data
]
        if publish_tasks:
await asyncio.gather(*publish_tasks)
            logger.debug(f"Published {len(publish_tasks)} price updates to the FaultBus.")

def _check_cache(self, symbols: List[str]) -> Tuple[Dict[str, CryptoData], set]:


    pass
    pass
        """Checks cache for fresh data, returns it and a set of symbols that still need fetching."""
fresh_data: Dict[str, CryptoData] = {}
symbols_to_fetch = set(symbols)
        now = datetime.now()

        for symbol in symbols:
            if symbol in self._cache:
entry = self._cache[symbol]
                if (now - entry.timestamp) < self._cache_ttl:
                    fresh_data[symbol] = entry.data
symbols_to_fetch.remove(symbol)

        if fresh_data:
logger.debug(f"Cache hit for symbols: {list(fresh_data.keys())}")

        return fresh_data, symbols_to_fetch

async def _fetch_with_retry(
        self, adapter: ApiAdapter, symbols: List[str], retries: int = 3, delay: float = 1.0
) -> List[CryptoData]:
"""Fetches data from an adapter with exponential backoff."""
self._request_stats[adapter.__class__.__name__]['attempts'] += 1
        for i in range(retries):
            try:
data = await adapter.get_crypto_data(symbols)
                self._request_stats[adapter.__class__.__name__]['successes'] += 1
                return data
            except aiohttp.ClientError as e:
logger.warning(
                    f"Attempt {i+1}/{retries} failed for {adapter.__class__.__name__}: {e}. Retrying in {delay}s..."

                if i < retries - 1:
await asyncio.sleep(delay)
                    delay *= 2

self._request_stats[adapter.__class__.__name__]['failures'] += 1
        return []

def get_api_statistics(self) -> Dict[str, Any]:


    pass
    pass
        """Returns statistics about the API usage."""
        return {
"cache_size": len(self._cache),
            "requests": dict(self._request_stats)
        }


async def main():
    """Demonstrates the functionality of the APIBridgeManager."""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bus = FaultBus()

async def price_listener(price: float, symbol: str, **kwargs):
        safe_print(f"[PriceListener] Received new price via FaultBus: {symbol} @ ${price:.2f} from {kwargs.get('source')}")

bus.subscribe("new_market_price", price_listener)

    # Pass the bus to the manager
manager = APIBridgeManager(fault_bus=bus)

    try:
await manager.initialize()
        # For CoinGecko, use IDs. For CMC, use symbols.
        # Example using CoinGecko IDs:
symbols_to_fetch = ["bitcoin", "ethereum"]

safe_print(f"\n--- Fetching data for {', '.join(symbols_to_fetch)} ---")
        data = await manager.get_crypto_data(symbols_to_fetch)

        if data:
            for symbol, crypto_data in data.items():
                safe_print(
                    f"  -> Fetched {crypto_data.name} ({symbol}): ${crypto_data.price:.2f} "
                    f"(Source: {crypto_data.source.value})"

        else:
safe_print("  -> Failed to fetch data. Check API keys and network.")

    finally:
await manager.close()


if __name__ == "__main__":
    pass
    pass
asyncio.run(main())
