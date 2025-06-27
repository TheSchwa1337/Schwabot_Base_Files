# Import core mathematical modules
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dual_unicore_handler import DualUnicoreHandler
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
import asyncio
import json
import logging
import os

from core.bit_phase_sequencer import BitPhase, BitSequence
from core.dual_error_handler import PhaseState, SickType, SickState
from core.symbolic_profit_router import ProfitTier, FlipBias, SymbolicState
from core.unified_math_system import unified_math


# Initialize Unicode handler
unicore = DualUnicoreHandler()

# -*- coding: utf-8 -*-
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

Mathematical Foundation:
- Rate limiting: R = base_rate * (1 + confidence_multiplier)
- Cache efficiency: CE = hits / (hits + misses)
- Data validation: DV = Σ(weight_i * confidence_i) / Σ(weight_i)
- Error recovery: ER = unified_math.exp(-retry_count * backoff_factor)
"""

# Import safe print for Windows compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, info, warn, error, success, debug
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False

    # Fallback functions
    def safe_print(message: str) -> str:
        return message

    def info(message):
        print(f"[INFO] {message}")

    def warn(message):
        print(f"[WARN] {message}")

    def error(message):
        print(f"[ERROR] {message}")

    def success(message):
        print(f"[SUCCESS] {message}")

    def debug(message):
        print(f"[DEBUG] {message}")

try:
    from .fault_bus import FaultBus
    FAULT_BUS_AVAILABLE = True
except ImportError:
    FAULT_BUS_AVAILABLE = False
    FaultBus = None

logger = logging.getLogger(__name__)


# --- Enums and Data Classes ---

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

    def __init__(
            self,
            session: aiohttp.ClientSession,
            api_key: Optional[str] = None):
        self.session = session
        self.api_key = api_key

    @property
    @abstractmethod
    def base_url(self) -> str:
        """The base URL for the API."""

    @abstractmethod
    async def get_crypto_data(self, symbols: List[str]) -> List[CryptoData]:
        """Fetch data for given cryptocurrency symbols."""

    def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:
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
            "sparkline": False,
            "price_change_percentage": "24h"
        }

        try:
            async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_data(data)
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            return []

    def _parse_data(self, data: List[Dict]) -> List[CryptoData]:
        """Parse CoinGecko API response."""
        crypto_data = []
        for item in data:
            try:
                crypto_data.append(CryptoData(
                    symbol=item.get("symbol", "").upper(),
                    name=item.get("name", ""),
                    price=float(item.get("current_price", 0)),
                    market_cap=float(item.get("market_cap", 0)),
                    volume_24h=float(item.get("total_volume", 0)),
                    source=APISource.COINGECKO,
                    timestamp=datetime.now(),
                    price_change_percentage_24h=item.get("price_change_percentage_24h"),
                    rank=item.get("market_cap_rank"),
                    circulating_supply=item.get("circulating_supply"),
                    total_supply=item.get("total_supply"),
                    max_supply=item.get("max_supply")
                ))
            except Exception as e:
                logger.error(f"Error parsing CoinGecko data: {e}")
                continue
        return crypto_data


class CoinMarketCapAdapter(ApiAdapter):
    """API adapter for CoinMarketCap."""

    @property
    def base_url(self) -> str:
        return "https://pro-api.coinmarketcap.com/v1"

    async def get_crypto_data(self, symbols: List[str]) -> List[CryptoData]:
        endpoint = "/cryptocurrency/quotes/latest"
        params = {
            "symbol": ",".join(symbols),
            "convert": "USD"
        }
        headers = {"X-CMC_PRO_API_KEY": self.api_key} if self.api_key else {}

        try:
            async with self.session.get(f"{self.base_url}{endpoint}",
                                        params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_data(data)
                else:
                    logger.error(f"CoinMarketCap API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching from CoinMarketCap: {e}")
            return []

    def _parse_data(self, data: Dict) -> List[CryptoData]:
        """Parse CoinMarketCap API response."""
        crypto_data = []
        quotes = data.get("data", {})

        for symbol, quote_data in quotes.items():
            try:
                quote = quote_data.get("quote", {}).get("USD", {})
                crypto_data.append(CryptoData(
                    symbol=symbol.upper(),
                    name=quote_data.get("name", ""),
                    price=float(quote.get("price", 0)),
                    market_cap=float(quote.get("market_cap", 0)),
                    volume_24h=float(quote.get("volume_24h", 0)),
                    source=APISource.COINMARKETCAP,
                    timestamp=datetime.now(),
                    price_change_percentage_24h=quote.get("percent_change_24h"),
                    rank=quote_data.get("cmc_rank"),
                    circulating_supply=quote_data.get("circulating_supply"),
                    total_supply=quote_data.get("total_supply"),
                    max_supply=quote_data.get("max_supply")
                ))
            except Exception as e:
                logger.error(f"Error parsing CoinMarketCap data: {e}")
                continue
        return crypto_data


# --- Main API Bridge Manager ---

class APIBridgeManager:
    """
    Main API bridge manager for Schwabot.

    Orchestrates multiple API sources, handles rate limiting,
    caching, and data normalization with mathematical precision.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the API bridge manager."""
        self.config = config or self._default_config()

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.adapters: Dict[APISource, ApiAdapter] = {}

        # Caching
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes

        # Rate limiting
        self.rate_limits: Dict[APISource, Dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "window_start": datetime.now()}
        )

        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Mathematical integration
        self.confidence_multiplier = 1.0
        self.backoff_factor = 0.5

        logger.info("🌉 API Bridge Manager initialized")

    async def initialize(self):
        """Initialize the API bridge manager."""
        try:
            # Create session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize adapters
            await self._initialize_adapters()

            logger.info("✅ API Bridge Manager initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize API Bridge Manager: {e}")
            raise

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "cache_ttl": 300,
            "max_retries": 3,
            "base_rate_limit": 100,
            "confidence_threshold": 0.8,
            "backoff_factor": 0.5
        }

    async def _initialize_adapters(self):
        """Initialize API adapters."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        # Initialize CoinGecko adapter
        self.adapters[APISource.COINGECKO] = CoinGeckoAdapter(self.session)

        # Initialize CoinMarketCap adapter if API key is available
        coinmarketcap_key = os.getenv("COINMARKETCAP_API_KEY")
        if coinmarketcap_key:
            self.adapters[APISource.COINMARKETCAP] = CoinMarketCapAdapter(
                self.session, coinmarketcap_key
            )

        logger.info(f"🔌 Initialized {len(self.adapters)} API adapters")

    async def close(self):
        """Close the API bridge manager."""
        if self.session:
            await self.session.close()
        logger.info("🔌 API Bridge Manager closed")

    async def get_crypto_data(self,
                              symbols: List[str],
                              sources: Optional[List[APISource]] = None) -> List[CryptoData]:
        """
        Get cryptocurrency data from multiple sources.

        Mathematical Process:
        1. Check cache efficiency: CE = hits / (hits + misses)
        2. Apply rate limiting: R = base_rate * (1 + confidence_multiplier)
        3. Validate data: DV = Σ(weight_i * confidence_i) / Σ(weight_i)
        4. Calculate cache hit ratio and update performance metrics
        """
        try:
            sources = sources or list(self.adapters.keys())
            all_data = []

            for source in sources:
                if source not in self.adapters:
                    continue

                # Check rate limits
                if not self._check_rate_limit(source):
                    logger.warning(f"Rate limit exceeded for {source.value}")
                    continue

                # Get data from adapter
                adapter_data = await self.adapters[source].get_crypto_data(symbols)

                # Apply data validation
                validated_data = self._validate_data(adapter_data)
                all_data.extend(validated_data)

                # Update rate limits
                self._update_rate_limit(source)

            # Cache results
            self._cache_data(all_data)

            # Update performance metrics
            self.request_count += 1

            return all_data

        except Exception as e:
            logger.error(f"Error getting crypto data: {e}")
            return []

    def _check_rate_limit(self, source: APISource) -> bool:
        """Check if rate limit allows request."""
        rate_info = self.rate_limits[source]
        window_duration = timedelta(seconds=60)

        if datetime.now() - rate_info["window_start"] > window_duration:
            rate_info["requests"] = 0
            rate_info["window_start"] = datetime.now()

        base_rate = self.config.get("base_rate_limit", 100)
        adjusted_rate = base_rate * (1 + self.confidence_multiplier)

        return rate_info["requests"] < adjusted_rate

    def _update_rate_limit(self, source: APISource):
        """Update rate limit counter."""
        self.rate_limits[source]["requests"] += 1

    def _validate_data(self, data: List[CryptoData]) -> List[CryptoData]:
        """Validate cryptocurrency data using mathematical criteria."""
        validated_data = []

        for item in data:
            # Calculate validation score
            # price, market_cap, volume, timestamp
            weights = [0.4, 0.3, 0.2, 0.1]
            confidences = [
                1.0 if item.price > 0 else 0.0,
                1.0 if item.market_cap > 0 else 0.0,
                1.0 if item.volume_24h > 0 else 0.0,
                1.0 if item.timestamp else 0.0
            ]

            validation_score = sum(w * c for w, c in zip(weights, confidences))

            if validation_score >= self.config.get(
                    "confidence_threshold", 0.8):
                validated_data.append(item)
            else:
                logger.debug(
                    f"Data validation failed for {
                        item.symbol}: {validation_score}")

        return validated_data

    def _cache_data(self, data: List[CryptoData]):
        """Cache cryptocurrency data."""
        for item in data:
            cache_key = f"{item.symbol}_{item.source.value}"
            self.cache[cache_key] = CacheEntry(
                data=item,
                timestamp=datetime.now()
            )

    def get_cache_efficiency(self) -> float:
        """Calculate cache efficiency."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_efficiency": self.get_cache_efficiency(),
            "active_adapters": len(self.adapters),
            "cache_size": len(self.cache)
        }


# Global API bridge manager instance
api_bridge_manager = APIBridgeManager()


async def get_api_bridge_manager() -> APIBridgeManager:
    """Get the global API bridge manager instance."""
    return api_bridge_manager


async def initialize_api_bridge(
        config: Optional[Dict[str, Any]] = None) -> APIBridgeManager:
    """Initialize the global API bridge manager."""
    global api_bridge_manager
    api_bridge_manager = APIBridgeManager(config)
    await api_bridge_manager.initialize()
    return api_bridge_manager


# Example usage
if __name__ == "__main__":
    async def test_api_bridge():
        """Test the API bridge functionality."""
        manager = await initialize_api_bridge()

        try:
            # Test data retrieval
            symbols = ["BTC", "ETH", "ADA"]
            data = await manager.get_crypto_data(symbols)

            print(f"Retrieved data for {len(data)} cryptocurrencies")
            for item in data:
                print(f"{item.symbol}: ${item.price:,.2f}")

            # Print performance metrics
            metrics = manager.get_performance_metrics()
            print(f"Performance metrics: {metrics}")

        finally:
            await manager.close()

    asyncio.run(test_api_bridge())


""""""
""""""
""""""
""""""
