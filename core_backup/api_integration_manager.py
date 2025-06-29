# -*- coding: utf-8 -*-
""""""
API Integration Manager - Unified API Coordination
=================================================

Comprehensive API integration manager for Coinbase, CoinMarketCap, and CCXT
with proper error handling, rate limiting, and Flake8 compliance.

Features:
- Coinbase API integration (CCXT-based)
- CoinMarketCap API for market data
- CoinGecko API as backup
- Unified error handling and retry logic
- Rate limiting and connection management
- Real-time data streaming
- Cross-platform compatibility
""""""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import external libraries
try:
    import aiohttp
    import ccxt
    import requests

    EXTERNAL_LIBS_AVAILABLE = True
except ImportError:
    EXTERNAL_LIBS_AVAILABLE = False
    logging.warning("External libraries not available. Install with: pip install ccxt aiohttp requests")

from core.type_defs import ExchangeType, MarketData, TradingPair
from utils.safe_print import debug, error, info, safe_print, success, warn

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API connections."""

    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    coinbase_passphrase: str = ""
    coinmarketcap_api_key: str = ""
    coingecko_api_key: str = ""
    sandbox_mode: bool = True
    rate_limit_multiplier: float = 1.0
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class ConnectionStatus:
    """Connection status for each API."""

    coinbase: bool = False
    coinmarketcap: bool = False
    coingecko: bool = False
    ccxt: bool = False
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimiter:
    """Rate limiter for API calls."""

    calls_per_minute: int = 60
    calls_per_second: int = 10
    last_call_time: float = field(default_factory=time.time)
    call_count: int = 0

    def wait_if_needed(self) -> None:
        """Wait if rate limit is exceeded."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time

        # Check per-second limit
        if time_since_last < 1.0 / self.calls_per_second:
            sleep_time = (1.0 / self.calls_per_second) - time_since_last
            time.sleep(sleep_time)

        # Check per-minute limit
        if self.call_count >= self.calls_per_minute:
            time.sleep(60)  # Wait one minute
            self.call_count = 0

        self.last_call_time = time.time()
        self.call_count += 1


class CoinbaseAPI:
    """Coinbase API integration using CCXT."""

    def __init__(self, config: APIConfig):
        """Initialize Coinbase API."""
        self.config = config
        self.exchange = None
        self.rate_limiter = RateLimiter(calls_per_minute=30, calls_per_second=3)
        self.connected = False

        if not EXTERNAL_LIBS_AVAILABLE:
            logger.error("CCXT not available for Coinbase API")
            return

        try:
            self.exchange = ccxt.coinbase()
                {}
                    "apiKey": config.coinbase_api_key,
                        "secret": config.coinbase_api_secret,
                            "password": config.coinbase_passphrase,
                            "sandbox": config.sandbox_mode,
                            "enableRateLimit": True,
                            "timeout": config.timeout_seconds * 1000,
                            "options": {"defaultType": "spot"},
}
            )
            self.connected = True
            logger.info("✅ Coinbase API initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Coinbase API: {e}")

    async def get_ticker(self, symbol: str) -> Optional[MarketData]:
        """Get ticker data from Coinbase."""
        if not self.connected or not self.exchange:
            return None

        try:
            self.rate_limiter.wait_if_needed()

            ticker = await asyncio.get_event_loop().run_in_executor(None, self.exchange.fetch_ticker, symbol)

            return MarketData()
                symbol=symbol,
                    price=Decimal(str(ticker["last"])),
                        volume=Decimal(str(ticker["baseVolume"])),
                        timestamp=datetime.fromtimestamp(ticker["timestamp"] / 1000),
                        bid=Decimal(str(ticker["bid"])) if ticker["bid"] else None,
                        ask=Decimal(str(ticker["ask"])) if ticker["ask"] else None,
                        high_24h=Decimal(str(ticker["high"])) if ticker["high"] else None,
                        low_24h=Decimal(str(ticker["low"])) if ticker["low"] else None,
                        change_24h=Decimal(str(ticker["change"])) if ticker["change"] else None,
                        )
        except Exception as e:
            logger.error(f"Error fetching Coinbase ticker for {symbol}: {e}")
            return None

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book from Coinbase."""
        if not self.connected or not self.exchange:
            return None

        try:
            self.rate_limiter.wait_if_needed()

            order_book = await asyncio.get_event_loop().run_in_executor()
                None, self.exchange.fetch_order_book, symbol, limit
            )

            return {}
                "bids": order_book["bids"],
                    "asks": order_book["asks"],
                        "timestamp": datetime.fromtimestamp(order_book["timestamp"] / 1000),
}
        except Exception as e:
            logger.error(f"Error fetching Coinbase order book for {symbol}: {e}")
            return None


class CoinMarketCapAPI:
    """CoinMarketCap API integration."""

    def __init__(self, config: APIConfig):
        """Initialize CoinMarketCap API."""
        self.config = config
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.rate_limiter = RateLimiter(calls_per_minute=30, calls_per_second=5)
        self.connected = bool(config.coinmarketcap_api_key)

        if self.connected:
            logger.info("✅ CoinMarketCap API initialized")
        else:
            logger.warning("❌ CoinMarketCap API key not provided")

    async def get_ticker(self, symbol: str) -> Optional[MarketData]:
        """Get ticker data from CoinMarketCap."""
        if not self.connected:
            return None

        try:
            self.rate_limiter.wait_if_needed()

            # Convert symbol format (e.g., "BTC/USDC" -> "BTC")
            base_symbol = symbol.split("/")[0]

            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            params = {"symbol": base_symbol, "convert": "USD"}
            headers = {"X-CMC_PRO_API_KEY": self.config.coinmarketcap_api_key, "Accept": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data.get("status", {}).get("error_code") == 0:
                            currency_data = data["data"].get(base_symbol)
                            if currency_data:
                                quote = currency_data["quote"].get("USD", {})

                                return MarketData()
                                    symbol=symbol,
                                        price=Decimal(str(quote.get("price", 0))),
                                            volume=Decimal(str(quote.get("volume_24h", 0))),
                                            timestamp=datetime.now(),
                                            change_24h=Decimal(str(quote.get("percent_change_24h", 0))),
                                            )

            return None
        except Exception as e:
            logger.error(f"Error fetching CoinMarketCap ticker for {symbol}: {e}")
            return None

    async def get_top_cryptocurrencies(self, limit: int = 10) -> List[MarketData]:
        """Get top cryptocurrencies from CoinMarketCap."""
        if not self.connected:
            return []

        try:
            self.rate_limiter.wait_if_needed()

            url = f"{self.base_url}/cryptocurrency/listings/latest"
            params = {"limit": limit, "convert": "USD"}
            headers = {"X-CMC_PRO_API_KEY": self.config.coinmarketcap_api_key, "Accept": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data.get("status", {}).get("error_code") == 0:
                            results = []
                            for currency in data["data"]:
                                quote = currency["quote"].get("USD", {})
                                results.append()
                                    MarketData(
                                        symbol=f"{currency['symbol']}/USD",
                                        price=Decimal(str(quote.get("price", 0))),
                                        volume=Decimal(str(quote.get("volume_24h", 0))),
                                        timestamp=datetime.now(),
                                        change_24h=Decimal(str(quote.get("percent_change_24h", 0))),
                                    )
                                                )
                                )
                            return results

            return []
        except Exception as e:
            logger.error(f"Error fetching top cryptocurrencies: {e}")
            return []


class CoinGeckoAPI:
    """CoinGecko API integration."""

    def __init__(self, config: APIConfig):
        """Initialize CoinGecko API."""
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
        self.rate_limiter = RateLimiter(calls_per_minute=50, calls_per_second=10)
        self.connected = True  # CoinGecko has a free tier

        logger.info("✅ CoinGecko API initialized")

    async def get_ticker(self, symbol: str) -> Optional[MarketData]:
        """Get ticker data from CoinGecko."""
        try:
            self.rate_limiter.wait_if_needed()

            # Convert symbol format (e.g., "BTC/USDC" -> "bitcoin")
            base_symbol = symbol.split("/")[0].lower()
            coin_id = self._get_coin_id(base_symbol)

            if not coin_id:
                return None

            url = f"{self.base_url}/simple/price"
            params = {"ids": coin_id, "vs_currencies": "usd", "include_24hr_vol": "true", "include_24hr_change": "true"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if coin_id in data:
                            coin_data = data[coin_id]
                            return MarketData()
                                symbol=symbol,
                                    price=Decimal(str(coin_data.get("usd", 0))),
                                        volume=Decimal(str(coin_data.get("usd_24h_vol", 0))),
                                        timestamp=datetime.now(),
                                        change_24h=Decimal(str(coin_data.get("usd_24h_change", 0))),
                                        )

            return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko ticker for {symbol}: {e}")
            return None

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol."""
        # Simplified mapping for common coins
        mapping = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "xrp": "ripple",
            "usdc": "usd-coin",
            "usdt": "tether",
            "sol": "solana",
            "matic": "polygon",
            "ada": "cardano",
            "dot": "polkadot",
            "link": "chainlink",
}
}
        return mapping.get(symbol.lower())


class APIIntegrationManager:
    """Unified API integration manager."""

    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize API integration manager."""
        self.config = config or APIConfig()
        self.connection_status = ConnectionStatus()
        self.data_cache: Dict[str, MarketData] = {}
        self.cache_expiry = 30  # seconds

        # Initialize APIs
        self.coinbase_api = CoinbaseAPI(self.config)
        self.coinmarketcap_api = CoinMarketCapAPI(self.config)
        self.coingecko_api = CoinGeckoAPI(self.config)

        # Update connection status
        self.connection_status.coinbase = self.coinbase_api.connected
        self.connection_status.coinmarketcap = self.coinmarketcap_api.connected
        self.connection_status.coingecko = self.coingecko_api.connected
        self.connection_status.ccxt = EXTERNAL_LIBS_AVAILABLE

        logger.info("🚀 API Integration Manager initialized")
        self._log_connection_status()

    def _log_connection_status(self) -> None:
        """Log connection status for all APIs."""
        status = self.connection_status
        safe_print("📡 API Connection Status:")
        safe_print(f"  Coinbase: {'✅' if status.coinbase else '❌'}")
        safe_print(f"  CoinMarketCap: {'✅' if status.coinmarketcap else '❌'}")
        safe_print(f"  CoinGecko: {'✅' if status.coingecko else '❌'}")
        safe_print(f"  CCXT: {'✅' if status.ccxt else '❌'}")

    async def get_market_data(self, symbol: str, preferred_source: str = "auto") -> Optional[MarketData]:
        """"""
        Get market data from preferred source or fallback.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDC")
            preferred_source: Preferred API source ("coinbase", "coinmarketcap", "coingecko", "auto")

        Returns:
            MarketData object or None if all sources fail
        """"""
        # Check cache first
        cache_key = f"{symbol}_{preferred_source}"
        if cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            if (datetime.now() - cached_data.timestamp).seconds < self.cache_expiry:
                return cached_data

        # Try preferred source first
        if preferred_source == "coinbase" or preferred_source == "auto":
            data = await self.coinbase_api.get_ticker(symbol)
            if data:
                self.data_cache[cache_key] = data
                return data

        if preferred_source == "coinmarketcap" or (preferred_source == "auto" and not data):
            data = await self.coinmarketcap_api.get_ticker(symbol)
            if data:
                self.data_cache[cache_key] = data
                return data

        if preferred_source == "coingecko" or (preferred_source == "auto" and not data):
            data = await self.coingecko_api.get_ticker(symbol)
            if data:
                self.data_cache[cache_key] = data
                return data

        return None

    async def get_multiple_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get market data for multiple symbols."""
        results = {}
        tasks = []

        for symbol in symbols:
            task = asyncio.create_task(self.get_market_data(symbol))
            tasks.append((symbol, task))

        for symbol, task in tasks:
            try:
                data = await task
                if data:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        return results

    async def get_top_cryptocurrencies(self, limit: int = 10) -> List[MarketData]:
        """Get top cryptocurrencies from available sources."""
        # Try CoinMarketCap first
        if self.connection_status.coinmarketcap:
            data = await self.coinmarketcap_api.get_top_cryptocurrencies(limit)
            if data:
                return data

        # Fallback to CoinGecko
        if self.connection_status.coingecko:
            # CoinGecko doesn't have a direct top cryptocurrencies endpoint in free tier'
            # We'll use a predefined list'
            common_symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "USDC/USD", "USDT/USD"]
            return await self.get_multiple_market_data(common_symbols[:limit])

        return []

    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self.connection_status

    async def test_connections(self) -> Dict[str, bool]:
        """Test all API connections."""
        results = {}

        # Test Coinbase
        try:
            test_data = await self.coinbase_api.get_ticker("BTC/USDC")
            results["coinbase"] = test_data is not None
        except Exception:
            results["coinbase"] = False

        # Test CoinMarketCap
        try:
            test_data = await self.coinmarketcap_api.get_ticker("BTC/USD")
            results["coinmarketcap"] = test_data is not None
        except Exception:
            results["coinmarketcap"] = False

        # Test CoinGecko
        try:
            test_data = await self.coingecko_api.get_ticker("BTC/USD")
            results["coingecko"] = test_data is not None
        except Exception:
            results["coingecko"] = False

        return results


# Example usage and testing
async def test_api_integration():
    """Test API integration functionality."""
    # Load configuration from environment variables
    config = APIConfig()
        coinbase_api_key=os.getenv("COINBASE_API_KEY", ""),
            coinbase_api_secret=os.getenv("COINBASE_API_SECRET", ""),
                coinbase_passphrase=os.getenv("COINBASE_PASSPHRASE", ""),
                coinmarketcap_api_key=os.getenv("COINMARKETCAP_API_KEY", ""),
                sandbox_mode=True,
                )

    # Initialize manager
    manager = APIIntegrationManager(config)

    # Test connections
    safe_print("🔍 Testing API connections...")
    connection_results = await manager.test_connections()

    for api, connected in connection_results.items():
        status = "✅ Connected" if connected else "❌ Failed"
        safe_print(f"  {api}: {status}")

    # Test market data retrieval
    safe_print("📊 Testing market data retrieval...")
    symbols = ["BTC/USDC", "ETH/USDC", "XRP/USDC"]

    for symbol in symbols:
        data = await manager.get_market_data(symbol)
        if data:
            safe_print(f"  {symbol}: ${data.price} (24h change: {data.change_24h}%)")
        else:
            safe_print(f"  {symbol}: No data available")

    # Test top cryptocurrencies
    safe_print("🏆 Testing top cryptocurrencies...")
    top_crypto = await manager.get_top_cryptocurrencies(5)
    for crypto in top_crypto:
        safe_print(f"  {crypto.symbol}: ${crypto.price}")


if __name__ == "__main__":
    asyncio.run(test_api_integration())
