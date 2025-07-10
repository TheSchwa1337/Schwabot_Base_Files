import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

# Add CCXT import at the top
import ccxt.async_support as ccxt
from secure_config_manager import get_secure_api_key

from core.secure_api_coordinator import APIProvider, SecureAPICoordinator
from utils.secure_config_manager import get_secure_api_key

#!/usr/bin/env python3


"""



Schwabot Price Bridge - Secure Price Data Integration



====================================================







Integrates with Schwabot's mathematical framework and secure API coordinator'



to provide reliable price data from multiple sources with fallback mechanisms.







Primary: CoinMarketCap API



Fallback: CoinGecko API



Emergency: CCXT Exchange APIs



"""


# Import Schwabot's secure systems'


try:
    pass


except ImportError:

    # Fallback for direct execution


logger = logging.getLogger(__name__)


@dataclass
    class PriceData:
    """Structured price data with Schwabot's mathematical framework integration."""'

    symbol: str

    price: float

    currency: str

    timestamp: int

    source: str

    volume_24h: Optional[float] = None

    market_cap: Optional[float] = None

    price_change_24h: Optional[float] = None

    price_change_percent_24h: Optional[float] = None

    high_24h: Optional[float] = None

    low_24h: Optional[float] = None

    circulating_supply: Optional[float] = None

    total_supply: Optional[float] = None

    max_supply: Optional[float] = None

    # Schwabot mathematical framework fields

    price_hash: Optional[str] = None

    market_state_hash: Optional[str] = None

    drift_field_value: Optional[float] = None

    entropy_level: Optional[float] = None

    quantum_state: Optional[str] = None

    def __post_init__(self):
        """Generate mathematical framework hashes after initialization."""

        if not self.price_hash:

            self.price_hash = self._generate_price_hash()

        if not self.market_state_hash:

            self.market_state_hash = self._generate_market_state_hash()

    def _generate_price_hash(self) -> str:
        """Generate SHA-256 hash of price data using Schwabot's framework."""'

        price_data = f"{self.symbol}:{self.price}:{self.currency}:{self.timestamp}"

        return hashlib.sha256(price_data.encode("utf-8")).hexdigest()

    def _generate_market_state_hash(self) -> str:
        """Generate comprehensive market state hash."""

        market_data = {}
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume_24h or 0,
            "market_cap": self.market_cap or 0,
            "timestamp": self.timestamp,
            "source": self.source,
        }

        market_json = json.dumps(market_data, sort_keys=True)

        return hashlib.sha256(market_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""

        return {}
            "symbol": self.symbol,
            "price": self.price,
            "currency": self.currency,
            "timestamp": self.timestamp,
            "source": self.source,
            "volume_24h": self.volume_24h,
            "market_cap": self.market_cap,
            "price_change_24h": self.price_change_24h,
            "price_change_percent_24h": self.price_change_percent_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "circulating_supply": self.circulating_supply,
            "total_supply": self.total_supply,
            "max_supply": self.max_supply,
            "price_hash": self.price_hash,
            "market_state_hash": self.market_state_hash,
            "drift_field_value": self.drift_field_value,
            "entropy_level": self.entropy_level,
            "quantum_state": self.quantum_state,
        }


class SchwabotPriceBridge:
    """



    Secure price bridge integrating with Schwabot's mathematical framework.'







    Features:



    - CoinMarketCap as primary source



    - CoinGecko as fallback



    - CCXT exchanges as emergency fallback



    - Secure API key management



    - Rate limiting and caching



    - Mathematical framework integration



    - Error recovery and logging



    """

    def __init__(self, use_secure_coordinator: bool = True):
        """Initialize the price bridge."""

        self.use_secure_coordinator = use_secure_coordinator

        # Initialize secure API coordinator if available

        if use_secure_coordinator:

            try:

                # Initialize with proper config to avoid storage_path issues

                config = {}
                    "storage_path": None,  # Use default path
                    "request_timeout": 30,
                    "connection_timeout": 10,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "enable_rate_limiting": True,
                    "enable_request_logging": True,
                    "auto_key_rotation": False,
                    "key_rotation_days": 90,
                }

                self.api_coordinator = SecureAPICoordinator(config)

                logger.info(" Using Schwabot Secure API Coordinator")

            except Exception as e:

                logger.warning(f"  Secure API Coordinator unavailable: {e}")

                self.api_coordinator = None

        else:

            self.api_coordinator = None

        # Price cache for rate limiting

        self.price_cache: Dict[str, Dict[str, Any]] = {}

        self.cache_duration = 30  # seconds

        # API endpoints

        self.endpoints = {}
            "coinmarketcap": {}
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "cryptocurrency_quotes": "/cryptocurrency/quotes/latest",
                "cryptocurrency_listings": "/cryptocurrency/listings/latest",
            },
            "coingecko": {}
                "base_url": "https://api.coingecko.com/api/v3",
                "simple_price": "/simple/price",
                "coins_markets": "/coins/markets",
            },
        }

        # Supported symbols

        self.supported_symbols = []
            "BTC",
            "ETH",
            "ADA",
            "DOT",
            "LINK",
            "LTC",
            "BCH",
            "XRP",
        ]

        logger.info(" Schwabot Price Bridge initialized")

    def _get_api_key(): -> Optional[str]:
        """Get API key from secure config manager."""

        try:

            return get_secure_api_key(service)

        except Exception as e:

            logger.error(f" Failed to get API key for {service}: {e}")

            return None

    def _is_cache_valid(): -> bool:
        """Check if cached price data is still valid."""

        if symbol not in self.price_cache:

            return False

        cache_time = self.price_cache[symbol].get("timestamp", 0)

        current_time = int(time.time())

        return (current_time - cache_time) < self.cache_duration

    def _update_cache(self, symbol: str, price_data: PriceData):
        """Update price cache."""

        self.price_cache[symbol] = {}
            "data": price_data.to_dict(),
            "timestamp": int(time.time()),
        }

    async def get_coinmarketcap_price(): -> Optional[PriceData]:
        """Get price from CoinMarketCap API."""

        try:

            api_key = self._get_api_key("COINMARKETCAP_API")

            if not api_key:

                logger.warning("  CoinMarketCap API key not configured")

                return None

            # Use secure API coordinator if available

            if self.api_coordinator:

                response = self.api_coordinator.make_request()
                    APIProvider.CUSTOM, f"{"}
                        self.endpoints['coinmarketcap']['base_url']}{
                        self.endpoints['coinmarketcap']['cryptocurrency_quotes']}", params={"
                        "symbol": symbol, "convert": "USD"}, headers={
                        "X-CMC_PRO_API_KEY": api_key}, )

            else:

                # Direct request with rate limiting

                url = f"{"}
                    self.endpoints['coinmarketcap']['base_url']}{
                    self.endpoints['coinmarketcap']['cryptocurrency_quotes']}"

                headers = {"X-CMC_PRO_API_KEY": api_key}

                params = {"symbol": symbol, "convert": "USD"}

                async with aiohttp.ClientSession() as session:

                    async with session.get(url, headers=headers, params=params) as response:

                        if response.status != 200:

                            logger.error()
                                f" CoinMarketCap API error: {"}
                                    response.status}")"

                            return None

                        response = await response.json()

            if not response or "data" not in response:

                logger.error(" Invalid response from CoinMarketCap")

                return None

            data = response["data"][symbol]

            quote = data["quote"]["USD"]

            price_data = PriceData()
                symbol=symbol,
                price=float(quote["price"]),
                currency="USD",
                timestamp=int(time.time()),
                source="coinmarketcap",
                volume_24h=float(quote.get("volume_24h", 0)),
                market_cap=float(quote.get("market_cap", 0)),
                price_change_24h=float(quote.get("volume_change_24h", 0)),
                price_change_percent_24h=float(quote.get("percent_change_24h", 0)),
                high_24h=float(quote.get("high_24h", 0)),
                low_24h=float(quote.get("low_24h", 0)),
                circulating_supply=float(data.get("circulating_supply", 0)),
                total_supply=float(data.get("total_supply", 0)),
                max_supply=float(data.get("max_supply", 0)),
            )

            logger.info()
                f" CoinMarketCap price for {symbol}: ${"}
                    price_data.price:,.2f}")"

            return price_data

        except Exception as e:

            logger.error(f" CoinMarketCap API error: {e}")

            return None

    async def get_coingecko_price(): -> Optional[PriceData]:
        """Get price from CoinGecko API (fallback)."""

        try:

            url = f"{"}
                self.endpoints['coingecko']['base_url']}{
                self.endpoints['coingecko']['simple_price']}"

            params = {}
                "ids": symbol,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            }

            async with aiohttp.ClientSession() as session:

                async with session.get(url, params=params) as response:

                    if response.status != 200:

                        logger.error()
                            f" CoinGecko API error: {"}
                                response.status}")"

                        return None

                    data = await response.json()

            if not data or symbol not in data:

                logger.error(" Invalid response from CoinGecko")

                return None

            coin_data = data[symbol]

            price_data = PriceData()
                symbol=symbol.upper(), price=float()
                    coin_data["usd"]), currency="USD", timestamp=int(
                    time.time()), source="coingecko", price_change_percent_24h=float(
                    coin_data.get()
                        "usd_24h_change", 0)), market_cap=float(
                        coin_data.get()
                            "usd_market_cap", 0)), )

            logger.info()
                f" CoinGecko price for {symbol}: ${"}
                    price_data.price:,.2f}")"

            return price_data

        except Exception as e:

            logger.error(f" CoinGecko API error: {e}")

            return None

    async def get_ccxt_price(self, symbol: str, exchange_name: str = 'coinbase') -> Optional[PriceData]:
        """Get price from CCXT exchange (emergency, fallback)."""
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({)}
                'enableRateLimit': True,
                'options': {}
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })

            # Fetch ticker
            ticker = await exchange.fetch_ticker(f"{symbol}/USD")
            await exchange.close()

            if ticker and ticker['last']:
                price_data = PriceData()
                    symbol=symbol,
                    price=float(ticker['last']),
                    currency="USD",
                    timestamp=int(ticker['timestamp'] / 1000) if ticker['timestamp'] else int(time.time()),
                    source=f"ccxt_{exchange_name}",
                    volume_24h=float(ticker.get('baseVolume', 0)),
                    high_24h=float(ticker.get('high', 0)),
                    low_24h=float(ticker.get('low', 0)),
                )

                logger.info(f"CCXT {exchange_name} price for {symbol}: ${price_data.price:,.2f}")
                return price_data

        except Exception as e:
            logger.error(f"CCXT {exchange_name} API error: {e}")

        return None

    async def get_price(self, symbol: str, use_cache: bool = True) -> Optional[PriceData]:
        """
        Get price data with comprehensive fallback mechanism.

        Priority:
        1. CoinMarketCap (if API key, configured)
        2. CoinGecko (free, fallback)
        3. CCXT Exchange APIs (emergency, fallback)
        4. Cached data (if, valid)
        """
        # Check cache first
        if use_cache and self._is_cache_valid(symbol):
            cached_data = self.price_cache[symbol]["data"]
            logger.info(f"Using cached price for {symbol}")
            return PriceData(**cached_data)

        # Try CoinMarketCap first
        price_data = await self.get_coinmarketcap_price(symbol)
        if price_data:
            self._update_cache(symbol, price_data)
            return price_data

        # Fallback to CoinGecko
        symbol_mapping = {}
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "ADA": "cardano",
            "SOL": "solana",
            "XRP": "ripple",
            "DOT": "polkadot",
            "DOGE": "dogecoin",
            "AVAX": "avalanche-2",
            "LINK": "chainlink"
        }
        coingecko_symbol = symbol_mapping.get(symbol, symbol.lower())
        price_data = await self.get_coingecko_price(coingecko_symbol)
        if price_data:
            self._update_cache(symbol, price_data)
            return price_data

        # Emergency fallback to CCXT exchanges
        ccxt_exchanges = ['coinbase', 'binance', 'bybit']
        for exchange_name in ccxt_exchanges:
            price_data = await self.get_ccxt_price(symbol, exchange_name)
            if price_data:
                self._update_cache(symbol, price_data)
                return price_data

        logger.error(f"Failed to get price for {symbol} from all sources")
        return None

    async def get_multiple_prices(): -> Dict[str, PriceData]:
        """Get prices for multiple symbols."""

        results = {}

        tasks = []

        for symbol in symbols:

            task = asyncio.create_task(self.get_price(symbol))

            tasks.append((symbol, task))

        for symbol, task in tasks:

            try:

                price_data = await task

                if price_data:

                    results[symbol] = price_data

            except Exception as e:

                logger.error(f" Error getting price for {symbol}: {e}")

        return results

    def get_cache_status(): -> Dict[str, Any]:
        """Get cache status and statistics."""

        cache_info = {}
            "total_cached_symbols": len(self.price_cache),
            "cache_duration_seconds": self.cache_duration,
            "cached_symbols": list(self.price_cache.keys()),
            "cache_validity": {},
        }

        for symbol in self.price_cache:

            cache_info["cache_validity"][symbol] = self._is_cache_valid(symbol)

        return cache_info

    def clear_cache(self):
        """Clear the price cache."""

        self.price_cache.clear()

        logger.info("  Price cache cleared")


# Global instance for easy access


price_bridge = SchwabotPriceBridge()


async def get_secure_price(): -> Optional[PriceData]:
    """Global function to get secure price data."""

    return await price_bridge.get_price(symbol)


async def get_multiple_secure_prices(): -> Dict[str, PriceData]:
    """Global function to get multiple secure prices."""

    return await price_bridge.get_multiple_prices(symbols)


if __name__ == "__main__":

    """Test the price bridge functionality."""

    async def test_price_bridge():

        print(" Testing Schwabot Price Bridge")

        print("=" * 50)

        # Test single price

        print("\n Testing single price (BTC):")

        btc_price = await get_secure_price("BTC")

        if btc_price:

            print(f" BTC Price: ${btc_price.price:,.2f}")

            print(f"   Source: {btc_price.source}")

            print(f"   Hash: {btc_price.price_hash[:16]}...")

        else:

            print(" Failed to get BTC price")

        # Test multiple prices

        print("\n Testing multiple prices:")

        symbols = ["BTC", "ETH", "ADA"]

        prices = await get_multiple_secure_prices(symbols)

        for symbol, price_data in prices.items():

            print(f" {symbol}: ${price_data.price:,.2f} ({price_data.source})")

        # Test cache status

        print("\n Cache status:")

        cache_status = price_bridge.get_cache_status()

        print(f"   Cached symbols: {cache_status['cached_symbols']}")

        print(f"   Cache validity: {cache_status['cache_validity']}")

    # Run the test

    asyncio.run(test_price_bridge())
