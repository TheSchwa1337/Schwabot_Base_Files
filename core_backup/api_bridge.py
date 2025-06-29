# -*- coding: utf-8 -*-
""""""
Unified API Access Layer.

Provides unified access to price, volume, and order book data from
multiple sources with fallback mechanisms for the Schwabot trading system.

Supports:
- CoinGecko API
- CoinMarketCap API
- CCXT exchange interface (Coinbase)
- Fallback mechanisms for reliability
- Mathematical pipeline integration
""""""

import asyncio
import logging
import time
import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import ccxt

import numpy as np

from .exchange_apis.coingecko_api import CoinGeckoAPI
from .exchange_apis.coinmarketcap_api import CoinMarketCapAPI
from .trading_models.containers import ExchangeConfig
from .trading_models.enums import ExchangeType

if TYPE_CHECKING:
    from typing import Self, List

logger = logging.getLogger(__name__)


class APIBridge:
    """"""
    Unified API Bridge for accessing multiple data sources.

    Provides fallback mechanisms and unified interfaces for:
    - Order book data
    - Price data
    - Volume data
    - News and sentiment data
    - Mathematical pipeline integration
    """"""

    def __init__()
        self: "Self",
            enable_coingecko: bool = True,
                enable_coinmarketcap: bool = True,
                enable_ccxt: bool = True,
                coinmarketcap_api_key: Optional[str] = None,
                coinbase_api_key: Optional[str] = None,
                coinbase_api_secret: Optional[str] = None,
                sandbox: bool = True,
                request_timeout: float = 10.0,
                max_retries: int = 3,
                rate_limit_delay: float = 1.0,
                ) -> None:
        """Initialize the API bridge."""

        Args:
            enable_coingecko: Enable CoinGecko API
            enable_coinmarketcap: Enable CoinMarketCap API
            enable_ccxt: Enable CCXT interface
            coinmarketcap_api_key: CoinMarketCap API key
            coinbase_api_key: Coinbase API key
            coinbase_api_secret: Coinbase API secret
            sandbox: Use sandbox mode for exchanges
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit_delay: Delay between requests
        """"""
        self.enable_coingecko = enable_coingecko
        self.enable_coinmarketcap = enable_coinmarketcap
        self.enable_ccxt = enable_ccxt
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        # Initialize API clients
        self.coingecko_api: Optional[CoinGeckoAPI] = None
        self.coinmarketcap_api: Optional[CoinMarketCapAPI] = None
        self.ccxt_exchange: Optional[ccxt.Exchange] = None

        # Initialize CoinGecko API
        if enable_coingecko:
            try:
                # CoinGecko public API - no credentials needed
                api_key = ""  # noqa: S105
                api_secret = ""  # noqa: S105
                config = ExchangeConfig()
                    exchange_type=ExchangeType.CUSTOM,
                        api_key=api_key,
                            api_secret=api_secret,
                            timeout=int(request_timeout),
                            )
                self.coingecko_api = CoinGeckoAPI(config)
                logger.info("CoinGecko API initialized")
            except Exception as e:
                logger.warning(f"CoinGecko initialization failed: {e}")

        # Initialize CoinMarketCap API
        if enable_coinmarketcap and coinmarketcap_api_key:
            try:
                # CMC doesn't use api_secret'
                api_secret = ""  # noqa: S105
                config = ExchangeConfig()
                    exchange_type=ExchangeType.CUSTOM,
                        api_key=coinmarketcap_api_key,
                            api_secret=api_secret,
                            timeout=int(request_timeout),
                            )
                self.coinmarketcap_api = CoinMarketCapAPI(config)
                logger.info("CoinMarketCap API initialized")
            except Exception as e:
                logger.warning(f"CoinMarketCap initialization failed: {e}")

        # Initialize CCXT exchange
        if enable_ccxt and coinbase_api_key and coinbase_api_secret:
            try:
                self.ccxt_exchange = ccxt.coinbase({)}
                    'apiKey': coinbase_api_key,
                        'secret': coinbase_api_secret,
                            'sandbox': sandbox,
                            'enableRateLimit': True,
                            'timeout': request_timeout * 1000,
                            })
                logger.info("CCXT Coinbase exchange initialized")
            except Exception as e:
                logger.warning(f"CCXT initialization failed: {e}")

        # Performance tracking
        self.api_stats = {}
            "total_requests": 0,
                "successful_requests": 0,
                    "failed_requests": 0,
                    "coingecko_requests": 0,
                    "coinmarketcap_requests": 0,
                    "ccxt_requests": 0,
                    "fallback_requests": 0,
                    "avg_response_time": 0.0,
}
        # Rate limiting
        self.last_request_time = 0.0

        logger.info()
            f"APIBridge initialized: "
            f"coingecko={enable_coingecko}, "
            f"coinmarketcap={enable_coinmarketcap}, "
            f"ccxt={enable_ccxt}, "
            f"sandbox={sandbox}"
        )

    async def fetch_order_book()
        self: "Self",
            symbol: str = "BTC/USDC",
                exchange: str = "coinbase",
                limit: int = 20
    ) -> Dict[str, List]:
        """"""
        Fetch order book data with fallback mechanisms.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDC")
            exchange: Exchange name
            limit: Number of order book levels

        Returns:
            Order book dictionary with 'bids' and 'asks'
        """"""
        start_time = time.time()

        try:
            # Respect rate limits
            await self._rate_limit_delay()

            # Try CCXT first (most reliable for order books)
            if self.ccxt_exchange and self.enable_ccxt:
                try:
                    order_book = await self._fetch_ccxt_order_book()
                        symbol, exchange, limit
                    )
                    if order_book and self._validate_order_book(order_book):
                        self._update_stats()
                            "ccxt", time.time() - start_time, True
                        )
                        return order_book
                except Exception as e:
                    logger.warning(f"CCXT order book fetch failed: {e}")

            # Fallback to mock data (CoinGecko/CMC don't provide order books)'
            logger.warning()
                f"No real order book available for {symbol}, using mock data"
            )
            self._update_stats("fallback", time.time() - start_time, False)
            return self._generate_mock_order_book(symbol, limit)

        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            self._update_stats("fallback", time.time() - start_time, False)
            return self._generate_mock_order_book(symbol, limit)

    async def fetch_price_data()
        self: "Self",
            symbol: str = "BTC/USDC",
                include_volume: bool = True,
                include_market_cap: bool = False
    ) -> Dict[str, Any]:
        """"""
        Fetch price data with fallback mechanisms.

        Args:
            symbol: Trading symbol
            include_volume: Include volume data
            include_market_cap: Include market cap data

        Returns:
            Dictionary with price, volume, and market data
        """"""
        start_time = time.time()

        try:
            await self._rate_limit_delay()

            # Try CCXT first for real-time data
            if self.ccxt_exchange and self.enable_ccxt:
                try:
                    price_data = await self._fetch_ccxt_price_data()
                        symbol, include_volume, include_market_cap
                    )
                    if price_data:
                        self._update_stats()
                            "ccxt", time.time() - start_time, True
                        )
                        return price_data
                except Exception as e:
                    logger.warning(f"CCXT price fetch failed: {e}")

            # Try CoinGecko as fallback
            if self.coingecko_api and self.enable_coingecko:
                try:
                    market_data = self.coingecko_api.get_ticker(symbol)
                    if market_data and market_data.price > 0:
                        price_data = {
                            "symbol": symbol,
                            "price": market_data.price,
                            "volume_24h": market_data.volume_24h,
                            "market_cap": market_data.market_cap,
                            "timestamp": time.time(),
}
}
                        self._update_stats()
                            "coingecko", time.time() - start_time, True
                        )
                        return price_data
                except Exception as e:
                    logger.warning(f"CoinGecko price fetch failed: {e}")

            # Try CoinMarketCap as fallback
            if self.coinmarketcap_api and self.enable_coinmarketcap:
                try:
                    latest_quote = self.coinmarketcap_api.get_latest_quotes(symbol)
                    if latest_quote:
                        price_data = {
                            "symbol": symbol,
                            "price": latest_quote.price,
                            "volume_24h": latest_quote.volume_24h,
                            "market_cap": latest_quote.market_cap,
                            "timestamp": time.time(),
}
}
                        self._update_stats()
                            "coinmarketcap", time.time() - start_time, True
                        )
                        return price_data
                except Exception as e:
                    logger.warning(f"CoinMarketCap price fetch failed: {e}")

            logger.warning(f"No real-time price data available for {symbol}, using mock data")
            self._update_stats("fallback", time.time() - start_time, False)
            return self._generate_mock_price_data(symbol)

        except Exception as e:
            logger.error(f"Price data fetch failed: {e}")
            self._update_stats("fallback", time.time() - start_time, False)
            return self._generate_mock_price_data(symbol)

    async def fetch_news_sentiment()
        self: "Self",
            symbol: str = "BTC",
                limit: int = 10
    ) -> List[Dict[str, Any]]:
        """"""
        Fetch news and sentiment data.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC")
            limit: Number of news items to fetch

        Returns:
            List of news items with sentiment data
        """"""
        start_time = time.time()
        try:
            await self._rate_limit_delay()
            # For now, return mock news data as most APIs require special access
            logger.warning()
                f"News API not fully implemented for {symbol}, using mock data"
            )
            self._update_stats("fallback", time.time() - start_time, True)
            return self._generate_mock_news_data(symbol, limit)
        except Exception as e:
            logger.error(f"News sentiment fetch failed: {e}")
            self._update_stats("fallback", time.time() - start_time, False)
            return self._generate_mock_news_data(symbol, limit)

    async def _fetch_ccxt_order_book()
        self: "Self",
            symbol: str,
                exchange: str,
                limit: int
    ) -> Optional[Dict[str, List]]:
        """"""
        Fetch order book data using CCXT.
        """"""
        if not self.ccxt_exchange:
            return None
        try:
            # CCXT expects symbol in format like 'BTC/USDT'
            ccxt_symbol = self._normalize_symbol(symbol)
            # Fetch L2 order book (bids and asks)
            ob = await self.ccxt_exchange.fetch_l2_order_book(ccxt_symbol, limit=limit)
            return ob
        except Exception as e:
            logger.error(f"Error fetching CCXT order book for {symbol}: {e}")
            return None

    async def _fetch_ccxt_price_data()
        self: "Self",
            symbol: str,
                include_volume: bool,
                include_market_cap: bool
    ) -> Optional[Dict[str, Any]]:
        """"""
        Fetch price data using CCXT.
        """"""
        if not self.ccxt_exchange:
            return None
        try:
            ccxt_symbol = self._normalize_symbol(symbol)
            ticker = await self.ccxt_exchange.fetch_ticker(ccxt_symbol)
            if ticker:
                return {}
                    "symbol": symbol,
                        "price": ticker.get("last"),
                            "volume_24h": ticker.get("quoteVolume") if include_volume else None,
                            "market_cap": None, # CCXT doesn't directly provide market cap'
                    "timestamp": time.time(),
}
            return None
        except Exception as e:
            logger.error(f"Error fetching CCXT price data for {symbol}: {e}")
            return None

    def _normalize_symbol(self: "Self", symbol: str) -> str:
        """"""
        Normalize symbol to CCXT format (e.g., BTC/USDC to BTC/USDT).
        """"""
        # This is a simplification; a real system would need a robust mapping
        return symbol.replace("USDC", "USDT") # Most exchanges use USDT pairs

    def _validate_order_book()
        self: "Self", order_book: Dict[str, List]
    ) -> bool:
        """"""
        Validate the structure and content of an order book.
        """"""
        if not isinstance(order_book, dict):
            logger.warning("Order book is not a dictionary.")
            return False
        if "bids" not in order_book or "asks" not in order_book:
            logger.warning("Order book missing 'bids' or 'asks' keys.")
            return False
        if not isinstance(order_book["bids"], list) or not isinstance(order_book["asks"], list):
            logger.warning("Order book bids or asks are not lists.")
            return False
        return True

    def _generate_mock_order_book()
        self: "Self", symbol: str, limit: int
    ) -> Dict[str, List]:
        """"""
        Generate mock order book data.
        """"""
        bids = [[random.uniform(30000, 31000), random.uniform(0.1, 5)] for _ in range(limit)]
        asks = [[random.uniform(31001, 32000), random.uniform(0.1, 5)] for _ in range(limit)]
        return {"bids": sorted(bids, key=lambda x: x[0], reverse=True), "asks": sorted(asks, key=lambda x: x[0])}

    def _generate_mock_price_data()
        self: "Self", symbol: str
    ) -> Dict[str, Any]:
        """"""
        Generate mock price data.
        """"""
        mock_price = random.uniform(30000, 35000)
        mock_volume = random.uniform(1000000, 5000000)
        mock_market_cap = random.uniform(500000000, 1000000000)
        return {}
            "symbol": symbol,
                "price": mock_price,
                    "volume_24h": mock_volume,
                    "market_cap": mock_market_cap,
                    "timestamp": time.time(),
}
    def _generate_mock_news_data()
        self: "Self", symbol: str, limit: int
    ) -> List[Dict[str, Any]]:
        """"""
        Generate mock news data.
        """"""
        news_items = []
        for i in range(limit):
            sentiment = random.choice(["positive", "negative", "neutral"])
            impact_level = random.choice(["low", "medium", "high", "critical"])
            news_item = {
                "title": f"Mock news about {symbol} #{i + 1} - {sentiment.upper()} impact",
                "content": f"This is mock news content about {symbol}. It has a {sentiment} sentiment and {impact_level} impact.",
                "published_at": (time.time() - i * 3600), # Older news items
                "source": random.choice(["MockNewsA", "MockNewsB"]),
                "url": f"https://mocknews.com/{symbol}/{i}",
                "category": random.choice(["crypto", "market", "general"]),
                "sentiment_score": random.uniform(-1.0, 1.0),
                "sentiment_type": sentiment,
                "impact_level": impact_level,
                "keywords": [symbol.lower(), sentiment, impact_level],
                "entities": [symbol.upper(), "Federal Reserve" if random.random() > 0.8 else ""] # Example entity
}
}
            news_items.append(news_item)
        return news_items

    async def _rate_limit_delay(self: "Self") -> None:
        """"""
        Enforce rate limiting between API requests.
        """"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _update_stats()
        self: "Self", api_type: str, response_time: float, success: bool
    ) -> None:
        """"""
        Update API request statistics.
        """"""
        self.api_stats["total_requests"] += 1
        if success:
            self.api_stats["successful_requests"] += 1
        else:
            self.api_stats["failed_requests"] += 1

        if api_type == "coingecko":
            self.api_stats["coingecko_requests"] += 1
        elif api_type == "coinmarketcap":
            self.api_stats["coinmarketcap_requests"] += 1
        elif api_type == "ccxt":
            self.api_stats["ccxt_requests"] += 1
        elif api_type == "fallback":
            self.api_stats["fallback_requests"] += 1

        # Update average response time (simple moving average)
        current_avg = self.api_stats["avg_response_time"]
        total_successful = self.api_stats["successful_requests"]

        if total_successful > 0:
            self.api_stats["avg_response_time"] = ()
                (current_avg * (total_successful - 1)) + response_time
            ) / total_successful
        else:
            self.api_stats["avg_response_time"] = 0.0

    def get_api_performance_summary()
        self: "Self",
            ) -> Dict[str, Union[int, float]]:
        """"""
        Get a summary of API performance statistics.
        """"""
        return self.api_stats

# Global APIBridge instance for convenience
# In a larger application, this would be managed via dependency injection
# or passed explicitly.
api_bridge = APIBridge()


async def fetch_price_data(symbol: str = "BTC/USDC") -> Dict[str, Any]:
    """"""
    Convenience function to fetch price data from the global APIBridge instance.
    """"""
    return await api_bridge.fetch_price_data(symbol)


async def fetch_order_book_data()
    symbol: str = "BTC/USDC", limit: int = 20
) -> Dict[str, List]:
    """"""
    Convenience function to fetch order book data from the global APIBridge instance.
    """"""
    return await api_bridge.fetch_order_book(symbol, limit=limit)


async def initialize_api_bridge()
    coinmarketcap_api_key: Optional[str] = None,
        coinbase_api_key: Optional[str] = None,
            coinbase_api_secret: Optional[str] = None,
            sandbox: bool = True
) -> APIBridge:
    """"""
    Initializes and returns a new APIBridge instance.
    This allows for flexible setup in different parts of the application.
    """"""
    new_bridge = APIBridge()
        coinmarketcap_api_key=coinmarketcap_api_key,
            coinbase_api_key=coinbase_api_key,
                coinbase_api_secret=coinbase_api_secret,
                sandbox=sandbox
    )
    # Update the global instance if needed, or return for local use
    # globals()['api_bridge'] = new_bridge # Uncomment if you want to replace the global instance
    return new_bridge

async def main() -> None:
    """"""
    Main function for testing APIBridge functionalities.
    """"""
    # Example of initializing with API keys (replace with your actual keys)
    # api_bridge_with_keys = await initialize_api_bridge()
    #     coinmarketcap_api_key="YOUR_CMC_API_KEY",
        #     coinbase_api_key="YOUR_COINBASE_API_KEY",
            #     coinbase_api_secret="YOUR_COINBASE_API_SECRET",
            #     sandbox=True
    # )
    # If using the global instance, no need to pass it around

    # Test fetching price data
    print("\n--- Fetching Price Data (BTC/USDC) ---")
    price_data = await api_bridge.fetch_price_data("BTC/USDC")
    print(f"Price Data: {price_data}")

    # Test fetching order book data
    print("\n--- Fetching Order Book (BTC/USDC) ---")
    order_book = await api_bridge.fetch_order_book("BTC/USDC")
    print(f"Order Book (bids): {order_book['bids'][:5]}...")
    print(f"Order Book (asks): {order_book['asks'][:5]}...")

    # Test fetching news sentiment data
    print("\n--- Fetching News Sentiment (BTC) ---")
    news_data = await api_bridge.fetch_news_sentiment("BTC", limit=3)
    print(f"News Data: {news_data}")

    # Get API performance summary
    print("\n--- API Performance Summary ---")
    summary = api_bridge.get_api_performance_summary()
    print(summary)

if __name__ == "__main__":
    asyncio.run(main())
