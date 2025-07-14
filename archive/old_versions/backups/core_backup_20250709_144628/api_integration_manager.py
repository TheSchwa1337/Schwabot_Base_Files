"""Module for Schwabot trading system."""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
import ccxt.async_support as ccxt

from utils.secure_config_manager import SecureConfigManager

#!/usr/bin/env python3
"""
API Integration Manager
======================

Comprehensive API integration system for Schwabot trading operations.
Handles CoinMarketCap, CoinGecko, Coinbase, and other exchange APIs
with fallback mechanisms, rate limiting, and error recovery.

    Features:
    - Multi-source price data with fallback logic
    - Coinbase portfolio management integration
    - Order book management and position sizing
    - Cross-platform compatibility (Windows, macOS, Linux)
    - Real-time data streaming and caching
    - Advanced error handling and retry logic
    """

    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logger = logging.getLogger(__name__)


        class APISource(Enum):
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """API data sources."""

        COINMARKETCAP = "coinmarketcap"
        COINGECKO = "coingecko"
        COINBASE = "coinbase"
        BINANCE = "binance"
        BYBIT = "bybit"
        FALLBACK = "fallback"


        @dataclass
            class PriceData:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Price data structure."""

            symbol: str
            price: float
            volume_24h: float
            market_cap: float
            price_change_24h: float
            price_change_percent_24h: float
            source: str
            timestamp: int
            confidence: float = 1.0


            @dataclass
                class OrderBookData:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Order book data structure."""

                symbol: str
                bids: List[List[float]]  # [price, quantity]
                asks: List[List[float]]  # [price, quantity]
                timestamp: int
                source: str


                @dataclass
                    class PortfolioData:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Portfolio data structure."""

                    total_balance: float
                    available_balance: float
                    positions: Dict[str, float]
                    unrealized_pnl: float
                    realized_pnl: float
                    source: str
                    timestamp: int


                        class APIIntegrationManager:
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """Comprehensive API integration manager for Schwabot."""

                            def __init__(self, config_path: str = "config/api_config.json") -> None:
                            self.config_manager = SecureConfigManager(config_path)
                            self.config = self.config_manager.load_config()

                            # API credentials
                            self.coinmarketcap_api_key = self.config.get("coinmarketcap_api_key")
                            self.coinbase_api_key = self.config.get("coinbase_api_key")
                            self.coinbase_secret = self.config.get("coinbase_secret")
                            self.coinbase_passphrase = self.config.get("coinbase_passphrase")

                            # Exchange instances
                            self.exchanges = {}
                            self.session = None
                            self.rate_limits = {}
                            self.cache = {}
                            self.cache_ttl = 30  # 30 seconds

                            # Initialize exchanges
                            self._initialize_exchanges()

                                def _initialize_exchanges(self) -> None:
                                """Initialize exchange connections."""
                                    try:
                                    # Coinbase Advanced Trade
                                        if all([self.coinbase_api_key, self.coinbase_secret, self.coinbase_passphrase]):
                                        self.exchanges['coinbase'] = ccxt.coinbase()
                                        self.exchanges['coinbase'].apiKey = self.coinbase_api_key
                                        self.exchanges['coinbase'].secret = self.coinbase_secret
                                        self.exchanges['coinbase'].password = self.coinbase_passphrase
                                        self.exchanges['coinbase'].sandbox = self.config.get('sandbox_mode', False)
                                        self.exchanges['coinbase'].enableRateLimit = True
                                        self.exchanges['coinbase'].options = {
                                        'defaultType': 'spot',
                                        'adjustForTimeDifference': True,
                                        }
                                        print("✅ Coinbase Advanced Trade initialized")

                                        # Binance (fallback)
                                        self.exchanges['binance'] = ccxt.binance()
                                        self.exchanges['binance'].enableRateLimit = True
                                        self.exchanges['binance'].options = {
                                        'defaultType': 'spot',
                                        'adjustForTimeDifference': True,
                                        }
                                        print("✅ Binance initialized (fallback)")

                                        # Bybit (additional, fallback)
                                        self.exchanges['bybit'] = ccxt.bybit()
                                        self.exchanges['bybit'].enableRateLimit = True
                                        self.exchanges['bybit'].options = {
                                        'defaultType': 'spot',
                                        'adjustForTimeDifference': True,
                                        }
                                        print("✅ Bybit initialized (additional fallback)")

                                            except Exception as e:
                                            print("❌ Failed to initialize exchanges: {0}".format(e))

                                                async def get_price_data(self, symbol: str, sources: List[APISource]=None) -> Optional[PriceData]:
                                                """Get price data with fallback mechanism."""
                                                    if sources is None:
                                                    sources = [APISource.COINMARKETCAP, APISource.COINGECKO, APISource.COINBASE]

                                                    # Check cache first
                                                    cache_key = "price_{0}".format(symbol)
                                                        if self._is_cache_valid(cache_key):
                                                    return self.cache[cache_key]['data']

                                                        for source in sources:
                                                            try:
                                                            price_data = await self._fetch_price_from_source(symbol, source)
                                                                if price_data:
                                                                self._update_cache(cache_key, price_data)
                                                            return price_data
                                                                except Exception as e:
                                                                print("⚠️ Failed to fetch price from {0}: {1}".format(source.value, e))
                                                            continue

                                                            print("❌ Failed to get price data for {0} from all sources".format(symbol))
                                                        return None

                                                            async def _fetch_price_from_source(self, symbol: str, source: APISource) -> Optional[PriceData]:
                                                            """Fetch price data from specific source."""
                                                                if source == APISource.COINMARKETCAP:
                                                            return await self._fetch_coinmarketcap_price(symbol)
                                                                elif source == APISource.COINGECKO:
                                                            return await self._fetch_coingecko_price(symbol)
                                                                elif source == APISource.COINBASE:
                                                            return await self._fetch_coinbase_price(symbol)
                                                                else:
                                                            return None

                                                                async def _fetch_coinmarketcap_price(self, symbol: str) -> Optional[PriceData]:
                                                                """Fetch price from CoinMarketCap API."""
                                                                    if not self.coinmarketcap_api_key:
                                                                return None

                                                                    try:
                                                                    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
                                                                    params = {'symbol': symbol, 'convert': 'USD'}
                                                                    headers = {'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key}

                                                                        async with aiohttp.ClientSession() as session:
                                                                            async with session.get(url, params=params, headers=headers) as response:
                                                                                if response.status == 200:
                                                                                data = await response.json()
                                                                                    if 'data' in data and symbol in data['data']:
                                                                                    quote = data['data'][symbol]['quote']['USD']
                                                                                return PriceData(
                                                                                symbol=symbol,
                                                                                price=float(quote['price']),
                                                                                volume_24h=float(quote['volume_24h']),
                                                                                market_cap=float(quote['market_cap']),
                                                                                price_change_24h=float(quote['volume_change_24h']),
                                                                                price_change_percent_24h=float(quote['percent_change_24h']),
                                                                                source=APISource.COINMARKETCAP.value,
                                                                                timestamp=int(time.time()),
                                                                                )
                                                                                    except Exception as e:
                                                                                    logger.error("CoinMarketCap API error: {0}".format(e))

                                                                                return None

                                                                                    async def _fetch_coingecko_price(self, symbol: str) -> Optional[PriceData]:
                                                                                    """Fetch price from CoinGecko API (free, fallback)."""
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
                                                                                        }

                                                                                        coin_id = symbol_mapping.get(symbol, symbol.lower())
                                                                                        url = f"https://api.coingecko.com/api/v3/simple/price"
                                                                                        params = {
                                                                                        'ids': coin_id,
                                                                                        'vs_currencies': 'usd',
                                                                                        'include_24hr_change': 'true',
                                                                                        'include_market_cap': 'true',
                                                                                        'include_24hr_vol': 'true',
                                                                                        }

                                                                                            async with aiohttp.ClientSession() as session:
                                                                                                async with session.get(url, params=params) as response:
                                                                                                    if response.status == 200:
                                                                                                    data = await response.json()
                                                                                                        if coin_id in data:
                                                                                                        coin_data = data[coin_id]
                                                                                                    return PriceData(
                                                                                                    symbol=symbol,
                                                                                                    price=float(coin_data['usd']),
                                                                                                    volume_24h=float(coin_data.get('usd_24h_vol', 0)),
                                                                                                    market_cap=float(coin_data.get('usd_market_cap', 0)),
                                                                                                    price_change_24h=0.0,  # Not provided by CoinGecko
                                                                                                    price_change_percent_24h=float(coin_data.get('usd_24h_change', 0)),
                                                                                                    source=APISource.COINGECKO.value,
                                                                                                    timestamp=int(time.time()),
                                                                                                    )
                                                                                                        except Exception as e:
                                                                                                        logger.error("CoinGecko API error: {0}".format(e))

                                                                                                    return None

                                                                                                        async def _fetch_coinbase_price(self, symbol: str) -> Optional[PriceData]:
                                                                                                        """Fetch price from Coinbase API."""
                                                                                                            try:
                                                                                                                if 'coinbase' not in self.exchanges:
                                                                                                            return None

                                                                                                            exchange = self.exchanges['coinbase']
                                                                                                            ticker = await exchange.fetch_ticker("{0}/USD".format(symbol))

                                                                                                        return PriceData(
                                                                                                        symbol=symbol,
                                                                                                        price=float(ticker['last']),
                                                                                                        volume_24h=float(ticker['baseVolume']),
                                                                                                        market_cap=0.0,  # Not provided by ticker
                                                                                                        price_change_24h=0.0,  # Not provided by ticker
                                                                                                        price_change_percent_24h=0.0,  # Not provided by ticker
                                                                                                        source=APISource.COINBASE.value,
                                                                                                        timestamp=int(ticker['timestamp'] / 1000),
                                                                                                        )
                                                                                                            except Exception as e:
                                                                                                            logger.error("Coinbase API error: {0}".format(e))

                                                                                                        return None

                                                                                                            async def get_order_book(self, symbol: str, exchange: str='coinbase', depth: int=20) -> Optional[OrderBookData]:
                                                                                                            """Get order book data from exchange."""
                                                                                                                try:
                                                                                                                    if exchange not in self.exchanges:
                                                                                                                return None

                                                                                                                exchange_instance = self.exchanges[exchange]
                                                                                                                order_book = await exchange_instance.fetch_order_book("{0}/USD".format(symbol), depth)

                                                                                                            return OrderBookData(
                                                                                                            symbol=symbol,
                                                                                                            bids=order_book['bids'][:depth],
                                                                                                            asks=order_book['asks'][:depth],
                                                                                                            timestamp=int(order_book['timestamp'] / 1000),
                                                                                                            source=exchange,
                                                                                                            )
                                                                                                                except Exception as e:
                                                                                                                logger.error("Order book fetch error: {0}".format(e))
                                                                                                            return None

                                                                                                                async def get_portfolio_data(self, exchange: str='coinbase') -> Optional[PortfolioData]:
                                                                                                                """Get portfolio data from exchange."""
                                                                                                                    try:
                                                                                                                        if exchange not in self.exchanges:
                                                                                                                    return None

                                                                                                                    exchange_instance = self.exchanges[exchange]
                                                                                                                    balance = await exchange_instance.fetch_balance()

                                                                                                                    # Calculate total balance
                                                                                                                    total_balance = 0.0
                                                                                                                    positions = {}

                                                                                                                        for currency, amount in balance['total'].items():
                                                                                                                            if amount > 0:
                                                                                                                                if currency == 'USD':
                                                                                                                                total_balance += amount
                                                                                                                                positions[currency] = amount
                                                                                                                                    else:
                                                                                                                                    # Get current price for crypto
                                                                                                                                        try:
                                                                                                                                        price_data = await self.get_price_data(currency)
                                                                                                                                            if price_data:
                                                                                                                                            usd_value = amount * price_data.price
                                                                                                                                            total_balance += usd_value
                                                                                                                                            positions[currency] = usd_value
                                                                                                                                                except Exception:
                                                                                                                                                positions[currency] = amount

                                                                                                                                            return PortfolioData(
                                                                                                                                            total_balance=total_balance,
                                                                                                                                            available_balance=float(balance['USD']['free']),
                                                                                                                                            positions=positions,
                                                                                                                                            unrealized_pnl=0.0,  # Would need position data
                                                                                                                                            realized_pnl=0.0,  # Would need trade history
                                                                                                                                            source=exchange,
                                                                                                                                            timestamp=int(time.time()),
                                                                                                                                            )
                                                                                                                                                except Exception as e:
                                                                                                                                                logger.error("Portfolio fetch error: {0}".format(e))
                                                                                                                                            return None

                                                                                                                                                async def calculate_position_size(self, symbol: str, entry_price: float, risk_amount: float, stop_loss_pct: float) -> float:
                                                                                                                                                """Calculate position size based on risk management."""
                                                                                                                                                    try:
                                                                                                                                                    # Get order book for slippage estimation
                                                                                                                                                    order_book = await self.get_order_book(symbol)
                                                                                                                                                        if not order_book:
                                                                                                                                                    return 0.0

                                                                                                                                                    # Calculate position size using risk management
                                                                                                                                                    risk_per_share = entry_price * stop_loss_pct
                                                                                                                                                    position_size = risk_amount / risk_per_share

                                                                                                                                                    # Check if order book can accommodate the position
                                                                                                                                                    available_liquidity = sum(bid[1] for bid in order_book.bids[:5])
                                                                                                                                                    max_position = available_liquidity * 0.1  # Use max 10% of available liquidity

                                                                                                                                                    position_size = min(position_size, max_position)

                                                                                                                                                return max(0.0, position_size)

                                                                                                                                                    except Exception as e:
                                                                                                                                                    logger.error("Position size calculation error: {0}".format(e))
                                                                                                                                                return 0.0

                                                                                                                                                    async def place_order(self, symbol: str, side: str, amount: float, price: float = None, order_type: str = 'market') -> Dict[str, Any]:
                                                                                                                                                    """Place order on exchange."""
                                                                                                                                                        try:
                                                                                                                                                            if 'coinbase' not in self.exchanges:
                                                                                                                                                        return {'success': False, 'error': 'Coinbase not available'}

                                                                                                                                                        exchange = self.exchanges['coinbase']

                                                                                                                                                        # Prepare order parameters
                                                                                                                                                        order_params = {'symbol': "{0}/USD".format(symbol), 'type': order_type, 'side': side, 'amount': amount}

                                                                                                                                                            if price and order_type == 'limit':
                                                                                                                                                            order_params['price'] = price

                                                                                                                                                            # Place order
                                                                                                                                                            order = await exchange.create_order(**order_params)

                                                                                                                                                        return {
                                                                                                                                                        'success': True,
                                                                                                                                                        'order_id': order['id'],
                                                                                                                                                        'status': order['status'],
                                                                                                                                                        'filled': order['filled'],
                                                                                                                                                        'remaining': order['remaining'],
                                                                                                                                                        'cost': order['cost'],
                                                                                                                                                        }

                                                                                                                                                            except Exception as e:
                                                                                                                                                            logger.error("Order placement error: {0}".format(e))
                                                                                                                                                        return {'success': False, 'error': str(e)}

                                                                                                                                                            def _is_cache_valid(self, key: str) -> bool:
                                                                                                                                                            """Check if cached data is still valid."""
                                                                                                                                                                if key not in self.cache:
                                                                                                                                                            return False

                                                                                                                                                            cache_time = self.cache[key]['timestamp']
                                                                                                                                                        return (time.time() - cache_time) < self.cache_ttl

                                                                                                                                                            def _update_cache(self, key: str, data: Any) -> None:
                                                                                                                                                            """Update cache with new data."""
                                                                                                                                                            self.cache[key] = {'data': data, 'timestamp': time.time()}

                                                                                                                                                                async def close(self):
                                                                                                                                                                """Close all exchange connections."""
                                                                                                                                                                    for exchange in self.exchanges.values():
                                                                                                                                                                        try:
                                                                                                                                                                        await exchange.close()
                                                                                                                                                                            except Exception:
                                                                                                                                                                        pass

                                                                                                                                                                            if self.session:
                                                                                                                                                                            await self.session.close()


                                                                                                                                                                            # CLI Interface
                                                                                                                                                                                class APIIntegrationCLI:
    """Class for Schwabot trading functionality."""
                                                                                                                                                                                """Class for Schwabot trading functionality."""
                                                                                                                                                                                """CLI interface for API integration manager."""

                                                                                                                                                                                    def __init__(self) -> None:
                                                                                                                                                                                    self.api_manager = None

                                                                                                                                                                                        async def initialize(self):
                                                                                                                                                                                        """Initialize the API manager."""
                                                                                                                                                                                            try:
                                                                                                                                                                                            self.api_manager = APIIntegrationManager()
                                                                                                                                                                                            print("✅ API Integration Manager initialized")
                                                                                                                                                                                        return True
                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                            print("❌ Failed to initialize API manager: {0}".format(e))
                                                                                                                                                                                        return False

                                                                                                                                                                                            async def cmd_price(self, symbol: str):
                                                                                                                                                                                            """Get price for symbol."""
                                                                                                                                                                                                if not self.api_manager:
                                                                                                                                                                                                print("❌ API manager not initialized")
                                                                                                                                                                                            return

                                                                                                                                                                                            print("💰 Fetching price for {0}...".format(symbol))
                                                                                                                                                                                            price_data = await self.api_manager.get_price_data(symbol)

                                                                                                                                                                                                if price_data:
                                                                                                                                                                                                print("   Source: {0}".format(price_data.source))
                                                                                                                                                                                                print("   24h, Change: {0:.2f}%".format(price_data.price_change_percent_24h))
                                                                                                                                                                                                print("   Volume: ${0}".format(price_data.volume_24h))
                                                                                                                                                                                                    else:
                                                                                                                                                                                                    print("❌ Failed to get price for {0}".format(symbol))

                                                                                                                                                                                                        async def cmd_orderbook(self, symbol: str):
                                                                                                                                                                                                        """Get order book for symbol."""
                                                                                                                                                                                                            if not self.api_manager:
                                                                                                                                                                                                            print("❌ API manager not initialized")
                                                                                                                                                                                                        return

                                                                                                                                                                                                        print("📚 Fetching order book for {0}...".format(symbol))
                                                                                                                                                                                                        order_book = await self.api_manager.get_order_book(symbol)

                                                                                                                                                                                                            if order_book:
                                                                                                                                                                                                            print("✅ Order book for {0}".format(symbol))
                                                                                                                                                                                                            print("   Top 5 Bids:")
                                                                                                                                                                                                                for i, (price, qty) in enumerate(order_book.bids[:5]):
                                                                                                                                                                                                                print("     {0}. ${1} - {2}".format(i + 1, price, qty))

                                                                                                                                                                                                                print("   Top 5 Asks:")
                                                                                                                                                                                                                    for i, (price, qty) in enumerate(order_book.asks[:5]):
                                                                                                                                                                                                                    print("     {0}. ${1} - {2}".format(i + 1, price, qty))
                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                        print("❌ Failed to get order book for {0}".format(symbol))

                                                                                                                                                                                                                            async def cmd_portfolio(self):
                                                                                                                                                                                                                            """Get portfolio data."""
                                                                                                                                                                                                                                if not self.api_manager:
                                                                                                                                                                                                                                print("❌ API manager not initialized")
                                                                                                                                                                                                                            return

                                                                                                                                                                                                                            print("💼 Fetching portfolio data...")
                                                                                                                                                                                                                            portfolio = await self.api_manager.get_portfolio_data()

                                                                                                                                                                                                                                if portfolio:
                                                                                                                                                                                                                                print("✅ Portfolio Total: ${0}".format(portfolio.total_balance))
                                                                                                                                                                                                                                print("   Available: ${0}".format(portfolio.available_balance))
                                                                                                                                                                                                                                print("   Positions: {0}".format(len(portfolio.positions)))
                                                                                                                                                                                                                                    for asset, value in portfolio.positions.items():
                                                                                                                                                                                                                                    print("     {0}: ${1}".format(asset, value))
                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                        print("❌ Failed to get portfolio data")

                                                                                                                                                                                                                                            async def cmd_position_size(self, symbol: str, entry_price: float, risk_amount: float, stop_loss_pct: float):
                                                                                                                                                                                                                                            """Calculate position size."""
                                                                                                                                                                                                                                                if not self.api_manager:
                                                                                                                                                                                                                                                print("❌ API manager not initialized")
                                                                                                                                                                                                                                            return

                                                                                                                                                                                                                                            print("🧮 Calculating position size for {0}...".format(symbol))
                                                                                                                                                                                                                                            position_size = await self.api_manager.calculate_position_size(symbol, entry_price, risk_amount, stop_loss_pct)

                                                                                                                                                                                                                                                if position_size > 0:
                                                                                                                                                                                                                                                print("   Entry price: ${0}".format(entry_price))
                                                                                                                                                                                                                                                print("   Risk amount: ${0}".format(risk_amount))
                                                                                                                                                                                                                                                print("   Stop loss: {0:.1f}%".format(stop_loss_pct * 100))
                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                    print("❌ Failed to calculate position size")

                                                                                                                                                                                                                                                        def show_banner(self) -> None:
                                                                                                                                                                                                                                                        """Show CLI banner."""
                                                                                                                                                                                                                                                        print()
                                                                                                                                                                                                                                                        print("""
                                                                                                                                                                                                                                                        🔗 SCHWABOT API INTEGRATION MANAGER
                                                                                                                                                                                                                                                        ==================================
                                                                                                                                                                                                                                                        Advanced API integration for crypto trading operations
                                                                                                                                                                                                                                                        """)

                                                                                                                                                                                                                                                            async def run_interactive_mode(self):
                                                                                                                                                                                                                                                            """Run interactive CLI mode."""
                                                                                                                                                                                                                                                            self.show_banner()

                                                                                                                                                                                                                                                                if not await self.initialize():
                                                                                                                                                                                                                                                            return

                                                                                                                                                                                                                                                            print("🎮 INTERACTIVE API INTEGRATION CLI")
                                                                                                                                                                                                                                                            print("=" * 40)
                                                                                                                                                                                                                                                            print("Type 'help' for commands, 'quit' to exit")

                                                                                                                                                                                                                                                                while True:
                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                    command = input("\n🔗 api> ").strip().lower()

                                                                                                                                                                                                                                                                        if command == 'quit' or command == 'exit':
                                                                                                                                                                                                                                                                    break
                                                                                                                                                                                                                                                                        elif command == 'help':
                                                                                                                                                                                                                                                                        self._show_help()
                                                                                                                                                                                                                                                                            elif command.startswith('price '):
                                                                                                                                                                                                                                                                            symbol = command.split(' ', 1)[1].upper()
                                                                                                                                                                                                                                                                            await self.cmd_price(symbol)
                                                                                                                                                                                                                                                                                elif command.startswith('orderbook '):
                                                                                                                                                                                                                                                                                symbol = command.split(' ', 1)[1].upper()
                                                                                                                                                                                                                                                                                await self.cmd_orderbook(symbol)
                                                                                                                                                                                                                                                                                    elif command == 'portfolio':
                                                                                                                                                                                                                                                                                    await self.cmd_portfolio()
                                                                                                                                                                                                                                                                                        elif command.startswith('position-size '):
                                                                                                                                                                                                                                                                                        parts = command.split()
                                                                                                                                                                                                                                                                                            if len(parts) >= 5:
                                                                                                                                                                                                                                                                                            symbol = parts[1].upper()
                                                                                                                                                                                                                                                                                            entry_price = float(parts[2])
                                                                                                                                                                                                                                                                                            risk_amount = float(parts[3])
                                                                                                                                                                                                                                                                                            stop_loss_pct = float(parts[4])
                                                                                                                                                                                                                                                                                            await self.cmd_position_size(symbol, entry_price, risk_amount, stop_loss_pct)
                                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                print("Usage: position-size <symbol> <entry_price> <risk_amount> <stop_loss_pct>")
                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                    print("Unknown command: {0}".format(command))

                                                                                                                                                                                                                                                                                                        except KeyboardInterrupt:
                                                                                                                                                                                                                                                                                                    break
                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                        print("Error: {0}".format(e))

                                                                                                                                                                                                                                                                                                            if self.api_manager:
                                                                                                                                                                                                                                                                                                            await self.api_manager.close()
                                                                                                                                                                                                                                                                                                            print("👋 Goodbye!")

                                                                                                                                                                                                                                                                                                                def _show_help(self) -> None:
                                                                                                                                                                                                                                                                                                                """Show help information."""
                                                                                                                                                                                                                                                                                                                print("📖 AVAILABLE COMMANDS:")
                                                                                                                                                                                                                                                                                                                print("  price <symbol>                    - Get price for symbol")
                                                                                                                                                                                                                                                                                                                print("  orderbook <symbol>                - Get order book for symbol")
                                                                                                                                                                                                                                                                                                                print("  portfolio                         - Get portfolio data")
                                                                                                                                                                                                                                                                                                                print("  position-size <symbol> <price> <risk> <stop_loss> - Calculate position size")
                                                                                                                                                                                                                                                                                                                print("  quit/exit                         - Exit CLI")


                                                                                                                                                                                                                                                                                                                    async def main():
                                                                                                                                                                                                                                                                                                                    """Main CLI entry point."""
                                                                                                                                                                                                                                                                                                                    parser = argparse.ArgumentParser(description="API Integration Manager CLI")
                                                                                                                                                                                                                                                                                                                    parser.add_argument("command", nargs="?", help="Command to execute")
                                                                                                                                                                                                                                                                                                                    parser.add_argument("args", nargs="*", help="Command arguments")
                                                                                                                                                                                                                                                                                                                    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")

                                                                                                                                                                                                                                                                                                                    args = parser.parse_args()

                                                                                                                                                                                                                                                                                                                    cli = APIIntegrationCLI()

                                                                                                                                                                                                                                                                                                                        if args.interactive:
                                                                                                                                                                                                                                                                                                                        await cli.run_interactive_mode()
                                                                                                                                                                                                                                                                                                                            elif args.command:
                                                                                                                                                                                                                                                                                                                                if not await cli.initialize():
                                                                                                                                                                                                                                                                                                                            return 1

                                                                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                                                                    if args.command == "price" and args.args:
                                                                                                                                                                                                                                                                                                                                    await cli.cmd_price(args.args[0].upper())
                                                                                                                                                                                                                                                                                                                                        elif args.command == "orderbook" and args.args:
                                                                                                                                                                                                                                                                                                                                        await cli.cmd_orderbook(args.args[0].upper())
                                                                                                                                                                                                                                                                                                                                            elif args.command == "portfolio":
                                                                                                                                                                                                                                                                                                                                            await cli.cmd_portfolio()
                                                                                                                                                                                                                                                                                                                                                elif args.command == "position-size" and len(args.args) >= 4:
                                                                                                                                                                                                                                                                                                                                                symbol = args.args[0].upper()
                                                                                                                                                                                                                                                                                                                                                entry_price = float(args.args[1])
                                                                                                                                                                                                                                                                                                                                                risk_amount = float(args.args[2])
                                                                                                                                                                                                                                                                                                                                                stop_loss_pct = float(args.args[3])
                                                                                                                                                                                                                                                                                                                                                await cli.cmd_position_size(symbol, entry_price, risk_amount, stop_loss_pct)
                                                                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                                                    parser.print_help()
                                                                                                                                                                                                                                                                                                                                                        finally:
                                                                                                                                                                                                                                                                                                                                                            if cli.api_manager:
                                                                                                                                                                                                                                                                                                                                                            await cli.api_manager.close()
                                                                                                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                                                                                parser.print_help()

                                                                                                                                                                                                                                                                                                                                                            return 0


                                                                                                                                                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                asyncio.run(main())
