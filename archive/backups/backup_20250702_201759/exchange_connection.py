from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import ccxt
import ccxt.async_support as ccxt_async

from .data_models import APICredentials, MarketData, OrderRequest, OrderResponse
from .enums import ConnectionStatus

try:

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)


"""Exchange Connection Manager ====================

Manages connection and communication with a single exchange via CCXT.
"""


class ExchangeConnection:
    """Manages the connection and communication with a single crypto exchange."""
    def __init__(self, credentials: APICredentials, config: Dict[str, any]):
        """Initializes the ExchangeConnection.

        Args:
            credentials: API credentials for the exchange.
            config: Configuration dictionary for the exchange connection.
        """
        self.credentials = credentials
        self.config = config
        self.exchange = None
        self.async_exchange = None
        self.status = ConnectionStatus.DISCONNECTED
        self.last_heartbeat = 0.0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.get("max_reconnect_attempts", 5)

        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_error: Optional[str] = None

        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.cache_expiry = config.get("market_data_cache_expiry", 30)

        logger.info(f"Exchange connection initialized for {credentials.exchange.value}")

    async def connect():-> bool:
        """Establishes connection to the exchange."""
        if not CCXT_AVAILABLE:
            logger.error("CCXT library not available. Cannot connect to exchange.")
            self.status = ConnectionStatus.ERROR
            self.last_error = "CCXT library not installed."
            return False

        self.status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to {self.credentials.exchange.value}...")

        try:
            exchange_name = self.credentials.exchange.value
            exchange_class = getattr(ccxt, exchange_name)
            async_exchange_class = getattr(ccxt_async, exchange_name)

            config = {
                "apiKey": self.credentials.api_key,
                "secret": self.credentials.secret,
                "enableRateLimit": True,
                "timeout": self.config.get("timeout", 30000),
                "options": {"defaultType": "spot", "adjustForTimeDifference": True},
            }
            if self.credentials.passphrase:
                config["password"] = self.credentials.passphrase

            # CCXT handles sandbox/testnet via a method on the instance
            self.exchange = exchange_class(config)
            self.async_exchange = async_exchange_class(config)

            if self.credentials.sandbox or self.credentials.testnet:
                self.exchange.set_sandbox_mode(True)
                self.async_exchange.set_sandbox_mode(True)

            await self.async_exchange.load_markets()

            self.status = ConnectionStatus.CONNECTED
            self.last_heartbeat = time.time()
            self.reconnect_attempts = 0
            logger.info(f"✅ Successfully connected to {exchange_name}")
            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(
                f"❌ Failed to connect to {self.credentials.exchange.value}: {e}", exc_info = True
            )
            return False

    async def disconnect(self):
        """Closes the connection to the exchange."""
        if self.status == ConnectionStatus.DISCONNECTED:
            return logger.info(f"Disconnecting from {self.credentials.exchange.value}...")
        try:
            if self.async_exchange:
                await self.async_exchange.close()
            self.status = ConnectionStatus.DISCONNECTED
            logger.info(f"Disconnected from {self.credentials.exchange.value}")
        except Exception as e:
            logger.error(f"Error during disconnection: {e}", exc_info = True)

    async def get_market_data():-> Optional[MarketData]:
        """Fetches market data for a given symbol, using a cache."""
        if self.status != ConnectionStatus.CONNECTED:
            return None

        # Check cache first
        cached_data = self.market_data_cache.get(symbol)
        if cached_data and (time.time() - cached_data.timestamp < self.cache_expiry):
            return cached_data

        try:
            # CCXT fetch_ticker is already rate-limited by the library
            ticker = await self.async_exchange.fetch_ticker(symbol)

            market_data = MarketData(
                symbol=symbol,
                price=float(ticker["last"]),
                volume=float(ticker.get("baseVolume", 0)),
                bid=float(ticker["bid"]),
                ask=float(ticker["ask"]),
                high_24h=float(ticker["high"]),
                low_24h=float(ticker["low"]),
                change_24h=float(ticker.get("change", 0)),
                timestamp=ticker["timestamp"] / 1000 if ticker["timestamp"] else time.time(),
                exchange=self.credentials.exchange.value,
                metadata=ticker,
            )

            self.market_data_cache[symbol] = market_data
            self.successful_requests += 1
            self.last_heartbeat = time.time()
            return market_data

        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            logger.error(
                f"Error fetching market data for {symbol} on {self.credentials.exchange.value}: {e}"
            )
            return None

    async def place_order():-> OrderResponse:
        """Places a trade order on the exchange."""
        if self.status != ConnectionStatus.CONNECTED:
            return self._create_error_response(order_request, "Exchange not connected.")

        try:
            params = {}
            if order_request.stop_loss:
                params["stopLossPrice"] = order_request.stop_loss
            if order_request.take_profit:
                params["takeProfitPrice"] = order_request.take_profit

            order = await self.async_exchange.create_order(
                symbol=order_request.symbol,
                type=order_request.order_type.value,
                side=order_request.side.value,
                amount=order_request.amount,
                price=order_request.price,
                params=params,
            )

            response = self._create_success_response(order)
            self.successful_requests += 1
            self.last_heartbeat = time.time()
            logger.info(
                f"✅ Order placed on {self.credentials.exchange.value}: {response.order_id}"
            )
            return response

        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            logger.error(
                f"❌ Error placing order on {self.credentials.exchange.value}: {e}", exc_info = True
            )
            return self._create_error_response(order_request, str(e))

    async def get_balance():-> Dict[str, float]:
        """Fetches the account balance from the exchange."""
        if self.status != ConnectionStatus.CONNECTED:
            return {}
        try:
            balance = await self.async_exchange.fetch_balance()
            free_balances = {
                currency: float(amount)
                for currency, amount in balance.get("free", {}).items()
                if float(amount) > 0
            }
            self.successful_requests += 1
            self.last_heartbeat = time.time()
            return free_balances
        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            logger.error(f"Error fetching balance from {self.credentials.exchange.value}: {e}")
            return {}

    def _create_success_response():-> OrderResponse:
        """Creates a successful OrderResponse from CCXT order data."""
        return OrderResponse(
            order_id=str(order.get("id")),
            client_order_id=order.get("clientOrderId"),
            symbol=order.get("symbol"),
            side=order.get("side"),
            order_type=order.get("type"),
            amount=float(order.get("amount")),
            price=float(order.get("price", 0.0)),
            filled=float(order.get("filled", 0.0)),
            remaining=float(order.get("remaining", 0.0)),
            cost=float(order.get("cost", 0.0)),
            status=order.get("status"),
            timestamp=order.get("timestamp") / 1000 if order.get("timestamp") else time.time(),
            fee=order.get("fee"),
            info=order,
            success=True,
            error_message=None,
        )

    def _create_error_response():-> OrderResponse:
        """Creates an error OrderResponse."""
        return OrderResponse(
            order_id="",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side.value,
            order_type=req.order_type.value,
            amount=req.amount,
            price=req.price if req.price is not None else 0.0,
            filled=0.0,
            remaining=req.amount,
            cost=0.0,
            status="failed",
            timestamp=time.time(),
            fee=None,
            info={"error": error_msg},
            success=False,
            error_message=error_msg,
        )
