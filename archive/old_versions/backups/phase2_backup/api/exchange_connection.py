"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exchange connection module for Schwabot API integration.
"""

import logging
import time
from typing import Any, Dict, Optional

from .data_models import APICredentials, MarketData, OrderRequest, OrderResponse
from .enums import ConnectionStatus

logger = logging.getLogger(__name__)

# Try to import CCXT
    try:
    import ccxt
    import ccxt.async_support as ccxt_async

    CCXT_AVAILABLE = True
        except ImportError:
        CCXT_AVAILABLE = False
        logger.warning("CCXT library not available. Exchange functionality will be limited.")


            class ExchangeConnection:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Manages connection to cryptocurrency exchanges via CCXT."""

                def __init__(self, credentials: APICredentials, config: Dict[str, Any]) -> None:
                self.credentials = credentials
                self.config = config
                self.status = ConnectionStatus.DISCONNECTED
                self.exchange = None
                self.async_exchange = None
                self.last_heartbeat = 0
                self.last_error = None
                self.reconnect_attempts = 0
                self.successful_requests = 0
                self.failed_requests = 0
                self.market_data_cache = {}
                self.cache_expiry = config.get("cache_expiry", 5)  # 5 seconds

                logger.info("Exchange connection initialized for {0}".format(credentials.exchange.value))

                    async def connect(self) -> bool:
                    """Establishes connection to the exchange."""
                        if not CCXT_AVAILABLE:
                        logger.error("CCXT library not available. Cannot connect to exchange.")
                        self.status = ConnectionStatus.ERROR
                        self.last_error = "CCXT library not installed."
                    return False

                    self.status = ConnectionStatus.CONNECTING
                    logger.info("Connecting to {0}...".format(self.credentials.exchange.value))

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
                                logger.info("Successfully connected to {0}".format(exchange_name))
                            return True

                                except Exception as e:
                                self.status = ConnectionStatus.ERROR
                                self.last_error = str(e)
                                logger.error(
                                "Failed to connect to {0}: {1}".format(self.credentials.exchange.value, e),
                                exc_info=True,
                                )
                            return False

                                async def disconnect(self):
                                """Closes the connection to the exchange."""
                                    if self.status == ConnectionStatus.DISCONNECTED:
                                return

                                logger.info("Disconnecting from {0}...".format(self.credentials.exchange.value))
                                    try:
                                        if self.async_exchange:
                                        await self.async_exchange.close()
                                        self.status = ConnectionStatus.DISCONNECTED
                                        logger.info("Disconnected from {0}".format(self.credentials.exchange.value))
                                            except Exception as e:
                                            logger.error("Error during disconnection: {0}".format(e), exc_info=True)

                                                async def get_market_data(self, symbol: str) -> Optional[MarketData]:
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
                                                    "Error fetching market data for {0} on {1}: {2}".format(symbol, self.credentials.exchange.value, e)
                                                    )
                                                return None

                                                    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
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

                                                                logger.info("Order placed on {0}: {1}".format(self.credentials.exchange.value, response.order_id))

                                                            return response

                                                                except Exception as e:
                                                                self.failed_requests += 1
                                                                self.last_error = str(e)
                                                                logger.error(
                                                                "Error placing order on {0}: {1}".format(self.credentials.exchange.value, e),
                                                                exc_info=True,
                                                                )
                                                            return self._create_error_response(order_request, str(e))

                                                                async def get_balance(self) -> Dict[str, float]:
                                                                """Fetches the account balance from the exchange."""
                                                                    if self.status != ConnectionStatus.CONNECTED:
                                                                return {}

                                                                    try:
                                                                    balance = await self.async_exchange.fetch_balance()
                                                                    free_balances = {
                                                                    currency: float(amount) for currency, amount in balance.get("free", {}).items() if float(amount) > 0
                                                                    }

                                                                    self.successful_requests += 1
                                                                    self.last_heartbeat = time.time()

                                                                return free_balances

                                                                    except Exception as e:
                                                                    self.failed_requests += 1
                                                                    self.last_error = str(e)
                                                                    logger.error("Error fetching balance from {0}: {1}".format(self.credentials.exchange.value, e))
                                                                return {}

                                                                    def _create_success_response(self, order) -> OrderResponse:
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
                                                                timestamp=order.get("timestamp", time.time() * 1000) / 1000,
                                                                fee=order.get("fee"),
                                                                info=order,
                                                                )

                                                                    def _create_error_response(self, order_request: OrderRequest, error_msg: str) -> OrderResponse:
                                                                    """Creates an error OrderResponse."""
                                                                return OrderResponse(
                                                                order_id="",
                                                                client_order_id=order_request.client_order_id,
                                                                symbol=order_request.symbol,
                                                                side=order_request.side.value,
                                                                order_type=order_request.order_type.value,
                                                                amount=order_request.amount,
                                                                price=order_request.price or 0.0,
                                                                filled=0.0,
                                                                remaining=order_request.amount,
                                                                cost=0.0,
                                                                status="error",
                                                                timestamp=time.time(),
                                                                success=False,
                                                                error_message=error_msg,
                                                                )
