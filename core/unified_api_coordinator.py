# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp

# Try to import CCXT
try:
    import ccxt
    import ccxt.async_support as ccxt_async

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Try to import AWS Secrets Manager
try:
    import boto3
    from botocore.exceptions import ClientError

    AWS_SECRETS_AVAILABLE = True
except ImportError:
    AWS_SECRETS_AVAILABLE = False

# Import core systems
try:
    from core.capital_controls import check_portfolio_limits, get_capital_controls
    from core.enhanced_risk_manager import get_enhanced_risk_manager
    from core.ops_observability import LogLevel, log_operation, record_api_request
    from core.risk_guard import check_circuit_breaker, get_risk_guard
    from core.secure_api_manager import get_secure_api_manager

    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False

# Import centralized CLI handler
from utils.safe_print import debug, error, info, safe_print, success, warn

# Initialize Unicode handler - assuming DualUnicoreHandler is correctly imported/defined elsewhere
# from dual_unicore_handler import DualUnicoreHandler
# unicore = DualUnicoreHandler()
# Temporarily commented out if not present, to prevent import errors.
# Re-enable if DualUnicoreHandler is confirmed to exist and be functional.

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Supported exchange types."""

    COINBASE = "coinbase"
    BINANCE = "binance"
    KRAKEN = "kraken"
    GEMINI = "gemini"
    POLONIEX = "poloniex"
    KUCOIN = "kucoin"
    BYBIT = "bybit"
    OKX = "okx"
    OTHER = "other"  # For data providers like CoinMarketCap/CoinGecko


class APIMethod(Enum):
    """API method enumeration."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class ConnectionStatus(Enum):
    """Connection status enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class APIEndpoint:
    """API endpoint configuration."""

    name: str
    url: str
    method: APIMethod
    rate_limit: int  # requests per minute
    timeout: float
    requires_auth: bool = False
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeConfig:
    """Exchange configuration."""

    exchange_type: ExchangeType
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # For some exchanges like Coinbase
    sandbox: bool = True
    enabled: bool = True
    rate_limit_multiplier: float = 1.0
    endpoints: Dict[str, APIEndpoint] = field(default_factory=dict)
    base_url: Optional[str] = None  # Added for generic API clients like CMC/CG
    timeout: float = 30.0  # Added for generic API clients
    retry_attempts: int = 3  # Added for generic API clients
    retry_delay: float = 1.0  # Added for generic API clients


@dataclass
class APIRequest:
    """API request container."""

    request_id: str
    endpoint: str
    method: APIMethod
    url: str
    headers: Dict[str, str]
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    timestamp: float
    exchange: str
    callback: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class APIResponse:
    """API response container."""

    request_id: str
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: float
    latency: float
    exchange: str
    success: bool
    error_message: Optional[str] = None


class RateLimiter:
    """Rate limiting implementation."""

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.requests: deque = deque()
        self.lock = threading.Lock()

    def can_make_request(self) -> bool:
        """Check if request can be made."""
        with self.lock:
            current_time = time.time()

            # Remove old requests (older than 1 minute)
            while self.requests and current_time - self.requests[0] > 60:
                self.requests.popleft()

            # Check if we're under the limit
            return len(self.requests) < self.requests_per_minute

    def record_request(self) -> None:
        """Record a request."""
        with self.lock:
            self.requests.append(time.time())


class UnifiedAPICoordinator:
    """Unified API coordination system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize API coordinator."""
        self.version = "1.0_0"
        self.config = config or self._default_config()

        # Exchange configurations
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.ccxt_exchanges: Dict[str, Any] = {}  # To store CCXT instances

        # Connection management
        self.connections: Dict[str, ConnectionStatus] = {}
        self.websocket_connections: Dict[str, Any] = {}

        # Request management
        self.request_queue: deque = deque(maxlen=self.config.get("max_queue_size", 1000))
        self.request_history: deque = deque(maxlen=self.config.get("max_history_size", 10000))
        self.pending_requests: Dict[str, APIRequest] = {}

        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0

        # Callbacks and hooks
        self.data_callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self.error_callbacks: List[Callable[[str, str], None]] = []

        # Threading and async
        self.request_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize default exchanges
        self._initialize_default_exchanges()

        logger.info(f"UnifiedAPICoordinator v{self.version} initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "max_queue_size": 1000,
            "max_history_size": 10000,
            "default_timeout": 30.0,
            "max_retries": 3,
            "retry_delay": 1.0,
            "enable_rate_limiting": True,
            "enable_websocket": True,
            "enable_rest_api": True,
            "enable_performance_monitoring": True,
            "default_rate_limit": 60,  # requests per minute
            "websocket_reconnect_delay": 5.0,
            "enable_ssl_verification": True,
            "ccxt_config": {  # Default CCXT config
                "timeout": 30000,  # 30 seconds
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                },
            },
        }

    def _initialize_default_exchanges(self) -> None:
        """Initialize default exchange configurations."""
        # Coinbase configuration (example, assuming ExchangeConfig handles CCXT params)
        coinbase_config = ExchangeConfig(
            exchange_type=ExchangeType.COINBASE,
            name="coinbase",
            sandbox=True,
            rate_limit_multiplier=1.0,
            endpoints={},  # Endpoints will be managed by CCXT
            api_key=os.getenv("COINBASE_API_KEY"),
            api_secret=os.getenv("COINBASE_API_SECRET"),
            passphrase=os.getenv("COINBASE_PASSPHRASE"),
        )
        self.register_exchange(coinbase_config)

        # Binance configuration
        binance_config = ExchangeConfig(
            exchange_type=ExchangeType.BINANCE,
            name="binance",
            sandbox=True,
            rate_limit_multiplier=1.0,
            endpoints={},  # Endpoints will be managed by CCXT
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )
        self.register_exchange(binance_config)

    def register_exchange(self, exchange_config: ExchangeConfig) -> bool:
        """Register an exchange configuration and initialize CCXT instance if available."""
        try:
            exchange_name = exchange_config.name
            self.exchanges[exchange_name] = exchange_config
            self.rate_limiters[exchange_name] = RateLimiter(
                int(exchange_config.rate_limit * exchange_config.rate_limit_multiplier)
            )
            self.connections[exchange_name] = ConnectionStatus.DISCONNECTED

            if CCXT_AVAILABLE and exchange_config.exchange_type != ExchangeType.OTHER:
                # Dynamically load CCXT exchange
                ccxt_exchange_class = getattr(ccxt, exchange_name)
                if ccxt_exchange_class:
                    ccxt_params = {
                        "apiKey": exchange_config.api_key,
                        "secret": exchange_config.api_secret,
                        "password": exchange_config.passphrase,  # For Coinbase Pro
                        "options": self.config["ccxt_config"]["options"],
                        "timeout": self.config["ccxt_config"]["timeout"],
                        "enableRateLimit": self.config["ccxt_config"]["enableRateLimit"],
                    }
                    if exchange_config.sandbox:
                        ccxt_params["options"]["defaultType"] = (
                            "future" if exchange_name == "binance" else "spot"
                        )  # Example sandbox default
                        ccxt_params["urls"] = {
                            "api": {
                                "public": f"https://testnet.binance.vision/api",  # Example for Binance testnet
                                "private": f"https://testnet.binance.vision/api",
                            },
                            "web": "https://testnet.binance.vision",
                        }  # Placeholder - CCXT usually handles sandbox URLs internally or via 'options'
                        if hasattr(ccxt_exchange_class, "sandbox"):
                            ccxt_exchange_instance = ccxt_exchange_class(ccxt_params).sandbox()
                        else:
                            ccxt_exchange_instance = ccxt_exchange_class(
                                ccxt_params
                            )  # Most CCXT exchanges handle sandbox via 'options' or direct instantiation

                    else:
                        ccxt_exchange_instance = ccxt_exchange_class(ccxt_params)

                    self.ccxt_exchanges[exchange_name] = ccxt_exchange_instance
                    self.connections[exchange_name] = ConnectionStatus.CONNECTED
                    logger.info(f"CCXT exchange {exchange_name} initialized and connected.")
                    return True
                else:
                    logger.warning(f"CCXT exchange class not found for {exchange_name}.")
                    return False
            elif exchange_config.exchange_type == ExchangeType.OTHER:
                # These are handled by specific API clients (e.g., CoinMarketCapAPI, CoinGeckoAPI)
                logger.info(f"Data provider {exchange_name} registered. No CCXT instance created.")
                return True
            else:
                logger.warning(f"CCXT not available or exchange type not supported for CCXT: {exchange_name}")
                return False

        except Exception as e:
            logger.error(f"Error registering exchange {exchange_config.name}: {e}")
            self.connections[exchange_config.name] = ConnectionStatus.ERROR
            return False

    def _load_ccxt_exchange(self, exchange_name: str) -> Optional[Any]:
        """Helper to load a CCXT exchange instance."""
        if not CCXT_AVAILABLE:
            logger.error("CCXT library is not available.")
            return None

        if exchange_name not in self.exchanges:
            logger.error(f"Exchange config for {exchange_name} not found.")
            return None

        exchange_config = self.exchanges[exchange_name]
        if exchange_config.exchange_type == ExchangeType.OTHER:
            logger.warning(f"Exchange type {exchange_name} is a data provider, not a CCXT exchange.")
            return None

        try:
            ccxt_exchange_class = getattr(ccxt, exchange_name)
            ccxt_params = {
                "apiKey": exchange_config.api_key,
                "secret": exchange_config.api_secret,
                "password": exchange_config.passphrase,
                "options": self.config["ccxt_config"]["options"],
                "timeout": self.config["ccxt_config"]["timeout"],
                "enableRateLimit": self.config["ccxt_config"]["enableRateLimit"],
            }
            if exchange_config.sandbox:
                if hasattr(ccxt_exchange_class, "sandbox"):
                    ccxt_exchange_instance = ccxt_exchange_class(ccxt_params).sandbox()
                else:
                    ccxt_exchange_instance = ccxt_exchange_class(ccxt_params)
            else:
                ccxt_exchange_instance = ccxt_exchange_class(ccxt_params)

            self.ccxt_exchanges[exchange_name] = ccxt_exchange_instance
            self.connections[exchange_name] = ConnectionStatus.CONNECTED
            logger.info(f"CCXT exchange {exchange_name} loaded and connected.")
            return ccxt_exchange_instance
        except Exception as e:
            logger.error(f"Error loading CCXT exchange {exchange_name}: {e}")
            self.connections[exchange_name] = ConnectionStatus.ERROR
            return None

    def get_exchange_instance(self, exchange_name: str) -> Optional[Any]:
        """Get a CCXT exchange instance by name."""
        return self.ccxt_exchanges.get(exchange_name)

    async def fetch_ticker(self, exchange_name: str, symbol: str) -> MarketData:
        """Fetch ticker data using CCXT."""
        exchange = self.get_exchange_instance(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found or not initialized.")

        try:
            # CCXT expects 'SYMBOL/QUOTE' format (e.g., 'BTC/USDT')
            ticker_data = await exchange.fetch_ticker(symbol)
            return MarketData(
                symbol=ticker_data["symbol"],
                timestamp=ticker_data["timestamp"],
                data_type=DataType.TICKER,
                price=float(ticker_data["last"]),
                volume_24h=float(ticker_data.get("quoteVolume", 0.0)),
            )
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} on {exchange_name}: {e}")
            return MarketData(symbol=symbol, timestamp=0, data_type=DataType.TICKER, price=0.0, volume_24h=0.0)

    async def place_order(
        self, exchange_name: str, order_request: Any
    ) -> Any:  # Returns Any for now, will be OrderResponse
        """Place an order using CCXT."""
        exchange = self.get_exchange_instance(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found or not initialized.")

        # Map our internal OrderRequest to CCXT params
        try:
            order = await exchange.create_order(
                order_request.symbol,
                order_request.type.value,  # 'market', 'limit'
                order_request.side.value,  # 'buy', 'sell'
                float(order_request.amount),
                float(order_request.price) if order_request.type == OrderType.LIMIT else None,
                params={},  # Additional params like stopLoss, takeProfit if supported by exchange
            )
            return OrderResponse(
                order_id=order["id"],
                client_order_id=order.get("clientOrderId"),
                symbol=order["symbol"],
                status=order["status"],  # 'open', 'closed', 'canceled'
                side=order["side"],
                type=order["type"],
                price=float(order.get("price", 0.0)),
                amount=float(order.get("amount", 0.0)),
                filled=float(order.get("filled", 0.0)),
                remaining=float(order.get("remaining", 0.0)),
                cost=float(order.get("cost", 0.0)),
                timestamp=order["timestamp"],
                datetime=order["datetime"],
                last_trade_timestamp=order.get("lastTradeTimestamp"),
                fee=order.get("fee"),
                info=order.get("info", {}),
                success=True,
            )
        except Exception as e:
            logger.error(f"Error placing order on {exchange_name} for {order_request.symbol}: {e}")
            return OrderResponse(
                order_id=None,
                symbol=order_request.symbol,
                status="failed",
                side=order_request.side.value,
                type=order_request.type.value,
                price=0.0,
                amount=0.0,
                filled=0.0,
                remaining=0.0,
                cost=0.0,
                timestamp=int(time.time() * 1000),
                datetime=datetime.now().isoformat(),
                success=False,
                error_message=str(e),
            )

    async def fetch_balance(
        self, exchange_name: str, asset: Optional[str] = None
    ) -> Dict[str, Any]:  # Returns Dict[str, Balance]
        """Fetch account balance using CCXT."""
        exchange = self.get_exchange_instance(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found or not initialized.")

        try:
            balances = await exchange.fetch_balance()
            result_balances = {}
            if asset:
                if asset in balances["total"]:
                    result_balances[asset] = Balance(
                        asset=asset,
                        total=float(balances["total"][asset]),
                        free=float(balances["free"][asset]),
                        used=float(balances["used"][asset]),
                    )
            else:
                for curr, bal_data in balances["total"].items():
                    result_balances[curr] = Balance(
                        asset=curr,
                        total=float(bal_data),
                        free=float(balances["free"].get(curr, 0.0)),
                        used=float(balances["used"].get(curr, 0.0)),
                    )
            return result_balances
        except Exception as e:
            logger.error(f"Error fetching balance on {exchange_name}: {e}")
            return {}

    # Placeholder for other CCXT operations (fetch_order_status, cancel_order, etc.)
    async def fetch_order_status(
        self, exchange_name: str, order_id: str, symbol: Optional[str] = None
    ) -> Any:  # Returns OrderResponse
        exchange = self.get_exchange_instance(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found or not initialized.")
        try:
            order_status = await exchange.fetch_order(order_id, symbol)
            return OrderResponse(
                order_id=order_status["id"],
                client_order_id=order_status.get("clientOrderId"),
                symbol=order_status["symbol"],
                status=order_status["status"],
                side=order_status["side"],
                type=order_status["type"],
                price=float(order_status.get("price", 0.0)),
                amount=float(order_status.get("amount", 0.0)),
                filled=float(order_status.get("filled", 0.0)),
                remaining=float(order_status.get("remaining", 0.0)),
                cost=float(order_status.get("cost", 0.0)),
                timestamp=order_status["timestamp"],
                datetime=order_status["datetime"],
                last_trade_timestamp=order_status.get("lastTradeTimestamp"),
                fee=order_status.get("fee"),
                info=order_status.get("info", {}),
                success=True,
            )
        except Exception as e:
            logger.error(f"Error fetching order status {order_id} on {exchange_name}: {e}")
            return OrderResponse(order_id=order_id, symbol=symbol, status="failed", success=False, error_message=str(e))

    async def cancel_order(self, exchange_name: str, order_id: str, symbol: Optional[str] = None) -> bool:
        exchange = self.get_exchange_instance(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found or not initialized.")
        try:
            await exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} on {exchange_name} cancelled successfully.")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} on {exchange_name}: {e}")
            return False

    async def get_all_tickers(self, exchange_name: str) -> List[MarketData]:
        exchange = self.get_exchange_instance(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found or not initialized.")
        try:
            tickers = await exchange.fetch_tickers()
            market_data_list = []
            for symbol, ticker_data in tickers.items():
                market_data_list.append(
                    MarketData(
                        symbol=ticker_data["symbol"],
                        timestamp=ticker_data["timestamp"],
                        data_type=DataType.TICKER,
                        price=float(ticker_data["last"]),
                        volume_24h=float(ticker_data.get("quoteVolume", 0.0)),
                    )
                )
            return market_data_list
        except Exception as e:
            logger.error(f"Error fetching all tickers on {exchange_name}: {e}")
            return []

    # Other utility methods from previous implementation (if any) should be here
    # For instance, start/stop for async operations, request handling loop etc.

    async def start(self) -> None:
        """Start the API coordinator and its background tasks."""
        if not self.is_running:
            self.is_running = True
            self.session = aiohttp.ClientSession()
            # Start request processing loop in a separate task/thread if needed
            logger.info("UnifiedAPICoordinator started.")

    async def stop(self) -> None:
        """Stop the API coordinator and clean up resources."""
        if self.is_running:
            self.is_running = False
            if self.session:
                await self.session.close()
            logger.info("UnifiedAPICoordinator stopped.")
