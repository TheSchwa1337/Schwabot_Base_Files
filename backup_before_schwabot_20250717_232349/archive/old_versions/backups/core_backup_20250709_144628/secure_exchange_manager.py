"""Module for Schwabot trading system."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

    try:
    import ccxt

    from utils.secure_config_manager import SecureConfigManager
    FillHandler,
    create_fill_handler,
    FillEvent,
    OrderState,
    )

    CCXT_AVAILABLE = True
        except ImportError:
        ccxt = None
        SecureConfigManager = None
        FillHandler = None
        create_fill_handler = None
        FillEvent = None
        OrderState = None
        CCXT_AVAILABLE = False

        #!/usr/bin/env python3
        """
        Secure Exchange Manager - Professional API Key & Exchange Integration

            Provides secure, properly labeled exchange integration with:
            - Environment variable support for secrets
            - Encrypted local storage fallback
            - Clear distinction between public/private keys
            - Validation and connectivity testing
            - Comprehensive logging without exposing secrets
            - Integration with automated trading pipeline
            - Advanced fill handling for partial fills and retries

                Security Features:
                - Never logs or displays actual secret keys
                - Validates keys before allowing trading
                - Supports multiple exchanges with proper isolation
                - Environment variable priority over local storage
                - Encrypted local storage for development/testing
                - Advanced fill management for crypto trading reliability

                    Usage:
                    # Environment variables (recommended for, production)
                    export BINANCE_API_KEY="your_public_api_key"
                    export BINANCE_API_SECRET="your_secret_key"

                    # Or use secure storage
                    exchange_manager = SecureExchangeManager()
                    exchange_manager.setup_exchange("binance", api_key="...", secret="...")
                    """

                    # CCXT for exchange integration
                        try:
                        CCXT_AVAILABLE = True
                            except ImportError:
                            CCXT_AVAILABLE = False
                            logging.warning("CCXT not available. Install with: pip install ccxt")

                            # Local secure storage
                                try:
                                SECURE_STORAGE_AVAILABLE = True
                                    except ImportError:
                                    SECURE_STORAGE_AVAILABLE = False
                                    logging.warning("Secure storage not available. Using environment variables only.")

                                    # Fill handler for advanced crypto trading
                                        try:
                                        FILL_HANDLER_AVAILABLE = True
                                            except ImportError:
                                            FILL_HANDLER_AVAILABLE = False
                                            logging.warning()
                                            "Fill handler not available. Install with: pip install -r requirements.txt"
                                            )

                                            logger = logging.getLogger(__name__)


                                                class ExchangeType(Enum):
    """Class for Schwabot trading functionality."""
                                                """Class for Schwabot trading functionality."""
                                                """Supported exchange types with proper labeling."""

                                                BINANCE = "binance"
                                                COINBASE = "coinbase"
                                                KRAKEN = "kraken"
                                                KUCOIN = "kucoin"
                                                OKX = "okx"


                                                @ dataclass
                                                    class ExchangeCredentials:
    """Class for Schwabot trading functionality."""
                                                    """Class for Schwabot trading functionality."""
                                                    """Securely stored exchange credentials with clear labeling."""

                                                    exchange: ExchangeType
                                                    api_key: str  # PUBLIC API KEY (can be logged, safely)
                                                    secret: str  # SECRET KEY (never, logged)
                                                passphrase: Optional[str] = None  # Additional secret for some exchanges
                                                sandbox: bool = True
                                                testnet: bool = True

                                                    def __post_init__(self) -> None:
                                                    """Validate credentials after initialization."""
                                                        if not self.api_key or not self.secret:
                                                    raise ValueError()
                                                    "API key and secret are required for {0}".format(self.exchange.value)
                                                    )

                                                    # Log only the public key (safe to, display)
                                                    logger.info()
                                                    "✅ Configured {0} with API key: {1}...".format()
                                                    self.exchange.value, self.api_key[:8]
                                                    )
                                                    )
                                                    logger.info("🔐 Secret key configured (length: {0})".format(len(self.secret)))

                                                        if self.passphrase:
                                                        logger.info()
                                                        "🔐 Passphrase configured (length: {0})".format(len(self.passphrase))
                                                        )


                                                        @ dataclass
                                                            class ExchangeStatus:
    """Class for Schwabot trading functionality."""
                                                            """Class for Schwabot trading functionality."""
                                                            """Exchange connection and trading status."""

                                                            exchange: ExchangeType
                                                            connected: bool = False
                                                            authenticated: bool = False
                                                            trading_enabled: bool = False
                                                            balance_available: bool = False
                                                            last_check: Optional[float] = None
                                                            error_message: Optional[str] = None

                                                                def __str__(self) -> str:
                                                                """Safe string representation without sensitive data."""
                                                                status = "🟢" if self.connected else "🔴"
                                                            return "{0} {1}: Connected={2}, Trading={3}".format()
                                                            status, self.exchange.value, self.connected, self.trading_enabled
                                                            )


                                                            @ dataclass
                                                                class TradeResult:
    """Class for Schwabot trading functionality."""
                                                                """Class for Schwabot trading functionality."""
                                                                """Result of a trade execution with fill handling."""

                                                                success: bool
                                                                order_id: Optional[str] = None
                                                                fill_events: List[FillEvent] = field(default_factory=list)
                                                                total_filled: float = 0.0
                                                                average_price: float = 0.0
                                                                total_fee: float = 0.0
                                                                error_message: Optional[str] = None
                                                                retry_count: int = 0
                                                                partial_fills: bool = False


                                                                    class SecureExchangeManager:
    """Class for Schwabot trading functionality."""
                                                                    """Class for Schwabot trading functionality."""
                                                                    """
                                                                    Secure exchange manager with proper key handling and validation.

                                                                        Security Features:
                                                                        - Environment variable priority
                                                                        - Encrypted local storage fallback
                                                                        - Never logs secrets
                                                                        - Validates connectivity before trading
                                                                        - Clear labeling of public vs private keys
                                                                        - Advanced fill handling for crypto trading reliability
                                                                        """

                                                                            def __init__(self, config_path: Optional[str]= None) -> None:
                                                                            """Initialize secure exchange manager."""
                                                                            self.exchanges: Dict[ExchangeType, ExchangeCredentials] = {}
                                                                            self.status: Dict[ExchangeType, ExchangeStatus] = {}
                                                                            self.ccxt_instances: Dict[ExchangeType, Any] = {}

                                                                            # Initialize fill handler for advanced crypto trading
                                                                            self.fill_handler: Optional[FillHandler] = None
                                                                                if FILL_HANDLER_AVAILABLE:
                                                                                self.fill_handler = None  # Will be initialized when needed
                                                                                logger.info("🔧 Fill handler available for advanced crypto trading")
                                                                                    else:
                                                                                    logger.warning("⚠️ Fill handler not available - basic trading only")

                                                                                    # Initialize secure storage if available
                                                                                        if SECURE_STORAGE_AVAILABLE:
                                                                                        self.secure_config = SecureConfigManager()
                                                                                            else:
                                                                                            self.secure_config = None
                                                                                            logger.warning()
                                                                                            "🔐 Secure storage not available. Using environment variables only."
                                                                                            )

                                                                                            # Load configuration
                                                                                            self._load_exchange_configs()

                                                                                            logger.info()
                                                                                            "🔐 Secure Exchange Manager initialized with advanced fill handling"
                                                                                            )

                                                                                                async def _initialize_fill_handler(self):
                                                                                                """Initialize the fill handler if available."""
                                                                                                    if FILL_HANDLER_AVAILABLE and self.fill_handler is None:
                                                                                                        try:
                                                                                                        self.fill_handler = await create_fill_handler()
                                                                                                        {}
                                                                                                        "retry_config": {}
                                                                                                        "max_retries": 3,
                                                                                                        "base_delay": 1.0,
                                                                                                        "max_delay": 30.0,
                                                                                                        "exponential_base": 2.0,
                                                                                                        "jitter_factor": 0.1,
                                                                                                        }
                                                                                                        }
                                                                                                        )
                                                                                                        logger.info("✅ Fill handler initialized for advanced crypto trading")
                                                                                                            except Exception as e:
                                                                                                            logger.error("❌ Failed to initialize fill handler: {0}".format(e))
                                                                                                            self.fill_handler = None

                                                                                                                def _load_exchange_configs(self) -> None:
                                                                                                                """Load exchange configurations from environment variables and secure storage."""
                                                                                                                logger.info("🔍 Loading exchange configurations...")

                                                                                                                    for exchange in ExchangeType:
                                                                                                                        try:
                                                                                                                        # Try environment variables first (most, secure)
                                                                                                                        env_credentials = self._load_from_environment(exchange)
                                                                                                                            if env_credentials:
                                                                                                                            self.exchanges[exchange] = env_credentials
                                                                                                                            self.status[exchange] = ExchangeStatus(exchange=exchange)
                                                                                                                            logger.info()
                                                                                                                            "✅ Loaded {0} from environment variables".format()
                                                                                                                            exchange.value
                                                                                                                            )
                                                                                                                            )
                                                                                                                        continue

                                                                                                                        # Try secure storage as fallback
                                                                                                                            if self.secure_config:
                                                                                                                            secure_credentials = self._load_from_secure_storage(exchange)
                                                                                                                                if secure_credentials:
                                                                                                                                self.exchanges[exchange] = secure_credentials
                                                                                                                                self.status[exchange] = ExchangeStatus(exchange=exchange)
                                                                                                                                logger.info()
                                                                                                                                "✅ Loaded {0} from secure storage".format(exchange.value)
                                                                                                                                )
                                                                                                                            continue

                                                                                                                            logger.info("⚠️ No credentials found for {0}".format(exchange.value))

                                                                                                                                except Exception as e:
                                                                                                                                logger.error("❌ Error loading {0}: {1}".format(exchange.value, e))

                                                                                                                                def _load_from_environment()
                                                                                                                                self, exchange: ExchangeType
                                                                                                                                    ) -> Optional[ExchangeCredentials]:
                                                                                                                                    """Load credentials from environment variables."""
                                                                                                                                    exchange_name = exchange.value.upper()

                                                                                                                                    # Environment variable naming convention
                                                                                                                                    api_key = os.environ.get("{0}_API_KEY".format(exchange_name))
                                                                                                                                    secret = os.environ.get("{0}_API_SECRET".format(exchange_name))
                                                                                                                                passphrase = os.environ.get("{0}_PASSPHRASE".format(exchange_name))

                                                                                                                                    if api_key and secret:
                                                                                                                                    logger.info()
                                                                                                                                    "🔍 Found {0} credentials in environment variables".format()
                                                                                                                                    exchange.value
                                                                                                                                    )
                                                                                                                                    )
                                                                                                                                return ExchangeCredentials()
                                                                                                                                exchange = exchange,
                                                                                                                                api_key = api_key,
                                                                                                                                secret = secret,
                                                                                                                            passphrase = passphrase,
                                                                                                                            sandbox = True,  # Default to sandbox for safety
                                                                                                                            testnet = True,
                                                                                                                            )

                                                                                                                        return None

                                                                                                                        def _load_from_secure_storage()
                                                                                                                        self, exchange: ExchangeType
                                                                                                                            ) -> Optional[ExchangeCredentials]:
                                                                                                                            """Load credentials from secure storage."""
                                                                                                                                if not self.secure_config:
                                                                                                                            return None

                                                                                                                                try:
                                                                                                                                exchange_name = exchange.value
                                                                                                                                api_key = self.secure_config.get_api_key()
                                                                                                                                "{0}_api_key".format(exchange_name)
                                                                                                                                )
                                                                                                                                secret = self.secure_config.get_api_key("{0}_secret".format(exchange_name))
                                                                                                                            passphrase = self.secure_config.get_api_key()
                                                                                                                            "{0}_passphrase".format(exchange_name)
                                                                                                                            )

                                                                                                                                if api_key and secret:
                                                                                                                                logger.info()
                                                                                                                                "🔍 Found {0} credentials in secure storage".format(exchange.value)
                                                                                                                                )
                                                                                                                            return ExchangeCredentials()
                                                                                                                            exchange = exchange,
                                                                                                                            api_key = api_key,
                                                                                                                            secret = secret,
                                                                                                                        passphrase = passphrase,
                                                                                                                        sandbox = True,
                                                                                                                        testnet = True,
                                                                                                                        )
                                                                                                                            except Exception as e:
                                                                                                                            logger.error("❌ Error loading from secure storage: {0}".format(e))

                                                                                                                        return None

                                                                                                                        def setup_exchange()
                                                                                                                        self,
                                                                                                                        exchange: ExchangeType,
                                                                                                                        api_key: str,
                                                                                                                        secret: str,
                                                                                                                    passphrase: Optional[str] = None,
                                                                                                                    sandbox: bool = True,
                                                                                                                        ) -> bool:
                                                                                                                        """Setup exchange with credentials."""
                                                                                                                            try:
                                                                                                                            credentials = ExchangeCredentials()
                                                                                                                            exchange = exchange,
                                                                                                                            api_key = api_key,
                                                                                                                            secret = secret,
                                                                                                                        passphrase = passphrase,
                                                                                                                        sandbox = sandbox,
                                                                                                                        testnet = sandbox,
                                                                                                                        )

                                                                                                                        self.exchanges[exchange] = credentials
                                                                                                                        self.status[exchange] = ExchangeStatus(exchange=exchange)

                                                                                                                        # Test connection
                                                                                                                            if self._test_exchange_connection(exchange):
                                                                                                                            logger.info("✅ {0} setup successful".format(exchange.value))
                                                                                                                        return True
                                                                                                                            else:
                                                                                                                            logger.error("❌ {0} connection test failed".format(exchange.value))
                                                                                                                        return False

                                                                                                                            except Exception as e:
                                                                                                                            logger.error("❌ Error setting up {0}: {1}".format(exchange.value, e))
                                                                                                                        return False

                                                                                                                            def _test_exchange_connection(self, exchange: ExchangeType) -> bool:
                                                                                                                            """Test exchange connection and authentication."""
                                                                                                                                if not CCXT_AVAILABLE:
                                                                                                                                logger.error("❌ CCXT not available for connection testing")
                                                                                                                            return False

                                                                                                                                try:
                                                                                                                                credentials = self.exchanges.get(exchange)
                                                                                                                                    if not credentials:
                                                                                                                                    logger.error("❌ No credentials found for {0}".format(exchange.value))
                                                                                                                                return False

                                                                                                                                # Create CCXT instance
                                                                                                                                exchange_class = getattr(ccxt, exchange.value)
                                                                                                                                ccxt_instance = exchange_class()
                                                                                                                                {}
                                                                                                                                "apiKey": credentials.api_key,
                                                                                                                                "secret": credentials.secret,
                                                                                                                                "passphrase": credentials.passphrase,
                                                                                                                                "sandbox": credentials.sandbox,
                                                                                                                                "testnet": credentials.testnet,
                                                                                                                                "enableRateLimit": True,
                                                                                                                                "options": {"defaultType": "spot"},
                                                                                                                                }
                                                                                                                                )

                                                                                                                                # Test connection
                                                                                                                                ccxt_instance.load_markets()

                                                                                                                                # Test authentication (try to fetch, balance)
                                                                                                                                    try:
                                                                                                                                    balance = ccxt_instance.fetch_balance()
                                                                                                                                    self.status[exchange].authenticated = True
                                                                                                                                    self.status[exchange].balance_available = True
                                                                                                                                    logger.info("✅ {0} authentication successful".format(exchange.value))
                                                                                                                                        except Exception as auth_error:
                                                                                                                                        logger.warning()
                                                                                                                                        "⚠️ {0} authentication failed: {1}".format()
                                                                                                                                        exchange.value, auth_error
                                                                                                                                        )
                                                                                                                                        )
                                                                                                                                        self.status[exchange].authenticated = False

                                                                                                                                        self.status[exchange].connected = True
                                                                                                                                        self.status[exchange].trading_enabled = True
                                                                                                                                        self.status[exchange].last_check = asyncio.get_event_loop().time()
                                                                                                                                        self.ccxt_instances[exchange] = ccxt_instance

                                                                                                                                    return True

                                                                                                                                        except Exception as e:
                                                                                                                                        self.status[exchange].connected = False
                                                                                                                                        self.status[exchange].error_message = str(e)
                                                                                                                                        logger.error("❌ {0} connection test failed: {1}".format(exchange.value, e))
                                                                                                                                    return False

                                                                                                                                        def get_exchange_status(self) -> Dict[str, ExchangeStatus]:
                                                                                                                                        """Get status of all configured exchanges."""
                                                                                                                                    return {exchange.value: status for exchange, status in self.status.items()}

                                                                                                                                        def get_available_exchanges(self) -> List[ExchangeType]:
                                                                                                                                        """Get list of available exchanges."""
                                                                                                                                    return []
                                                                                                                                    exchange
                                                                                                                                    for exchange in self.exchanges.keys()
                                                                                                                                    if self.status[exchange].connected
                                                                                                                                    ]

                                                                                                                                    async def execute_trade()
                                                                                                                                    self,
                                                                                                                                    exchange: ExchangeType,
                                                                                                                                    symbol: str,
                                                                                                                                    side: str,
                                                                                                                                    amount: float,
                                                                                                                                    order_type: str = "market",
                                                                                                                                        ) -> TradeResult:
                                                                                                                                        """Execute a trade with advanced fill handling."""
                                                                                                                                            try:
                                                                                                                                            # Initialize fill handler if needed
                                                                                                                                            await self._initialize_fill_handler()

                                                                                                                                            # Validate exchange
                                                                                                                                                if not self.status[exchange].connected:
                                                                                                                                            return TradeResult()
                                                                                                                                            success = False,
                                                                                                                                            error_message = "Exchange {0} not connected".format(exchange.value),
                                                                                                                                            )

                                                                                                                                                if not self.status[exchange].authenticated:
                                                                                                                                            return TradeResult()
                                                                                                                                            success = False,
                                                                                                                                            error_message = "Exchange {0} not authenticated".format()
                                                                                                                                            exchange.value
                                                                                                                                            ),
                                                                                                                                            )

                                                                                                                                            # Get CCXT instance
                                                                                                                                            ccxt_instance = self.ccxt_instances.get(exchange)
                                                                                                                                                if not ccxt_instance:
                                                                                                                                            return TradeResult()
                                                                                                                                            success = False,
                                                                                                                                            error_message = "CCXT instance not available for {0}".format()
                                                                                                                                            exchange.value
                                                                                                                                            ),
                                                                                                                                            )

                                                                                                                                            # Execute order
                                                                                                                                            logger.info()
                                                                                                                                            "🚀 Executing {0} {1} {2} on {3}".format()
                                                                                                                                            side, amount, symbol, exchange.value
                                                                                                                                            )
                                                                                                                                            )

                                                                                                                                            order_params = {}
                                                                                                                                            "symbol": symbol,
                                                                                                                                            "type": order_type,
                                                                                                                                            "side": side,
                                                                                                                                            "amount": amount,
                                                                                                                                            }

                                                                                                                                            # Add exchange-specific parameters
                                                                                                                                                if exchange == ExchangeType.BINANCE:
                                                                                                                                                order_params["newOrderRespType"] = ()
                                                                                                                                                "FULL"  # Get detailed response with fills
                                                                                                                                                )

                                                                                                                                                order = await ccxt_instance.create_order(**order_params)

                                                                                                                                                # Process with fill handler if available
                                                                                                                                                    if self.fill_handler and order:
                                                                                                                                                return await self._process_order_with_fill_handler()
                                                                                                                                                order, exchange, symbol
                                                                                                                                                )
                                                                                                                                                    else:
                                                                                                                                                    # Basic processing without fill handler
                                                                                                                                                return self._process_basic_order(order)

                                                                                                                                                    except Exception as e:
                                                                                                                                                    logger.error("❌ Trade execution failed: {0}".format(e))
                                                                                                                                                return TradeResult(success=False, error_message=str(e))

                                                                                                                                                async def _process_order_with_fill_handler()
                                                                                                                                                self, order: Dict[str, Any], exchange: ExchangeType, symbol: str
                                                                                                                                                    ) -> TradeResult:
                                                                                                                                                    """Process order with advanced fill handling."""
                                                                                                                                                        try:
                                                                                                                                                        order_id = order.get("id", "")
                                                                                                                                                        status = order.get("status", "").lower()

                                                                                                                                                        # Initialize order state in fill handler
                                                                                                                                                            if self.fill_handler:
                                                                                                                                                            # Create initial order state
                                                                                                                                                            order_state = OrderState()
                                                                                                                                                            order_id = order_id,
                                                                                                                                                            symbol = symbol,
                                                                                                                                                            side = order.get("side", ""),
                                                                                                                                                            order_type = order.get("type", ""),
                                                                                                                                                            original_amount = float(order.get("amount", 0)),
                                                                                                                                                            )
                                                                                                                                                            self.fill_handler.active_orders[order_id] = order_state

                                                                                                                                                            # Process fills if present
                                                                                                                                                            fill_events = []
                                                                                                                                                                if "fills" in order and order["fills"]:
                                                                                                                                                                    for fill_data in order["fills"]:
                                                                                                                                                                        try:
                                                                                                                                                                        fill_event = await self.fill_handler.process_fill_event()
                                                                                                                                                                        {}
                                                                                                                                                                        "orderId": order_id,
                                                                                                                                                                        "symbol": symbol,
                                                                                                                                                                        "side": order.get("side", ""),
                                                                                                                                                                        **fill_data,
                                                                                                                                                                        }
                                                                                                                                                                        )
                                                                                                                                                                        fill_events.append(fill_event)
                                                                                                                                                                            except Exception as e:
                                                                                                                                                                            logger.error("Error processing fill: {0}".format(e))

                                                                                                                                                                            # Calculate totals
                                                                                                                                                                            total_filled = sum(float(fill.amount) for fill in fill_events)
                                                                                                                                                                            total_fee = sum(float(fill.fee) for fill in fill_events)
                                                                                                                                                                            average_price = 0.0
                                                                                                                                                                                if total_filled > 0:
                                                                                                                                                                                total_cost = sum()
                                                                                                                                                                                float(fill.amount * fill.price) for fill in fill_events
                                                                                                                                                                                )
                                                                                                                                                                                average_price = total_cost / total_filled

                                                                                                                                                                                # Check for partial fills
                                                                                                                                                                                original_amount = float(order.get("amount", 0))
                                                                                                                                                                                partial_fills = total_filled < original_amount

                                                                                                                                                                            return TradeResult()
                                                                                                                                                                            success = status in ["filled", "closed", "partial"],
                                                                                                                                                                            order_id = order_id,
                                                                                                                                                                            fill_events = fill_events,
                                                                                                                                                                            total_filled = total_filled,
                                                                                                                                                                            average_price = average_price,
                                                                                                                                                                            total_fee = total_fee,
                                                                                                                                                                            partial_fills = partial_fills,
                                                                                                                                                                            )

                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                logger.error("Error processing order with fill handler: {0}".format(e))
                                                                                                                                                                            return TradeResult(success=False, error_message=str(e))

                                                                                                                                                                                def _process_basic_order(self, order: Dict[str, Any]) -> TradeResult:
                                                                                                                                                                                """Process order without fill handler (basic, mode)."""
                                                                                                                                                                                    try:
                                                                                                                                                                                    order_id = order.get("id", "")
                                                                                                                                                                                    status = order.get("status", "").lower()

                                                                                                                                                                                    # Basic fill processing
                                                                                                                                                                                    fill_events = []
                                                                                                                                                                                        if "fills" in order and order["fills"]:
                                                                                                                                                                                            for fill_data in order["fills"]:
                                                                                                                                                                                            # Create basic fill event
                                                                                                                                                                                            fill_event = FillEvent()
                                                                                                                                                                                            order_id = order_id,
                                                                                                                                                                                            trade_id = fill_data.get("tradeId", ""),
                                                                                                                                                                                            symbol = order.get("symbol", ""),
                                                                                                                                                                                            side = order.get("side", ""),
                                                                                                                                                                                            amount = float(fill_data.get("qty", 0)),
                                                                                                                                                                                            price = float(fill_data.get("price", 0)),
                                                                                                                                                                                            fee = float(fill_data.get("commission", 0)),
                                                                                                                                                                                            fee_currency = fill_data.get("commissionAsset", ""),
                                                                                                                                                                                            timestamp = int(asyncio.get_event_loop().time() * 1000),
                                                                                                                                                                                            )
                                                                                                                                                                                            fill_events.append(fill_event)

                                                                                                                                                                                            total_filled = sum(float(fill.amount) for fill in fill_events)
                                                                                                                                                                                            total_fee = sum(float(fill.fee) for fill in fill_events)
                                                                                                                                                                                            average_price = 0.0
                                                                                                                                                                                                if total_filled > 0:
                                                                                                                                                                                                total_cost = sum()
                                                                                                                                                                                                float(fill.amount * fill.price) for fill in fill_events
                                                                                                                                                                                                )
                                                                                                                                                                                                average_price = total_cost / total_filled

                                                                                                                                                                                            return TradeResult()
                                                                                                                                                                                            success = status in ["filled", "closed", "partial"],
                                                                                                                                                                                            order_id = order_id,
                                                                                                                                                                                            fill_events = fill_events,
                                                                                                                                                                                            total_filled = total_filled,
                                                                                                                                                                                            average_price = average_price,
                                                                                                                                                                                            total_fee = total_fee,
                                                                                                                                                                                            partial_fills = total_filled < float(order.get("amount", 0)),
                                                                                                                                                                                            )

                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                logger.error("Error processing basic order: {0}".format(e))
                                                                                                                                                                                            return TradeResult(success=False, error_message=str(e))

                                                                                                                                                                                            async def handle_partial_fill()
                                                                                                                                                                                            self, order_id: str, fill_data: Dict[str, Any]
                                                                                                                                                                                                ) -> Dict[str, Any]:
                                                                                                                                                                                                """Handle partial fill scenario with retry logic."""
                                                                                                                                                                                                    if not self.fill_handler:
                                                                                                                                                                                                return {"status": "error", "message": "Fill handler not available"}

                                                                                                                                                                                                    try:
                                                                                                                                                                                                return await self.fill_handler.handle_partial_fill(order_id, fill_data)
                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                    logger.error("Error handling partial fill: {0}".format(e))
                                                                                                                                                                                                return {"status": "error", "message": str(e)}

                                                                                                                                                                                                def get_balance()
                                                                                                                                                                                                self, exchange: ExchangeType, currency: str = None
                                                                                                                                                                                                    ) -> Dict[str, Any]:
                                                                                                                                                                                                    """Get account balance."""
                                                                                                                                                                                                        try:
                                                                                                                                                                                                            if not self.status[exchange].connected:
                                                                                                                                                                                                        return {"error": "Exchange {0} not connected".format(exchange.value)}

                                                                                                                                                                                                        ccxt_instance = self.ccxt_instances.get(exchange)
                                                                                                                                                                                                            if not ccxt_instance:
                                                                                                                                                                                                        return {}
                                                                                                                                                                                                        "error": "CCXT instance not available for {0}".format()
                                                                                                                                                                                                        exchange.value
                                                                                                                                                                                                        )
                                                                                                                                                                                                        }

                                                                                                                                                                                                        balance = ccxt_instance.fetch_balance()

                                                                                                                                                                                                            if currency:
                                                                                                                                                                                                        return {}
                                                                                                                                                                                                        "currency": currency,
                                                                                                                                                                                                        "free": balance.get(currency, {}).get("free", 0),
                                                                                                                                                                                                        "used": balance.get(currency, {}).get("used", 0),
                                                                                                                                                                                                        "total": balance.get(currency, {}).get("total", 0),
                                                                                                                                                                                                        }
                                                                                                                                                                                                            else:
                                                                                                                                                                                                        return balance

                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                            logger.error("❌ Error fetching balance: {0}".format(e))
                                                                                                                                                                                                        return {"error": str(e)}

                                                                                                                                                                                                            def validate_trading_ready(self) -> Tuple[bool, List[str]]:
                                                                                                                                                                                                            """Validate that trading is ready."""
                                                                                                                                                                                                            errors = []

                                                                                                                                                                                                                if not CCXT_AVAILABLE:
                                                                                                                                                                                                                errors.append("CCXT library not available")

                                                                                                                                                                                                                    if not self.exchanges:
                                                                                                                                                                                                                    errors.append("No exchanges configured")

                                                                                                                                                                                                                        for exchange, status in self.status.items():
                                                                                                                                                                                                                            if not status.connected:
                                                                                                                                                                                                                            errors.append("{0} not connected".format(exchange.value))
                                                                                                                                                                                                                                elif not status.authenticated:
                                                                                                                                                                                                                                errors.append("{0} not authenticated".format(exchange.value))

                                                                                                                                                                                                                                    if not FILL_HANDLER_AVAILABLE:
                                                                                                                                                                                                                                    errors.append("Fill handler not available - limited functionality")

                                                                                                                                                                                                                                return len(errors) == 0, errors

                                                                                                                                                                                                                                    def get_secure_summary(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                    """Get secure summary without exposing sensitive data."""
                                                                                                                                                                                                                                    summary = {}
                                                                                                                                                                                                                                    "exchanges_configured": len(self.exchanges),
                                                                                                                                                                                                                                    "exchanges_connected": len()
                                                                                                                                                                                                                                    [s for s in self.status.values() if s.connected]
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                    "exchanges_authenticated": len()
                                                                                                                                                                                                                                    [s for s in self.status.values() if s.authenticated]
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                    "fill_handler_available": FILL_HANDLER_AVAILABLE,
                                                                                                                                                                                                                                    "secure_storage_available": SECURE_STORAGE_AVAILABLE,
                                                                                                                                                                                                                                    "ccxt_available": CCXT_AVAILABLE,
                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                    # Add fill handler statistics if available
                                                                                                                                                                                                                                        if self.fill_handler:
                                                                                                                                                                                                                                        summary["fill_statistics"] = self.fill_handler.get_fill_statistics()

                                                                                                                                                                                                                                    return summary

                                                                                                                                                                                                                                        async def get_fill_statistics(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                        """Get fill handling statistics."""
                                                                                                                                                                                                                                            if self.fill_handler:
                                                                                                                                                                                                                                        return self.fill_handler.get_fill_statistics()
                                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                                        return {"error": "Fill handler not available"}

                                                                                                                                                                                                                                            async def export_fill_state(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                            """Export fill handler state for persistence."""
                                                                                                                                                                                                                                                if self.fill_handler:
                                                                                                                                                                                                                                            return self.fill_handler.export_state()
                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                            return {"error": "Fill handler not available"}

                                                                                                                                                                                                                                                async def import_fill_state(self, state_data: Dict[str, Any]):
                                                                                                                                                                                                                                                """Import fill handler state from persistence."""
                                                                                                                                                                                                                                                    if self.fill_handler:
                                                                                                                                                                                                                                                    self.fill_handler.import_state(state_data)
                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                        logger.warning("Fill handler not available for state import")


                                                                                                                                                                                                                                                        # Convenience functions
                                                                                                                                                                                                                                                            def get_exchange_manager() -> SecureExchangeManager:
                                                                                                                                                                                                                                                            """Get the global exchange manager instance."""
                                                                                                                                                                                                                                                        return SecureExchangeManager()


                                                                                                                                                                                                                                                            def setup_exchange_from_env(exchange_name: str) -> bool:
                                                                                                                                                                                                                                                            """Setup exchange from environment variables."""
                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                exchange = ExchangeType(exchange_name.lower())
                                                                                                                                                                                                                                                                manager = get_exchange_manager()
                                                                                                                                                                                                                                                            return manager._test_exchange_connection(exchange)
                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                logger.error("Error setting up exchange {0}: {1}".format(exchange_name, e))
                                                                                                                                                                                                                                                            return False


                                                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                                                # Test the secure exchange manager
                                                                                                                                                                                                                                                                logging.basicConfig(level=logging.INFO)

                                                                                                                                                                                                                                                                manager = SecureExchangeManager()

                                                                                                                                                                                                                                                                print("\n🔐 SECURE EXCHANGE MANAGER TEST")
                                                                                                                                                                                                                                                                print("=" * 40)

                                                                                                                                                                                                                                                                # Show status
                                                                                                                                                                                                                                                                status = manager.get_secure_summary()
                                                                                                                                                                                                                                                                print("Total exchanges: {0}".format(status["exchanges_configured"]))
                                                                                                                                                                                                                                                                print("Connected: {0}".format(status["exchanges_connected"]))
                                                                                                                                                                                                                                                                print("Trading ready: {0}".format(status["exchanges_authenticated"]))

                                                                                                                                                                                                                                                                # Show individual exchange status
                                                                                                                                                                                                                                                                    for exchange_name, exchange_status in status.items():
                                                                                                                                                                                                                                                                    print("\n{0}:".format(exchange_name))
                                                                                                                                                                                                                                                                        for key, value in exchange_status.items():
                                                                                                                                                                                                                                                                        print("  {0}: {1}".format(key, value))

                                                                                                                                                                                                                                                                        # Validate trading readiness
                                                                                                                                                                                                                                                                        is_ready, issues = manager.validate_trading_ready()
                                                                                                                                                                                                                                                                        print("\nTrading ready: {0}".format(is_ready))
                                                                                                                                                                                                                                                                            if issues:
                                                                                                                                                                                                                                                                            print("Issues:")
                                                                                                                                                                                                                                                                                for issue in issues:
                                                                                                                                                                                                                                                                                print("  - {0}".format(issue))
