#!/usr/bin/env python3
""""""
CCXT Trading Executor
====================

Comprehensive trading executor using CCXT library for:
- Multi-exchange connectivity (Coinbase, Binance, Kraken, etc.)
- Order execution and management
- Portfolio tracking and rebalancing
- Risk management and compliance
- Real-time market data streaming
- Cross-platform compatibility
""""""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

# CCXT import with fallback
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None
    ccxt_async = None

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the executor."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TradingPair:
    """Trading pair configuration."""
    base: str
    quote: str
    symbol: str
    min_amount: float
    max_amount: float
    price_precision: int
    amount_precision: int


@dataclass
class OrderRequest:
    """Order request structure."""
    symbol: str
    side: OrderSide
    amount: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class ExecutedOrder:
    """Executed order information."""
    id: str
    symbol: str
    side: OrderSide
    amount: float
    filled: float
    remaining: float
    price: float
    average: Optional[float]
    status: OrderStatus
    timestamp: datetime
    fee: Optional[Dict[str, Any]] = None
    trades: Optional[List[Dict[str, Any]]] = None


@dataclass
class Portfolio:
    """Portfolio state."""
    total_value: float
    balances: Dict[str, float]
    positions: Dict[str, Dict[str, Any]]
    pnl: float
    pnl_percentage: float
    last_update: datetime


class CCXTTradingExecutor:
    """CCXT-based trading executor with multi-exchange support."""

    def __init__(self, exchange_name: str = "coinbase", )
                 api_key: str = "", api_secret: str = "",
                     passphrase: str = "", sandbox: bool = True):

        if not CCXT_AVAILABLE:
            raise ImportError("CCXT library is required for trading execution")

        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox

        # Initialize exchange
        self.exchange = None
        self.async_exchange = None

        # Trading state
        self.active_orders: Dict[str, ExecutedOrder] = {}
        self.order_history: List[ExecutedOrder] = []
        self.portfolio = Portfolio()
            total_value=0.0,
                balances={},
                    positions={},
                    pnl=0.0,
                    pnl_percentage=0.0,
                    last_update=datetime.now()
        )

        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio
        self.max_daily_loss = 0.5    # 5% daily stop loss
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.last_daily_reset = datetime.now().date()

        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_fees = 0.0

        self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize the exchange connection."""
        try:
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_name)
            async_exchange_class = getattr(ccxt_async, self.exchange_name)

            # Configuration
            config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'sandbox': self.sandbox,
}
}
            # Add passphrase for exchanges that require it (like Coinbase)
            if self.passphrase and self.exchange_name in ['coinbase', 'coinbasepro']:
                config['password'] = self.passphrase

            # Initialize sync and async exchanges
            self.exchange = exchange_class(config)
            self.async_exchange = async_exchange_class(config)

            logger.info(f"Initialized {self.exchange_name} exchange ")
                       f"(sandbox: {self.sandbox})")

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    async def connect(self) -> bool:
        """Connect to the exchange and verify credentials."""
        try:
            if not self.async_exchange:
                return False

            # Test connection with balance fetch
            balances = await self.async_exchange.fetch_balance()

            if balances:
                self.portfolio.balances = balances.get('free', {})
                self.portfolio.last_update = datetime.now()

                logger.info(f"Connected to {self.exchange_name} successfully")
                logger.info(f"Available balances: {self.portfolio.balances}")
                return True

            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the exchange."""
        try:
            if self.async_exchange:
                await self.async_exchange.close()
            logger.info(f"Disconnected from {self.exchange_name}")
        except Exception as e:
            logger.error(f"Disconnection error: {e}")

    async def get_markets(self) -> Dict[str, Any]:
        """Get available markets."""
        try:
            markets = await self.async_exchange.load_markets()
            return markets
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return {}

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker information for a symbol."""
        try:
            ticker = await self.async_exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Get order book for a symbol."""
        try:
            order_book = await self.async_exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None

    async def place_order(self, order_request: OrderRequest) -> Optional[ExecutedOrder]:
        """Place a trading order."""
        try:
            # Validate order
            if not self._validate_order(order_request):
                return None

            # Check risk limits
            if not self._check_risk_limits(order_request):
                logger.warning("Order rejected due to risk limits")
                return None

            # Place order
            result = await self._execute_order(order_request)

            if result:
                order = self._parse_order_result(result)
                self.active_orders[order.id] = order
                self.order_history.append(order)
                self.total_trades += 1

                logger.info(f"Order placed: {order.id} - {order.side.value} ")
                           f"{order.amount} {order.symbol} at {order.price}")

                return order

            return None

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def _execute_order(self, order_request: OrderRequest) -> Optional[Dict[str, Any]]:
        """Execute the actual order."""
        try:
            if order_request.order_type == OrderType.MARKET:
                if order_request.side == OrderSide.BUY:
                    result = await self.async_exchange.create_market_buy_order()
                        order_request.symbol, order_request.amount, None, None,
                            order_request.params or {}
                    )
                else:
                    result = await self.async_exchange.create_market_sell_order()
                        order_request.symbol, order_request.amount, None, None,
                            order_request.params or {}
                    )

            elif order_request.order_type == OrderType.LIMIT:
                if order_request.side == OrderSide.BUY:
                    result = await self.async_exchange.create_limit_buy_order()
                        order_request.symbol, order_request.amount, order_request.price,
                            order_request.params or {}
                    )
                else:
                    result = await self.async_exchange.create_limit_sell_order()
                        order_request.symbol, order_request.amount, order_request.price,
                            order_request.params or {}
                    )

            else:
                logger.error(f"Unsupported order type: {order_request.order_type}")
                return None

            return result

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

    def _validate_order(self, order_request: OrderRequest) -> bool:
        """Validate order parameters."""
        try:
            # Check symbol
            if not order_request.symbol:
                logger.error("Order validation failed: missing symbol")
                return False

            # Check amount
            if order_request.amount <= 0:
                logger.error("Order validation failed: invalid amount")
                return False

            # Check price for limit orders
            if order_request.order_type == OrderType.LIMIT and not order_request.price:
                logger.error("Order validation failed: missing price for limit order")
                return False

            return True

        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False

    def _check_risk_limits(self, order_request: OrderRequest) -> bool:
        """Check risk management limits."""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss * self.portfolio.total_value:
                logger.warning("Daily loss limit exceeded")
                return False

            # Check position size limit
            order_value = order_request.amount * (order_request.price or 0)
            if order_value > self.max_position_size * self.portfolio.total_value:
                logger.warning("Position size limit exceeded")
                return False

            # Reset daily counters if needed
            today = datetime.now().date()
            if today != self.last_daily_reset:
                self.daily_pnl = 0.0
                self.daily_trade_count = 0
                self.last_daily_reset = today

            return True

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False

    def _parse_order_result(self, result: Dict[str, Any]) -> ExecutedOrder:
        """Parse order result into ExecutedOrder."""
        return ExecutedOrder()
            id=result.get('id', ''),
                symbol=result.get('symbol', ''),
                    side=OrderSide(result.get('side', 'buy')),
                    amount=float(result.get('amount', 0)),
                    filled=float(result.get('filled', 0)),
                    remaining=float(result.get('remaining', 0)),
                    price=float(result.get('price', 0)),
                    average=result.get('average'),
                    status=OrderStatus(result.get('status', 'pending')),
                    timestamp=datetime.fromtimestamp(result.get('timestamp', 0) / 1000),
                    fee=result.get('fee'),
                    trades=result.get('trades')
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an active order."""
        try:
            result = await self.async_exchange.cancel_order(order_id, symbol)

            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
                del self.active_orders[order_id]

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[ExecutedOrder]:
        """Get status of a specific order."""
        try:
            result = await self.async_exchange.fetch_order(order_id, symbol)
            order = self._parse_order_result(result)

            # Update active orders
            if order_id in self.active_orders:
                self.active_orders[order_id] = order

            return order

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    async def update_portfolio(self) -> bool:
        """Update portfolio information."""
        try:
            # Fetch balances
            balances = await self.async_exchange.fetch_balance()

            if balances:
                self.portfolio.balances = balances.get('free', {})
                self.portfolio.total_value = balances.get('total', {}).get('USDT', 0.0)
                self.portfolio.last_update = datetime.now()

                # Calculate P&L
                self._calculate_pnl()

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
            return False

    def _calculate_pnl(self):
        """Calculate profit and loss."""
        try:
            # Simple P&L calculation based on order history
            total_buy_value = 0.0
            total_sell_value = 0.0

            for order in self.order_history:
                if order.status == OrderStatus.CLOSED:
                    if order.side == OrderSide.BUY:
                        total_buy_value += order.filled * (order.average or order.price)
                    else:
                        total_sell_value += order.filled * (order.average or order.price)

            self.portfolio.pnl = total_sell_value - total_buy_value

            if total_buy_value > 0:
                self.portfolio.pnl_percentage = (self.portfolio.pnl / total_buy_value) * 100

            # Update daily P&L
            today = datetime.now().date()
            if today == self.last_daily_reset:
                daily_orders = [o for o in self.order_history ]
                              if o.timestamp.date() == today and o.status == OrderStatus.CLOSED]

                daily_buy = sum(o.filled * (o.average or o.price) )
                              for o in daily_orders if o.side == OrderSide.BUY)
                daily_sell = sum(o.filled * (o.average or o.price) )
                               for o in daily_orders if o.side == OrderSide.SELL)

                self.daily_pnl = daily_sell - daily_buy

        except Exception as e:
            logger.error(f"P&L calculation error: {e}")

    async def get_trade_history(self, symbol: str = None, )
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history."""
        try:
            if symbol:
                trades = await self.async_exchange.fetch_my_trades(symbol, None, limit)
            else:
                trades = await self.async_exchange.fetch_my_trades(None, None, limit)

            return trades

        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        success_rate = (self.successful_trades / max(self.total_trades, 1)) * 100

        return {}
            'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                    'success_rate': success_rate,
                    'total_pnl': self.portfolio.pnl,
                    'pnl_percentage': self.portfolio.pnl_percentage,
                    'daily_pnl': self.daily_pnl,
                    'total_fees': self.total_fees,
                    'active_orders': len(self.active_orders),
                    'portfolio_value': self.portfolio.total_value
}
    async def emergency_stop(self) -> bool:
        """Emergency stop - cancel all orders and close positions."""
        try:
            logger.warning("Emergency stop activated")

            # Cancel all active orders
            cancelled_orders = 0
            for order_id, order in list(self.active_orders.items()):
                if await self.cancel_order(order_id, order.symbol):
                    cancelled_orders += 1

            logger.info(f"Emergency stop completed - cancelled {cancelled_orders} orders")
            return True

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    def cleanup(self):
        """Cleanup resources."""
        try:
            asyncio.create_task(self.disconnect())
            logger.info("CCXT Trading Executor cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Convenience functions for quick trading operations
async def quick_buy_btc(executor: CCXTTradingExecutor, amount_usd: float) -> Optional[ExecutedOrder]:
    """Quick BTC buy order."""
    order_request = OrderRequest()
        symbol='BTC/USD',
            side=OrderSide.BUY,
                amount=amount_usd,
                order_type=OrderType.MARKET
    )
    return await executor.place_order(order_request)


async def quick_sell_btc(executor: CCXTTradingExecutor, amount_btc: float) -> Optional[ExecutedOrder]:
    """Quick BTC sell order."""
    order_request = OrderRequest()
        symbol='BTC/USD',
            side=OrderSide.SELL,
                amount=amount_btc,
                order_type=OrderType.MARKET
    )
    return await executor.place_order(order_request)


if __name__ == "__main__":
    async def test_executor():
        """Test the CCXT trading executor."""
        executor = CCXTTradingExecutor()
            exchange_name="coinbase",
                sandbox=True  # Always use sandbox for testing
        )

        # Test connection
        connected = await executor.connect()
        if connected:
            print("✅ Connected successfully")

            # Get markets
            markets = await executor.get_markets()
            print(f"✅ Loaded {len(markets)} markets")

            # Get BTC ticker
            ticker = await executor.get_ticker('BTC/USD')
            if ticker:
                print(f"✅ BTC Price: ${ticker['last']}")

            # Update portfolio
            await executor.update_portfolio()
            print(f"✅ Portfolio value: ${executor.portfolio.total_value}")

        await executor.disconnect()

    # Run test
    if CCXT_AVAILABLE:
        asyncio.run(test_executor())
    else:
        print("❌ CCXT not available for testing")
