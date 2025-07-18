"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCXT Trading Executor for Schwabot Trading System
================================================

CCXT-based trading executor for real trading operations.

Handles:
- Exchange connections and authentication
- Order book management
- Trade execution and order management
- Portfolio balance tracking
- Risk management and position sizing
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ccxt

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderBook:
    """Order book representation."""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, amount)
    asks: List[Tuple[float, float]]  # (price, amount)
    timestamp: float = field(default_factory=time.time)

    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (highest price)."""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None

    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        return None


@dataclass
class TradeOrder:
    """Trade order representation."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    average_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeResult:
    """Trade execution result."""
    order_id: str
    symbol: str
    side: OrderSide
    amount: float
    price: float
    cost: float
    fee: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class IntegratedTradingSignal:
    """Integrated trading signal with mathematical components."""
    signal_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    quantity: float
    price: float
    timestamp: float
    mathematical_score: float
    thermal_state: str
    bit_phase: int
    profit_vector: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingPair:
    """Trading pair configuration."""
    symbol: str
    base_currency: str
    quote_currency: str
    min_amount: float
    max_amount: float
    price_precision: int
    amount_precision: int
    enabled: bool = True


class CCXTTradingExecutor:
    """
    CCXT-based trading executor for real trading operations.

    Handles:
    - Exchange connections and authentication
    - Order book management
    - Trade execution and order management
    - Portfolio balance tracking
    - Risk management and position sizing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the CCXT trading executor."""
        self.config = config or {}
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.active_orders: Dict[str, TradeOrder] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.balances: Dict[str, Dict[str, float]] = {}

        # Trading parameters
        self.max_position_size = self.config.get("max_position_size", 0.1)
        self.max_daily_trades = self.config.get("max_daily_trades", 100)
        self.slippage_tolerance = self.config.get("slippage_tolerance", 0.001)
        self.order_timeout = self.config.get("order_timeout", 30.0)

        # Performance tracking
        self.daily_trades = 0
        self.daily_volume = 0.0
        self.last_reset = time.time()

        # Initialize exchanges
        self._initialize_exchanges()

    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        exchanges_config = self.config.get("exchanges", [])

        for exchange_config in exchanges_config:
            if not exchange_config.get("enabled", False):
                continue

            exchange_name = exchange_config.get("name", "").lower()
            api_key = exchange_config.get("api_key")
            secret = exchange_config.get("secret")
            sandbox = exchange_config.get("sandbox", False)

            try:
                # Get exchange class
                exchange_class = getattr(ccxt, exchange_name)

                # Create exchange instance
                exchange = exchange_class(
                    {
                        'apiKey': api_key,
                        'secret': secret,
                        'sandbox': sandbox,
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'spot',
                        },
                    }
                )

                self.exchanges[exchange_name] = exchange
                logger.info(f"✅ Initialized {exchange_name} exchange")

            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_name}: {e}")

    async def connect_exchanges(self):
        """Connect to all configured exchanges."""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.load_markets()
                logger.info(f"✅ Connected to {exchange_name}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to {exchange_name}: {e}")

    async def disconnect_exchanges(self):
        """Disconnect from all exchanges."""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"✅ Disconnected from {exchange_name}")
            except Exception as e:
                logger.error(f"❌ Failed to disconnect from {exchange_name}: {e}")

    async def fetch_order_book(self, symbol: str, exchange_name: str = "binance") -> Optional[OrderBook]:
        """Fetch order book for a symbol."""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                logger.error(f"Exchange {exchange_name} not found")
                return None

            order_book_data = await exchange.fetch_order_book(symbol)
            
            order_book = OrderBook(
                symbol=symbol,
                bids=order_book_data['bids'],
                asks=order_book_data['asks'],
                timestamp=order_book_data['timestamp'] / 1000.0
            )
            
            self.order_books[symbol] = order_book
            return order_book

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    async def fetch_balance(self, exchange_name: str = "binance") -> Dict[str, float]:
        """Fetch account balance."""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                logger.error(f"Exchange {exchange_name} not found")
                return {}

            balance_data = await exchange.fetch_balance()
            
            # Extract free balances
            balances = {}
            for currency, balance_info in balance_data['free'].items():
                if balance_info > 0:
                    balances[currency] = balance_info

            self.balances[exchange_name] = balances
            return balances

        except Exception as e:
            logger.error(f"Error fetching balance from {exchange_name}: {e}")
            return {}

    async def fetch_positions(self, exchange_name: str = "binance") -> Dict[str, Dict[str, Any]]:
        """Fetch current positions."""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                logger.error(f"Exchange {exchange_name} not found")
                return {}

            positions_data = await exchange.fetch_positions()
            
            positions = {}
            for position in positions_data:
                if position['size'] != 0:  # Only non-zero positions
                    positions[position['symbol']] = {
                        'size': position['size'],
                        'side': position['side'],
                        'entry_price': position['entryPrice'],
                        'unrealized_pnl': position['unrealizedPnl'],
                        'margin_type': position.get('marginType', 'cross'),
                    }

            self.positions[exchange_name] = positions
            return positions

        except Exception as e:
            logger.error(f"Error fetching positions from {exchange_name}: {e}")
            return {}

    async def place_order(self, order: TradeOrder, exchange_name: str = "binance") -> TradeResult:
        """Place a trade order."""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return TradeResult(
                    order_id="",
                    symbol=order.symbol,
                    side=order.side,
                    amount=order.amount,
                    price=order.price or 0.0,
                    cost=0.0,
                    success=False,
                    error_message=f"Exchange {exchange_name} not found"
                )

            # Prepare order parameters
            order_params = {
                'symbol': order.symbol,
                'type': order.order_type.value,
                'side': order.side.value,
                'amount': order.amount,
            }

            if order.price:
                order_params['price'] = order.price

            if order.stop_price:
                order_params['stopPrice'] = order.stop_price

            # Place order
            result = await exchange.create_order(**order_params)

            # Create trade result
            trade_result = TradeResult(
                order_id=result['id'],
                symbol=result['symbol'],
                side=OrderSide(result['side']),
                amount=result['amount'],
                price=result['price'],
                cost=result['cost'],
                fee=result.get('fee', {}).get('cost'),
                success=True
            )

            # Update tracking
            self.active_orders[result['id']] = order
            self.daily_trades += 1
            self.daily_volume += result['cost']

            logger.info(f"✅ Order placed: {result['id']} for {result['symbol']}")
            return trade_result

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return TradeResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=order.price or 0.0,
                cost=0.0,
                success=False,
                error_message=str(e)
            )

    async def cancel_order(self, order_id: str, symbol: str, exchange_name: str = "binance") -> bool:
        """Cancel an active order."""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                logger.error(f"Exchange {exchange_name} not found")
                return False

            result = await exchange.cancel_order(order_id, symbol)
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            logger.info(f"✅ Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_order_status(
        self, order_id: str, symbol: str, exchange_name: str = "binance"
    ) -> Optional[Dict[str, Any]]:
        """Get order status."""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                logger.error(f"Exchange {exchange_name} not found")
                return None

            order_status = await exchange.fetch_order(order_id, symbol)
            return order_status

        except Exception as e:
            logger.error(f"Error fetching order status for {order_id}: {e}")
            return None

    async def calculate_position_size(
        self, symbol: str, available_balance: float, risk_per_trade: float = 0.02
    ) -> float:
        """Calculate position size based on risk management."""
        try:
            # Get current price
            order_book = await self.fetch_order_book(symbol)
            if not order_book:
                return 0.0

            current_price = order_book.get_mid_price()
            if not current_price:
                return 0.0

            # Calculate position size using Kelly Criterion
            win_rate = 0.6  # Estimated win rate
            avg_win = 0.02  # Average win percentage
            avg_loss = 0.01  # Average loss percentage

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.0, min(kelly_fraction, self.max_position_size))

            position_size = (available_balance * kelly_fraction) / current_price
            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _reset_daily_counters(self) -> None:
        """Reset daily trading counters."""
        current_time = time.time()
        if current_time - self.last_reset > 86400:  # 24 hours
            self.daily_trades = 0
            self.daily_volume = 0.0
            self.last_reset = current_time

    async def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading summary and statistics."""
        self._reset_daily_counters()
        
        return {
            "exchanges": list(self.exchanges.keys()),
            "active_orders": len(self.active_orders),
            "daily_trades": self.daily_trades,
            "daily_volume": self.daily_volume,
            "max_position_size": self.max_position_size,
            "max_daily_trades": self.max_daily_trades,
            "slippage_tolerance": self.slippage_tolerance,
            "order_timeout": self.order_timeout,
        }

    def cleanup(self) -> None:
        """Clean up trading executor resources."""
        try:
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'close'):
                    exchange.close()
            
            self.exchanges.clear()
            self.order_books.clear()
            self.active_orders.clear()
            
            logger.info("CCXT Trading Executor cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up trading executor: {e}")


def create_ccxt_trading_executor(config: Dict[str, Any]) -> CCXTTradingExecutor:
    """Create a CCXT trading executor instance."""
    return CCXTTradingExecutor(config)


# Global instance for easy access
ccxt_trading_executor = CCXTTradingExecutor()
