import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional


class TradingVisualizer:
    """
    Trading-specific visualizer for Schwabot.

    Handles visualization of:
    - Order book data
    - Trade executions
    - Portfolio positions
    - Market data
    - Strategy performance
    """

    def __init__(
        self,
        max_history: int = 1000,
        update_interval: float = 0.5,
        enable_order_book: bool = True,
        enable_portfolio_tracking: bool = True,
    ):
        """
        Initialize the trading visualizer.

        Args:
            max_history: Maximum number of data points to keep
            update_interval: Update interval in seconds
            enable_order_book: Enable order book visualization
            enable_portfolio_tracking: Enable portfolio tracking
        """
        self.logger = logging.getLogger("TradingVisualizer")

        # Configuration
        self.max_history = max_history
        self.update_interval = update_interval
        self.enable_order_book = enable_order_book
        self.enable_portfolio_tracking = enable_portfolio_tracking

        # Data storage
        self.order_book_data = defaultdict(lambda: deque(maxlen=max_history))
        self.trade_history = deque(maxlen=max_history)
        self.portfolio_positions = {}
        self.market_data = defaultdict(lambda: deque(maxlen=max_history))
        self.strategy_performance = defaultdict(lambda: deque(maxlen=max_history))

        # Current state
        self.current_prices = {}
        self.active_orders = {}
        self.completed_trades = []

        # Callbacks
        self.trading_callbacks: List[Callable] = []

        # Async state
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

    def add_order_book_data(self, symbol: str, bids: List[Dict], asks: List[Dict]):
        """Add order book data for a symbol"""
        if not self.enable_order_book:
            return

        data_point = {
            "timestamp": time.time(),
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "spread": asks[0]["price"] - bids[0]["price"] if asks and bids else 0,
        }

        self.order_book_data[symbol].append(data_point)

        # Update current price
        if asks and bids:
            self.current_prices[symbol] = (asks[0]["price"] + bids[0]["price"]) / 2

    def add_trade_execution(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_id: str,
        timestamp: float = None,
    ):
        """Add a trade execution"""
        trade_data = {
            "timestamp": timestamp or time.time(),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "order_id": order_id,
            "value": amount * price,
        }

        self.trade_history.append(trade_data)
        self.completed_trades.append(trade_data)

        # Update portfolio positions
        if self.enable_portfolio_tracking:
            self._update_portfolio_position(symbol, side, amount, price)

        # Trigger callbacks
        self._trigger_trading_callbacks("trade_execution", trade_data)

    def add_market_data(self, symbol: str, data: Dict[str, Any]):
        """Add market data for a symbol"""
        data_point = {"timestamp": time.time(), "symbol": symbol, **data}

        self.market_data[symbol].append(data_point)

        # Update current price if available
        if "price" in data:
            self.current_prices[symbol] = data["price"]

    def add_strategy_performance(self, strategy_id: str, performance_data: Dict[str, Any]):
        """Add strategy performance data"""
        data_point = {"timestamp": time.time(), "strategy_id": strategy_id, **performance_data}

        self.strategy_performance[strategy_id].append(data_point)

    def update_portfolio_position(self, symbol: str, side: str, amount: float, price: float):
        """Update portfolio position"""
        if not self.enable_portfolio_tracking:
            return

        if symbol not in self.portfolio_positions:
            self.portfolio_positions[symbol] = {
                "amount": 0,
                "avg_price": 0,
                "total_value": 0,
                "last_update": time.time(),
            }

        position = self.portfolio_positions[symbol]

        if side == "buy":
            # Add to position
            total_amount = position["amount"] + amount
            total_value = position["total_value"] + (amount * price)
            position["amount"] = total_amount
            position["total_value"] = total_value
            position["avg_price"] = total_value / total_amount if total_amount > 0 else 0
        else:
            # Reduce position
            position["amount"] -= amount
            if position["amount"] <= 0:
                position["amount"] = 0
                position["avg_price"] = 0
                position["total_value"] = 0

        position["last_update"] = time.time()

    def _update_portfolio_position(self, symbol: str, side: str, amount: float, price: float):
        """Internal method to update portfolio position"""
        self.update_portfolio_position(symbol, side, amount, price)

    def get_order_book_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get order book summary"""
        if symbol:
            if symbol not in self.order_book_data:
                return {}

            data = list(self.order_book_data[symbol])
            if not data:
                return {}

            latest = data[-1]
            return {
                "symbol": symbol,
                "current_price": self.current_prices.get(symbol, 0),
                "spread": latest["spread"],
                "bid_depth": len(latest["bids"]),
                "ask_depth": len(latest["asks"]),
                "top_bid": latest["bids"][0] if latest["bids"] else None,
                "top_ask": latest["asks"][0] if latest["asks"] else None,
            }
        else:
            return {
                symbol: self.get_order_book_summary(symbol)
                for symbol in self.order_book_data.keys()
            }

    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading activity summary"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "total_volume": 0,
                "total_value": 0,
                "buy_trades": 0,
                "sell_trades": 0,
            }

        total_trades = len(self.trade_history)
        total_volume = sum(trade["amount"] for trade in self.trade_history)
        total_value = sum(trade["value"] for trade in self.trade_history)
        buy_trades = len([t for t in self.trade_history if t["side"] == "buy"])
        sell_trades = len([t for t in self.trade_history if t["side"] == "sell"])

        return {
            "total_trades": total_trades,
            "total_volume": total_volume,
            "total_value": total_value,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_trade_size": total_volume / total_trades if total_trades > 0 else 0,
            "avg_trade_value": total_value / total_trades if total_trades > 0 else 0,
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        if not self.enable_portfolio_tracking:
            return {}

        total_positions = len(self.portfolio_positions)
        total_value = sum(pos["total_value"] for pos in self.portfolio_positions.values())
        active_positions = len(
            [pos for pos in self.portfolio_positions.values() if pos["amount"] > 0]
        )

        return {
            "total_positions": total_positions,
            "active_positions": active_positions,
            "total_value": total_value,
            "positions": self.portfolio_positions.copy(),
        }

    def get_market_summary(self) -> Dict[str, Any]:
        """Get market data summary"""
        summary = {}

        for symbol, data in self.market_data.items():
            if not data:
                continue

            latest = data[-1]
            summary[symbol] = {
                "current_price": self.current_prices.get(symbol, 0),
                "last_update": latest["timestamp"],
                "data_points": len(data),
            }

        return summary

    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        summary = {}

        for strategy_id, data in self.strategy_performance.items():
            if not data:
                continue

            latest = data[-1]
            summary[strategy_id] = {
                "last_update": latest["timestamp"],
                "data_points": len(data),
                "latest_performance": latest,
            }

        return summary

    def get_all_trading_data(self) -> Dict[str, Any]:
        """Get all trading-related data"""
        return {
            "order_book": self.get_order_book_summary(),
            "trading_summary": self.get_trading_summary(),
            "portfolio": self.get_portfolio_summary(),
            "market_summary": self.get_market_summary(),
            "strategy_performance": self.get_strategy_performance_summary(),
            "current_prices": self.current_prices.copy(),
            "active_orders": self.active_orders.copy(),
        }

    def register_trading_callback(self, callback: Callable):
        """Register a callback for trading events"""
        self.trading_callbacks.append(callback)

    def _trigger_trading_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger trading callbacks"""
        for callback in self.trading_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Trading callback error: {e}")

    async def start(self):
        """Start the trading visualizer"""
        if self.is_running:
            return

        self.is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Trading Visualizer started")

    async def stop(self):
        """Stop the trading visualizer"""
        self.is_running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Trading Visualizer stopped")

    async def _update_loop(self):
        """Main update loop"""
        while self.is_running:
            try:
                # Get all trading data
                trading_data = self.get_all_trading_data()

                # Trigger periodic callbacks
                for callback in self.trading_callbacks:
                    try:
                        callback("periodic_update", trading_data)
                    except Exception as e:
                        self.logger.error(f"Periodic callback error: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Trading visualizer update loop error: {e}")
                await asyncio.sleep(1)

    def get_trading_alerts(self) -> List[Dict[str, Any]]:
        """Get trading-related alerts"""
        alerts = []
        current_time = time.time()

        # Check for large trades
        recent_trades = [
            t for t in self.trade_history if current_time - t["timestamp"] <= 300
        ]  # Last 5 minutes
        if recent_trades:
            large_trades = [t for t in recent_trades if t["value"] > 10000]  # Trades > $10k
            if large_trades:
                alerts.append(
                    {
                        "type": "large_trade",
                        "severity": "info",
                        "message": f"Large trade detected: {len(large_trades)} trades > $10k in last 5 minutes",
                        "timestamp": current_time,
                    }
                )

        # Check for high spread
        for symbol, data in self.order_book_data.items():
            if data:
                latest = data[-1]
                if latest["spread"] > latest["bids"][0]["price"] * 0.01:  # Spread > 1%
                    alerts.append(
                        {
                            "type": "high_spread",
                            "severity": "warning",
                            "message": f"High spread detected for {symbol}: {latest['spread']:.4f}",
                            "timestamp": current_time,
                        }
                    )
