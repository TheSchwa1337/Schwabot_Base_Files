"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Order Book Manager for Schwabot
==================================

Real-time order book management and analysis for trading operations.
Handles order book data, market depth analysis, and liquidity calculations.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
    class OrderBookLevel:
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    """Individual order book level."""

    price: float
    amount: float
    count: int = 1
    timestamp: float = field(default_factory=time.time)


    @dataclass
        class OrderBookSnapshot:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Complete order book snapshot."""

        symbol: str
        bids: List[OrderBookLevel]
        asks: List[OrderBookLevel]
        timestamp: float = field(default_factory=time.time)
        sequence_number: Optional[int] = None

            def get_best_bid(self) -> Optional[OrderBookLevel]:
            """Get best bid (highest price)."""
        return self.bids[0] if self.bids else None

            def get_best_ask(self) -> Optional[OrderBookLevel]:
            """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else None

            def get_spread(self) -> Optional[float]:
            """Calculate bid-ask spread."""
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
                if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

            def get_mid_price(self) -> Optional[float]:
            """Calculate mid price."""
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
                if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None

            def get_spread_percentage(self) -> Optional[float]:
            """Calculate spread as percentage of mid price."""
            spread = self.get_spread()
            mid_price = self.get_mid_price()
                if spread and mid_price:
            return (spread / mid_price) * 100
        return None


        @dataclass
            class MarketDepth:
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Market depth analysis."""

            symbol: str
            bid_depth: Dict[float, float]  # price -> cumulative amount
            ask_depth: Dict[float, float]  # price -> cumulative amount
            timestamp: float = field(default_factory=time.time)

                def get_liquidity_at_price(self, price: float, side: str) -> float:
                """Get liquidity available at specific price level."""
                    if side.lower() == 'bid':
                return self.bid_depth.get(price, 0.0)
                    elif side.lower() == 'ask':
                return self.ask_depth.get(price, 0.0)
            return 0.0

                def get_impact_price(self, amount: float, side: str) -> Optional[float]:
                """Calculate price impact for given order size."""
                    if side.lower() == 'bid':
                    cumulative = 0.0
                        for price, level_amount in sorted(self.ask_depth.items()):
                        cumulative += level_amount
                            if cumulative >= amount:
                        return price
                            elif side.lower() == 'ask':
                            cumulative = 0.0
                                for price, level_amount in sorted(self.bid_depth.items(), reverse=True):
                                cumulative += level_amount
                                    if cumulative >= amount:
                                return price
                            return None


                                class OrderBookManager:
    """Class for Schwabot trading functionality."""
                                """Class for Schwabot trading functionality."""
                                """
                                Real-time order book manager for trading operations.

                                    Features:
                                    - Real-time order book updates
                                    - Market depth analysis
                                    - Liquidity calculations
                                    - Price impact estimation
                                    - Order book imbalance detection
                                    - Historical order book tracking
                                    """

                                        def __init__(self, config: Dict[str, Any]) -> None:
                                        """Initialize the order book manager."""
                                        self.config = config
                                        self.order_books: Dict[str, OrderBookSnapshot] = {}
                                        self.market_depth: Dict[str, MarketDepth] = {}
                                        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
                                        self.update_callbacks: Dict[str, List[Callable]] = defaultdict(list)

                                        # Configuration
                                        self.max_levels = config.get("max_levels", 20)
                                        self.update_interval = config.get("update_interval", 1.0)
                                        self.enable_historical_tracking = config.get("enable_historical_tracking", True)
                                        self.liquidity_threshold = config.get("liquidity_threshold", 0.001)

                                        # Performance tracking
                                        self.update_count = 0
                                        self.last_update = time.time()

                                        # Start update loop
                                        self.running = False
                                        self.update_task = None

                                            async def start(self):
                                            """Start the order book manager."""
                                            self.running = True
                                            self.update_task = asyncio.create_task(self._update_loop())
                                            logger.info("✅ Order book manager started")

                                                async def stop(self):
                                                """Stop the order book manager."""
                                                self.running = False
                                                    if self.update_task:
                                                    self.update_task.cancel()
                                                        try:
                                                        await self.update_task
                                                            except asyncio.CancelledError:
                                                        pass
                                                        logger.info("🛑 Order book manager stopped")

                                                            async def _update_loop(self):
                                                            """Main update loop for order book processing."""
                                                                while self.running:
                                                                    try:
                                                                    # Process any pending updates
                                                                    await self._process_updates()

                                                                    # Wait for next cycle
                                                                    await asyncio.sleep(self.update_interval)

                                                                        except Exception as e:
                                                                        logger.error(f"Error in order book update loop: {e}")
                                                                        await asyncio.sleep(5.0)  # Wait before retrying

                                                                            async def _process_updates(self):
                                                                            """Process pending order book updates."""
                                                                            # This would typically process updates from exchange websockets
                                                                            # For now, we'll just update timestamps
                                                                            current_time = time.time()
                                                                            self.last_update = current_time
                                                                            self.update_count += 1

                                                                                def update_order_book(self, symbol: str, order_book_data: Dict[str, Any]) -> bool:
                                                                                """Update order book for a symbol."""
                                                                                    try:
                                                                                    # Parse order book data
                                                                                    bids = self._parse_levels(order_book_data.get('bids', []))
                                                                                    asks = self._parse_levels(order_book_data.get('asks', []))

                                                                                    # Create order book snapshot
                                                                                    snapshot = OrderBookSnapshot(
                                                                                    symbol=symbol,
                                                                                    bids=bids[: self.max_levels],
                                                                                    asks=asks[: self.max_levels],
                                                                                    timestamp=order_book_data.get('timestamp', time.time()) / 1000.0,
                                                                                    sequence_number=order_book_data.get('sequence', None),
                                                                                    )

                                                                                    # Update order book
                                                                                    self.order_books[symbol] = snapshot

                                                                                    # Update market depth
                                                                                    self._update_market_depth(symbol, snapshot)

                                                                                    # Store historical data
                                                                                        if self.enable_historical_tracking:
                                                                                        self.historical_data[symbol].append(snapshot)

                                                                                        # Trigger callbacks
                                                                                        self._trigger_callbacks(symbol, snapshot)

                                                                                    return True

                                                                                        except Exception as e:
                                                                                        logger.error(f"Error updating order book for {symbol}: {e}")
                                                                                    return False

                                                                                        def _parse_levels(self, levels: List[List[float]]) -> List[OrderBookLevel]:
                                                                                        """Parse order book levels from exchange data."""
                                                                                        parsed_levels = []

                                                                                            for level in levels:
                                                                                                if len(level) >= 2:
                                                                                                price = float(level[0])
                                                                                                amount = float(level[1])
                                                                                                count = int(level[2]) if len(level) > 2 else 1

                                                                                                parsed_levels.append(OrderBookLevel(price=price, amount=amount, count=count))

                                                                                            return parsed_levels

                                                                                                def _update_market_depth(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
                                                                                                """Update market depth for a symbol."""
                                                                                                    try:
                                                                                                    # Calculate cumulative bid depth
                                                                                                    bid_depth = {}
                                                                                                    cumulative = 0.0
                                                                                                        for level in sorted(snapshot.bids, key=lambda x: x.price, reverse=True):
                                                                                                        cumulative += level.amount
                                                                                                        bid_depth[level.price] = cumulative

                                                                                                        # Calculate cumulative ask depth
                                                                                                        ask_depth = {}
                                                                                                        cumulative = 0.0
                                                                                                            for level in sorted(snapshot.asks, key=lambda x: x.price):
                                                                                                            cumulative += level.amount
                                                                                                            ask_depth[level.price] = cumulative

                                                                                                            # Create market depth object
                                                                                                            market_depth = MarketDepth(
                                                                                                            symbol=symbol,
                                                                                                            bid_depth=bid_depth,
                                                                                                            ask_depth=ask_depth,
                                                                                                            timestamp=snapshot.timestamp,
                                                                                                            )

                                                                                                            self.market_depth[symbol] = market_depth

                                                                                                                except Exception as e:
                                                                                                                logger.error(f"Error updating market depth for {symbol}: {e}")

                                                                                                                    def _trigger_callbacks(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
                                                                                                                    """Trigger registered callbacks for order book updates."""
                                                                                                                        for callback in self.update_callbacks[symbol]:
                                                                                                                            try:
                                                                                                                            callback(snapshot)
                                                                                                                                except Exception as e:
                                                                                                                                logger.error(f"Error in order book callback: {e}")

                                                                                                                                    def register_callback(self, symbol: str, callback: Callable[[OrderBookSnapshot], None]) -> None:
                                                                                                                                    """Register a callback for order book updates."""
                                                                                                                                    self.update_callbacks[symbol].append(callback)
                                                                                                                                    logger.info(f"Registered callback for {symbol}")

                                                                                                                                        def unregister_callback(self, symbol: str, callback: Callable[[OrderBookSnapshot], None]) -> None:
                                                                                                                                        """Unregister a callback for order book updates."""
                                                                                                                                            if symbol in self.update_callbacks:
                                                                                                                                                try:
                                                                                                                                                self.update_callbacks[symbol].remove(callback)
                                                                                                                                                logger.info(f"Unregistered callback for {symbol}")
                                                                                                                                                    except ValueError:
                                                                                                                                                pass

                                                                                                                                                    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
                                                                                                                                                    """Get current order book for a symbol."""
                                                                                                                                                return self.order_books.get(symbol)

                                                                                                                                                    def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
                                                                                                                                                    """Get market depth for a symbol."""
                                                                                                                                                return self.market_depth.get(symbol)

                                                                                                                                                    def calculate_price_impact(self, symbol: str, amount: float, side: str) -> Dict[str, Any]:
                                                                                                                                                    """Calculate price impact for a given order size."""
                                                                                                                                                        try:
                                                                                                                                                        market_depth = self.get_market_depth(symbol)
                                                                                                                                                            if not market_depth:
                                                                                                                                                        return {"error": "No market depth data available"}

                                                                                                                                                        # Get current mid price
                                                                                                                                                        order_book = self.get_order_book(symbol)
                                                                                                                                                            if not order_book:
                                                                                                                                                        return {"error": "No order book data available"}

                                                                                                                                                        mid_price = order_book.get_mid_price()
                                                                                                                                                            if not mid_price:
                                                                                                                                                        return {"error": "Cannot calculate mid price"}

                                                                                                                                                        # Calculate impact price
                                                                                                                                                        impact_price = market_depth.get_impact_price(amount, side)
                                                                                                                                                            if not impact_price:
                                                                                                                                                        return {"error": "Insufficient liquidity"}

                                                                                                                                                        # Calculate impact
                                                                                                                                                            if side.lower() == 'buy':
                                                                                                                                                            price_impact = (impact_price - mid_price) / mid_price
                                                                                                                                                            else:  # sell
                                                                                                                                                            price_impact = (mid_price - impact_price) / mid_price

                                                                                                                                                        return {
                                                                                                                                                        "symbol": symbol,
                                                                                                                                                        "side": side,
                                                                                                                                                        "amount": amount,
                                                                                                                                                        "mid_price": mid_price,
                                                                                                                                                        "impact_price": impact_price,
                                                                                                                                                        "price_impact": price_impact,
                                                                                                                                                        "price_impact_percentage": price_impact * 100,
                                                                                                                                                        }

                                                                                                                                                            except Exception as e:
                                                                                                                                                            logger.error(f"Error calculating price impact: {e}")
                                                                                                                                                        return {"error": str(e)}

                                                                                                                                                            def detect_order_book_imbalance(self, symbol: str) -> Dict[str, Any]:
                                                                                                                                                            """Detect order book imbalance."""
                                                                                                                                                                try:
                                                                                                                                                                order_book = self.get_order_book(symbol)
                                                                                                                                                                    if not order_book:
                                                                                                                                                                return {"error": "No order book data available"}

                                                                                                                                                                # Calculate bid and ask volumes
                                                                                                                                                                bid_volume = sum(level.amount for level in order_book.bids)
                                                                                                                                                                ask_volume = sum(level.amount for level in order_book.asks)

                                                                                                                                                                total_volume = bid_volume + ask_volume
                                                                                                                                                                    if total_volume == 0:
                                                                                                                                                                return {"error": "No volume data available"}

                                                                                                                                                                # Calculate imbalance
                                                                                                                                                                bid_ratio = bid_volume / total_volume
                                                                                                                                                                ask_ratio = ask_volume / total_volume
                                                                                                                                                                imbalance = bid_ratio - ask_ratio

                                                                                                                                                                # Determine imbalance direction
                                                                                                                                                                    if imbalance > self.liquidity_threshold:
                                                                                                                                                                    direction = "bid_heavy"
                                                                                                                                                                        elif imbalance < -self.liquidity_threshold:
                                                                                                                                                                        direction = "ask_heavy"
                                                                                                                                                                            else:
                                                                                                                                                                            direction = "balanced"

                                                                                                                                                                        return {
                                                                                                                                                                        "symbol": symbol,
                                                                                                                                                                        "bid_volume": bid_volume,
                                                                                                                                                                        "ask_volume": ask_volume,
                                                                                                                                                                        "total_volume": total_volume,
                                                                                                                                                                        "bid_ratio": bid_ratio,
                                                                                                                                                                        "ask_ratio": ask_ratio,
                                                                                                                                                                        "imbalance": imbalance,
                                                                                                                                                                        "direction": direction,
                                                                                                                                                                        "timestamp": order_book.timestamp,
                                                                                                                                                                        }

                                                                                                                                                                            except Exception as e:
                                                                                                                                                                            logger.error(f"Error detecting order book imbalance: {e}")
                                                                                                                                                                        return {"error": str(e)}

                                                                                                                                                                            def get_liquidity_analysis(self, symbol: str, price_levels: List[float] = None) -> Dict[str, Any]:
                                                                                                                                                                            """Analyze liquidity at different price levels."""
                                                                                                                                                                                try:
                                                                                                                                                                                market_depth = self.get_market_depth(symbol)
                                                                                                                                                                                    if not market_depth:
                                                                                                                                                                                return {"error": "No market depth data available"}

                                                                                                                                                                                # Default price levels if not provided
                                                                                                                                                                                    if price_levels is None:
                                                                                                                                                                                    order_book = self.get_order_book(symbol)
                                                                                                                                                                                        if order_book:
                                                                                                                                                                                        mid_price = order_book.get_mid_price()
                                                                                                                                                                                            if mid_price:
                                                                                                                                                                                            # Create price levels around mid price
                                                                                                                                                                                            price_levels = [
                                                                                                                                                                                            mid_price * (1 - 0.01),  # -1%
                                                                                                                                                                                            mid_price * (1 - 0.005),  # -0.5%
                                                                                                                                                                                            mid_price,  # Mid price
                                                                                                                                                                                            mid_price * (1 + 0.005),  # +0.5%
                                                                                                                                                                                            mid_price * (1 + 0.01),  # +1%
                                                                                                                                                                                            ]

                                                                                                                                                                                            analysis = {"symbol": symbol, "price_levels": {}, "timestamp": market_depth.timestamp}

                                                                                                                                                                                                for price in price_levels:
                                                                                                                                                                                                bid_liquidity = market_depth.get_liquidity_at_price(price, 'bid')
                                                                                                                                                                                                ask_liquidity = market_depth.get_liquidity_at_price(price, 'ask')

                                                                                                                                                                                                analysis["price_levels"][price] = {
                                                                                                                                                                                                "bid_liquidity": bid_liquidity,
                                                                                                                                                                                                "ask_liquidity": ask_liquidity,
                                                                                                                                                                                                "total_liquidity": bid_liquidity + ask_liquidity,
                                                                                                                                                                                                }

                                                                                                                                                                                            return analysis

                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                logger.error(f"Error analyzing liquidity: {e}")
                                                                                                                                                                                            return {"error": str(e)}

                                                                                                                                                                                                def get_historical_data(self, symbol: str, limit: int = 100) -> List[OrderBookSnapshot]:
                                                                                                                                                                                                """Get historical order book data."""
                                                                                                                                                                                                    if symbol in self.historical_data:
                                                                                                                                                                                                return list(self.historical_data[symbol])[-limit:]
                                                                                                                                                                                            return []

                                                                                                                                                                                                def get_statistics(self) -> Dict[str, Any]:
                                                                                                                                                                                                """Get order book manager statistics."""
                                                                                                                                                                                            return {
                                                                                                                                                                                            "active_symbols": len(self.order_books),
                                                                                                                                                                                            "update_count": self.update_count,
                                                                                                                                                                                            "last_update": self.last_update,
                                                                                                                                                                                            "historical_tracking_enabled": self.enable_historical_tracking,
                                                                                                                                                                                            "total_historical_records": sum(len(data) for data in self.historical_data.values()),
                                                                                                                                                                                            "registered_callbacks": sum(len(callbacks) for callbacks in self.update_callbacks.values()),
                                                                                                                                                                                            }


                                                                                                                                                                                                def create_order_book_manager(config: Dict[str, Any]) -> OrderBookManager:
                                                                                                                                                                                                """Factory function to create order book manager."""
                                                                                                                                                                                            return OrderBookManager(config)
