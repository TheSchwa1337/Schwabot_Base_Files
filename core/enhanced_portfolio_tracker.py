"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Portfolio Tracker
==========================
Enhanced portfolio tracker with real-time price integration and automatic rebalancing.
Integrates with exchange APIs and real-time market data feeds.
"""

import logging
import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Union, Callable
from .portfolio_tracker import PortfolioTracker, Position
from .api.exchange_connection import ExchangeManager
from .real_time_market_data_integration import RealTimeMarketDataIntegration, PriceUpdate


logger = logging.getLogger(__name__)
getcontext().prec = 18


@dataclass
class RebalancingAction:
    """Represents a rebalancing action."""
    symbol: str
    action: str  # 'buy' or 'sell'
    amount: float
    current_allocation: float
    target_allocation: float
    deviation: float
    priority: int = 1  # 1 = high, 2 = medium, 3 = low


@dataclass
class PortfolioRebalancingConfig:
    """Configuration for portfolio rebalancing."""
    enabled: bool = True
    threshold: float = 0.05  # 5% deviation threshold
    interval: int = 3600  # Check every hour
    max_rebalancing_cost: float = 0.01  # 1% max cost
    target_allocation: Dict[str, float] = field(default_factory=dict)
    rebalancing_strategy: str = 'threshold_based'  # 'threshold_based', 'time_based', 'risk_adjusted'


class EnhancedPortfolioTracker(PortfolioTracker):
    """Enhanced portfolio tracker with real-time price integration and rebalancing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        # Initialize components
        self.market_data_integration = RealTimeMarketDataIntegration(config)
        self.exchange_manager = ExchangeManager(config)

        # Rebalancing configuration
        self.rebalancing_config = PortfolioRebalancingConfig(**config.get('rebalancing', {}))

        # Price update settings
        self.price_update_interval = config.get('price_update_interval', 5)
        self.last_price_update = 0
        self.last_rebalance_check = 0
        self.last_rebalance_execution = 0

        # Statistics
        self.rebalancing_actions_executed = 0
        self.total_rebalancing_cost = Decimal('0')
        self.price_updates_received = 0

        # Callbacks
        self.rebalancing_callbacks = []
        self.price_update_callbacks = []

        # Setup automatic updates
        self.setup_automatic_updates()

        logger.info("Enhanced portfolio tracker initialized")

    def setup_automatic_updates(self) -> None:
        """Setup automatic price updates and rebalancing checks."""
        async def automatic_update_loop():
            while True:
                try:
                    await self.update_all_prices()
                    await self.check_rebalancing_needs()
                    await asyncio.sleep(self.price_update_interval)
                except Exception as e:
                    logger.error(f"Automatic update loop error: {e}")
                    await asyncio.sleep(10)

        # Start the automatic update loop
        asyncio.create_task(automatic_update_loop())

    async def start(self):
        """Start the enhanced portfolio tracker."""
        try:
            # Start market data integration
            await self.market_data_integration.start()

            # Connect to exchanges
            await self.exchange_manager.connect_all()

            # Add price update callback
            self.market_data_integration.add_price_callback(self._on_price_update)

            logger.info("✅ Enhanced portfolio tracker started")

        except Exception as e:
            logger.error(f"❌ Failed to start enhanced portfolio tracker: {e}")

    async def stop(self):
        """Stop the enhanced portfolio tracker."""
        try:
            # Stop market data integration
            await self.market_data_integration.stop()

            # Disconnect from exchanges
            await self.exchange_manager.disconnect_all()

            logger.info("✅ Enhanced portfolio tracker stopped")

        except Exception as e:
            logger.error(f"❌ Error stopping enhanced portfolio tracker: {e}")

    async def update_all_prices(self):
        """Update prices for all tracked positions."""
        try:
            symbols = self.get_tracked_symbols()
            price_updates = {}

            for symbol in symbols:
                price_data = self.market_data_integration.get_price(symbol)
                if price_data:
                    price_updates[symbol] = price_data['price']

            if price_updates:
                self.update_prices(price_updates)
                self.price_updates_received += 1
                self.last_price_update = time.time()

            logger.debug(f"Updated prices for {len(price_updates)} symbols")

        except Exception as e:
            logger.error(f"Price update error: {e}")

    def get_tracked_symbols(self) -> List[str]:
        """Get list of symbols to track for price updates."""
        symbols = set()

        # Add symbols from open positions
        for position in self.positions.values():
            symbols.add(position.symbol)

        # Add symbols from configuration
        config_symbols = self.config.get('tracked_symbols', [])
        symbols.update(config_symbols)

        # Add symbols from target allocation
        symbols.update(self.rebalancing_config.target_allocation.keys())

        return list(symbols)

    async def sync_with_exchanges(self):
        """Sync portfolio with all connected exchanges."""
        try:
            await self.exchange_manager.connect_all()

            for exchange_name, connection in self.exchange_manager.connections.items():
                try:
                    balance = await connection.get_balance()
                    if balance:
                        self.sync_balances(balance)
                        logger.info(f"✅ Synced {exchange_name} balances")
                except Exception as e:
                    logger.error(f"❌ Failed to sync {exchange_name}: {e}")

            logger.info("✅ Portfolio synced with exchanges")

        except Exception as e:
            logger.error(f"Exchange sync error: {e}")

    async def check_rebalancing_needs(self) -> Dict[str, Any]:
        """Check if portfolio rebalancing is needed."""
        try:
            if not self.rebalancing_config.enabled:
                return {'needs_rebalancing': False, 'reason': 'Rebalancing disabled'}

            # Check time-based rebalancing
            if time.time() - self.last_rebalance_execution < self.rebalancing_config.interval:
                return {'needs_rebalancing': False, 'reason': 'Too soon since last rebalance'}

            # Analyze current portfolio
            analysis = self._analyze_portfolio_for_rebalancing()

            if analysis['needs_rebalancing']:
                logger.info(f"🔄 Rebalancing needed: {len(analysis['rebalancing_actions'])} actions")
                return analysis
            else:
                return {'needs_rebalancing': False, 'reason': 'Portfolio within thresholds'}

        except Exception as e:
            logger.error(f"Rebalancing check error: {e}")
            return {'needs_rebalancing': False, 'reason': f'Error: {e}'}

    def _analyze_portfolio_for_rebalancing(self) -> Dict[str, Any]:
        """Analyze portfolio for rebalancing needs."""
        try:
            summary = self.get_portfolio_summary()
            target_allocation = self.rebalancing_config.target_allocation
            threshold = self.rebalancing_config.threshold

            needs_rebalancing = []
            total_value = summary['total_value']

            if total_value <= 0:
                return {
                    'needs_rebalancing': False, 'rebalancing_actions': []}

            for asset, target_pct in target_allocation.items():
                current_value = summary['balances'].get(
                    asset, 0)
                current_pct = current_value / total_value if total_value > 0 else 0
                deviation = abs(
                    current_pct - target_pct)

                if deviation > threshold:
                    needs_rebalancing.append(RebalancingAction(
                        symbol=asset,
                        action='buy' if current_pct < target_pct else 'sell',
                        amount=abs(target_pct - current_pct) * total_value,
                        current_allocation=current_pct,
                        target_allocation=target_pct,
                        deviation=deviation,
                        priority=1 if deviation > threshold * 2 else 2
                    ))

            return {
                'needs_rebalancing': len(needs_rebalancing) > 0,
                'rebalancing_actions': needs_rebalancing,
                'total_value': total_value,
                'current_allocation': {asset: summary['balances'].get(asset, 0) / total_value for asset in target_allocation.keys()}
            }

        except Exception as e:
            logger.error(
                f"Portfolio analysis error: {e}")
            return {
                'needs_rebalancing': False, 'rebalancing_actions': []}

    async def execute_rebalancing(self, rebalancing_actions: List[RebalancingAction]):
        """Execute rebalancing actions."""
        try:
            if not rebalancing_actions:
                logger.info(
                    "No rebalancing actions to execute")
                return True

            logger.info(
                f"🔄 Executing {len(rebalancing_actions)} rebalancing actions")

            executed_actions = []
            total_cost = Decimal(
                '0')

            # Sort actions by priority
            sorted_actions = sorted(
                rebalancing_actions, key=lambda x: x.priority)

            for action in sorted_actions:
                try:
                    # Find best exchange for this symbol
                    best_exchange = await self._find_best_exchange(action.symbol)
                    if not best_exchange:
                        logger.warning(
                            f"No suitable exchange found for {action.symbol}")
                        continue

                    # Calculate order parameters
                    current_price = self._get_current_price(
                        action.symbol)
                    if not current_price:
                        logger.warning(
                            f"No price data for {action.symbol}")
                        continue

                    # Calculate quantity
                    quantity = action.amount / current_price

                    # Place order
                    order_request = {
                        'symbol': f"{action.symbol}/USD",
                        'type': 'market',
                        'side': action.action,
                        'amount': quantity
                    }

                    result = await best_exchange.place_order(order_request)

                    if result.get('success'):
                        executed_actions.append(
                            action)
                        self.rebalancing_actions_executed += 1

                        # Calculate cost
                        order = result['order']
                        cost = float(
                            order.get('cost', 0))
                        self.total_rebalancing_cost += Decimal(
                            str(cost))

                        logger.info(
                            f"✅ Rebalancing order executed: {
                                action.symbol} {
                                action.action} {
                                quantity:.6f}")

                        # Notify callbacks
                        await self._notify_rebalancing_callbacks(action, result)

                    else:
                        logger.error(
                            f"❌ Rebalancing order failed: {result.get('error')}")

                except Exception as e:
                    logger.error(
                        f"❌ Error executing rebalancing action for {action.symbol}: {e}")

            self.last_rebalance_execution = time.time()

            logger.info(
                f"✅ Rebalancing completed: {len(executed_actions)}/{len(rebalancing_actions)} actions executed")
            return len(
                executed_actions) > 0

        except Exception as e:
            logger.error(
                f"Rebalancing execution error: {e}")
            return False

    async def _find_best_exchange(self, symbol: str):
        """Find the best exchange for a given symbol."""
        try:
            # Check which exchanges support this symbol
            available_exchanges = []

            for exchange_name, connection in self.exchange_manager.connections.items():
                if connection.status == "CONNECTED":
                    try:
                        # Check if symbol is available on this exchange
                        markets = await connection.async_exchange.fetch_markets()
                        symbol_exists = any(
                            market['symbol'] == symbol for market in markets)

                        if symbol_exists:
                            available_exchanges.append(
                                (exchange_name, connection))
                    except Exception as e:
                        logger.debug(
                            f"Could not check {exchange_name} for {symbol}: {e}")

            if not available_exchanges:
                return None

            # For now, return the first available exchange
            # In the future, could implement logic to choose based on fees, liquidity, etc.
            return available_exchanges[
                0][1]

        except Exception as e:
            logger.error(
                f"Error finding best exchange for {symbol}: {e}")
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Try to get price from market data integration
            price_data = self.market_data_integration.get_price(
                symbol)
            if price_data:
                return price_data[
                    'price']

            # Fallback to portfolio positions
            for position in self.positions.values():
                if position.symbol == symbol:
                    return float(
                        position.current_price)

            return None

        except Exception as e:
            logger.error(
                f"Error getting current price for {symbol}: {e}")
            return None

    async def _notify_rebalancing_callbacks(self, action: RebalancingAction, result: Dict[str, Any]):
        """Notify rebalancing callbacks."""
        for callback in self.rebalancing_callbacks:
            try:
                if asyncio.iscoroutinefunction(
                    callback):
                    await callback(action, result)
                else:
                    callback(
                        action, result)
            except Exception as e:
                logger.error(
                    f"Rebalancing callback error: {e}")


    def _on_price_update(self, price_update: PriceUpdate) -> None:
        """Handle real-time price updates."""
        try:
            # Update portfolio with new price
            self.update_prices({price_update.symbol: price_update.price})

            # Notify price update callbacks
            for callback in self.price_update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(price_update))
                    else:
                        callback(price_update)
                except Exception as e:
                    logger.error(f"Price update callback error: {e}")

            # Check if rebalancing is needed
            asyncio.create_task(self.check_rebalancing_needs())

        except Exception as e:
            logger.error(f"Price update handling error: {e}")


    def add_rebalancing_callback(self, callback: Callable[[RebalancingAction, Dict[str, Any]], None]) -> None:
        """Add a callback for rebalancing events."""
        self.rebalancing_callbacks.append(callback)
        logger.info(f"Added rebalancing callback, total callbacks: {len(self.rebalancing_callbacks)}")


    def add_price_update_callback(self, callback: Callable[[PriceUpdate], None]) -> None:
        """Add a callback for price update events."""
        self.price_update_callbacks.append(callback)
        logger.info(f"Added price update callback, total callbacks: {len(self.price_update_callbacks)}")

    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced portfolio summary with rebalancing information."""
        try:
            base_summary = self.get_portfolio_summary()

            # Add rebalancing information
            rebalancing_analysis = self._analyze_portfolio_for_rebalancing()

            enhanced_summary = {
                **base_summary,
                'rebalancing': {
                    'enabled': self.rebalancing_config.enabled,
                    'threshold': self.rebalancing_config.threshold,
                    'target_allocation': self.rebalancing_config.target_allocation,
                    'needs_rebalancing': rebalancing_analysis['needs_rebalancing'],
                    'rebalancing_actions': [
                        {
                            'symbol': action.symbol,
                            'action': action.action,
                            'amount': action.amount,
                            'deviation': action.deviation,
                            'priority': action.priority
                        }
                        for action in rebalancing_analysis['rebalancing_actions']
                    ],
                    'current_allocation': rebalancing_analysis.get('current_allocation', {}),
                    'last_rebalance': self.last_rebalance_execution,
                    'actions_executed': self.rebalancing_actions_executed,
                    'total_cost': float(self.total_rebalancing_cost)
                },
                'market_data': {
                    'price_updates_received': self.price_updates_received,
                    'last_price_update': self.last_price_update,
                    'connection_status': self.market_data_integration.get_connection_status(),
                    'statistics': self.market_data_integration.get_statistics()
                },
                'exchanges': {
                    'status': self.exchange_manager.get_all_status(),
                    'connected_count': len([c for c in self.exchange_manager.connections.values() if c.status == "CONNECTED"])
                }
            }

            return enhanced_summary

        except Exception as e:
            logger.error(f"Error getting enhanced summary: {e}")
            return self.get_portfolio_summary()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the portfolio."""
        try:
            summary = self.get_portfolio_summary()

            # Calculate performance metrics
            total_pnl = summary['realized_pnl'] + summary['unrealized_pnl']
            total_value = summary['total_value']

            # Calculate rebalancing efficiency
            rebalancing_efficiency = 0.0
            if self.total_rebalancing_cost > 0:
                rebalancing_efficiency = float(total_pnl) / float(self.total_rebalancing_cost)

            metrics = {
                'total_pnl': float(total_pnl),
                'total_pnl_percentage': (float(total_pnl) / total_value * 100) if total_value > 0 else 0,
                'realized_pnl': summary['realized_pnl'],
                'unrealized_pnl': summary['unrealized_pnl'],
                'total_value': total_value,
                'rebalancing_efficiency': rebalancing_efficiency,
                'rebalancing_cost': float(self.total_rebalancing_cost),
                'rebalancing_actions': self.rebalancing_actions_executed,
                'price_updates': self.price_updates_received,
                'uptime': time.time() - self.last_price_update if self.last_price_update > 0 else 0
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    async def test_enhanced_portfolio_tracker():
        """Test the enhanced portfolio tracker."""
        config = {
            'exchanges': {
                'binance': {
                    'enabled': True,
                    'websocket_enabled': True,
                    'symbols': ['btcusdt', 'ethusdt']
                }
            },
            'tracked_symbols': ['BTC/USD', 'ETH/USD'],
            'price_update_interval': 5,
            'rebalancing': {
                'enabled': True,
                'threshold': 0.05,
                'interval': 60,
                'target_allocation': {
                    'BTC': 0.6,
                    'ETH': 0.4
                }
            }
        }

        tracker = EnhancedPortfolioTracker(config)

        def price_callback(price_update: PriceUpdate):
            print(f"Price update: {price_update.symbol} = ${price_update.price}")

        def rebalancing_callback(action: RebalancingAction, result: Dict[str, Any]):
            print(f"Rebalancing: {action.symbol} {action.action} {action.amount}")

        tracker.add_price_update_callback(price_callback)
        tracker.add_rebalancing_callback(rebalancing_callback)

        await tracker.start()

        try:
            # Run for 60 seconds
            await asyncio.sleep(60)

            # Print summary
            summary = tracker.get_enhanced_summary()
            print(f"Portfolio Summary: {summary}")

        finally:
            await tracker.stop()
            print("Test completed")

    if __name__ == "__main__":
        asyncio.run(test_enhanced_portfolio_tracker())
