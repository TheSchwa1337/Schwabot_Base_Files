"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trading System
===================
Comprehensive live trading system with portfolio rebalancing, mathematical decision engine,
and direct exchange integration for production trading.
"""

from .math import MathematicalDecisionEngine, MathSignal
from .api.coinbase_direct import CoinbaseDirectAPI, CoinbaseWebSocket
from .api.exchange_connection import ExchangeManager
from .real_time_market_data_integration import RealTimeMarketDataIntegration, PriceUpdate
from .enhanced_portfolio_tracker import EnhancedPortfolioTracker, RebalancingAction
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import time
import os
import asyncio
import logging


logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for live trading system."""
    # Exchange settings
    exchanges: Dict[str, Dict[str, Any]]

    # Portfolio settings
    tracked_symbols: List[str]
    price_update_interval: int = 5

    # Rebalancing settings
    rebalancing_enabled: bool = True
    rebalancing_threshold: float = 0.05
    rebalancing_interval: int = 3600
    target_allocation: Dict[str, float] = None

    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_trades: int = 10
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.05  # 5%

    # Mathematical settings
    math_decision_enabled: bool = True
    math_confidence_threshold: float = 0.7
    math_risk_threshold: float = 0.8

    # Live trading settings
    live_trading_enabled: bool = False
    sandbox_mode: bool = True
    max_slippage: float = 0.001  # 0.1%

    # Monitoring
    enable_logging: bool = True
    enable_alerts: bool = True
    performance_tracking: bool = True


class LiveTradingSystem:
    """Comprehensive live trading system with all components integrated."""

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.is_running = False
        self.start_time = time.time()

        # Initialize components
        self.portfolio_tracker = None
        self.market_data_integration = None
        self.exchange_manager = None
        self.coinbase_api = None
        self.math_engine = None

        # Trading state
        self.daily_trades = 0
        self.last_trade_time = 0
        self.total_pnl = 0.0
        self.open_positions = {}
        self.order_history = []

        # Callbacks
        self.trade_callbacks = []
        self.alert_callbacks = []
        self.performance_callbacks = []

        # Initialize system
        self._initialize_components()

        logger.info("Live trading system initialized")


def _initialize_components(self) -> None:
    """Initialize all trading system components."""
    try:
        # Initialize portfolio tracker
        portfolio_config = {
            'exchanges': self.config.exchanges,
            'tracked_symbols': self.config.tracked_symbols,
            'price_update_interval': self.config.price_update_interval,
            'rebalancing': {
                'enabled': self.config.rebalancing_enabled,
                'threshold': self.config.rebalancing_threshold,
                'interval': self.config.rebalancing_interval,
                'target_allocation': self.config.target_allocation or {}
            }
        }
        self.portfolio_tracker = EnhancedPortfolioTracker(portfolio_config)

        # Initialize market data integration
        self.market_data_integration = RealTimeMarketDataIntegration(portfolio_config)

        # Initialize exchange manager
        self.exchange_manager = ExchangeManager(portfolio_config)

        # Initialize Coinbase direct API if configured
        if 'coinbase' in self.config.exchanges and self.config.exchanges['coinbase'].get('enabled', False):
            coinbase_config = self.config.exchanges['coinbase']
            api_key = os.getenv('COINBASE_API_KEY', '')
            secret = os.getenv('COINBASE_SECRET', '')
            passphrase = os.getenv('COINBASE_PASSPHRASE', '')

            if api_key and secret and passphrase:
                self.coinbase_api = CoinbaseDirectAPI(
                    api_key=api_key,
                    secret=secret,
                    passphrase=passphrase,
                    sandbox=self.config.sandbox_mode
                )
                logger.info("Coinbase direct API initialized")

        # Initialize mathematical decision engine
        if self.config.math_decision_enabled:
            self.math_engine = MathematicalDecisionEngine()
            logger.info("Mathematical decision engine initialized")

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

async def start(self):
    """Start the live trading system."""
    if self.is_running:
        logger.warning("Trading system already running")
        return

    try:
        logger.info("Starting live trading system...")
        self.is_running = True

        # Start portfolio tracker
        await self.portfolio_tracker.start()

        # Start market data integration
        await self.market_data_integration.start()

        # Connect to exchanges
        await self.exchange_manager.connect_all()

        # Connect to Coinbase if available
        if self.coinbase_api:
            await self.coinbase_api.connect()

        # Add callbacks
        self._setup_callbacks()

        # Start trading loop
        asyncio.create_task(self._trading_loop())

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

        logger.info("Live trading system started successfully")

    except Exception as e:
        logger.error(f"Failed to start trading system: {e}")
        self.is_running = False
        raise

async def stop(self):
    """Stop the live trading system."""
    if not self.is_running:
        return

    logger.info("Stopping live trading system...")
    self.is_running = False

    try:
        # Stop portfolio tracker
        if self.portfolio_tracker:
            await self.portfolio_tracker.stop()

        # Stop market data integration
        if self.market_data_integration:
            await self.market_data_integration.stop()

        # Disconnect from exchanges
        if self.exchange_manager:
            await self.exchange_manager.disconnect_all()

        # Disconnect from Coinbase
        if self.coinbase_api:
            await self.coinbase_api.disconnect()

        logger.info("Live trading system stopped")

    except Exception as e:
        logger.error(f"Error stopping trading system: {e}")

def _setup_callbacks(self) -> None:
    """Setup all system callbacks."""
    # Portfolio rebalancing callback
    def rebalancing_callback(action: RebalancingAction, result: Dict[str, Any]):
        logger.info(f"Rebalancing: {action.symbol} {action.action} ${action.amount:.2f}")
        self._notify_trade_callbacks('rebalancing', action, result)

    # Price update callback
    def price_callback(price_update: PriceUpdate):
        self._process_price_update(price_update)

    # Add callbacks
    if self.portfolio_tracker:
        self.portfolio_tracker.add_rebalancing_callback(rebalancing_callback)

    if self.market_data_integration:
        self.market_data_integration.add_price_callback(price_callback)

def _process_price_update(self, price_update: PriceUpdate) -> None:
    """Process incoming price updates."""
    try:
        # Update portfolio with new price
        if self.portfolio_tracker:
            self.portfolio_tracker.update_prices({price_update.symbol: price_update.price})

        # Check for mathematical signals
        if self.math_engine and self.config.math_decision_enabled:
            self._check_math_signals(price_update)

        # Check for stop loss/take profit
        self._check_risk_management(price_update)

    except Exception as e:
        logger.error(f"Error processing price update: {e}")

def _check_math_signals(self, price_update: PriceUpdate) -> None:
    """Check for mathematical trading signals."""
    try:
        # Get historical data for analysis
        # This would need to be implemented based on your data storage
        historical_data = self._get_historical_data(price_update.symbol)

        if historical_data and len(historical_data) > 100:
            # Analyze with mathematical engine
            math_signal = self.math_engine.analyze_market_mathematics(
                price_data=historical_data['prices'],
                volume_data=historical_data['volumes'],
                current_price=price_update.price,
                current_volume=price_update.volume
            )

            # Check if signal meets confidence threshold
            if (math_signal.confidence >= self.config.math_confidence_threshold and
                math_signal.risk_level <= self.config.math_risk_threshold):

                # Execute mathematical signal
                self._execute_math_signal(math_signal)

    except Exception as e:
        logger.error(f"Error checking math signals: {e}")

def _check_risk_management(self, price_update: PriceUpdate) -> None:
    """Check stop loss and take profit levels."""
    try:
        if price_update.symbol in self.open_positions:
            position = self.open_positions[price_update.symbol]
            entry_price = position['entry_price']
            current_price = price_update.price

            # Calculate P&L
            if position['side'] == 'buy':
                pnl_percentage = (current_price - entry_price) / entry_price
            else:
                pnl_percentage = (entry_price - current_price) / entry_price

            # Check stop loss
            if pnl_percentage <= -self.config.stop_loss_percentage:
                logger.warning(f"Stop loss triggered for {price_update.symbol}")
                self._close_position(price_update.symbol, 'stop_loss')

            # Check take profit
            elif pnl_percentage >= self.config.take_profit_percentage:
                logger.info(f"Take profit triggered for {price_update.symbol}")
                self._close_position(price_update.symbol, 'take_profit')

    except Exception as e:
        logger.error(f"Error in risk management: {e}")

async def _trading_loop(self):
    """Main trading loop."""
    while self.is_running:
        try:
            # Check if we can trade
            if not self._can_trade():
                await asyncio.sleep(10)
                continue

            # Check for rebalancing needs
            if self.config.rebalancing_enabled:
                rebalancing_check = await self.portfolio_tracker.check_rebalancing_needs()
                if rebalancing_check['needs_rebalancing']:
                    await self._execute_rebalancing(rebalancing_check['rebalancing_actions'])

            # Check for new opportunities
            await self._check_trading_opportunities()

            # Wait before next iteration
            await asyncio.sleep(self.config.price_update_interval)

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(30)  # Wait longer on error

async def _monitoring_loop(self):
    """Monitoring and performance tracking loop."""
    while self.is_running:
        try:
            # Update performance metrics
            if self.config.performance_tracking:
                await self._update_performance_metrics()

            # Check system health
            await self._health_check()

            # Send alerts if needed
            if self.config.enable_alerts:
                await self._check_alerts()

            # Wait before next check
            await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await asyncio.sleep(60)

def _can_trade(self) -> bool:
    """Check if trading is allowed."""
    # Check if live trading is enabled
    if not self.config.live_trading_enabled:
        return False

    # Check daily trade limit
    if self.daily_trades >= self.config.max_daily_trades:
        return False

    # Check if enough time has passed since last trade
    if time.time() - self.last_trade_time < 60:  # Minimum 1 minute between trades
        return False

    return True

async def _execute_rebalancing(self, rebalancing_actions: List[RebalancingAction]):
    """Execute portfolio rebalancing."""
    try:
        for action in rebalancing_actions:
            if not self._can_trade():
                logger.warning("Cannot execute rebalancing - trading not allowed")
                break

            # Execute the rebalancing action
            success = await self._execute_trade(
                symbol=action.symbol,
                side=action.action,
                amount=action.amount,
                order_type='market',
                reason='rebalancing'
            )

            if success:
                self.daily_trades += 1
                self.last_trade_time = time.time()
                logger.info(f"Rebalancing trade executed: {action.symbol} {action.action} ${action.amount:.2f}")
            else:
                logger.error(f"Failed to execute rebalancing trade: {action.symbol}")

    except Exception as e:
        logger.error(f"Error executing rebalancing: {e}")

async def _execute_trade(self, symbol: str, side: str, amount: float,
order_type: str = 'market', price: float = None,
reason: str = 'manual') -> bool:
    """Execute a trade order."""
    try:
        # Validate trade parameters
        if not self._validate_trade(symbol, side, amount):
            return False

        # Choose best exchange for this trade
        exchange = self._select_best_exchange(symbol)
        if not exchange:
            logger.error(f"No suitable exchange found for {symbol}")
            return False

        # Prepare order
        order_data = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount
        }

        if price:
            order_data['price'] = price

        # Place order
        if exchange == 'coinbase' and self.coinbase_api:
            # Use direct Coinbase API
            order = await self.coinbase_api.place_order(
                product_id=symbol.replace('/', '-'),
                side=side,
                order_type=order_type,
                size=str(amount),
                price=str(price) if price else None
            )
        else:
            # Use CCXT exchange
            connection = self.exchange_manager.get_connection(exchange)
            if connection:
                order = await connection.place_order(order_data)
            else:
                logger.error(f"Exchange connection not available: {exchange}")
                return False

        if order:
            # Record the trade
            self._record_trade(symbol, side, amount, price, order, reason)
            self._notify_trade_callbacks('trade_executed', order, {'reason': reason})
            return True
        else:
            logger.error(f"Failed to place order for {symbol}")
            return False

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def _validate_trade(self, symbol: str, side: str, amount: float) -> bool:
    """Validate trade parameters."""
    # Check position size limit
    portfolio_value = self.portfolio_tracker.get_enhanced_summary()['total_value']
    max_position_value = portfolio_value * self.config.max_position_size

    # Get current price
    current_price = self._get_current_price(symbol)
    if not current_price:
        return False

    position_value = amount * current_price
    if position_value > max_position_value:
        logger.warning(f"Position size too large: ${position_value:.2f} > ${max_position_value:.2f}")
        return False

    return True

def _select_best_exchange(self, symbol: str) -> Optional[str]:
    """Select the best exchange for a given symbol."""
    # Simple logic - can be enhanced with liquidity, fees, etc.
    if symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD'] and self.coinbase_api:
        return 'coinbase'
    elif symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        return 'binance'
    else:
        # Return first available exchange
        for exchange_name, connection in self.exchange_manager.connections.items():
            if connection.status == "CONNECTED":
                return exchange_name

        return None

def _record_trade(self, symbol: str, side: str, amount: float, price: float, order: Dict[str, Any], reason: str):
    """Record a completed trade."""
    trade_record = {
        'timestamp': time.time(),
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'price': price or order.get('price', 0),
        'order_id': order.get('id', ''),
        'reason': reason,
        'exchange': order.get('exchange', 'unknown')
    }

    self.order_history.append(trade_record)

    # Update open positions
    if side == 'buy':
        self.open_positions[symbol] = {
            'side': 'buy',
            'amount': amount,
            'entry_price': price or order.get('price', 0),
            'timestamp': time.time()
        }
    elif side == 'sell' and symbol in self.open_positions:
        # Close position
        del self.open_positions[symbol]

def _get_current_price(self, symbol: str) -> Optional[float]:
    """Get current price for a symbol."""
    try:
        # Try market data integration first
        if self.market_data_integration:
            price_data = self.market_data_integration.get_price(symbol)
            if price_data:
                return price_data['price']

        # Fallback to exchange
        exchange = self._select_best_exchange(symbol)
        if exchange and exchange != 'coinbase':
            connection = self.exchange_manager.get_connection(exchange)
            if connection:
                market_data = connection.market_data_cache.get(symbol)
                if market_data:
                    return market_data['price']

        return None

    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None

def _get_historical_data(self, symbol: str) -> Optional[Dict[str, List[float]]]:
    """Get historical data for mathematical analysis."""
    # This would need to be implemented based on your data storage
    # For now, return None to indicate no data available
    return None

def _execute_math_signal(self, math_signal: MathSignal) -> None:
    """Execute a mathematical trading signal."""
    try:
        if math_signal.decision.value.startswith('enter'):
            # Open position
            asyncio.create_task(self._execute_trade(
                symbol=math_signal.symbol,
                side='buy' if 'buy' in math_signal.decision.value else 'sell',
                amount=self._calculate_position_size(math_signal),
                order_type='market',
                reason='math_signal'
            ))
        elif math_signal.decision.value.startswith('exit'):
            # Close position
            if math_signal.symbol in self.open_positions:
                asyncio.create_task(self._close_position(math_signal.symbol, 'math_signal'))

    except Exception as e:
        logger.error(f"Error executing math signal: {e}")

def _calculate_position_size(self, math_signal: MathSignal) -> float:
    """Calculate position size based on mathematical signal."""
    portfolio_value = self.portfolio_tracker.get_enhanced_summary()['total_value']
    base_size = portfolio_value * self.config.max_position_size

    # Adjust based on confidence
    confidence_multiplier = math_signal.confidence
    risk_adjustment = 1.0 - math_signal.risk_level

    final_size = base_size * confidence_multiplier * risk_adjustment

    # Get current price to convert to amount
    current_price = self._get_current_price(math_signal.symbol)
    if current_price:
        return final_size / current_price

    return 0.0

async def _close_position(self, symbol: str, reason: str):
    """Close an open position."""
    if symbol in self.open_positions:
        position = self.open_positions[symbol]
        side = 'sell' if position['side'] == 'buy' else 'buy'

        await self._execute_trade(
            symbol=symbol,
            side=side,
            amount=position['amount'],
            order_type='market',
            reason=reason
        )

async def _update_performance_metrics(self):
    """Update performance tracking metrics."""
    try:
        summary = self.portfolio_tracker.get_enhanced_summary()
        self.total_pnl = summary.get('total_pnl', 0.0)

        # Calculate additional metrics
        metrics = {
            'total_pnl': self.total_pnl,
            'daily_trades': self.daily_trades,
            'open_positions': len(self.open_positions),
            'uptime': time.time() - self.start_time,
            'portfolio_value': summary.get('total_value', 0.0)
        }

        self._notify_performance_callbacks(metrics)

    except Exception as e:
        logger.error(f"Error updating performance metrics: {e}")

async def _health_check(self):
    """Perform system health check."""
    try:
        # Check exchange connections
        health_status = await self.exchange_manager.health_check_all()

        # Check Coinbase API
        if self.coinbase_api:
            coinbase_health = await self.coinbase_api.health_check()
            health_status['coinbase'] = coinbase_health

        # Log health status
        unhealthy_exchanges = [k for k, v in health_status.items() if not v]
        if unhealthy_exchanges:
            logger.warning(f"Unhealthy exchanges: {unhealthy_exchanges}")

    except Exception as e:
        logger.error(f"Error in health check: {e}")

async def _check_alerts(self):
    """Check and send alerts."""
    try:
        # Check for significant P&L changes
        if abs(self.total_pnl) > 100:  # $100 threshold
            self._notify_alert_callbacks('significant_pnl', {
                'pnl': self.total_pnl,
                'timestamp': time.time()
            })

        # Check for high trade frequency
        if self.daily_trades > self.config.max_daily_trades * 0.8:
            self._notify_alert_callbacks('high_trade_frequency', {
                'trades': self.daily_trades,
                'limit': self.config.max_daily_trades
            })

    except Exception as e:
        logger.error(f"Error checking alerts: {e}")

async def _check_trading_opportunities(self):
    """Check for new trading opportunities."""
    # Real mathematical decision logic
    try:
        from core.clean_unified_math import CleanUnifiedMathSystem
        math_system = CleanUnifiedMathSystem()

        # Get historical price data for analysis
        price_data = self._get_historical_data("BTC/USDT")
        if not price_data or 'prices' not in price_data:
            return False  # No data available

        prices = price_data['prices']
        if len(prices) < 10:  # Need minimum data points
            return False

        # Calculate decision metrics
        volatility = math_system.calculate_volatility(prices)
        momentum = math_system.calculate_momentum(prices)
        trend_strength = math_system.calculate_trend_strength(prices)

        # Combine metrics for decision
        decision_score = (momentum * 0.4 + trend_strength * 0.4 + (1 - volatility) * 0.2)

        return decision_score > 0.6  # Threshold for action

    except Exception as e:
        logger.error(f"Error in mathematical decision: {e}")
        return False  # Conservative fallback

def add_trade_callback(self, callback: Callable) -> None:
    """Add trade event callback."""
    self.trade_callbacks.append(callback)

def add_alert_callback(self, callback: Callable) -> None:
    """Add alert callback."""
    self.alert_callbacks.append(callback)

def add_performance_callback(self, callback: Callable) -> None:
    """Add performance tracking callback."""
    self.performance_callbacks.append(callback)

def _notify_trade_callbacks(self, event_type: str, data: Any, metadata: Dict[str, Any] = None) -> None:
    """Notify trade callbacks."""
    for callback in self.trade_callbacks:
        try:
            callback(event_type, data, metadata or {})
        except Exception as e:
            logger.error(f"Trade callback error: {e}")

def _notify_alert_callbacks(self, alert_type: str, data: Dict[str, Any]) -> None:
    """Notify alert callbacks."""
    for callback in self.alert_callbacks:
        try:
            callback(alert_type, data)
        except Exception as e:
            logger.error(f"Alert callback error: {e}")

def _notify_performance_callbacks(self, metrics: Dict[str, Any]) -> None:
    """Notify performance callbacks."""
    for callback in self.performance_callbacks:
        try:
            callback(metrics)
        except Exception as e:
            logger.error(f"Performance callback error: {e}")

def get_system_status(self) -> Dict[str, Any]:
    """Get comprehensive system status."""
    return {
        'is_running': self.is_running,
        'uptime': time.time() - self.start_time,
        'daily_trades': self.daily_trades,
        'total_pnl': self.total_pnl,
        'open_positions': len(self.open_positions),
        'portfolio_value': self.portfolio_tracker.get_enhanced_summary()['total_value'] if self.portfolio_tracker else 0.0,
        'exchanges': self.exchange_manager.get_all_status() if self.exchange_manager else {},
        'coinbase_status': self.coinbase_api.get_status() if self.coinbase_api else None,
        'config': {
            'live_trading_enabled': self.config.live_trading_enabled,
            'sandbox_mode': self.config.sandbox_mode,
            'rebalancing_enabled': self.config.rebalancing_enabled,
            'math_decision_enabled': self.config.math_decision_enabled
        }
    }

def get_trading_history(self) -> List[Dict[str, Any]]:
    """Get trading history."""
    return self.order_history.copy()

def reset_daily_trades(self) -> None:
    """Reset daily trade counter."""
    self.daily_trades = 0
    logger.info("Daily trade counter reset")


def create_live_trading_system(config: TradingConfig) -> LiveTradingSystem:
    """Factory function to create a live trading system."""
    return LiveTradingSystem(config)