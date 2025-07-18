"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 UNIFIED MARKET DATA INTEGRATION SERVICE - SCHWABOT COMPLETE INTEGRATION
==========================================================================

Complete integration layer that connects real market data feed to ALL trading components:
- Portfolio Tracker → Real-time position updates and rebalancing
- Risk Manager → Live risk assessment and position sizing
- Trading Logic → Mathematical signal generation and execution
- Order Execution → Real-time order placement and management
- Performance Tracking → Live P&L and performance metrics

This service ensures NO trading components are left behind while maintaining
your sophisticated mathematical architecture and enhancing real-time performance.

Features:
- Real-time market data from multiple exchanges (Coinbase, Binance, Kraken)
- Live portfolio tracking with automatic rebalancing
- Real-time risk assessment with position sizing
- Mathematical signal generation with confidence scoring
- Live order execution with slippage protection
- Performance monitoring with real-time metrics
- Error handling and recovery mechanisms
- Circuit breakers and safety controls

Author: Schwabot Team
Date: 2025-01-02
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union

# Import core components
from .real_market_data_feed import RealMarketDataFeed, MarketDataPoint, OrderBookData, TradeData
from .enhanced_portfolio_tracker import EnhancedPortfolioTracker, RebalancingAction
from .risk_manager import RiskManager, PortfolioRisk, PositionRisk
# Fix circular import - use lazy import
# from .unified_mathematical_bridge import UnifiedMathematicalBridge
from .live_trading_system import LiveTradingSystem, TradingConfig
from .pure_profit_calculator import PureProfitCalculator

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
"""Class for Schwabot trading functionality."""
"""Trading signal with mathematical confidence and risk assessment."""
symbol: str
signal_type: str  # 'BUY', 'SELL', 'HOLD'
confidence: float
strength: float
price: float
volume: float
timestamp: float
mathematical_signature: str
risk_assessment: Dict[str, Any]
position_size: float
stop_loss: float
take_profit: float
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
"""Class for Schwabot trading functionality."""
"""Current portfolio state with real-time data."""
total_value: float
available_balance: float
positions: Dict[str, float]
unrealized_pnl: float
realized_pnl: float
risk_metrics: Dict[str, Any]
timestamp: float
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketState:
"""Class for Schwabot trading functionality."""
"""Current market state with real-time data."""
symbol: str
current_price: float
volume_24h: float
change_24h: float
bid: float
ask: float
spread: float
order_book_depth: Dict[str, float]
timestamp: float
exchange: str
metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedMarketDataIntegrationService:
"""Class for Schwabot trading functionality."""
"""
Unified Market Data Integration Service - Complete Trading System Integration

This service connects the real market data feed to ALL trading components,
ensuring no components are left behind while maintaining your sophisticated
mathematical architecture and enhancing real-time performance.
"""

def __init__(self, config: Dict[str, Any]) -> None:
"""Initialize the unified market data integration service."""
self.config = config
self.is_running = False
self.start_time = time.time()

# Initialize core components
self.market_data_feed = None
self.portfolio_tracker = None
self.risk_manager = None
self.mathematical_bridge = None
self.trading_system = None
self.profit_calculator = None

# Data storage
self.current_market_states: Dict[str, MarketState] = {}
self.current_portfolio_state: Optional[PortfolioState] = None
self.trading_signals: List[TradingSignal] = []

# Performance tracking
self.total_signals_generated = 0
self.total_trades_executed = 0
self.total_pnl = 0.0
self.performance_metrics = {}

# Callbacks
self.signal_callbacks = []
self.portfolio_callbacks = []
self.risk_callbacks = []
self.trade_callbacks = []

# Initialize system
self._initialize_components()

logger.info("🧠 Unified Market Data Integration Service initialized")

def _initialize_components(self) -> None:
"""Initialize all trading system components."""
try:
# Initialize real market data feed
market_data_config = {
'symbols': self.config.get('symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD']),
'coinbase': {'enabled': True},
'binance': {'enabled': True},
'kraken': {'enabled': True}
}
self.market_data_feed = RealMarketDataFeed(market_data_config)
logger.info("✅ Real Market Data Feed initialized")

# Initialize enhanced portfolio tracker
portfolio_config = {
'exchanges': self.config.get('exchanges', {}),
'tracked_symbols': self.config.get('symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD']),
'price_update_interval': self.config.get('price_update_interval', 5),
'rebalancing': {
'enabled': self.config.get('rebalancing_enabled', True),
'threshold': self.config.get('rebalancing_threshold', 0.05),
'interval': self.config.get('rebalancing_interval', 3600),
'target_allocation': self.config.get('target_allocation', {})
}
}
self.portfolio_tracker = EnhancedPortfolioTracker(portfolio_config)
logger.info("✅ Enhanced Portfolio Tracker initialized")

# Initialize risk manager
risk_config = {
'risk_tolerance': self.config.get('risk_tolerance', 0.02),
'max_portfolio_risk': self.config.get('max_portfolio_risk', 0.05),
'position_sizing_enabled': True,
'circuit_breakers_enabled': True
}
self.risk_manager = RiskManager(**risk_config)
logger.info("✅ Risk Manager initialized")

# Initialize unified mathematical bridge
math_config = {
'enable_quantum_integration': True,
'enable_phantom_integration': True,
'enable_homology_integration': True,
'enable_tensor_integration': True,
'enable_unified_math_integration': True,
'enable_risk_integration': True,
'enable_profit_integration': True,
'confidence_threshold': self.config.get('math_confidence_threshold', 0.7)
}
# Lazy import to avoid circular import
from .unified_mathematical_bridge import UnifiedMathematicalBridge
self.mathematical_bridge = UnifiedMathematicalBridge(math_config)
logger.info("✅ Unified Mathematical Bridge initialized")

# Initialize trading system
trading_config = TradingConfig(
exchanges=self.config.get('exchanges', {}),
tracked_symbols=self.config.get('symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD']),
price_update_interval=self.config.get('price_update_interval', 5),
rebalancing_enabled=self.config.get('rebalancing_enabled', True),
rebalancing_threshold=self.config.get('rebalancing_threshold', 0.05),
rebalancing_interval=self.config.get('rebalancing_interval', 3600),
target_allocation=self.config.get('target_allocation', {}),
max_position_size=self.config.get('max_position_size', 0.1),
max_daily_trades=self.config.get('max_daily_trades', 10),
stop_loss_percentage=self.config.get('stop_loss_percentage', 0.02),
take_profit_percentage=self.config.get('take_profit_percentage', 0.05),
math_decision_enabled=True,
math_confidence_threshold=self.config.get('math_confidence_threshold', 0.7),
math_risk_threshold=self.config.get('math_risk_threshold', 0.8),
live_trading_enabled=self.config.get('live_trading_enabled', False),
sandbox_mode=self.config.get('sandbox_mode', True),
max_slippage=self.config.get('max_slippage', 0.001),
enable_logging=True,
enable_alerts=True,
performance_tracking=True
)
self.trading_system = LiveTradingSystem(trading_config)
logger.info("✅ Live Trading System initialized")

# Initialize profit calculator
profit_config = {
'strategy_params': self.config.get('strategy_params', {}),
'risk_adjustment': True,
'portfolio_optimization': True
}
self.profit_calculator = PureProfitCalculator(profit_config)
logger.info("✅ Pure Profit Calculator initialized")

logger.info("✅ All components initialized successfully")

except Exception as e:
logger.error(f"❌ Failed to initialize components: {e}")
raise

async def start(self):
"""Start the unified market data integration service."""
if self.is_running:
logger.warning("Service already running")
return

try:
logger.info("🚀 Starting Unified Market Data Integration Service...")
self.is_running = True

# Start market data feed
await self.market_data_feed.initialize()

# Start portfolio tracker
await self.portfolio_tracker.start()

# Start trading system
await self.trading_system.start()

# Setup callbacks
self._setup_callbacks()

# Start integration loops
asyncio.create_task(self._market_data_integration_loop())
asyncio.create_task(self._portfolio_integration_loop())
asyncio.create_task(self._risk_integration_loop())
asyncio.create_task(self._trading_integration_loop())
asyncio.create_task(self._performance_monitoring_loop())

logger.info("✅ Unified Market Data Integration Service started successfully")

except Exception as e:
logger.error(f"❌ Failed to start service: {e}")
self.is_running = False
raise

async def stop(self):
"""Stop the unified market data integration service."""
if not self.is_running:
return

logger.info("🛑 Stopping Unified Market Data Integration Service...")
self.is_running = False

try:
# Stop all components
if self.market_data_feed:
# Stop market data feed
pass

if self.portfolio_tracker:
await self.portfolio_tracker.stop()

if self.trading_system:
await self.trading_system.stop()

logger.info("✅ Service stopped successfully")

except Exception as e:
logger.error(f"❌ Error stopping service: {e}")

def _setup_callbacks(self) -> None:
"""Setup callbacks for all components."""

# Market data callbacks
def market_data_callback(data: MarketDataPoint):
asyncio.create_task(self._process_market_data_update(data))

def order_book_callback(order_book: OrderBookData):
asyncio.create_task(self._process_order_book_update(order_book))

def trade_callback(trade: TradeData):
asyncio.create_task(self._process_trade_update(trade))

# Register callbacks
self.market_data_feed.register_data_callback(market_data_callback)
self.market_data_feed.register_order_book_callback(order_book_callback)
self.market_data_feed.register_trade_callback(trade_callback)

# Portfolio callbacks
def portfolio_callback(action: RebalancingAction, result: Dict[str, Any]):
asyncio.create_task(self._process_portfolio_update(action, result))

self.portfolio_tracker.add_rebalancing_callback(portfolio_callback)

logger.info("✅ All callbacks registered")

async def _market_data_integration_loop(self):
"""Main market data integration loop."""
while self.is_running:
try:
# Update market states
await self._update_market_states()

# Generate trading signals
await self._generate_trading_signals()

# Wait for next iteration
await asyncio.sleep(self.config.get('market_data_interval', 1))

except Exception as e:
logger.error(f"❌ Market data integration loop error: {e}")
await asyncio.sleep(5)

async def _portfolio_integration_loop(self):
"""Main portfolio integration loop."""
while self.is_running:
try:
# Update portfolio state
await self._update_portfolio_state()

# Check rebalancing needs
await self._check_rebalancing_needs()

# Wait for next iteration
await asyncio.sleep(self.config.get('portfolio_interval', 5))

except Exception as e:
logger.error(f"❌ Portfolio integration loop error: {e}")
await asyncio.sleep(10)

async def _risk_integration_loop(self):
"""Main risk integration loop."""
while self.is_running:
try:
# Update risk assessment
await self._update_risk_assessment()

# Check circuit breakers
await self._check_circuit_breakers()

# Wait for next iteration
await asyncio.sleep(self.config.get('risk_interval', 10))

except Exception as e:
logger.error(f"❌ Risk integration loop error: {e}")
await asyncio.sleep(15)

async def _trading_integration_loop(self):
"""Main trading integration loop."""
while self.is_running:
try:
# Process trading signals
await self._process_trading_signals()

# Execute pending orders
await self._execute_pending_orders()

# Wait for next iteration
await asyncio.sleep(self.config.get('trading_interval', 2))

except Exception as e:
logger.error(f"❌ Trading integration loop error: {e}")
await asyncio.sleep(5)

async def _performance_monitoring_loop(self):
"""Main performance monitoring loop."""
while self.is_running:
try:
# Update performance metrics
await self._update_performance_metrics()

# Generate performance reports
await self._generate_performance_reports()

# Wait for next iteration
await asyncio.sleep(self.config.get('performance_interval', 60))

except Exception as e:
logger.error(f"❌ Performance monitoring loop error: {e}")
await asyncio.sleep(30)

async def _process_market_data_update(self, data: MarketDataPoint):
"""Process market data update."""
try:
# Update market state
market_state = MarketState(
symbol=data.symbol,
current_price=data.price,
volume_24h=data.volume_24h,
change_24h=data.change_24h,
bid=data.bid,
ask=data.ask,
spread=data.spread,
order_book_depth={},
timestamp=data.timestamp,
exchange=data.exchange,
metadata={
'high_24h': data.high_24h,
'low_24h': data.low_24h,
'change_percent_24h': data.change_percent_24h,
'market_cap': data.market_cap,
'circulating_supply': data.circulating_supply
}
)

self.current_market_states[data.symbol] = market_state

# Notify callbacks
self._notify_signal_callbacks('market_data_update', market_state)

logger.debug(f"📊 Market data updated: {data.symbol} @ {data.price}")

except Exception as e:
logger.error(f"❌ Market data processing error: {e}")

async def _process_order_book_update(self, order_book: OrderBookData):
"""Process order book update."""
try:
# Update order book depth in market state
if order_book.symbol in self.current_market_states:
self.current_market_states[order_book.symbol].order_book_depth = {
'bids': order_book.bids,
'asks': order_book.asks,
'best_bid': order_book.best_bid,
'best_ask': order_book.best_ask,
'spread': order_book.spread,
'total_bid_volume': order_book.total_bid_volume,
'total_ask_volume': order_book.total_ask_volume
}

logger.debug(f"📚 Order book updated: {order_book.symbol}")

except Exception as e:
logger.error(f"❌ Order book processing error: {e}")

async def _process_trade_update(self, trade: TradeData):
"""Process trade update."""
try:
# Update market state with trade data
if trade.symbol in self.current_market_states:
self.current_market_states[trade.symbol].current_price = trade.price
self.current_market_states[trade.symbol].timestamp = trade.timestamp

logger.debug(f"💱 Trade executed: {trade.symbol} {trade.side} {trade.volume} @ {trade.price}")

except Exception as e:
logger.error(f"❌ Trade processing error: {e}")

async def _process_portfolio_update(self, action: RebalancingAction, result: Dict[str, Any]):
"""Process portfolio update."""
try:
# Update portfolio state
await self._update_portfolio_state()

# Notify callbacks
self._notify_portfolio_callbacks('rebalancing_action', action, result)

logger.info(f"🔄 Portfolio rebalancing: {action.symbol} {action.action} {action.amount}")

except Exception as e:
logger.error(f"❌ Portfolio processing error: {e}")

async def _update_market_states(self):
"""Update all market states."""
try:
for symbol in self.config.get('symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD']):
# Get latest price from market data feed
price = self.market_data_feed.get_latest_price(symbol)
if price:
# Update market state if exists
if symbol in self.current_market_states:
self.current_market_states[symbol].current_price = price
self.current_market_states[symbol].timestamp = time.time()

except Exception as e:
logger.error(f"❌ Market states update error: {e}")

async def _update_portfolio_state(self):
"""Update current portfolio state."""
try:
# Get portfolio summary
portfolio_summary = self.portfolio_tracker.get_enhanced_summary()

# Create portfolio state
self.current_portfolio_state = PortfolioState(
total_value=portfolio_summary.get('total_value', 0.0),
available_balance=portfolio_summary.get('available_balance', 0.0),
positions=portfolio_summary.get('positions', {}),
unrealized_pnl=portfolio_summary.get('unrealized_pnl', 0.0),
realized_pnl=portfolio_summary.get('realized_pnl', 0.0),
risk_metrics=portfolio_summary.get('risk_metrics', {}),
timestamp=time.time(),
metadata=portfolio_summary
)

# Notify callbacks
self._notify_portfolio_callbacks('portfolio_update', self.current_portfolio_state)

except Exception as e:
logger.error(f"❌ Portfolio state update error: {e}")

async def _update_risk_assessment(self):
"""Update risk assessment."""
try:
if not self.current_portfolio_state:
return

# Get portfolio data for risk assessment
portfolio_data = {
'total_value': self.current_portfolio_state.total_value,
'positions': self.current_portfolio_state.positions,
'market_states': self.current_market_states
}

# Assess portfolio risk
portfolio_risk = self.risk_manager.assess_portfolio_risk(portfolio_data)

# Update portfolio state with risk metrics
if self.current_portfolio_state:
self.current_portfolio_state.risk_metrics = {
'var_95': portfolio_risk.risk_metrics.var_95,
'var_99': portfolio_risk.risk_metrics.var_99,
'cvar_95': portfolio_risk.risk_metrics.cvar_95,
'cvar_99': portfolio_risk.risk_metrics.cvar_99,
'sharpe_ratio': portfolio_risk.risk_metrics.sharpe_ratio,
'max_drawdown': portfolio_risk.risk_metrics.max_drawdown,
'volatility': portfolio_risk.risk_metrics.volatility,
'risk_level': portfolio_risk.risk_level.value
}

# Notify callbacks
self._notify_risk_callbacks('risk_update', portfolio_risk)

except Exception as e:
logger.error(f"❌ Risk assessment update error: {e}")

async def _generate_trading_signals(self):
"""Generate trading signals using mathematical bridge."""
try:
if not self.current_market_states or not self.current_portfolio_state:
return

# Prepare market data for mathematical bridge
market_data = {}
for symbol, market_state in self.current_market_states.items():
market_data[symbol] = {
'price': market_state.current_price,
'volume': market_state.volume_24h,
'change_24h': market_state.change_24h,
'bid': market_state.bid,
'ask': market_state.ask,
'spread': market_state.spread,
'timestamp': market_state.timestamp
}

# Prepare portfolio state
portfolio_state = {
'total_value': self.current_portfolio_state.total_value,
'available_balance': self.current_portfolio_state.available_balance,
'positions': self.current_portfolio_state.positions,
'risk_metrics': self.current_portfolio_state.risk_metrics
}

# Integrate mathematical systems
math_result = self.mathematical_bridge.integrate_all_mathematical_systems(
market_data, portfolio_state
)

if math_result.success:
# Generate trading signals based on mathematical integration
signals = await self._create_trading_signals_from_math_result(math_result, market_data)

# Add signals to list
self.trading_signals.extend(signals)
self.total_signals_generated += len(signals)

# Notify callbacks
for signal in signals:
self._notify_signal_callbacks('trading_signal', signal)

logger.info(f"🎯 Generated {len(signals)} trading signals")

except Exception as e:
logger.error(f"❌ Trading signal generation error: {e}")

async def _create_trading_signals_from_math_result(self, math_result, market_data: Dict[str, Any]) -> List[TradingSignal]:
"""Create trading signals from mathematical integration result."""
signals = []

try:
for connection in math_result.connections:
# Extract signal information from mathematical connection
if connection.connection_type.value in ['quantum_to_phantom', 'homology_to_signal']:
# Create trading signal
for symbol, data in market_data.items():
signal = TradingSignal(
symbol=symbol,
signal_type=self._determine_signal_type(connection),
confidence=connection.connection_strength,
strength=connection.connection_strength,
price=data['price'],
volume=data['volume'],
timestamp=time.time(),
mathematical_signature=connection.mathematical_signature,
risk_assessment=self._assess_signal_risk(connection, symbol),
position_size=self._calculate_position_size(connection, symbol),
stop_loss=self._calculate_stop_loss(connection, symbol, data['price']),
take_profit=self._calculate_take_profit(connection, symbol, data['price']),
metadata={
'connection_type': connection.connection_type.value,
'source_system': connection.source_system,
'target_system': connection.target_system,
'performance_metrics': connection.performance_metrics
}
)
signals.append(signal)

except Exception as e:
logger.error(f"❌ Signal creation error: {e}")

return signals

def _determine_signal_type(self, connection) -> str:
"""Determine signal type from mathematical connection."""
# This is a simplified logic - you can enhance this based on your mathematical systems
if connection.connection_strength > 0.7:
return 'BUY'
elif connection.connection_strength < 0.3:
return 'SELL'
else:
return 'HOLD'

def _assess_signal_risk(self, connection, symbol: str) -> Dict[str, Any]:
"""Assess risk for a trading signal."""
try:
# Get current risk metrics
risk_metrics = self.current_portfolio_state.risk_metrics if self.current_portfolio_state else {}

return {
'risk_level': risk_metrics.get('risk_level', 'medium'),
'var_95': risk_metrics.get('var_95', 0.0),
'max_drawdown': risk_metrics.get('max_drawdown', 0.0),
'volatility': risk_metrics.get('volatility', 0.0),
'signal_confidence': connection.connection_strength,
'symbol_risk': self._calculate_symbol_risk(symbol)
}
except Exception as e:
logger.error(f"❌ Signal risk assessment error: {e}")
return {'risk_level': 'high', 'signal_confidence': 0.0}

def _calculate_symbol_risk(self, symbol: str) -> float:
"""Calculate symbol-specific risk."""
try:
if symbol in self.current_market_states:
market_state = self.current_market_states[symbol]
# Calculate risk based on volatility and spread
volatility_risk = abs(market_state.change_24h) / 100.0
spread_risk = market_state.spread / market_state.current_price
return min(volatility_risk + spread_risk, 1.0)
return 0.5
except Exception as e:
logger.error(f"❌ Symbol risk calculation error: {e}")
return 0.5

def _calculate_position_size(self, connection, symbol: str) -> float:
"""Calculate position size for trading signal."""
try:
if not self.current_portfolio_state:
return 0.0

# Base position size on confidence and available balance
base_size = self.current_portfolio_state.available_balance * 0.1  # 10% of available balance
confidence_multiplier = connection.connection_strength

# Apply risk adjustment
risk_metrics = self.current_portfolio_state.risk_metrics
risk_multiplier = 1.0 - risk_metrics.get('var_95', 0.0)

position_size = base_size * confidence_multiplier * risk_multiplier

# Apply maximum position size limit
max_position = self.current_portfolio_state.total_value * self.config.get('max_position_size', 0.1)

return min(position_size, max_position)

except Exception as e:
logger.error(f"❌ Position size calculation error: {e}")
return 0.0

def _calculate_stop_loss(self, connection, symbol: str, current_price: float) -> float:
"""Calculate stop loss for trading signal."""
try:
# Base stop loss percentage
base_stop_loss = self.config.get('stop_loss_percentage', 0.02)

# Adjust based on signal confidence
confidence_adjustment = 1.0 - connection.connection_strength
adjusted_stop_loss = base_stop_loss * (1.0 + confidence_adjustment)

return current_price * (1.0 - adjusted_stop_loss)

except Exception as e:
logger.error(f"❌ Stop loss calculation error: {e}")
return current_price * 0.98  # Default 2% stop loss

def _calculate_take_profit(self, connection, symbol: str, current_price: float) -> float:
"""Calculate take profit for trading signal."""
try:
# Base take profit percentage
base_take_profit = self.config.get('take_profit_percentage', 0.05)

# Adjust based on signal confidence
confidence_adjustment = connection.connection_strength
adjusted_take_profit = base_take_profit * (1.0 + confidence_adjustment)

return current_price * (1.0 + adjusted_take_profit)

except Exception as e:
logger.error(f"❌ Take profit calculation error: {e}")
return current_price * 1.05  # Default 5% take profit

async def _check_rebalancing_needs(self):
"""Check if portfolio rebalancing is needed."""
try:
if not self.portfolio_tracker:
return

# Check rebalancing needs
rebalancing_result = await self.portfolio_tracker.check_rebalancing_needs()

if rebalancing_result.get('needs_rebalancing', False):
logger.info(f"🔄 Rebalancing needed: {rebalancing_result.get('reason', 'Unknown')}")

# Execute rebalancing
if rebalancing_result.get('rebalancing_actions'):
await self.portfolio_tracker.execute_rebalancing(
rebalancing_result['rebalancing_actions']
)

except Exception as e:
logger.error(f"❌ Rebalancing check error: {e}")

async def _check_circuit_breakers(self):
"""Check circuit breakers."""
try:
if not self.risk_manager:
return

# Get system status
system_status = self.risk_manager.get_system_status()

if system_status.get('safe_mode') != 'normal':
logger.warning(f"⚠️ Circuit breaker active: {system_status.get('safe_mode')}")

# Notify callbacks
self._notify_risk_callbacks('circuit_breaker', system_status)

except Exception as e:
logger.error(f"❌ Circuit breaker check error: {e}")

async def _process_trading_signals(self):
"""Process trading signals and execute trades."""
try:
# Process recent signals
recent_signals = [s for s in self.trading_signals
if time.time() - s.timestamp < 60]  # Last minute

for signal in recent_signals:
# Check if signal meets execution criteria
if self._should_execute_signal(signal):
# Execute trade
await self._execute_trade_signal(signal)

# Remove signal from list
self.trading_signals.remove(signal)

except Exception as e:
logger.error(f"❌ Trading signal processing error: {e}")

def _should_execute_signal(self, signal: TradingSignal) -> bool:
"""Check if signal should be executed."""
try:
# Check confidence threshold
if signal.confidence < self.config.get('signal_confidence_threshold', 0.7):
return False

# Check risk level
if signal.risk_assessment.get('risk_level') == 'critical':
return False

# Check position size
if signal.position_size <= 0:
return False

# Check if we can trade this symbol
if not self.risk_manager.can_trade_symbol(signal.symbol):
return False

return True

except Exception as e:
logger.error(f"❌ Signal execution check error: {e}")
return False

async def _execute_trade_signal(self, signal: TradingSignal):
"""Execute a trading signal."""
try:
logger.info(f"🚀 Executing trade: {signal.symbol} {signal.signal_type} "
f"${signal.position_size:.2f} @ ${signal.price:.2f}")

# Execute trade through trading system
if self.trading_system:
success = await self.trading_system._execute_trade(
symbol=signal.symbol,
side=signal.signal_type.lower(),
amount=signal.position_size,
order_type='market',
price=signal.price,
reason=f"mathematical_signal_{signal.confidence:.3f}"
)

if success:
self.total_trades_executed += 1
logger.info(f"✅ Trade executed successfully")
else:
logger.warning(f"⚠️ Trade execution failed")

# Notify callbacks
self._notify_trade_callbacks('trade_executed', signal)

except Exception as e:
logger.error(f"❌ Trade execution error: {e}")

async def _execute_pending_orders(self):
"""Execute pending orders."""
try:
# This would integrate with your order execution system
# For now, we'll just log that we're checking for pending orders
pass

except Exception as e:
logger.error(f"❌ Pending orders execution error: {e}")

async def _update_performance_metrics(self):
"""Update performance metrics."""
try:
# Calculate performance metrics
self.performance_metrics = {
'total_signals_generated': self.total_signals_generated,
'total_trades_executed': self.total_trades_executed,
'total_pnl': self.total_pnl,
'uptime': time.time() - self.start_time,
'active_connections': len(self.current_market_states),
'portfolio_value': self.current_portfolio_state.total_value if self.current_portfolio_state else 0.0,
'available_balance': self.current_portfolio_state.available_balance if self.current_portfolio_state else 0.0
}

except Exception as e:
logger.error(f"❌ Performance metrics update error: {e}")

async def _generate_performance_reports(self):
"""Generate performance reports."""
try:
# Generate comprehensive performance report
report = {
'timestamp': time.time(),
'performance_metrics': self.performance_metrics,
'market_states': {symbol: {
'price': state.current_price,
'volume': state.volume_24h,
'change_24h': state.change_24h
} for symbol, state in self.current_market_states.items()},
'portfolio_state': {
'total_value': self.current_portfolio_state.total_value if self.current_portfolio_state else 0.0,
'available_balance': self.current_portfolio_state.available_balance if self.current_portfolio_state else 0.0,
'positions': self.current_portfolio_state.positions if self.current_portfolio_state else {},
'risk_metrics': self.current_portfolio_state.risk_metrics if self.current_portfolio_state else {}
} if self.current_portfolio_state else None,
'active_signals': len(self.trading_signals),
'system_health': 'healthy' if self.is_running else 'stopped'
}

logger.info(f"📊 Performance Report: {report['performance_metrics']}")

except Exception as e:
logger.error(f"❌ Performance report generation error: {e}")

# Callback notification methods
def _notify_signal_callbacks(self, event_type: str, data: Any) -> None:
"""Notify signal callbacks."""
for callback in self.signal_callbacks:
try:
callback(event_type, data)
except Exception as e:
logger.error(f"❌ Signal callback error: {e}")

def _notify_portfolio_callbacks(self, event_type: str, data: Any, metadata: Dict[str, Any] = None) -> None:
"""Notify portfolio callbacks."""
for callback in self.portfolio_callbacks:
try:
callback(event_type, data, metadata or {})
except Exception as e:
logger.error(f"❌ Portfolio callback error: {e}")

def _notify_risk_callbacks(self, event_type: str, data: Any) -> None:
"""Notify risk callbacks."""
for callback in self.risk_callbacks:
try:
callback(event_type, data)
except Exception as e:
logger.error(f"❌ Risk callback error: {e}")

def _notify_trade_callbacks(self, event_type: str, data: Any) -> None:
"""Notify trade callbacks."""
for callback in self.trade_callbacks:
try:
callback(event_type, data)
except Exception as e:
logger.error(f"❌ Trade callback error: {e}")

# Public methods for external access
def add_signal_callback(self, callback: Callable) -> None:
"""Add signal callback."""
self.signal_callbacks.append(callback)

def add_portfolio_callback(self, callback: Callable) -> None:
"""Add portfolio callback."""
self.portfolio_callbacks.append(callback)

def add_risk_callback(self, callback: Callable) -> None:
"""Add risk callback."""
self.risk_callbacks.append(callback)

def add_trade_callback(self, callback: Callable) -> None:
"""Add trade callback."""
self.trade_callbacks.append(callback)

def get_current_market_states(self) -> Dict[str, MarketState]:
"""Get current market states."""
return self.current_market_states.copy()

def get_current_portfolio_state(self) -> Optional[PortfolioState]:
"""Get current portfolio state."""
return self.current_portfolio_state

def get_trading_signals(self) -> List[TradingSignal]:
"""Get current trading signals."""
return self.trading_signals.copy()

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get performance metrics."""
return self.performance_metrics.copy()

def get_system_status(self) -> Dict[str, Any]:
"""Get system status."""
return {
'is_running': self.is_running,
'uptime': time.time() - self.start_time,
'total_signals_generated': self.total_signals_generated,
'total_trades_executed': self.total_trades_executed,
'total_pnl': self.total_pnl,
'active_market_states': len(self.current_market_states),
'active_signals': len(self.trading_signals),
'portfolio_value': self.current_portfolio_state.total_value if self.current_portfolio_state else 0.0
}


# Factory function
def create_unified_market_data_integration_service(config: Dict[str, Any]) -> UnifiedMarketDataIntegrationService:
"""Create a unified market data integration service instance."""
return UnifiedMarketDataIntegrationService(config)


# Singleton instance for global use
unified_market_data_integration_service = UnifiedMarketDataIntegrationService({})


async def main():
"""Main function for testing the unified market data integration service."""
logger.info("🧠 Testing Unified Market Data Integration Service")

# Test configuration
test_config = {
'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
'exchanges': {
'coinbase': {'enabled': True},
'binance': {'enabled': True},
'kraken': {'enabled': True}
},
'price_update_interval': 5,
'rebalancing_enabled': True,
'rebalancing_threshold': 0.05,
'rebalancing_interval': 3600,
'target_allocation': {'BTC/USD': 0.4, 'ETH/USD': 0.4, 'SOL/USD': 0.2},
'max_position_size': 0.1,
'max_daily_trades': 10,
'stop_loss_percentage': 0.02,
'take_profit_percentage': 0.05,
'math_confidence_threshold': 0.7,
'math_risk_threshold': 0.8,
'live_trading_enabled': False,
'sandbox_mode': True,
'max_slippage': 0.001,
'risk_tolerance': 0.02,
'max_portfolio_risk': 0.05,
'signal_confidence_threshold': 0.7,
'market_data_interval': 1,
'portfolio_interval': 5,
'risk_interval': 10,
'trading_interval': 2,
'performance_interval': 60
}

# Create service
service = create_unified_market_data_integration_service(test_config)

# Add callbacks for monitoring
def signal_callback(event_type: str, data: Any):
logger.info(f"📡 Signal event: {event_type}")

def portfolio_callback(event_type: str, data: Any, metadata: Dict[str, Any]):
logger.info(f"💼 Portfolio event: {event_type}")

def risk_callback(event_type: str, data: Any):
logger.info(f"🛡️ Risk event: {event_type}")

def trade_callback(event_type: str, data: Any):
logger.info(f"💱 Trade event: {event_type}")

service.add_signal_callback(signal_callback)
service.add_portfolio_callback(portfolio_callback)
service.add_risk_callback(risk_callback)
service.add_trade_callback(trade_callback)

try:
# Start service
await service.start()

# Run for a period
logger.info("🔄 Running service for 60 seconds...")
await asyncio.sleep(60)

# Get status
status = service.get_system_status()
logger.info(f"📊 Final status: {status}")

except KeyboardInterrupt:
logger.info("🛑 Interrupted by user")
except Exception as e:
logger.error(f"❌ Service error: {e}")
finally:
# Stop service
await service.stop()
logger.info("✅ Service stopped")


if __name__ == "__main__":
asyncio.run(main())