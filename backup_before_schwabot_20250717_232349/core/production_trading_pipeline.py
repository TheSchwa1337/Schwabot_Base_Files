"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Trading Pipeline
===========================
Complete production-ready trading system that integrates:
- Real CCXT exchange connections with API keys
- Portfolio tracking and position management
- Live market data feeds
- Risk management and circuit breakers
- Order execution and balance synchronization
- Performance monitoring and reporting

This system provides a complete, production-ready trading environment
for live cryptocurrency trading with proper risk management and
portfolio tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import ccxt

# Core imports
from core.portfolio_tracker import PortfolioTracker
from core.risk_manager import RiskManager
from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
from core.enhanced_ccxt_trading_engine import EnhancedCCXTTradingEngine

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for production trading."""
exchange_name: str
api_key: str
secret: str
sandbox: bool = True
symbols: List[str] = field(default_factory=lambda: ['BTC/USDC'])
base_currency: str = 'USDC'
initial_balance: Dict[str, float] = field(default_factory=dict)
risk_tolerance: float = 0.2
max_position_size: float = 0.1
max_daily_loss: float = 0.05
enable_circuit_breakers: bool = True
portfolio_sync_interval: int = 30  # seconds
price_update_interval: int = 5    # seconds

@dataclass
class TradingStatus:
"""Class for Schwabot trading functionality."""
"""Current trading system status."""
is_running: bool = False
last_sync: float = 0.0
last_trade: float = 0.0
total_trades: int = 0
successful_trades: int = 0
total_pnl: float = 0.0
current_risk_level: str = 'normal'
circuit_breaker_active: bool = False
error_count: int = 0

class ProductionTradingPipeline:
"""Class for Schwabot trading functionality."""
"""
Complete production-ready trading pipeline.

This class orchestrates the entire trading system:
1. Exchange connection management
2. Portfolio tracking and synchronization
3. Real-time market data processing
4. Risk management and circuit breakers
5. Order execution and position management
6. Performance monitoring and reporting
"""


def __init__(self, config: TradingConfig) -> None:
"""Initialize the production trading pipeline."""
self.config = config
self.status = TradingStatus()

# Initialize core components
self._initialize_exchange()
self._initialize_portfolio_tracker()
self._initialize_risk_manager()
self._initialize_trading_executor()

# Performance tracking
self.performance_history: List[Dict[str, Any]] = []
self.error_log: List[Dict[str, Any]] = []

logger.info("🚀 Production Trading Pipeline initialized")

def _initialize_exchange(self) -> None:
"""Initialize CCXT exchange connection."""
try:
exchange_class = getattr(ccxt, self.config.exchange_name)

self.exchange = exchange_class({
'apiKey': self.config.api_key,
'secret': self.config.secret,
'sandbox': self.config.sandbox,
'enableRateLimit': True,
'options': {
'defaultType': 'spot',
'adjustForTimeDifference': True,
}
})

logger.info(f"✅ Exchange connection established: {self.config.exchange_name}")

except Exception as e:
logger.error(f"❌ Failed to initialize exchange: {e}")
raise

def _initialize_portfolio_tracker(self) -> None:
"""Initialize portfolio tracker with initial balances."""
try:
self.portfolio_tracker = PortfolioTracker(
base_currency=self.config.base_currency,
initial_balances=self.config.initial_balance
)

logger.info(f"✅ Portfolio tracker initialized with base currency: {self.config.base_currency}")

except Exception as e:
logger.error(f"❌ Failed to initialize portfolio tracker: {e}")
raise

def _initialize_risk_manager(self) -> None:
"""Initialize risk manager with configuration."""
try:
risk_config = {
'risk_tolerance': self.config.risk_tolerance,
'max_position_size': self.config.max_position_size,
'max_daily_loss': self.config.max_daily_loss,
'enable_circuit_breakers': self.config.enable_circuit_breakers,
'symbols': self.config.symbols
}

self.risk_manager = RiskManager(risk_config)

logger.info(f"✅ Risk manager initialized with tolerance: {self.config.risk_tolerance}")

except Exception as e:
logger.error(f"❌ Failed to initialize risk manager: {e}")
raise

def _initialize_trading_executor(self) -> None:
"""Initialize trading executor with all components."""
try:
exchange_config = {
'exchange': self.config.exchange_name,
'api_key': self.config.api_key,
'secret': self.config.secret,
'sandbox': self.config.sandbox
}

strategy_config = {
'strategy_type': 'production_entropy',
'symbols': self.config.symbols,
'timeframe': '1m'
}

entropy_config = {
'entropy_threshold': 0.7,
'signal_strength_min': 0.3,
'timing_window': 300
}

risk_config = {
'risk_tolerance': self.config.risk_tolerance,
'max_position_size': self.config.max_position_size,
'max_daily_loss': self.config.max_daily_loss,
'enable_circuit_breakers': self.config.enable_circuit_breakers
}

self.trading_executor = EntropyEnhancedTradingExecutor(
exchange_config=exchange_config,
strategy_config=strategy_config,
entropy_config=entropy_config,
risk_config=risk_config
)

# Override portfolio tracker with our production instance
self.trading_executor.portfolio_tracker = self.portfolio_tracker

logger.info("✅ Trading executor initialized with production components")

except Exception as e:
logger.error(f"❌ Failed to initialize trading executor: {e}")
raise

async def start_trading(self) -> None:
"""Start the production trading system."""
try:
logger.info("🚀 Starting production trading system...")
self.status.is_running = True

# Initial portfolio synchronization
await self.sync_portfolio()

# Start background tasks
tasks = [
asyncio.create_task(self._portfolio_sync_loop()),
asyncio.create_task(self._price_update_loop()),
asyncio.create_task(self._trading_loop()),
asyncio.create_task(self._monitoring_loop())
]

# Wait for all tasks
await asyncio.gather(*tasks)

except Exception as e:
logger.error(f"❌ Trading system failed: {e}")
self.status.is_running = False
raise

async def stop_trading(self) -> None:
"""Stop the production trading system."""
logger.info("🛑 Stopping production trading system...")
self.status.is_running = False

# Close all open positions if needed
await self._close_all_positions()

# Final portfolio sync
await self.sync_portfolio()

async def sync_portfolio(self) -> None:
"""Synchronize portfolio with exchange balances."""
try:
if not self.status.is_running:
return

# Fetch current balances from exchange
balance = await self.exchange.fetch_balance()

# Update portfolio tracker
self.portfolio_tracker.sync_balances(balance)

self.status.last_sync = time.time()

logger.info(f"🔄 Portfolio synchronized - Balances: {self.portfolio_tracker.balances}")

except Exception as e:
logger.error(f"❌ Portfolio sync failed: {e}")
self._log_error("portfolio_sync", str(e))

async def _portfolio_sync_loop(self) -> None:
"""Background loop for portfolio synchronization."""
while self.status.is_running:
try:
await self.sync_portfolio()
await asyncio.sleep(self.config.portfolio_sync_interval)
except Exception as e:
logger.error(f"❌ Portfolio sync loop error: {e}")
await asyncio.sleep(10)

async def _price_update_loop(self) -> None:
"""Background loop for position price updates."""
while self.status.is_running:
try:
# Fetch current prices for all symbols
price_data = {}
for symbol in self.config.symbols:
ticker = await self.exchange.fetch_ticker(symbol)
price_data[symbol] = ticker['last']

# Update position prices
self.portfolio_tracker.update_prices(price_data)

await asyncio.sleep(self.config.price_update_interval)

except Exception as e:
logger.error(f"❌ Price update loop error: {e}")
await asyncio.sleep(5)

async def _trading_loop(self) -> None:
"""Main trading loop."""
while self.status.is_running:
try:
# Check circuit breakers
if self.status.circuit_breaker_active:
logger.warning("⚠️ Circuit breaker active - skipping trading cycle")
await asyncio.sleep(30)
continue

# Execute trading cycle
result = await self.trading_executor.execute_trading_cycle()

if result.success:
self.status.total_trades += 1
self.status.successful_trades += 1
self.status.last_trade = time.time()

logger.info(f"✅ Trade executed: {result.action.value} {result.executed_quantity} at ${result.executed_price}")
else:
logger.info(f"ℹ️ No trade executed: {result.metadata.get('reason', 'unknown')}")

# Update performance metrics
self._update_performance_metrics(result)

# Wait for next cycle
await asyncio.sleep(60)  # 1-minute trading cycles

except Exception as e:
logger.error(f"❌ Trading loop error: {e}")
self.status.error_count += 1
self._log_error("trading_loop", str(e))
await asyncio.sleep(30)

async def _monitoring_loop(self) -> None:
"""Background monitoring and health check loop."""
while self.status.is_running:
try:
# Check system health
health_status = await self._check_system_health()

# Update status
self.status.current_risk_level = health_status['risk_level']
self.status.circuit_breaker_active = health_status['circuit_breaker_active']

# Log performance summary
if self.status.total_trades > 0:
win_rate = self.status.successful_trades / self.status.total_trades
logger.info(f"📊 Performance: {self.status.total_trades} trades, "
f"{win_rate:.2%} win rate, ${self.status.total_pnl:.2f} PnL")

await asyncio.sleep(300)  # 5-minute monitoring cycles

except Exception as e:
logger.error(f"❌ Monitoring loop error: {e}")
await asyncio.sleep(60)

async def _check_system_health(self) -> Dict[str, Any]:
"""Check overall system health and risk status."""
try:
# Get portfolio status
portfolio_status = self.portfolio_tracker.get_portfolio_summary()

# Check risk manager status
risk_status = self.risk_manager.get_system_status()

# Determine overall health
circuit_breaker_active = (
risk_status.get('circuit_breaker_active', False) or
portfolio_status.get('unrealized_pnl', 0) < -self.config.max_daily_loss
)

risk_level = 'high' if circuit_breaker_active else 'normal'

return {
'risk_level': risk_level,
'circuit_breaker_active': circuit_breaker_active,
'portfolio_value': portfolio_status.get('total_value', 0),
'unrealized_pnl': portfolio_status.get('unrealized_pnl', 0),
'open_positions': portfolio_status.get('open_positions_count', 0)
}

except Exception as e:
logger.error(f"❌ Health check failed: {e}")
return {
'risk_level': 'unknown',
'circuit_breaker_active': True,
'error': str(e)
}

async def _close_all_positions(self) -> None:
"""Close all open positions."""
try:
open_positions = [pos_id for pos_id, pos in self.portfolio_tracker.positions.items()
if not pos.closed]

if not open_positions:
logger.info("ℹ️ No open positions to close")
return

logger.info(f"🔄 Closing {len(open_positions)} open positions...")

for pos_id in open_positions:
position = self.portfolio_tracker.positions[pos_id]

# Get current market price
ticker = await self.exchange.fetch_ticker(position.symbol)
current_price = ticker['last']

# Close position
closed_pos = self.portfolio_tracker.close_position(pos_id, current_price)

if closed_pos:
logger.info(f"✅ Closed position {pos_id}: {closed_pos.quantity} at ${current_price}")

# Execute actual trade if needed
if position.side == 'buy':
# Sell to close long position
await self.exchange.create_market_order(
symbol=position.symbol,
side='sell',
amount=position.quantity
)
else:
# Buy to close short position
await self.exchange.create_market_order(
symbol=position.symbol,
side='buy',
amount=position.quantity
)

logger.info("✅ All positions closed")

except Exception as e:
logger.error(f"❌ Failed to close positions: {e}")

def _update_performance_metrics(self, result) -> None:
"""Update performance tracking metrics."""
try:
if result.success:
# Update total PnL from portfolio
summary = self.portfolio_tracker.get_portfolio_summary()
self.status.total_pnl = summary['realized_pnl']

# Record performance snapshot
self.performance_history.append({
'timestamp': time.time(),
'total_trades': self.status.total_trades,
'successful_trades': self.status.successful_trades,
'total_pnl': self.status.total_pnl,
'portfolio_value': self.portfolio_tracker.get_portfolio_summary()['total_value']
})

# Keep only last 1000 records
if len(self.performance_history) > 1000:
self.performance_history = self.performance_history[-1000:]

except Exception as e:
logger.error(f"❌ Failed to update performance metrics: {e}")

def _log_error(self, error_type: str, error_message: str) -> None:
"""Log error for monitoring."""
error_record = {
'timestamp': time.time(),
'type': error_type,
'message': error_message,
'status': self.status.__dict__.copy()
}

self.error_log.append(error_record)

# Keep only last 100 error records
if len(self.error_log) > 100:
self.error_log = self.error_log[-100:]

def get_system_status(self) -> Dict[str, Any]:
"""Get comprehensive system status."""
try:
portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
risk_status = self.risk_manager.get_system_status()

return {
'status': self.status.__dict__,
'portfolio': portfolio_summary,
'risk': risk_status,
'config': {
'exchange': self.config.exchange_name,
'symbols': self.config.symbols,
'sandbox': self.config.sandbox
},
'performance': {
'total_trades': self.status.total_trades,
'successful_trades': self.status.successful_trades,
'win_rate': (self.status.successful_trades / self.status.total_trades
if self.status.total_trades > 0 else 0.0),
'total_pnl': self.status.total_pnl,
'error_count': self.status.error_count
}
}

except Exception as e:
logger.error(f"❌ Failed to get system status: {e}")
return {'error': str(e)}

def export_trading_report(self) -> Dict[str, Any]:
"""Export comprehensive trading report."""
try:
return {
'system_status': self.get_system_status(),
'performance_history': self.performance_history[-100:],  # Last 100 records
'error_log': self.error_log[-50:],  # Last 50 errors
'portfolio_history': self.portfolio_tracker.transaction_history[-100:],  # Last 100 transactions
'export_timestamp': time.time()
}

except Exception as e:
logger.error(f"❌ Failed to export trading report: {e}")
return {'error': str(e)}

# Factory function for easy initialization

def create_production_pipeline(
exchange_name: str,
api_key: str,
secret: str,
sandbox: bool = True,
symbols: List[str] = None,
**kwargs
) -> ProductionTradingPipeline:
"""Create a production trading pipeline with the given configuration."""

if symbols is None:
symbols = ['BTC/USDC']

config = TradingConfig(
exchange_name=exchange_name,
api_key=api_key,
secret=secret,
sandbox=sandbox,
symbols=symbols,
**kwargs
)

return ProductionTradingPipeline(config)

# Demo function

async def demo_production_pipeline():
"""Demonstrate the production trading pipeline."""
logger.info("🎯 DEMO: Production Trading Pipeline")

# Create pipeline with demo configuration
pipeline = create_production_pipeline(
exchange_name='coinbase',
api_key='demo_key',
secret='demo_secret',
sandbox=True,
symbols=['BTC/USDC'],
risk_tolerance=0.1,
max_position_size=0.05
)

try:
# Start trading for a short period
logger.info("🚀 Starting demo trading...")

# Run for 5 minutes
await asyncio.wait_for(pipeline.start_trading(), timeout=300)

except asyncio.TimeoutError:
logger.info("⏰ Demo completed after 5 minutes")
except Exception as e:
logger.error(f"❌ Demo failed: {e}")
finally:
await pipeline.stop_trading()

# Export final report
report = pipeline.export_trading_report()
logger.info(f"📊 Final Report: {report['system_status']['performance']}")

if __name__ == "__main__":
# Run demo
asyncio.run(demo_production_pipeline())
