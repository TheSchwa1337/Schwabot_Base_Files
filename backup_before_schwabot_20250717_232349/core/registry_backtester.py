"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry Backtester for Schwabot
================================

Master backtester that runs logic from registry hashes, not random function calls.
Integrates with Schwabot's mathematical framework to provide recursive AI strategy emulation.

This backtester implements the Ferris Wheel tick cycle simulation,
enabling full mathematical pipeline testing before live deployment.
"""

import logging
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Schwabot components
try:
from ..backtesting.historical_data_manager import HistoricalDataManager
from core.btc_usdc_trading_engine import BTCTradingEngine
from core.profit_feedback_engine import FeedbackCycle, ProfitFeedbackEngine
from core.quad_bit_strategy_array import TradingPair
from core.registry_strategy import RegistryStrategy, StrategyResult, TickData

SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
SCHWABOT_COMPONENTS_AVAILABLE = False
logger.warning(f"Schwabot components not available: {e}")


@dataclass
class BacktestConfig:
"""Configuration for registry backtesting."""

registry_hashes: List[str]
start_date: datetime
end_date: datetime
trading_pair: TradingPair
initial_capital: Decimal
tick_interval_minutes: int = 1
ferris_wheel_cycles: int = 16  # Number of ticks per Ferris Wheel cycle
enable_feedback: bool = True
enable_registry_updates: bool = True
output_path: str = "backtest_results"
memory_path: str = "memory/cycle_feedback.json"
registry_path: str = "registry/hashed_strategies.json"


@dataclass
class BacktestResult:
"""Result of a registry backtest."""

config: BacktestConfig
total_cycles: int
successful_cycles: int
total_profit: float
average_profit: float
max_drawdown: float
sharpe_ratio: float
win_rate: float
profit_factor: float
execution_time: float
strategy_results: List[StrategyResult]
feedback_cycles: List[FeedbackCycle]
performance_summary: Dict[str, Any]


class RegistryBacktester:
"""
Registry Backtester Implementation

Master backtester that runs strategies from registry hashes,
implementing the Ferris Wheel tick cycle simulation.
"""

def __init__(self, config: BacktestConfig) -> None:
"""Initialize the registry backtester."""
self.config = config
self.strategies: Dict[str, RegistryStrategy] = {}
self.feedback_engine = None
self.trading_engine = None
self.historical_data_manager = None
self.results: List[StrategyResult] = []
self.feedback_cycles: List[FeedbackCycle] = []

# Performance tracking
self.start_time = 0.0
self.end_time = 0.0
self.cycle_count = 0

# Initialize components
self._initialize_components()

logger.info("✅ Registry Backtester initialized")

def _initialize_components(self) -> None:
"""Initialize all backtesting components."""
try:
if not SCHWABOT_COMPONENTS_AVAILABLE:
raise RuntimeError("Schwabot components not available")

# Initialize feedback engine
if self.config.enable_feedback:
self.feedback_engine = ProfitFeedbackEngine(
memory_log_path=self.config.memory_path,
registry_path=self.config.registry_path,
)

# Initialize trading engine for price processing
self.trading_engine = BTCTradingEngine(
config={"testnet": True, "enable_live_trading": False}
)

# Initialize historical data manager
self.historical_data_manager = HistoricalDataManager(
start_date=self.config.start_date,
end_date=self.config.end_date,
interval_minutes=self.config.tick_interval_minutes,
initial_price=Decimal("50000.0"),  # Default BTC price
)

# Load strategies from registry
self._load_strategies()

logger.info(f"✅ Loaded {len(self.strategies)} strategies from registry")

except Exception as e:
logger.error(f"❌ Error initializing components: {e}")
raise

def _load_strategies(self) -> None:
"""Load strategies from registry hashes."""
try:
for hash_id in self.config.registry_hashes:
try:
strategy = RegistryStrategy(
hash_id=hash_id, registry_path=self.config.registry_path
)
self.strategies[hash_id] = strategy
logger.info(f"✅ Loaded strategy {hash_id[:8]}...")
except Exception as e:
logger.warning(f"⚠️ Failed to load strategy {hash_id}: {e}")
continue

if not self.strategies:
raise RuntimeError("No strategies loaded from registry")

except Exception as e:
logger.error(f"❌ Error loading strategies: {e}")
raise

async def run_backtest(self) -> BacktestResult:
"""
Run the registry backtest.

Returns:
BacktestResult with comprehensive results
"""
try:
self.start_time = time.time()
logger.info("🚀 Starting Registry Backtest")
logger.info(f"📊 Testing {len(self.strategies)} strategies")
logger.info(
f"📅 Period: {self.config.start_date.date()} to {self.config.end_date.date()}"
)
logger.info(f"💰 Initial Capital: ${self.config.initial_capital:,.2f}")

# Process historical data in Ferris Wheel cycles
async for data_point in self.historical_data_manager.get_historical_data(
source="mock"
):
await self._process_tick_cycle(data_point)

# Simulate processing delay
await asyncio.sleep(0.001)

self.end_time = time.time()

# Generate results
result = self._generate_backtest_result()

# Save results
self._save_results(result)

logger.info("✅ Registry Backtest completed successfully")
return result

except Exception as e:
logger.error(f"❌ Registry backtest failed: {e}")
raise

async def _process_tick_cycle(self, data_point: Dict[str, Any]) -> None:
"""Process a single tick cycle through all strategies."""
try:
# Convert data point to tick data
tick_data = self._create_tick_data(data_point)

# Process through each strategy
for hash_id, strategy in self.strategies.items():
try:
# Execute strategy
strategy_result = strategy.run([tick_data])

# Add to results
self.results.append(strategy_result)

# Process feedback if enabled
if self.config.enable_feedback and self.feedback_engine:
market_data = {
"price": tick_data.price,
"volume": tick_data.volume,
"timestamp": tick_data.timestamp,
"thermal_state": tick_data.thermal_state,
"entropy": tick_data.entropy,
}

feedback_result = self.feedback_engine.process(
strategy_result, market_data
)

if "error" not in feedback_result:
self.cycle_count += 1

# Log cycle result
logger.debug(
f"Cycle {self.cycle_count}: Strategy {hash_id[:8]} | "
f"Decision: {strategy_result.decision} | "
f"Confidence: {strategy_result.confidence:.2f} | "
f"Profit: {strategy_result.actual_profit:.2f}%"
)

except Exception as e:
logger.warning(f"⚠️ Error processing strategy {hash_id}: {e}")
continue

except Exception as e:
logger.error(f"❌ Error processing tick cycle: {e}")

def _create_tick_data(self, data_point: Dict[str, Any]) -> TickData:
"""Create tick data from historical data point."""
try:
# Process through trading engine to get mathematical analysis
price = float(data_point["close"])
volume = float(data_point["volume"])
timestamp = int(
datetime.fromisoformat(data_point["datetime"]).timestamp() * 1000
)

# Get mathematical analysis from trading engine
btc_price_data = self.trading_engine.process_btc_price(
price=price, volume=volume, thermal_state=65.0  # Default thermal state
)

return TickData(
timestamp=timestamp,
price=price,
volume=volume,
hash_value=btc_price_data.hash_value,
bit_phase=btc_price_data.bit_phase,
tensor_score=btc_price_data.tensor_score,
entropy=btc_price_data.entropy,
thermal_state=btc_price_data.thermal_state,
)

except Exception as e:
logger.error(f"❌ Error creating tick data: {e}")
# Return fallback tick data
return TickData(
timestamp=int(time.time() * 1000),
price=50000.0,
volume=1000.0,
hash_value="fallback_hash",
bit_phase=0,
tensor_score=0.0,
entropy=0.0,
thermal_state=65.0,
)

def _generate_backtest_result(self) -> BacktestResult:
"""Generate comprehensive backtest results."""
try:
execution_time = self.end_time - self.start_time

# Calculate performance metrics
successful_results = [r for r in self.results if r.decision != "error"]
total_cycles = len(successful_results)
successful_cycles = len(
[r for r in successful_results if r.actual_profit > 0]
)

if total_cycles == 0:
return BacktestResult(
config=self.config,
total_cycles=0,
successful_cycles=0,
total_profit=0.0,
average_profit=0.0,
max_drawdown=0.0,
sharpe_ratio=0.0,
win_rate=0.0,
profit_factor=0.0,
execution_time=execution_time,
strategy_results=self.results,
feedback_cycles=self.feedback_cycles,
performance_summary={},
)

# Calculate metrics
profits = [r.actual_profit for r in successful_results]
total_profit = sum(profits)
average_profit = total_profit / total_cycles
win_rate = successful_cycles / total_cycles

# Calculate max drawdown
cumulative_profits = []
running_total = 0
for profit in profits:
running_total += profit
cumulative_profits.append(running_total)

max_drawdown = 0.0
peak = 0.0
for cumulative in cumulative_profits:
if cumulative > peak:
peak = cumulative
drawdown = peak - cumulative
if drawdown > max_drawdown:
max_drawdown = drawdown

# Calculate Sharpe ratio (simplified)
if len(profits) > 1:
returns = [profits[i] - profits[i - 1] for i in range(1, len(profits))]
avg_return = sum(returns) / len(returns)
std_return = (
sum((r - avg_return) ** 2 for r in returns) / len(returns)
) ** 0.5
sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
else:
sharpe_ratio = 0.0

# Calculate profit factor
winning_trades = [p for p in profits if p > 0]
losing_trades = [p for p in profits if p < 0]
profit_factor = (
sum(winning_trades) / abs(sum(losing_trades))
if sum(losing_trades) != 0
else float("inf")
)

# Performance summary
performance_summary = {
"total_strategies": len(self.strategies),
"total_ticks_processed": len(self.results),
"average_confidence": sum(r.confidence for r in successful_results)
/ total_cycles,
"strategy_performance": {
hash_id: {
"executions": len(
[r for r in self.results if r.strategy_hash == hash_id]
),
"success_rate": len(
[
r
for r in self.results
if r.strategy_hash == hash_id and r.actual_profit > 0
]
)
/ len([r for r in self.results if r.strategy_hash == hash_id]),
}
for hash_id in self.strategies.keys()
},
}

return BacktestResult(
config=self.config,
total_cycles=total_cycles,
successful_cycles=successful_cycles,
total_profit=total_profit,
average_profit=average_profit,
max_drawdown=max_drawdown,
sharpe_ratio=sharpe_ratio,
win_rate=win_rate,
profit_factor=profit_factor,
execution_time=execution_time,
strategy_results=self.results,
feedback_cycles=self.feedback_cycles,
performance_summary=performance_summary,
)

except Exception as e:
logger.error(f"❌ Error generating backtest result: {e}")
raise

def _save_results(self, result: BacktestResult) -> None:
"""Save backtest results to file."""
try:
output_dir = Path(self.config.output_path)
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed results
results_file = output_dir / f"registry_backtest_{int(time.time())}.json"

results_data = {
"config": {
"registry_hashes": result.config.registry_hashes,
"start_date": result.config.start_date.isoformat(),
"end_date": result.config.end_date.isoformat(),
"trading_pair": result.config.trading_pair.value,
"initial_capital": float(result.config.initial_capital),
},
"summary": {
"total_cycles": result.total_cycles,
"successful_cycles": result.successful_cycles,
"total_profit": result.total_profit,
"average_profit": result.average_profit,
"max_drawdown": result.max_drawdown,
"sharpe_ratio": result.sharpe_ratio,
"win_rate": result.win_rate,
"profit_factor": result.profit_factor,
"execution_time": result.execution_time,
},
"performance_summary": result.performance_summary,
"strategy_results": [
{
"strategy_hash": r.strategy_hash,
"decision": r.decision,
"confidence": r.confidence,
"actual_profit": r.actual_profit,
"profit_delta": r.profit_delta,
"execution_timestamp": r.execution_timestamp,
}
for r in result.strategy_results
],
}

import json

with open(results_file, "w", encoding="utf-8") as f:
json.dump(results_data, f, indent=2, ensure_ascii=False)

logger.info(f"✅ Results saved to {results_file}")

except Exception as e:
logger.error(f"❌ Error saving results: {e}")

def print_summary(self, result: BacktestResult) -> None:
"""Print backtest summary to console."""
try:
print("\n" + "=" * 60)
print("🎯 REGISTRY BACKTEST RESULTS")
print("=" * 60)
print(f"📊 Total Cycles: {result.total_cycles}")
print(f"✅ Successful Cycles: {result.successful_cycles}")
print(f"📈 Win Rate: {result.win_rate:.2%}")
print(f"💰 Total Profit: {result.total_profit:.2f}%")
print(f"📊 Average Profit: {result.average_profit:.2f}%")
print(f"📉 Max Drawdown: {result.max_drawdown:.2f}%")
print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"📊 Profit Factor: {result.profit_factor:.2f}")
print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
print("=" * 60)

# Strategy performance breakdown
print("\n📋 STRATEGY PERFORMANCE BREAKDOWN:")
for hash_id, perf in result.performance_summary[
"strategy_performance"
].items():
print(
f"  {hash_id[:8]}... | Executions: {perf['executions']} | "
f"Success Rate: {perf['success_rate']:.2%}"
)

print("=" * 60)

except Exception as e:
logger.error(f"❌ Error printing summary: {e}")


# Factory function
def create_registry_backtester(config: BacktestConfig) -> RegistryBacktester:
"""Create a registry backtester instance."""
return RegistryBacktester(config)
