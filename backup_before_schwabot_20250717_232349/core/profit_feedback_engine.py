"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profit Feedback Engine for Schwabot
===================================

Evaluates strategy outcomes and generates profit hashes for recursive learning.
Integrates with Schwabot's mathematical framework to provide intelligent feedback
for strategy evolution and registry updates.

This engine implements the Ferris Wheel tick cycle feedback mechanism,
enabling continuous strategy optimization based on mathematical performance metrics.
"""

import json
import hashlib
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from core.hash_config_manager import generate_hash_from_string
import numpy as np


# Import centralized hash configuration

logger = logging.getLogger(__name__)

# Import Schwabot mathematical components
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator
from core.registry_strategy import StrategyResult

MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")

@dataclass
class FeedbackCycle:
"""Class for Schwabot trading functionality."""
"""Represents a single feedback cycle in the Ferris Wheel system."""
cycle_id: int
strategy_hash: str
entry_timestamp: int
exit_timestamp: int
expected_profit: float
actual_profit: float
profit_delta: float
confidence_score: float
execution_efficiency: float
market_conditions: Dict[str, Any]
feedback_hash: str
registry_update: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
"""Class for Schwabot trading functionality."""
"""Performance metrics for strategy evaluation."""
total_cycles: int
successful_cycles: int
success_rate: float
average_profit: float
average_profit_delta: float
total_profit: float
max_drawdown: float
sharpe_ratio: float
win_rate: float
profit_factor: float
average_confidence: float
execution_efficiency: float

class ProfitFeedbackEngine:
"""Class for Schwabot trading functionality."""
"""
Profit Feedback Engine Implementation

Evaluates strategy outcomes and generates intelligent feedback
for recursive strategy optimization.
"""


def __init__(self, memory_log_path: str = "memory/cycle_feedback.json", -> None
registry_path: str = "registry/hashed_strategies.json"):
"""Initialize the profit feedback engine."""
self.memory_log_path = Path(memory_log_path)
self.registry_path = Path(registry_path)
self.cycle_count = 0
self.feedback_cycles: List[FeedbackCycle] = []

# Initialize mathematical infrastructure
self.math_config = None
self.math_cache = None
self.math_orchestrator = None

if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()

# Ensure memory directory exists
self.memory_log_path.parent.mkdir(parents=True, exist_ok=True)

# Load existing feedback cycles
self._load_existing_feedback()

logger.info("✅ Profit Feedback Engine initialized")

def _load_existing_feedback(self) -> None:
"""Load existing feedback cycles from memory."""
try:
if self.memory_log_path.exists():
with open(self.memory_log_path, 'r', encoding='utf-8') as f:
data = json.load(f)
self.feedback_cycles = [FeedbackCycle(**cycle) for cycle in data.get('cycles', [])]
self.cycle_count = data.get('cycle_count', 0)
logger.info(f"✅ Loaded {len(self.feedback_cycles)} existing feedback cycles")
except Exception as e:
logger.warning(f"⚠️ Could not load existing feedback: {e}")


def process(self, strategy_result: StrategyResult, -> None
market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
"""
Process strategy result and generate feedback.

Args:
strategy_result: Result from strategy execution
market_data: Optional market conditions data

Returns:
Dictionary containing feedback information
"""
try:
self.cycle_count += 1

# Calculate execution efficiency
execution_efficiency = self._calculate_execution_efficiency(strategy_result)

# Generate feedback hash
feedback_hash = self._generate_feedback_hash(strategy_result, market_data)

# Create feedback cycle
feedback_cycle = FeedbackCycle(
cycle_id=self.cycle_count,
strategy_hash=strategy_result.strategy_hash,
entry_timestamp=strategy_result.execution_timestamp,
exit_timestamp=int(time.time() * 1000),
expected_profit=strategy_result.expected_profit,
actual_profit=strategy_result.actual_profit,
profit_delta=strategy_result.profit_delta,
confidence_score=strategy_result.confidence,
execution_efficiency=execution_efficiency,
market_conditions=market_data or {},
feedback_hash=feedback_hash,
registry_update=self._generate_registry_update(strategy_result, feedback_hash)
)

# Add to feedback cycles
self.feedback_cycles.append(feedback_cycle)

# Save feedback
self._save_feedback()

# Update registry if needed
if self._should_update_registry(feedback_cycle):
self._update_registry(feedback_cycle)

# Calculate performance metrics
metrics = self._calculate_performance_metrics(strategy_result.strategy_hash)

feedback_result = {
'cycle_id': self.cycle_count,
'strategy_hash': strategy_result.strategy_hash,
'profit_delta': strategy_result.profit_delta,
'actual_profit': strategy_result.actual_profit,
'feedback_hash': feedback_hash,
'execution_efficiency': execution_efficiency,
'performance_metrics': metrics,
'registry_updated': self._should_update_registry(feedback_cycle)
}

logger.info(f"✅ Cycle {self.cycle_count}: Strategy {strategy_result.strategy_hash[:8]} | "
f"Δ: {strategy_result.profit_delta:.3f} | PnL: {strategy_result.actual_profit:.2f}%")

return feedback_result

except Exception as e:
logger.error(f"❌ Error processing feedback: {e}")
return {
'error': str(e),
'cycle_id': self.cycle_count,
'strategy_hash': strategy_result.strategy_hash
}

def _calculate_execution_efficiency(self, strategy_result: StrategyResult) -> float:
"""Calculate execution efficiency based on strategy result."""
try:
# Base efficiency on confidence and profit delta
confidence_factor = strategy_result.confidence
profit_factor = min(1.0, max(0.0, (strategy_result.actual_profit + 10) / 20))  # Normalize to 0-1

# Consider decision quality
decision_quality = 1.0
if strategy_result.decision == 'hold' and abs(strategy_result.profit_delta) < 0.1:
decision_quality = 0.8  # Good hold decision
elif strategy_result.decision in ['buy', 'sell'] and strategy_result.profit_delta > 0:
decision_quality = 1.0  # Good trade decision
elif strategy_result.decision in ['buy', 'sell'] and strategy_result.profit_delta < -1.0:
decision_quality = 0.3  # Poor trade decision

efficiency = (confidence_factor * 0.4 + profit_factor * 0.4 + decision_quality * 0.2)
return min(1.0, max(0.0, efficiency))

except Exception as e:
logger.warning(f"⚠️ Error calculating execution efficiency: {e}")
return 0.5


def _generate_feedback_hash(self, strategy_result: StrategyResult, -> None
market_data: Optional[Dict[str, Any]]) -> str:
"""Generate feedback hash for recursive learning."""
try:
# Create feedback data string
feedback_data = f"{strategy_result.strategy_hash}:{strategy_result.actual_profit}:"
feedback_data += f"{strategy_result.profit_delta}:{strategy_result.confidence}:"
feedback_data += f"{strategy_result.execution_timestamp}"

if market_data:
feedback_data += f":{hash(str(market_data))}"

# Generate SHA-256 hash
feedback_hash = generate_hash_from_string(feedback_data)
return feedback_hash

except Exception as e:
logger.error(f"❌ Error generating feedback hash: {e}")
return generate_hash_from_string(str(time.time()))


def _generate_registry_update(self, strategy_result: StrategyResult, -> None
feedback_hash: str) -> Dict[str, Any]:
"""Generate registry update based on feedback."""
try:
update = {
'last_updated': time.time(),
'last_feedback_hash': feedback_hash,
'last_profit_delta': strategy_result.profit_delta,
'last_execution_timestamp': strategy_result.execution_timestamp
}

# Update success rate and average profit
strategy_cycles = [c for c in self.feedback_cycles
if c.strategy_hash == strategy_result.strategy_hash]

if strategy_cycles:
successful_cycles = len([c for c in strategy_cycles if c.actual_profit > 0])
total_cycles = len(strategy_cycles)
avg_profit = sum(c.actual_profit for c in strategy_cycles) / total_cycles

update['success_rate'] = successful_cycles / total_cycles
update['average_profit'] = avg_profit
update['execution_count'] = total_cycles

return update

except Exception as e:
logger.error(f"❌ Error generating registry update: {e}")
return {}

def _should_update_registry(self, feedback_cycle: FeedbackCycle) -> bool:
"""Determine if registry should be updated based on feedback."""
try:
# Update if significant performance deviation
if abs(feedback_cycle.profit_delta) > 2.0:  # 2% deviation
return True

# Update if low execution efficiency
if feedback_cycle.execution_efficiency < 0.3:
return True

# Update every 10 cycles for continuous learning
if feedback_cycle.cycle_id % 10 == 0:
return True

return False

except Exception as e:
logger.error(f"❌ Error determining registry update: {e}")
return False

def _update_registry(self, feedback_cycle: FeedbackCycle) -> None:
"""Update registry with feedback information."""
try:
if not self.registry_path.exists():
logger.warning(f"⚠️ Registry file not found: {self.registry_path}")
return

# Load current registry
with open(self.registry_path, 'r', encoding='utf-8') as f:
registry = json.load(f)

strategy_hash = feedback_cycle.strategy_hash
if strategy_hash in registry:
# Update strategy metadata
registry[strategy_hash].update(feedback_cycle.registry_update)

# Save updated registry
with open(self.registry_path, 'w', encoding='utf-8') as f:
json.dump(registry, f, indent=2, ensure_ascii=False)

logger.info(f"✅ Registry updated for strategy {strategy_hash[:8]}")

except Exception as e:
logger.error(f"❌ Error updating registry: {e}")

def _calculate_performance_metrics(self, strategy_hash: str) -> PerformanceMetrics:
"""Calculate performance metrics for a strategy."""
try:
strategy_cycles = [c for c in self.feedback_cycles
if c.strategy_hash == strategy_hash]

if not strategy_cycles:
return PerformanceMetrics(
total_cycles=0, successful_cycles=0, success_rate=0.0,
average_profit=0.0, average_profit_delta=0.0, total_profit=0.0,
max_drawdown=0.0, sharpe_ratio=0.0, win_rate=0.0,
profit_factor=0.0, average_confidence=0.0, execution_efficiency=0.0
)

total_cycles = len(strategy_cycles)
successful_cycles = len([c for c in strategy_cycles if c.actual_profit > 0])
success_rate = successful_cycles / total_cycles if total_cycles > 0 else 0.0

profits = [c.actual_profit for c in strategy_cycles]
profit_deltas = [c.profit_delta for c in strategy_cycles]
confidences = [c.confidence_score for c in strategy_cycles]
efficiencies = [c.execution_efficiency for c in strategy_cycles]

average_profit = np.mean(profits) if profits else 0.0
average_profit_delta = np.mean(profit_deltas) if profit_deltas else 0.0
total_profit = sum(profits)
average_confidence = np.mean(confidences) if confidences else 0.0
execution_efficiency = np.mean(efficiencies) if efficiencies else 0.0

# Calculate max drawdown
cumulative_profits = np.cumsum(profits)
running_max = np.maximum.accumulate(cumulative_profits)
drawdowns = running_max - cumulative_profits
max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

# Calculate Sharpe ratio (simplified)
returns = np.diff(profits) if len(profits) > 1 else [0.0]
sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

# Calculate win rate and profit factor
winning_trades = [p for p in profits if p > 0]
losing_trades = [p for p in profits if p < 0]

win_rate = len(winning_trades) / total_cycles if total_cycles > 0 else 0.0
profit_factor = (sum(winning_trades) / abs(sum(losing_trades))
if sum(losing_trades) != 0 else float('inf'))

return PerformanceMetrics(
total_cycles=total_cycles,
successful_cycles=successful_cycles,
success_rate=success_rate,
average_profit=average_profit,
average_profit_delta=average_profit_delta,
total_profit=total_profit,
max_drawdown=max_drawdown,
sharpe_ratio=sharpe_ratio,
win_rate=win_rate,
profit_factor=profit_factor,
average_confidence=average_confidence,
execution_efficiency=execution_efficiency
)

except Exception as e:
logger.error(f"❌ Error calculating performance metrics: {e}")
return PerformanceMetrics(
total_cycles=0, successful_cycles=0, success_rate=0.0,
average_profit=0.0, average_profit_delta=0.0, total_profit=0.0,
max_drawdown=0.0, sharpe_ratio=0.0, win_rate=0.0,
profit_factor=0.0, average_confidence=0.0, execution_efficiency=0.0
)

def _save_feedback(self) -> None:
"""Save feedback cycles to memory."""
try:
data = {
'cycle_count': self.cycle_count,
'cycles': [cycle.__dict__ for cycle in self.feedback_cycles]
}

with open(self.memory_log_path, 'w', encoding='utf-8') as f:
json.dump(data, f, indent=2, ensure_ascii=False)

except Exception as e:
logger.error(f"❌ Error saving feedback: {e}")

def get_feedback_summary(self, strategy_hash: Optional[str] = None) -> Dict[str, Any]:
"""Get feedback summary for all strategies or a specific strategy."""
try:
if strategy_hash:
cycles = [c for c in self.feedback_cycles if c.strategy_hash == strategy_hash]
else:
cycles = self.feedback_cycles

if not cycles:
return {'error': 'No feedback cycles found'}

metrics = self._calculate_performance_metrics(cycles[0].strategy_hash if cycles else '')

return {
'total_cycles': len(cycles),
'performance_metrics': metrics.__dict__,
'recent_cycles': [c.__dict__ for c in cycles[-5:]]  # Last 5 cycles
}

except Exception as e:
logger.error(f"❌ Error getting feedback summary: {e}")
return {'error': str(e)}

def get_status(self) -> Dict[str, Any]:
"""Get engine status."""
return {
'cycle_count': self.cycle_count,
'feedback_cycles': len(self.feedback_cycles),
'memory_path': str(self.memory_log_path),
'registry_path': str(self.registry_path),
'math_infrastructure': MATH_INFRASTRUCTURE_AVAILABLE
}

# Factory function

def create_profit_feedback_engine(memory_log_path: str = "memory/cycle_feedback.json",
registry_path: str = "registry/hashed_strategies.json") -> ProfitFeedbackEngine:
"""Create a profit feedback engine instance."""
return ProfitFeedbackEngine(memory_log_path, registry_path)
