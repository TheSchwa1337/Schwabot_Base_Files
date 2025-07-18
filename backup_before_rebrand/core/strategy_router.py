"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Router Module
======================
Provides strategy routing functionality for the Schwabot trading system.

Mathematical Core:
R(s_i) = argmax_j {w_j * f_j(s_i) + λ_j * g_j(s_i)}
Where:
- w_j: strategy weights
- f_j: strategy performance functions
- λ_j: risk adjustment factors
- g_j: market condition functions

This module intelligently routes trading signals to optimal execution strategies
based on mathematical optimization and market conditions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator

MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")

class RoutingStrategy(Enum):
"""Class for Schwabot trading functionality."""
"""Routing strategy types."""
PERFORMANCE_BASED = "performance_based"
RISK_ADJUSTED = "risk_adjusted"
MARKET_CONDITION = "market_condition"
HYBRID = "hybrid"
ADAPTIVE = "adaptive"


class MarketCondition(Enum):
"""Class for Schwabot trading functionality."""
"""Market condition types."""
BULL_TRENDING = "bull_trending"
BEAR_TRENDING = "bear_trending"
SIDEWAYS = "sideways"
VOLATILE = "volatile"
CALM = "calm"
CRISIS = "crisis"


class SignalPriority(Enum):
"""Class for Schwabot trading functionality."""
"""Signal priority levels."""
LOW = "low"
MEDIUM = "medium"
HIGH = "high"
CRITICAL = "critical"


@dataclass
class RoutingDecision:
"""Class for Schwabot trading functionality."""
"""Routing decision with mathematical analysis."""
signal_id: str
selected_strategy: str
routing_score: float
confidence: float
reasoning: str
mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
timestamp: float = field(default_factory=time.time)
routing_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
"""Class for Schwabot trading functionality."""
"""Strategy performance metrics for routing."""
strategy_name: str
total_signals: int
successful_routes: int
average_score: float
win_rate: float
risk_score: float
mathematical_signature: str = ""


@dataclass
class MarketConditionData:
"""Class for Schwabot trading functionality."""
"""Market condition data for routing."""
condition: MarketCondition
volatility: float
trend_strength: float
liquidity_score: float
mathematical_signature: str = ""


@dataclass
class StrategyRouterConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for strategy router."""
enabled: bool = True
timeout: float = 30.0
retries: int = 3
debug: bool = False
max_concurrent_routes: int = 20
routing_threshold: float = 0.6  # Minimum routing score
mathematical_analysis_enabled: bool = True
adaptive_routing_enabled: bool = True
strategy_weights: Dict[str, float] = field(default_factory=lambda: {
'momentum': 0.25,
'mean_reversion': 0.20,
'scalping': 0.15,
'arbitrage': 0.15,
'grid': 0.10,
'quantum': 0.10,
'phantom': 0.05
})
risk_factors: Dict[str, float] = field(default_factory=lambda: {
'volatility_penalty': 0.1,
'liquidity_penalty': 0.05,
'trend_alignment_bonus': 0.15
})


@dataclass
class StrategyRouterMetrics:
"""Class for Schwabot trading functionality."""
"""Strategy router metrics."""
signals_routed: int = 0
successful_routes: int = 0
routing_errors: int = 0
average_routing_time: float = 0.0
routing_accuracy: float = 0.0
mathematical_analyses: int = 0
last_updated: float = 0.0


class StrategyRouter:
"""Class for Schwabot trading functionality."""
"""
Strategy Router System

Implements intelligent signal routing:
R(s_i) = argmax_j {w_j * f_j(s_i) + λ_j * g_j(s_i)}

Intelligently routes trading signals to optimal execution strategies
based on mathematical optimization and market conditions.
"""

def __init__(self, config: Optional[StrategyRouterConfig] = None) -> None:
"""Initialize the strategy router system."""
self.config = config or StrategyRouterConfig()
self.logger = logging.getLogger(__name__)

# Routing state
self.active_routes: Dict[str, RoutingDecision] = {}
self.routing_history: List[RoutingDecision] = []
self.strategy_performance: Dict[str, StrategyPerformance] = {}
self.market_conditions: Dict[str, MarketConditionData] = {}

# Signal processing
self.signal_queue: asyncio.Queue = asyncio.Queue()
self.routing_queue: asyncio.Queue = asyncio.Queue()

# Mathematical infrastructure
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()
else:
self.math_config = None
self.math_cache = None
self.math_orchestrator = None

# Performance tracking
self.metrics = StrategyRouterMetrics()

# System state
self.initialized = False
self.active = False

self._initialize_system()

def _initialize_system(self) -> None:
"""Initialize the strategy router system."""
try:
self.logger.info("Initializing Strategy Router System")

# Initialize strategy performance tracking
for strategy_name in self.config.strategy_weights.keys():
self.strategy_performance[strategy_name] = StrategyPerformance(
strategy_name=strategy_name,
total_signals=0,
successful_routes=0,
average_score=0.0,
win_rate=0.0,
risk_score=0.0
)

self.initialized = True
self.logger.info("✅ Strategy Router System initialized successfully")

except Exception as e:
self.logger.error(f"❌ Error initializing Strategy Router System: {e}")
self.initialized = False

async def start_router(self) -> bool:
"""Start the strategy router."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True

# Start processing tasks
asyncio.create_task(self._process_signal_queue())
asyncio.create_task(self._process_routing_queue())

self.logger.info("✅ Strategy Router started")
return True

except Exception as e:
self.logger.error(f"❌ Error starting strategy router: {e}")
return False

async def stop_router(self) -> bool:
"""Stop the strategy router."""
try:
self.active = False
self.logger.info("✅ Strategy Router stopped")
return True

except Exception as e:
self.logger.error(f"❌ Error stopping strategy router: {e}")
return False

async def route_signal(self, signal_data: Dict[str, Any]) -> bool:
"""Route a trading signal to optimal strategy."""
if not self.active:
self.logger.error("Strategy router not active")
return False

try:
# Validate signal data
if not self._validate_signal_data(signal_data):
self.logger.error(f"Invalid signal data: {signal_data}")
return False

# Add mathematical analysis
if self.config.mathematical_analysis_enabled:
await self._analyze_signal_mathematically(signal_data)

# Queue for processing
await self.signal_queue.put(signal_data)

self.logger.info(f"✅ Signal queued for routing: {signal_data.get('signal_id', 'unknown')}")
return True

except Exception as e:
self.logger.error(f"❌ Error routing signal: {e}")
return False

def _validate_signal_data(self, signal_data: Dict[str, Any]) -> bool:
"""Validate signal data."""
try:
# Check required fields
required_fields = ['signal_id', 'symbol', 'signal_type', 'strength']
for field in required_fields:
if field not in signal_data:
return False

# Validate signal strength
strength = signal_data.get('strength', 0.0)
if not isinstance(strength, (int, float)) or strength < 0.0 or strength > 1.0:
return False

# Validate signal type
valid_types = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
if signal_data.get('signal_type') not in valid_types:
return False

return True

except Exception as e:
self.logger.error(f"❌ Error validating signal data: {e}")
return False

async def _analyze_signal_mathematically(self, signal_data: Dict[str, Any]) -> None:
"""Perform mathematical analysis on signal."""
try:
if not self.math_orchestrator:
return

# Prepare signal data for mathematical analysis
signal_vector = np.array([
signal_data.get('strength', 0.0),
signal_data.get('confidence', 0.0),
signal_data.get('price', 0.0),
signal_data.get('volume', 0.0),
time.time()
])

# Perform mathematical orchestration
result = self.math_orchestrator.process_data(signal_vector)

# Update signal data with mathematical analysis
signal_data['mathematical_analysis'] = {
'confidence': float(result),
'timestamp': time.time()
}

# Update metrics
self.metrics.mathematical_analyses += 1

except Exception as e:
self.logger.error(f"❌ Error analyzing signal mathematically: {e}")

async def _process_signal_queue(self) -> None:
"""Process signals from the queue."""
try:
while self.active:
try:
# Get signal from queue
signal_data = await asyncio.wait_for(
self.signal_queue.get(),
timeout=1.0
)

# Process signal
await self._process_signal(signal_data)

# Mark task as done
self.signal_queue.task_done()

except asyncio.TimeoutError:
continue
except Exception as e:
self.logger.error(f"❌ Error processing signal: {e}")

except Exception as e:
self.logger.error(f"❌ Error in signal processing loop: {e}")

async def _process_signal(self, signal_data: Dict[str, Any]) -> None:
"""Process a trading signal."""
try:
# Update performance metrics
self.metrics.signals_routed += 1

# Make routing decision
decision = await self._make_routing_decision(signal_data)

# Store decision
self.routing_history.append(decision)
self.active_routes[decision.signal_id] = decision

# Update strategy performance
self._update_strategy_performance(decision)

# Queue for execution
await self.routing_queue.put(decision)

self.logger.info(f"✅ Signal routed: {signal_data.get('signal_id')} -> {decision.selected_strategy}")

except Exception as e:
self.logger.error(f"❌ Error processing signal: {e}")
self.metrics.routing_errors += 1

async def _make_routing_decision(self, signal_data: Dict[str, Any]) -> RoutingDecision:
"""Make routing decision based on signal and market conditions."""
try:
signal_id = signal_data.get('signal_id', 'unknown')
symbol = signal_data.get('symbol', 'BTC/USDT')

# Get market condition
market_condition = self._get_market_condition(symbol)

# Calculate routing scores for all strategies
routing_scores = {}
for strategy_name in self.config.strategy_weights.keys():
score = await self._calculate_routing_score(signal_data, strategy_name, market_condition)
routing_scores[strategy_name] = score

# Select best strategy
selected_strategy = max(routing_scores.items(), key=lambda x: x[1])[0]
routing_score = routing_scores[selected_strategy]

# Generate routing parameters
routing_parameters = self._generate_routing_parameters(signal_data, selected_strategy, market_condition)

# Perform mathematical analysis on decision
mathematical_analysis = await self._analyze_decision_mathematically(
signal_data, selected_strategy, routing_score, market_condition
)

# Generate reasoning
reasoning = self._generate_routing_reasoning(signal_data, selected_strategy, routing_score, market_condition)

# Create decision
decision = RoutingDecision(
signal_id=signal_id,
selected_strategy=selected_strategy,
routing_score=routing_score,
confidence=signal_data.get('confidence', 0.5),
reasoning=reasoning,
mathematical_analysis=mathematical_analysis,
routing_parameters=routing_parameters
)

return decision

except Exception as e:
self.logger.error(f"❌ Error making routing decision: {e}")

# Return safe decision
return RoutingDecision(
signal_id=signal_data.get('signal_id', 'unknown'),
selected_strategy='momentum',  # Default strategy
routing_score=0.5,
confidence=0.0,
reasoning="Error in decision making - using default strategy",
mathematical_analysis={},
routing_parameters={}
)

def _get_market_condition(self, symbol: str) -> MarketConditionData:
"""Get market condition for a symbol."""
try:
# Return cached market condition or default
if symbol in self.market_conditions:
return self.market_conditions[symbol]

# Return default market condition
return MarketConditionData(
condition=MarketCondition.SIDEWAYS,
volatility=0.02,  # 2% volatility
trend_strength=0.0,
liquidity_score=0.8
)

except Exception as e:
self.logger.error(f"❌ Error getting market condition: {e}")
return MarketConditionData(
condition=MarketCondition.SIDEWAYS,
volatility=0.02,
trend_strength=0.0,
liquidity_score=0.8
)

async def _calculate_routing_score(self, signal_data: Dict[str, Any],
strategy_name: str,
market_condition: MarketConditionData) -> float:
"""Calculate routing score for a strategy."""
try:
# Get strategy performance
performance = self.strategy_performance[strategy_name]

# Get strategy weight
weight = self.config.strategy_weights.get(strategy_name, 0.1)

# Calculate performance function
performance_score = self._calculate_performance_function(signal_data, performance, market_condition)

# Calculate market condition function
market_score = self._calculate_market_condition_function(signal_data, strategy_name, market_condition)

# Calculate risk factor
risk_factor = self._calculate_risk_factor(signal_data, strategy_name, market_condition)

# Apply mathematical analysis if available
mathematical_boost = 1.0
if self.math_orchestrator and 'mathematical_analysis' in signal_data:
math_confidence = signal_data['mathematical_analysis'].get('confidence', 0.5)
mathematical_boost = 0.8 + (math_confidence * 0.4)  # 0.8 to 1.2 range

# Calculate final score: R(s_i) = argmax_j {w_j * f_j(s_i) + λ_j * g_j(s_i)}
routing_score = (
weight * performance_score +
risk_factor * market_score
) * mathematical_boost

return max(0.0, min(1.0, routing_score))  # Clamp to [0, 1]

except Exception as e:
self.logger.error(f"❌ Error calculating routing score: {e}")
return 0.5

def _calculate_performance_function(self, signal_data: Dict[str, Any], -> None
performance: StrategyPerformance,
market_condition: MarketConditionData) -> float:
"""Calculate performance function f_j(s_i)."""
try:
# Base performance score
base_score = performance.win_rate if performance.total_signals > 0 else 0.5

# Adjust for signal strength
signal_strength = signal_data.get('strength', 0.5)
strength_adjustment = signal_strength * 0.3

# Adjust for market condition alignment
market_alignment = self._calculate_market_alignment(signal_data, market_condition)

# Calculate final performance score
performance_score = base_score + strength_adjustment + market_alignment

return max(0.0, min(1.0, performance_score))

except Exception as e:
self.logger.error(f"❌ Error calculating performance function: {e}")
return 0.5

def _calculate_market_condition_function(self, signal_data: Dict[str, Any], -> None
strategy_name: str,
market_condition: MarketConditionData) -> float:
"""Calculate market condition function g_j(s_i)."""
try:
# Base market score
market_score = 0.5

# Adjust based on strategy type and market condition
if strategy_name == 'momentum':
if market_condition.condition in [MarketCondition.BULL_TRENDING, MarketCondition.BEAR_TRENDING]:
market_score += 0.2
elif market_condition.condition == MarketCondition.SIDEWAYS:
market_score -= 0.1

elif strategy_name == 'mean_reversion':
if market_condition.condition == MarketCondition.SIDEWAYS:
market_score += 0.2
elif market_condition.condition in [MarketCondition.BULL_TRENDING, MarketCondition.BEAR_TRENDING]:
market_score -= 0.1

elif strategy_name == 'scalping':
if market_condition.condition == MarketCondition.VOLATILE:
market_score += 0.2
elif market_condition.condition == MarketCondition.CALM:
market_score -= 0.1

# Adjust for volatility
volatility_adjustment = market_condition.volatility * 2.0  # Scale volatility
market_score += volatility_adjustment

# Adjust for liquidity
liquidity_adjustment = (market_condition.liquidity_score - 0.5) * 0.2
market_score += liquidity_adjustment

return max(0.0, min(1.0, market_score))

except Exception as e:
self.logger.error(f"❌ Error calculating market condition function: {e}")
return 0.5

def _calculate_risk_factor(self, signal_data: Dict[str, Any], -> None
strategy_name: str,
market_condition: MarketConditionData) -> float:
"""Calculate risk adjustment factor λ_j."""
try:
# Base risk factor
risk_factor = 1.0

# Adjust for volatility
volatility_penalty = self.config.risk_factors.get('volatility_penalty', 0.1)
risk_factor -= market_condition.volatility * volatility_penalty

# Adjust for liquidity
liquidity_penalty = self.config.risk_factors.get('liquidity_penalty', 0.05)
risk_factor -= (1.0 - market_condition.liquidity_score) * liquidity_penalty

# Adjust for signal strength
signal_strength = signal_data.get('strength', 0.5)
if signal_strength > 0.8:
risk_factor += 0.1  # Bonus for strong signals
elif signal_strength < 0.3:
risk_factor -= 0.1  # Penalty for weak signals

return max(0.1, min(2.0, risk_factor))  # Clamp to reasonable range

except Exception as e:
self.logger.error(f"❌ Error calculating risk factor: {e}")
return 1.0

def _calculate_market_alignment(self, signal_data: Dict[str, Any], -> None
market_condition: MarketConditionData) -> float:
"""Calculate market alignment score."""
try:
signal_type = signal_data.get('signal_type', 'hold')
condition = market_condition.condition

# Calculate alignment based on signal type and market condition
if signal_type in ['buy', 'strong_buy']:
if condition == MarketCondition.BULL_TRENDING:
return 0.2
elif condition == MarketCondition.BEAR_TRENDING:
return -0.1
else:
return 0.0

elif signal_type in ['sell', 'strong_sell']:
if condition == MarketCondition.BEAR_TRENDING:
return 0.2
elif condition == MarketCondition.BULL_TRENDING:
return -0.1
else:
return 0.0

else:  # hold
if condition == MarketCondition.SIDEWAYS:
return 0.1
else:
return 0.0

except Exception as e:
self.logger.error(f"❌ Error calculating market alignment: {e}")
return 0.0

def _generate_routing_parameters(self, signal_data: Dict[str, Any], -> None
selected_strategy: str,
market_condition: MarketConditionData) -> Dict[str, Any]:
"""Generate routing parameters for selected strategy."""
try:
base_params = {
'strategy': selected_strategy,
'signal_type': signal_data.get('signal_type', 'hold'),
'signal_strength': signal_data.get('strength', 0.5),
'market_condition': market_condition.condition.value,
'volatility': market_condition.volatility,
'liquidity_score': market_condition.liquidity_score
}

# Add strategy-specific parameters
if selected_strategy == 'momentum':
base_params.update({
'lookback_period': 20,
'momentum_threshold': 0.02,
'position_size_multiplier': 1.0
})

elif selected_strategy == 'mean_reversion':
base_params.update({
'mean_period': 50,
'std_dev_multiplier': 2.0,
'position_size_multiplier': 0.8
})

elif selected_strategy == 'scalping':
base_params.update({
'timeframe': '1m',
'profit_target': 0.005,
'stop_loss': 0.003,
'position_size_multiplier': 0.5
})

elif selected_strategy == 'arbitrage':
base_params.update({
'min_spread': 0.001,
'execution_timeout': 30,
'position_size_multiplier': 0.3
})

else:  # Default parameters
base_params.update({
'position_size_multiplier': 0.7,
'risk_adjustment': 1.0
})

return base_params

except Exception as e:
self.logger.error(f"❌ Error generating routing parameters: {e}")
return {'strategy': selected_strategy, 'error': str(e)}

async def _analyze_decision_mathematically(self, signal_data: Dict[str, Any],
selected_strategy: str,
routing_score: float,
market_condition: MarketConditionData) -> Dict[str, Any]:
"""Perform mathematical analysis on routing decision."""
try:
if not self.math_orchestrator:
return {}

# Prepare decision data for mathematical analysis
decision_data = np.array([
routing_score,
signal_data.get('strength', 0.0),
signal_data.get('confidence', 0.0),
market_condition.volatility,
market_condition.liquidity_score,
len(selected_strategy)
])

# Perform mathematical orchestration
result = self.math_orchestrator.process_data(decision_data)

return {
'mathematical_score': float(result),
'routing_confidence': routing_score,
'strategy_complexity': len(selected_strategy),
'market_volatility': market_condition.volatility,
'timestamp': time.time()
}

except Exception as e:
self.logger.error(f"❌ Error analyzing decision mathematically: {e}")
return {}

def _generate_routing_reasoning(self, signal_data: Dict[str, Any], -> None
selected_strategy: str,
routing_score: float,
market_condition: MarketConditionData) -> str:
"""Generate human-readable reasoning for routing decision."""
try:
signal_type = signal_data.get('signal_type', 'hold')
signal_strength = signal_data.get('strength', 0.5)

reasoning_parts = []

# Strategy selection reasoning
if routing_score > 0.8:
reasoning_parts.append(f"High confidence routing to {selected_strategy}")
elif routing_score > 0.6:
reasoning_parts.append(f"Moderate confidence routing to {selected_strategy}")
else:
reasoning_parts.append(f"Low confidence routing to {selected_strategy}")

# Market condition reasoning
condition_name = market_condition.condition.value.replace('_', ' ').title()
reasoning_parts.append(f"Market condition: {condition_name}")

# Signal strength reasoning
if signal_strength > 0.8:
reasoning_parts.append("Strong signal detected")
elif signal_strength > 0.6:
reasoning_parts.append("Moderate signal strength")
else:
reasoning_parts.append("Weak signal - conservative routing")

# Strategy-specific reasoning
if selected_strategy == 'momentum':
reasoning_parts.append("Momentum strategy selected for trend following")
elif selected_strategy == 'mean_reversion':
reasoning_parts.append("Mean reversion strategy for range-bound markets")
elif selected_strategy == 'scalping':
reasoning_parts.append("Scalping strategy for high-frequency opportunities")
elif selected_strategy == 'arbitrage':
reasoning_parts.append("Arbitrage strategy for price inefficiencies")

return " | ".join(reasoning_parts)

except Exception as e:
self.logger.error(f"❌ Error generating routing reasoning: {e}")
return f"Routed to {selected_strategy} with score {routing_score:.3f}"

def _update_strategy_performance(self, decision: RoutingDecision) -> None:
"""Update strategy performance metrics."""
try:
strategy_name = decision.selected_strategy
if strategy_name not in self.strategy_performance:
return

performance = self.strategy_performance[strategy_name]

# Update basic metrics
performance.total_signals += 1

# Update average score
current_avg = performance.average_score
total_signals = performance.total_signals
performance.average_score = (
(current_avg * (total_signals - 1) + decision.routing_score) / total_signals
)

# Update mathematical signature
performance.mathematical_signature = decision.mathematical_analysis.get('mathematical_signature', '')

# Note: Win rate would be updated after actual execution results

except Exception as e:
self.logger.error(f"❌ Error updating strategy performance: {e}")

async def _process_routing_queue(self) -> None:
"""Process routing decisions from the queue."""
try:
while self.active:
try:
# Get decision from queue
decision = await asyncio.wait_for(
self.routing_queue.get(),
timeout=1.0
)

# Execute routing decision
await self._execute_routing_decision(decision)

# Mark task as done
self.routing_queue.task_done()

except asyncio.TimeoutError:
continue
except Exception as e:
self.logger.error(f"❌ Error processing routing decision: {e}")

except Exception as e:
self.logger.error(f"❌ Error in routing processing loop: {e}")

async def _execute_routing_decision(self, decision: RoutingDecision) -> None:
"""Execute a routing decision."""
try:
# Update metrics
self.metrics.successful_routes += 1

# Log execution
self.logger.info(f"✅ Routing decision executed: {decision.signal_id} -> {decision.selected_strategy}")

# In production, this would trigger actual strategy execution
# For now, just simulate execution
await asyncio.sleep(0.1)  # Simulate execution time

# Remove from active routes
if decision.signal_id in self.active_routes:
del self.active_routes[decision.signal_id]

self.logger.info(f"✅ Routing decision completed: {decision.selected_strategy}")

except Exception as e:
self.logger.error(f"❌ Error executing routing decision: {e}")

def update_market_condition(self, symbol: str, condition_data: Dict[str, Any]) -> bool:
"""Update market condition for a symbol."""
try:
condition = MarketCondition(condition_data.get('condition', 'sideways'))

market_condition = MarketConditionData(
condition=condition,
volatility=float(condition_data.get('volatility', 0.02)),
trend_strength=float(condition_data.get('trend_strength', 0.0)),
liquidity_score=float(condition_data.get('liquidity_score', 0.8))
)

self.market_conditions[symbol] = market_condition

self.logger.info(f"✅ Updated market condition for {symbol}: {condition.value}")
return True

except Exception as e:
self.logger.error(f"❌ Error updating market condition: {e}")
return False

def get_strategy_performance(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
"""Get strategy performance metrics."""
try:
if strategy_name:
if strategy_name not in self.strategy_performance:
return {}

performance = self.strategy_performance[strategy_name]
return {
'strategy_name': performance.strategy_name,
'total_signals': performance.total_signals,
'successful_routes': performance.successful_routes,
'average_score': performance.average_score,
'win_rate': performance.win_rate,
'risk_score': performance.risk_score,
'mathematical_signature': performance.mathematical_signature
}
else:
# Return all strategy performance
return {
strategy_name: {
'total_signals': perf.total_signals,
'successful_routes': perf.successful_routes,
'average_score': perf.average_score,
'win_rate': perf.win_rate,
'risk_score': perf.risk_score
}
for strategy_name, perf in self.strategy_performance.items()
}

except Exception as e:
self.logger.error(f"❌ Error getting strategy performance: {e}")
return {}

def get_recent_routing_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
"""Get recent routing decisions."""
try:
recent_decisions = self.routing_history[-limit:]
return [
{
'signal_id': decision.signal_id,
'selected_strategy': decision.selected_strategy,
'routing_score': decision.routing_score,
'confidence': decision.confidence,
'reasoning': decision.reasoning,
'timestamp': decision.timestamp,
'mathematical_analysis': decision.mathematical_analysis
}
for decision in recent_decisions
]
except Exception as e:
self.logger.error(f"❌ Error getting recent routing decisions: {e}")
return []

def get_performance_metrics(self) -> Dict[str, Any]:
"""Get system performance metrics."""
return {
'signals_routed': self.metrics.signals_routed,
'successful_routes': self.metrics.successful_routes,
'routing_errors': self.metrics.routing_errors,
'average_routing_time': self.metrics.average_routing_time,
'routing_accuracy': self.metrics.routing_accuracy,
'mathematical_analyses': self.metrics.mathematical_analyses,
'last_updated': time.time()
}

def activate(self) -> bool:
"""Activate the system."""
if not self.initialized:
self.logger.error("System not initialized")
return False

try:
self.active = True
self.logger.info("✅ Strategy Router System activated")
return True
except Exception as e:
self.logger.error(f"❌ Error activating Strategy Router System: {e}")
return False

def deactivate(self) -> bool:
"""Deactivate the system."""
try:
self.active = False
self.logger.info("✅ Strategy Router System deactivated")
return True
except Exception as e:
self.logger.error(f"❌ Error deactivating Strategy Router System: {e}")
return False

def get_status(self) -> Dict[str, Any]:
"""Get system status."""
return {
'active': self.active,
'initialized': self.initialized,
'active_routes_count': len(self.active_routes),
'signals_queued': self.signal_queue.qsize(),
'routing_queued': self.routing_queue.qsize(),
'performance_metrics': self.get_performance_metrics(),
'config': {
'enabled': self.config.enabled,
'max_concurrent_routes': self.config.max_concurrent_routes,
'routing_threshold': self.config.routing_threshold,
'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled,
'adaptive_routing_enabled': self.config.adaptive_routing_enabled
}
}

def calculate_mathematical_result(self, data: Union[List, np.ndarray]) -> float:
"""Calculate mathematical result with proper data handling and strategy routing integration."""
try:
if not isinstance(data, np.ndarray):
data = np.array(data)

if MATH_INFRASTRUCTURE_AVAILABLE and self.math_orchestrator:
# Use the actual mathematical modules for calculation
if len(data) > 0:
# Use mathematical orchestration for strategy routing analysis
result = self.math_orchestrator.process_data(data)
return float(result)
else:
return 0.0
else:
# Fallback to basic calculation
result = np.sum(data) / len(data) if len(data) > 0 else 0.0
return float(result)
except Exception as e:
self.logger.error(f"Mathematical calculation error: {e}")
return 0.0


def create_strategy_router(config: Optional[StrategyRouterConfig] = None) -> StrategyRouter:
"""Factory function to create StrategyRouter instance."""
return StrategyRouter(config)


async def main():
"""Main function for testing."""
# Create configuration
config = StrategyRouterConfig(
enabled=True,
debug=True,
max_concurrent_routes=10,
routing_threshold=0.6,
mathematical_analysis_enabled=True,
adaptive_routing_enabled=True
)

# Create strategy router
router = create_strategy_router(config)

# Activate system
router.activate()

# Start router
await router.start_router()

# Submit test signals
test_signals = [
{
'signal_id': 'test_001',
'symbol': 'BTC/USDT',
'signal_type': 'buy',
'strength': 0.8,
'confidence': 0.9,
'price': 50000.0,
'volume': 100.0
},
{
'signal_id': 'test_002',
'symbol': 'ETH/USDT',
'signal_type': 'sell',
'strength': 0.7,
'confidence': 0.8,
'price': 3000.0,
'volume': 50.0
}
]

# Submit signals
for signal in test_signals:
await router.route_signal(signal)

# Wait for processing
await asyncio.sleep(5)

# Get status
status = router.get_status()
print(f"System Status: {json.dumps(status, indent=2)}")

# Get strategy performance
performance = router.get_strategy_performance()
print(f"Strategy Performance: {json.dumps(performance, indent=2)}")

# Get recent decisions
decisions = router.get_recent_routing_decisions()
print(f"Recent Decisions: {json.dumps(decisions, indent=2)}")

# Stop router
await router.stop_router()

# Deactivate system
router.deactivate()


if __name__ == "__main__":
asyncio.run(main())
