"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual State Router - ZPE/ZBE Strategy Routing & Compute Mode Selection

Implements Nexus mathematics for dual-state strategy routing:
- ZPE Mode: CPU-based Zero Point Efficiency calculations
- ZBE Mode: GPU-based Zero Bottleneck Entropy processing
- Strategy tier routing based on time sensitivity and compute requirements
- Automatic mode selection based on performance metrics and system load
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

class StrategyTier(Enum):
"""Class for Schwabot trading functionality."""
"""Strategy time tiers for routing decisions."""
SHORT = "short"  # <300ms, tick-level, high-urgency
MID = "mid"      # 300ms-2s, volume curves, pattern matching
LONG = "long"    # >2s, fractal sequences, timing overlays

class ComputeMode(Enum):
"""Class for Schwabot trading functionality."""
"""Available computation modes."""
ZPE = "zpe"  # CPU-based, Zero Point Efficiency
ZBE = "zbe"  # GPU-based, Zero Bottleneck Entropy

class RouterState(Enum):
"""Class for Schwabot trading functionality."""
"""Router operational states."""
IDLE = "idle"
ROUTING = "routing"
EXECUTING = "executing"
ERROR = "error"

@dataclass
class StrategyMetadata:
"""Class for Schwabot trading functionality."""
"""Metadata for strategy routing decisions."""
strategy_id: str
tier: StrategyTier
priority: float  # 0.0 to 1.0 (profit, density)
avg_compute_time_ms: float
avg_profit_margin: float
success_rate: float
last_execution: datetime
preferred_mode: ComputeMode
execution_count: int = 0
total_profit: float = 0.0

def __post_init__(self) -> None:
"""Validate strategy metadata constraints."""
if not (0.0 <= self.priority <= 1.0):
raise ValueError("Priority must be in [0.0, 1.0]")
if not (0.0 <= self.success_rate <= 1.0):
raise ValueError("Success rate must be in [0.0, 1.0]")

@dataclass
class ExecutionResult:
"""Class for Schwabot trading functionality."""
"""Result of strategy execution."""
strategy_id: str
success: bool
execution_time_ms: float
profit: float
mode_used: ComputeMode
timestamp: datetime
metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingDecision:
"""Class for Schwabot trading functionality."""
"""Routing decision for strategy execution."""
strategy_id: str
selected_mode: ComputeMode
confidence: float
reasoning: str
timestamp: datetime
metadata: Dict[str, Any] = field(default_factory=dict)

class DualStateRouter:
"""Class for Schwabot trading functionality."""
"""
Dual State Router - ZPE/ZBE Strategy Routing & Compute Mode Selection

Implements the Nexus mathematics for dual-state strategy routing:
- ZPE Mode: CPU-based Zero Point Efficiency calculations
- ZBE Mode: GPU-based Zero Bottleneck Entropy processing
- Strategy tier routing based on time sensitivity and compute requirements
"""


def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
"""Initialize the Dual State Router."""
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)
self.state = RouterState.IDLE
self.initialized = False

# Router parameters
self.zpe_threshold = self.config.get('zpe_threshold', 0.5)
self.zbe_threshold = self.config.get('zbe_threshold', 0.7)
self.timeout_ms = self.config.get('timeout_ms', 1000)
self.max_retries = self.config.get('max_retries', 3)

# Strategy tracking
self.strategies: Dict[str, StrategyMetadata] = {}
self.execution_history: List[ExecutionResult] = []
self.routing_history: List[RoutingDecision] = []

# Performance metrics
self.zpe_performance = {'avg_time': 0.0, 'success_rate': 0.0, 'total_executions': 0}
self.zbe_performance = {'avg_time': 0.0, 'success_rate': 0.0, 'total_executions': 0}

self._initialize_router()

def _default_config(self) -> Dict[str, Any]:
"""Default configuration for Dual State Router."""
return {
'zpe_threshold': 0.5,      # ZPE mode selection threshold
'zbe_threshold': 0.7,      # ZBE mode selection threshold
'timeout_ms': 1000,        # Execution timeout
'max_retries': 3,          # Maximum retry attempts
'performance_window': 100,  # Performance tracking window
'load_balancing': True,    # Enable load balancing
'adaptive_routing': True,  # Enable adaptive routing
}


def _initialize_router(self) -> None:
"""Initialize the router system."""
try:
self.logger.info("Initializing Dual State Router...")

# Validate thresholds
if not (0.0 <= self.zpe_threshold <= 1.0):
raise ValueError("zpe_threshold must be between 0.0 and 1.0")
if not (0.0 <= self.zbe_threshold <= 1.0):
raise ValueError("zbe_threshold must be between 0.0 and 1.0")

# Initialize performance tracking
self.performance_window = self.config.get('performance_window', 100)
self.load_balancing = self.config.get('load_balancing', True)
self.adaptive_routing = self.config.get('adaptive_routing', True)

self.initialized = True
self.state = RouterState.IDLE
self.logger.info("[SUCCESS] Dual State Router initialized successfully")

except Exception as e:
self.logger.error(f"[FAIL] Error initializing Dual State Router: {e}")
self.initialized = False
self.state = RouterState.ERROR


def register_strategy(self, strategy_metadata: StrategyMetadata) -> None:
"""Register a strategy with the router."""
try:
self.strategies[strategy_metadata.strategy_id] = strategy_metadata
self.logger.info(f"Registered strategy: {strategy_metadata.strategy_id}")
except Exception as e:
self.logger.error(f"Error registering strategy: {e}")


def unregister_strategy(self, strategy_id: str) -> None:
"""Unregister a strategy from the router."""
try:
if strategy_id in self.strategies:
del self.strategies[strategy_id]
self.logger.info(f"Unregistered strategy: {strategy_id}")
except Exception as e:
self.logger.error(f"Error unregistering strategy: {e}")

def compute_routing_score(self, strategy_metadata: StrategyMetadata) -> float:
"""
Compute routing score for strategy execution mode selection.

Args:
strategy_metadata: Strategy metadata

Returns:
Routing score (0.0 to 1.0)
"""
try:
# Base score from priority and success rate
base_score = (strategy_metadata.priority + strategy_metadata.success_rate) / 2.0

# Time sensitivity factor
if strategy_metadata.tier == StrategyTier.SHORT:
time_factor = 1.0  # High urgency
elif strategy_metadata.tier == StrategyTier.MID:
time_factor = 0.7  # Medium urgency
else:  # LONG
time_factor = 0.4  # Low urgency

# Compute time factor (faster is better)
compute_factor = max(0.1, 1.0 - (strategy_metadata.avg_compute_time_ms / 1000.0))

# Profit margin factor
profit_factor = min(1.0, strategy_metadata.avg_profit_margin * 10.0)

# Combined routing score
routing_score = (base_score * 0.4 +
time_factor * 0.3 +
compute_factor * 0.2 +
profit_factor * 0.1)

return np.clip(routing_score, 0.0, 1.0)

except Exception as e:
self.logger.error(f"Error computing routing score: {e}")
return 0.5

def select_compute_mode(self, strategy_metadata: StrategyMetadata) -> ComputeMode:
"""
Select the appropriate compute mode for strategy execution.

Args:
strategy_metadata: Strategy metadata

Returns:
Selected compute mode
"""
try:
# Compute routing score
routing_score = self.compute_routing_score(strategy_metadata)

# Mode selection logic
if routing_score >= self.zbe_threshold:
selected_mode = ComputeMode.ZBE
reasoning = f"High routing score ({
routing_score:.3f}) >= ZBE threshold ({
self.zbe_threshold})"
elif routing_score >= self.zpe_threshold:
selected_mode = ComputeMode.ZPE
reasoning = f"Medium routing score ({
routing_score:.3f}) >= ZPE threshold ({
self.zpe_threshold})"
else:
# Default to preferred mode or ZPE
selected_mode = strategy_metadata.preferred_mode or ComputeMode.ZPE
reasoning = f"Low routing score ({
routing_score:.3f}), using preferred mode"

# Load balancing override
if self.load_balancing:
selected_mode = self._apply_load_balancing(
selected_mode, routing_score)

return selected_mode

except Exception as e:
self.logger.error(f"Error selecting compute mode: {e}")
return ComputeMode.ZPE

def _apply_load_balancing(
self, selected_mode: ComputeMode, routing_score: float) -> ComputeMode:
"""
Apply load balancing to mode selection.

Args:
selected_mode: Initially selected mode
routing_score: Routing score

Returns:
Load-balanced mode selection
"""
try:
# Get current performance metrics
zpe_load = self.zpe_performance['total_executions']
zbe_load = self.zbe_performance['total_executions']

# Load balancing threshold
load_threshold = 0.3

if selected_mode == ComputeMode.ZBE and zbe_load > zpe_load * \
(1 + load_threshold):
# ZBE is overloaded, consider ZPE
if routing_score >= self.zpe_threshold:
return ComputeMode.ZPE

elif selected_mode == ComputeMode.ZPE and zpe_load > zbe_load * (1 + load_threshold):
# ZPE is overloaded, consider ZBE
if routing_score >= self.zbe_threshold:
return ComputeMode.ZBE

return selected_mode

except Exception as e:
self.logger.error(f"Error applying load balancing: {e}")
return selected_mode

def route_strategy(self, strategy_id: str, input_data: Dict[str, Any] = None) -> RoutingDecision:
"""
Route a strategy for execution.

Args:
strategy_id: Strategy identifier
input_data: Input data for strategy execution

Returns:
Routing decision
"""
try:
if not self.initialized:
raise RuntimeError("Router not initialized")

if strategy_id not in self.strategies:
raise ValueError(
f"Strategy not registered: {strategy_id}")

self.state = RouterState.ROUTING

# Get strategy metadata
strategy_metadata = self.strategies[strategy_id]

# Select compute mode
selected_mode = self.select_compute_mode(
strategy_metadata)

# Compute routing score for reasoning
routing_score = self.compute_routing_score(
strategy_metadata)

# Create routing decision
decision = RoutingDecision(
strategy_id=strategy_id,
selected_mode=selected_mode,
confidence=routing_score,
reasoning=f"Selected {selected_mode.value} mode with score {routing_score:.3f}",
timestamp=datetime.now()
)

# Store decision
self.routing_history.append(decision)

# Keep history manageable
max_history = 1000
if len(self.routing_history) > max_history:
self.routing_history = self.routing_history[-max_history:]

self.state = RouterState.IDLE
return decision

except Exception as e:
self.logger.error(
f"Error routing strategy: {e}")
self.state = RouterState.ERROR
return RoutingDecision(
strategy_id=strategy_id,
selected_mode=ComputeMode.ZPE,
confidence=0.0,
reasoning=f"Error: {str(e)}",
timestamp=datetime.now()
)

def execute_strategy(self, strategy_id: str, input_data: Dict[str, Any] = None) -> ExecutionResult:
"""
Execute a strategy using the selected compute mode.

Args:
strategy_id: Strategy identifier
input_data: Input data for strategy execution

Returns:
Execution result
"""
try:
if not self.initialized:
raise RuntimeError("Router not initialized")

self.state = RouterState.EXECUTING
start_time = time.time()

# Route the strategy
routing_decision = self.route_strategy(
strategy_id, input_data)

# Execute based on selected mode
if routing_decision.selected_mode == ComputeMode.ZPE:
success, result_data = self._execute_zpe_mode(
strategy_id, input_data)
else:  # ZBE mode
success, result_data = self._execute_zbe_mode(
strategy_id, input_data)

# Calculate execution time
execution_time_ms = (
time.time() - start_time) * 1000.0

# Create execution result
result = ExecutionResult(
strategy_id=strategy_id,
success=success,
execution_time_ms=execution_time_ms,
profit=result_data.get('profit', 0.0),
mode_used=routing_decision.selected_mode,
timestamp=datetime.now(),
metadata=result_data
)

# Update performance metrics
self._update_performance_metrics(result)

# Store result
self.execution_history.append(result)

# Keep history manageable
max_history = 1000
if len(self.execution_history) > max_history:
self.execution_history = self.execution_history[-max_history:]

self.state = RouterState.IDLE
return result

except Exception as e:
self.logger.error(
f"Error executing strategy: {e}")
self.state = RouterState.ERROR
return ExecutionResult(
strategy_id=strategy_id,
success=False,
execution_time_ms=0.0,
profit=0.0,
mode_used=ComputeMode.ZPE,
timestamp=datetime.now(),
metadata={'error': str(e)}
)

def _execute_zpe_mode(self, strategy_id: str, input_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
"""Execute strategy in ZPE (CPU) mode."""
try:
# Placeholder for ZPE execution
# In a real implementation, this would call the actual ZPE strategy
# Simulate computation time
time.sleep(0.01)

return True, {
'profit': 0.1,
'mode': 'zpe',
'computation_units': 100
}
except Exception as e:
self.logger.error(
f"Error in ZPE execution: {e}")
return False, {'error': str(e)}

def _execute_zbe_mode(self, strategy_id: str, input_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
"""Execute strategy in ZBE (GPU) mode."""
try:
# Placeholder for ZBE execution
# In a real implementation, this would call the actual ZBE strategy
# Simulate faster computation time
time.sleep(0.005)

return True, {
'profit': 0.15,
'mode': 'zbe',
'computation_units': 200
}
except Exception as e:
self.logger.error(
f"Error in ZBE execution: {e}")
return False, {'error': str(e)}


def _update_performance_metrics(self, result: ExecutionResult) -> None:
"""Update performance metrics based on execution result."""
try:
if result.mode_used == ComputeMode.ZPE:
metrics = self.zpe_performance
else:
metrics = self.zbe_performance

# Update metrics
total_executions = metrics['total_executions'] + 1
current_avg_time = metrics['avg_time']
current_success_rate = metrics['success_rate']

# Update average execution time
new_avg_time = (current_avg_time * metrics['total_executions'] + \
result.execution_time_ms) / total_executions

# Update success rate
successful_executions = int(current_success_rate * \
metrics['total_executions']) + (1 if result.success else 0)
new_success_rate = successful_executions / total_executions

# Store updated metrics
metrics['avg_time'] = new_avg_time
metrics['success_rate'] = new_success_rate
metrics['total_executions'] = total_executions

except Exception as e:
self.logger.error(f"Error updating performance metrics: {e}")

def get_router_summary(self) -> Dict[str, Any]:
"""Get comprehensive router summary."""
if not self.strategies:
return {'status': 'no_strategies'}

# Compute router statistics
total_strategies = len(self.strategies)
total_executions = len(self.execution_history)
total_routings = len(self.routing_history)

# Mode distribution
mode_distribution = {}
for mode in ComputeMode:
mode_distribution[mode.value] = sum(
1 for r in self.routing_history if r.selected_mode == mode)

# Tier distribution
tier_distribution = {}
for tier in StrategyTier:
tier_distribution[tier.value] = sum(
1 for s in self.strategies.values() if s.tier == tier)

return {
'total_strategies': total_strategies,
'total_executions': total_executions,
'total_routings': total_routings,
'router_state': self.state.value,
'mode_distribution': mode_distribution,
'tier_distribution': tier_distribution,
'zpe_performance': self.zpe_performance,
'zbe_performance': self.zbe_performance,
'initialized': self.initialized,
'load_balancing': self.load_balancing,
'adaptive_routing': self.adaptive_routing
}


def reset_router(self) -> None:
"""Reset the router to initial state."""
self.strategies.clear()
self.execution_history.clear()
self.routing_history.clear()
self.zpe_performance = {'avg_time': 0.0, 'success_rate': 0.0, 'total_executions': 0}
self.zbe_performance = {'avg_time': 0.0, 'success_rate': 0.0, 'total_executions': 0}
self.state = RouterState.IDLE
self.logger.info("Dual State Router reset")

# Factory function
def create_dual_state_router(config: Optional[Dict[str, Any]] = None) -> DualStateRouter:
"""Create a Dual State Router instance."""
return DualStateRouter(config)
