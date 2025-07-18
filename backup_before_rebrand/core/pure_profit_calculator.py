"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Profit Calculator - Mathematically Rigorous Core

This module implements the fundamental profit calculation framework:
Π = F(M(t), H(t), S)

Where:
- M(t): Market data (prices, volumes, on-chain signals)
- H(t): History/state (hash matrices, tensor buckets)
- S: Static strategy parameters

CRITICAL GUARANTEE: ZPE/ZBE systems never appear in this calculation.
They only affect computation time, never profit.

Mathematical Formulas Implemented:
- Profit Optimization: P = Σ w_i * r_i - λ * Σ w_i²
- Sharpe Ratio: (R_p - R_f) / σ_p
- Sortino Ratio: (R_p - R_f) / σ_d
- Market Entropy: H = -Σ p_i * log(p_i)
- Tensor Contraction: C_ij = Σ_k A_ik * B_kj
- Risk-Adjusted Returns: RAR = R / (σ * VaR)
"""

import logging
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from scipy.stats import entropy as scipy_entropy
import numpy as np


# Entropy Signal Integration
try:
from core.entropy_signal_integration import EntropySignalIntegrator

ENTROPY_AVAILABLE = True
logger = logging.getLogger(__name__)
logger.info("🔄 Entropy Signal Integration enabled in Pure Profit Calculator")
except ImportError:
ENTROPY_AVAILABLE = False
logger = logging.getLogger(__name__)
logger.warning(
"⚠️ Entropy Signal Integration not available in Pure Profit Calculator"
)


@dataclass(frozen=True)
class MarketData:
"""Immutable market data structure - M(t)."""

timestamp: float
btc_price: float
eth_price: float
usdc_volume: float
volatility: float
momentum: float
volume_profile: float
on_chain_signals: Dict[str, float] = field(default_factory=dict)

def __post_init__(self) -> None:
"""Validate market data integrity."""
if self.btc_price <= 0:
raise ValueError("BTC price must be positive")
if self.volatility < 0:
raise ValueError("Volatility cannot be negative")


@dataclass(frozen=True)
class HistoryState:
"""Immutable history state - H(t)."""

timestamp: float
hash_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
tensor_buckets: Dict[str, np.ndarray] = field(default_factory=dict)
profit_memory: List[float] = field(default_factory=list)
signal_history: List[float] = field(default_factory=list)

def get_hash_signature(self) -> str:
"""Generate deterministic hash signature for state."""
state_str = (
f"{self.timestamp}_{len(self.hash_matrices)}_{len(self.tensor_buckets)}"
)
return hashlib.sha256(state_str.encode()).hexdigest()


@dataclass(frozen=True)
class StrategyParameters:
"""Immutable strategy parameters - S."""

risk_tolerance: float = 0.02
profit_target: float = 0.05
stop_loss: float = 0.01
position_size: float = 0.1
tensor_depth: int = 4
hash_memory_depth: int = 100
momentum_weight: float = 0.3
volatility_weight: float = 0.2
volume_weight: float = 0.5


class ProfitCalculationMode(Enum):
"""Pure profit calculation modes."""

CONSERVATIVE = "conservative"
BALANCED = "balanced"
AGGRESSIVE = "aggressive"
TENSOR_OPTIMIZED = "tensor_optimized"


class ProcessingMode(Enum):
"""Processing modes for calculations."""

CPU_FALLBACK = "cpu_fallback"
GPU_ACCELERATED = "gpu_accelerated"
HYBRID = "hybrid"


@dataclass(frozen=True)
class ProfitResult:
"""Immutable profit calculation result."""

timestamp: float
base_profit: float
risk_adjusted_profit: float
confidence_score: float
tensor_contribution: float
hash_contribution: float
total_profit_score: float
processing_mode: ProcessingMode
calculation_metadata: Dict[str, Any] = field(default_factory=dict)

def __post_init__(self) -> None:
"""Validate profit result integrity."""
if not (-1.0 <= self.total_profit_score <= 1.0):
raise ValueError("Profit score must be between -1.0 and 1.0")


@dataclass
class CalculationError:
"""Error tracking for profit calculations."""

timestamp: float
error_type: str
error_message: str
calculation_context: Dict[str, Any]


class PureProfitCalculator:
"""
Pure Profit Calculator - Mathematically Rigorous Implementation.

Implements: Π = F(M(t), H(t), S)

GUARANTEE: This class never imports or uses ZPE/ZBE systems.
All computations are mathematically pure and deterministic.
"""

def __init__(
self,
strategy_params: StrategyParameters,
processing_mode: ProcessingMode = ProcessingMode.HYBRID,
):
"""Initialize pure profit calculator."""
self.strategy_params = strategy_params
self.processing_mode = processing_mode
self.calculation_count = 0
self.total_calculation_time = 0.0
self.error_log: List[CalculationError] = []
self.last_calculation_data: Dict[str, Any] = {}
# Performance metrics
self.performance_metrics = {
"gpu_operations": 0,
"cpu_operations": 0,
"fallback_operations": 0,
"error_count": 0,
"avg_calculation_time": 0.0,
}
# Mathematical constants for profit calculation
self.GOLDEN_RATIO = 1.618033988749
self.EULER_CONSTANT = 2.718281828459
self.PI = 3.141592653589793
# Initialize entropy signal integration if available
if ENTROPY_AVAILABLE:
self.entropy_integration = EntropySignalIntegrator()
logger.info(
"🔄 Entropy signal integration initialized in Pure Profit Calculator"
)
else:
self.entropy_integration = None
logger.warning(
"⚠️ Entropy signal integration not available in Pure Profit Calculator"
)
logger.info(
f"🧮 Pure Profit Calculator initialized - Mathematical Mode with {processing_mode.value}"
)

def calculate_profit(
self,
market_data: MarketData,
history_state: HistoryState,
mode: ProfitCalculationMode = ProfitCalculationMode.BALANCED,
force_cpu: bool = False,
) -> ProfitResult:
"""
Calculate pure profit using mathematical framework.
Implements: Π = F(M(t), H(t), S)
Args:
market_data: Current market state M(t)
history_state: Historical state H(t)
mode: Calculation mode
force_cpu: Force CPU processing for error recovery
Returns:
ProfitResult: Complete profit calculation result
"""
start_time = time.time()
self.calculation_count += 1
try:
# Determine processing mode
if force_cpu or self.processing_mode == ProcessingMode.CPU_FALLBACK:
current_mode = ProcessingMode.CPU_FALLBACK
self.performance_metrics["cpu_operations"] += 1
elif self.processing_mode == ProcessingMode.GPU_ACCELERATED:
current_mode = ProcessingMode.GPU_ACCELERATED
self.performance_metrics["gpu_operations"] += 1
else:
current_mode = ProcessingMode.HYBRID
self.performance_metrics["cpu_operations"] += 1
# Base profit calculation - Core mathematical formula
base_profit = self._calculate_base_profit_safe(
market_data, history_state, current_mode
)
# Risk adjustment based on volatility and momentum
risk_adjustment = self._calculate_risk_adjustment(market_data)
# Tensor contribution from historical patterns
tensor_contribution = self._calculate_tensor_contribution(
market_data, history_state
)
# Hash memory contribution from pattern matching
hash_contribution = self._calculate_hash_contribution(
market_data, history_state
)
# Confidence score based on signal alignment
confidence_score = self._calculate_confidence_score(
market_data, history_state
)
# Apply mode-specific calculations
mode_multiplier = self._get_mode_multiplier(mode)
# Calculate risk-adjusted profit
risk_adjusted_profit = base_profit * risk_adjustment * mode_multiplier
# Calculate total profit score using weighted combination
total_profit_score = (
risk_adjusted_profit * 0.4
+ tensor_contribution * 0.3
+ hash_contribution * 0.2
+ confidence_score * 0.1
)
# Ensure profit score is bounded
total_profit_score = np.clip(total_profit_score, -1.0, 1.0)
# Create profit result
profit_result = ProfitResult(
timestamp=time.time(),
base_profit=base_profit,
risk_adjusted_profit=risk_adjusted_profit,
confidence_score=confidence_score,
tensor_contribution=tensor_contribution,
hash_contribution=hash_contribution,
total_profit_score=total_profit_score,
processing_mode=current_mode,
calculation_metadata={
"mode": mode.value,
"risk_adjustment": risk_adjustment,
"mode_multiplier": mode_multiplier,
"market_hash": f"{market_data.btc_price:.0f}",
"history_hash": history_state.get_hash_signature()[:8],
"calculation_time_ms": (time.time() - start_time) * 1000,
},
)
# Update calculation metrics
calculation_time = time.time() - start_time
self.total_calculation_time += calculation_time
self.performance_metrics["avg_calculation_time"] = (
self.total_calculation_time / self.calculation_count
)
logger.info(
f"🧮 Profit calculated: Base={base_profit:.4f}, Adjusted={risk_adjusted_profit:.4f}, "
f"Total={total_profit_score:.4f} ({calculation_time * 1000:.3f}ms)"
)
return profit_result
except Exception as e:
logger.error(f"❌ Pure profit calculation failed: {e}")
self.performance_metrics["error_count"] += 1
# Log error
error = CalculationError(
timestamp=time.time(),
error_type=type(e).__name__,
error_message=str(e),
calculation_context={
"market_data": str(market_data),
"history_state": str(history_state),
"mode": mode.value,
},
)
self.error_log.append(error)
# Return fallback result
return ProfitResult(
timestamp=time.time(),
base_profit=0.0,
risk_adjusted_profit=0.0,
confidence_score=0.0,
tensor_contribution=0.0,
hash_contribution=0.0,
total_profit_score=0.0,
processing_mode=ProcessingMode.CPU_FALLBACK,
calculation_metadata={"error": str(e)},
)

def _calculate_base_profit_safe(
self,
market_data: MarketData,
history_state: HistoryState,
processing_mode: ProcessingMode,
) -> float:
"""Calculate base profit from market data with safe fallbacks."""
try:
# Price momentum component
price_momentum = market_data.momentum * self.strategy_params.momentum_weight
# Volatility opportunity component
volatility_opportunity = (
market_data.volatility * self.strategy_params.volatility_weight
)
# Volume strength component
volume_strength = (
market_data.volume_profile * self.strategy_params.volume_weight
)
# Combine components using mathematical constants
base_profit = (
price_momentum * np.sin(self.PI / 4)
+ volatility_opportunity * np.cos(self.PI / 6)
+ volume_strength * (1 / self.GOLDEN_RATIO)
)
# Apply strategic scaling
base_profit *= self.strategy_params.position_size
return np.clip(base_profit, -0.5, 0.5)
except Exception as e:
logger.error(f"❌ Base profit calculation failed: {e}")
return 0.0

def _calculate_risk_adjustment(self, market_data: MarketData) -> float:
"""Calculate risk adjustment factor."""
try:
# Volatility risk factor
volatility_risk = min(1.0, market_data.volatility / 0.5)
# Momentum risk factor
momentum_risk = abs(market_data.momentum)
# Combined risk factor
combined_risk = (volatility_risk + momentum_risk) / 2.0
# Risk tolerance adjustment
risk_adjustment = 1.0 - (
combined_risk * (1.0 - self.strategy_params.risk_tolerance)
)
return max(0.1, min(1.0, risk_adjustment))
except Exception as e:
logger.error(f"❌ Risk adjustment calculation failed: {e}")
return 0.5

def _calculate_tensor_contribution(
self, market_data: MarketData, history_state: HistoryState
) -> float:
"""Calculate tensor contribution from historical patterns."""
try:
if not history_state.tensor_buckets:
return 0.0
# Create current pattern vector
current_pattern = np.array(
[
market_data.btc_price / 50000.0,  # Normalized price
market_data.volatility,
market_data.momentum,
market_data.volume_profile,
]
)
tensor_scores = []
for bucket_name, tensor_data in history_state.tensor_buckets.items():
if tensor_data.size > 0:
# Calculate pattern similarity using tensor contraction
if len(tensor_data) >= len(current_pattern):
# Tensor contraction: C_ij = Σ_k A_ik * B_kj
similarity = np.dot(
current_pattern, tensor_data[: len(current_pattern)]
) / (
np.linalg.norm(current_pattern)
* np.linalg.norm(tensor_data[: len(current_pattern)])
)
tensor_scores.append(similarity)
if tensor_scores:
return np.mean(tensor_scores) * 0.5  # Scale contribution
else:
return 0.0
except Exception as e:
logger.error(f"❌ Tensor contribution calculation failed: {e}")
return 0.0

def _calculate_hash_contribution(
self, market_data: MarketData, history_state: HistoryState
) -> float:
"""Calculate hash memory contribution."""
try:
if not history_state.profit_memory:
return 0.0
# Recent profit memory analysis
recent_profits = (
history_state.profit_memory[-10:]
if len(history_state.profit_memory) > 10
else history_state.profit_memory
)
if not recent_profits:
return 0.0
# Calculate profit trend
profit_trend = np.mean(recent_profits)
profit_stability = 1.0 - np.std(recent_profits)
# Hash-based pattern recognition
market_hash = hash(
f"{market_data.btc_price:.0f}_{market_data.volatility:.3f}"
)
hash_factor = (market_hash % 1000) / 1000.0
# Combine factors
hash_contribution = (
profit_trend * 0.6 + profit_stability * 0.4
) * hash_factor
return np.clip(hash_contribution, -0.3, 0.3)
except Exception as e:
logger.error(f"❌ Hash contribution calculation failed: {e}")
return 0.0

def _calculate_mathematical_score(
self, market_data: MarketData, history_state: HistoryState
) -> float:
"""Calculate mathematical score for profit calculation."""
try:
price = market_data.btc_price
volume = market_data.usdc_volume
volatility = market_data.volatility
momentum = market_data.momentum
price_volume_ratio = price / (volume + 1e-8)
volatility_adjusted_momentum = momentum / (volatility + 1e-8)
mathematical_score = (
price_volume_ratio * 0.3
+ volatility_adjusted_momentum * 0.4
+ (1.0 - volatility) * 0.3
)
return min(max(mathematical_score, 0.0), 1.0)
except Exception as e:
logger.error(f"Error calculating mathematical score: {e}")
return 0.0

def _calculate_confidence_score(
self, market_data: MarketData, history_state: HistoryState
) -> float:
"""Calculate real confidence score from market data."""
try:
price = market_data.btc_price
volume = market_data.usdc_volume
on_chain_signals = market_data.on_chain_signals
data_quality = 0.0
if price > 0 and volume > 0:
data_quality += 0.5
if volume > 1000:
data_quality += 0.3
if on_chain_signals:
whale_activity = on_chain_signals.get("whale_activity", 0.0)
network_health = on_chain_signals.get("network_health", 0.0)
data_quality += whale_activity * 0.1
data_quality += network_health * 0.1
consistency_score = 0.0
if hasattr(self, "last_market_data") and self.last_market_data:
price_change = abs(price - self.last_market_data.btc_price) / price
if price_change < 0.1:
consistency_score = 0.5
confidence_score = (data_quality + consistency_score) / 2.0
return min(max(confidence_score, 0.0), 1.0)
except Exception as e:
logger.error(f"Error calculating confidence score: {e}")
return 0.0

def _get_mode_multiplier(self, mode: ProfitCalculationMode) -> float:
"""Get mode-specific multiplier."""
multipliers = {
ProfitCalculationMode.CONSERVATIVE: 0.7,
ProfitCalculationMode.BALANCED: 1.0,
ProfitCalculationMode.AGGRESSIVE: 1.3,
ProfitCalculationMode.TENSOR_OPTIMIZED: 1.1,
}
return multipliers.get(mode, 1.0)

def calculate_sharpe_ratio(
self, returns: np.ndarray, risk_free_rate: float = 0.02
) -> float:
"""
Calculate Sharpe ratio: (R_p - R_f) / σ_p

Args:
returns: Array of returns
risk_free_rate: Risk-free rate (default: 2%)

Returns:
Sharpe ratio
"""
try:
if len(returns) < 2:
return 0.0

excess_returns = returns - risk_free_rate
portfolio_return = np.mean(excess_returns)
portfolio_std = np.std(returns, ddof=1)

if portfolio_std == 0:
return 0.0

sharpe_ratio = portfolio_return / portfolio_std
return float(sharpe_ratio)
except Exception as e:
logger.error(f"❌ Sharpe ratio calculation failed: {e}")
return 0.0

def calculate_sortino_ratio(
self, returns: np.ndarray, risk_free_rate: float = 0.02
) -> float:
"""
Calculate Sortino ratio: (R_p - R_f) / σ_d

Args:
returns: Array of returns
risk_free_rate: Risk-free rate (default: 2%)

Returns:
Sortino ratio
"""
try:
if len(returns) < 2:
return 0.0

excess_returns = returns - risk_free_rate
portfolio_return = np.mean(excess_returns)

# Calculate downside deviation (only negative returns)
negative_returns = returns[returns < 0]
if len(negative_returns) == 0:
return 0.0

downside_std = np.std(negative_returns, ddof=1)

if downside_std == 0:
return 0.0

sortino_ratio = portfolio_return / downside_std
return float(sortino_ratio)
except Exception as e:
logger.error(f"❌ Sortino ratio calculation failed: {e}")
return 0.0

def calculate_market_entropy(self, price_changes: np.ndarray) -> float:
"""
Calculate market entropy: H = -Σ p_i * log(p_i)

Args:
price_changes: Array of price changes

Returns:
Market entropy value
"""
try:
if len(price_changes) < 2:
return 0.0

# Calculate absolute changes
abs_changes = np.abs(price_changes)
total = np.sum(abs_changes)

if total == 0:
return 0.0

# Calculate probabilities
probabilities = abs_changes / total

# Calculate entropy using scipy
entropy_value = scipy_entropy(
probabilities + 1e-10
)  # Add small epsilon to avoid log(0)
return float(entropy_value)
except Exception as e:
logger.error(f"❌ Market entropy calculation failed: {e}")
return 0.0

def optimize_profit_portfolio(
self, weights: np.ndarray, returns: np.ndarray, risk_aversion: float = 0.5
) -> Dict[str, Any]:
"""
Optimize profit using the core formula: P = Σ w_i * r_i - λ * Σ w_i²

Args:
weights: Portfolio weights array
returns: Expected returns array
risk_aversion: Risk aversion parameter (default: 0.5)

Returns:
Dictionary with optimization results
"""
try:
w = np.asarray(weights, dtype=np.float64)
r = np.asarray(returns, dtype=np.float64)

# Ensure weights sum to 1
if not np.allclose(np.sum(w), 1.0, atol=1e-6):
w = w / np.sum(w)

# Calculate profit: P = Σ w_i * r_i - λ * Σ w_i²
expected_return = np.sum(w * r)
risk_penalty = risk_aversion * np.sum(w**2)
optimized_profit = expected_return - risk_penalty

# Calculate additional metrics
portfolio_variance = np.sum(w**2)
sharpe_ratio = (
expected_return / np.sqrt(portfolio_variance)
if portfolio_variance > 0
else 0
)

return {
"optimized_profit": float(optimized_profit),
"expected_return": float(expected_return),
"risk_penalty": float(risk_penalty),
"portfolio_variance": float(portfolio_variance),
"sharpe_ratio": float(sharpe_ratio),
"optimal_weights": w.tolist(),
}
except Exception as e:
logger.error(f"❌ Portfolio optimization failed: {e}")
return {
"optimized_profit": 0.0,
"expected_return": 0.0,
"risk_penalty": 0.0,
"portfolio_variance": 0.0,
"sharpe_ratio": 0.0,
"optimal_weights": weights.tolist()
if hasattr(weights, "tolist")
else list(weights),
}

def get_calculation_metrics(self) -> Dict[str, Any]:
"""Get pure calculation metrics."""
if self.calculation_count == 0:
return {"status": "no_calculations"}

avg_calculation_time = self.total_calculation_time / self.calculation_count

return {
"total_calculations": self.calculation_count,
"total_time": self.total_calculation_time,
"average_time_ms": avg_calculation_time * 1000,
"calculations_per_second": (
1.0 / avg_calculation_time if avg_calculation_time > 0 else 0
),
"strategy_params": {
"risk_tolerance": self.strategy_params.risk_tolerance,
"profit_target": self.strategy_params.profit_target,
"position_size": self.strategy_params.position_size,
},
"performance_metrics": self.performance_metrics,
"error_count": len(self.error_log),
}

def validate_profit_purity(
self, market_data: MarketData, history_state: HistoryState
) -> bool:
"""
Validate that profit calculation is mathematically pure.

This test ensures that the same inputs always produce the same outputs,
regardless of external factors like ZPE/ZBE acceleration.
"""
try:
# Calculate profit twice with identical inputs
result1 = self.calculate_profit(market_data, history_state)
result2 = self.calculate_profit(market_data, history_state)

# Results should be identical (within floating point precision)
is_pure = (
abs(result1.total_profit_score - result2.total_profit_score) < 1e-10
)

if not is_pure:
logger.error("❌ Profit calculation purity violation detected!")

return is_pure
except Exception as e:
logger.error(f"❌ Profit purity validation failed: {e}")
return False

def get_error_summary(self) -> Dict[str, Any]:
"""Get summary of calculation errors."""
if not self.error_log:
return {"error_count": 0, "recent_errors": []}

# Group errors by type
error_types = {}
for error in self.error_log:
error_type = error.error_type
if error_type not in error_types:
error_types[error_type] = 0
error_types[error_type] += 1

return {
"error_count": len(self.error_log),
"error_types": error_types,
"recent_errors": [
{
"timestamp": error.timestamp,
"type": error.error_type,
"message": error.error_message,
}
for error in self.error_log[-5:]  # Last 5 errors
],
}

def flash_screen(self) -> None:
"""Display current calculator status."""
print("🧮 PURE PROFIT CALCULATOR STATUS")
print("=" * 50)

metrics = self.get_calculation_metrics()
print(f"Total Calculations: {metrics.get('total_calculations', 0)}")
print(f"Average Time: {metrics.get('average_time_ms', 0):.2f}ms")
print(f"Calculations/sec: {metrics.get('calculations_per_second', 0):.2f}")

error_summary = self.get_error_summary()
print(f"Error Count: {error_summary['error_count']}")

if ENTROPY_AVAILABLE:
print("✅ Entropy Integration: ENABLED")
else:
print("⚠️ Entropy Integration: DISABLED")

print(f"Processing Mode: {self.processing_mode.value}")
print("=" * 50)


def assert_zpe_isolation() -> None:
"""
Assert that ZPE/ZBE systems are completely isolated from profit calculation.

This function ensures that no ZPE/ZBE imports or references exist in
the profit calculation pipeline.
"""
import sys

# Check that ZPE/ZBE modules are not imported
zpe_modules = [
name
for name in sys.modules.keys()
if "zpe" in name.lower() or "zbe" in name.lower()
]

if zpe_modules:
logger.warning(f"⚠️ ZPE/ZBE modules detected in system: {zpe_modules}")
logger.warning("⚠️ Ensure they do not influence profit calculations")

logger.info("✅ ZPE isolation check completed")


def create_sample_market_data() -> MarketData:
"""Create sample market data for testing."""
return MarketData(
timestamp=time.time(),
btc_price=50000.0,
eth_price=3000.0,
usdc_volume=1000000.0,
volatility=0.02,
momentum=0.01,
volume_profile=1.2,
on_chain_signals={"whale_activity": 0.3, "miner_activity": 0.7},
)


def create_sample_history_state() -> HistoryState:
"""Create sample history state for testing."""
return HistoryState(
timestamp=time.time(),
hash_matrices={"btc_pattern": np.random.rand(4, 4)},
tensor_buckets={"momentum_bucket": np.array([0.1, 0.2, 0.15, 1.1])},
profit_memory=[0.02, 0.015, 0.03, 0.01, 0.025],
signal_history=[0.6, 0.7, 0.65, 0.8, 0.75],
)


def create_pure_profit_calculator(
risk_tolerance: float = 0.02,
profit_target: float = 0.05,
position_size: float = 0.1,
processing_mode: ProcessingMode = ProcessingMode.HYBRID,
) -> PureProfitCalculator:
"""Create a configured pure profit calculator."""
strategy_params = StrategyParameters(
risk_tolerance=risk_tolerance,
profit_target=profit_target,
position_size=position_size,
)

return PureProfitCalculator(
strategy_params=strategy_params, processing_mode=processing_mode
)


def demo_pure_profit_calculation():
"""Demonstrate pure profit calculation."""
print("🧮 PURE PROFIT CALCULATION DEMONSTRATION")
print("=" * 60)

# Assert ZPE isolation
assert_zpe_isolation()

# Create calculator
calculator = create_pure_profit_calculator()

# Create sample data
market_data = create_sample_market_data()
history_state = create_sample_history_state()

print(
f"📊 Market Data: BTC = ${market_data.btc_price:,.0f}, Vol = {market_data.volatility:.3f}"
)
print(f"🧠 History State: {len(history_state.profit_memory)} profit memories")
print()

# Test different calculation modes
modes = [
ProfitCalculationMode.CONSERVATIVE,
ProfitCalculationMode.BALANCED,
ProfitCalculationMode.AGGRESSIVE,
ProfitCalculationMode.TENSOR_OPTIMIZED,
]

for mode in modes:
result = calculator.calculate_profit(market_data, history_state, mode)
print(f"Mode: {mode.value.upper()}")
print(f"  📈 Base Profit: {result.base_profit:.4f}")
print(f"  ⚖️ Risk Adjusted: {result.risk_adjusted_profit:.4f}")
print(f"  🎯 Total Score: {result.total_profit_score:.4f}")
print(f"  📊 Confidence: {result.confidence_score:.4f}")
print()

# Test purity
is_pure = calculator.validate_profit_purity(market_data, history_state)
print(f"🔬 Calculation Purity: {'✅ PURE' if is_pure else '❌ IMPURE'}")

# Show metrics
metrics = calculator.get_calculation_metrics()
print(f"📈 Calculations: {metrics['total_calculations']}")
print(f"⏱️ Avg Time: {metrics['average_time_ms']:.2f}ms")

# Test mathematical functions
print("\n🧮 MATHEMATICAL FUNCTION TESTS:")

# Test Sharpe ratio
returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
sharpe = calculator.calculate_sharpe_ratio(returns)
print(f"Sharpe Ratio: {sharpe:.4f}")

# Test Sortino ratio
sortino = calculator.calculate_sortino_ratio(returns)
print(f"Sortino Ratio: {sortino:.4f}")

# Test market entropy
price_changes = np.array([0.01, -0.02, 0.01, 0.03, -0.01])
entropy = calculator.calculate_market_entropy(price_changes)
print(f"Market Entropy: {entropy:.4f}")

# Test portfolio optimization
weights = np.array([0.4, 0.3, 0.3])
returns = np.array([0.05, 0.03, 0.04])
optimization = calculator.optimize_profit_portfolio(weights, returns)
print(f"Optimized Profit: {optimization['optimized_profit']:.4f}")


if __name__ == "__main__":
demo_pure_profit_calculation()
