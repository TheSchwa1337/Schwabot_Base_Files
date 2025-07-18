"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomic Limit Layer Module
============================
Provides self-governing execution constraints for Schwabot trading system.

This module enforces internal limits and prevents overreach while
maintaining Schwabot's integrity through memory-based strategy validation
and profit tensor routing.

Features:
- Drawdown and cycle repetition guards
- Memory-based strategy validation
- Profit tensor routing logic
- Cross-chain fallback protection
- Thermal and entropy constraints
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up logger first
logger = logging.getLogger(__name__)

# Import existing infrastructure
try:
from core.profit_echo_cache import profit_echo_cache
from core.drift_band_profiler import drift_band_profiler
from core.thermal_strategy_router import thermal_strategy_router
INFRASTRUCTURE_AVAILABLE = True
except ImportError:
INFRASTRUCTURE_AVAILABLE = False
logger.warning("Some infrastructure not available")

class AutonomicLimitLayer:
"""Class for Schwabot trading functionality."""
"""
Autonomic limit layer for Schwabot trading system.

Enforces internal limits and prevents overreach while maintaining
Schwabot's integrity through memory-based strategy validation and
profit tensor routing.
"""


def __init__(self) -> None:
"""Initialize autonomic limit layer."""
self.logger = logging.getLogger(f"{__name__}.AutonomicLimitLayer")

# Initialize infrastructure components
self.echo = profit_echo_cache if 'profit_echo_cache' in globals() else None
self.profiler = drift_band_profiler if 'drift_band_profiler' in globals() else None
self.router = thermal_strategy_router if 'thermal_strategy_router' in globals() else None

# Limit thresholds
self.limits = {
"max_drawdown": 0.04,  # 4% maximum drawdown
"max_cycle_repeats": 5,  # Maximum consecutive cycles per tag
"min_profit_threshold": 0.01,  # 1% minimum profit
"max_thermal_pressure": 0.9,  # Maximum thermal pressure
"max_entropy_rate": 0.8,  # Maximum entropy rate
"max_memory_usage": 0.85,  # Maximum memory usage
"min_confidence": 0.6,  # Minimum strategy confidence
"max_volume_multiplier": 2.0,  # Maximum volume multiplier
}

# Strategy tracking
self.strategy_history = {}
self.cycle_counts = {}
self.last_executions = {}

# Performance tracking
self.total_validations = 0
self.blocked_strategies = 0
self.successful_executions = 0

self.logger.info("✅ Autonomic limit layer initialized")

def validate_strategy_execution(
self, tag: str, strategy_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
"""
Validate strategy execution against autonomic limits.

Args:
tag: Strategy tag identifier
strategy_data: Strategy execution data

Returns:
Tuple of (is_valid, reason, validation_data)
"""
try:
self.total_validations += 1

# Initialize validation data
validation_data = {
"tag": tag,
"timestamp": datetime.utcnow().isoformat(),
"checks_passed": [],
"checks_failed": [],
"warnings": [],
"recommendations": []
}

# Check 1: Drawdown limit
if not self._check_drawdown_limit(tag, strategy_data):
reason = "Drawdown limit exceeded"
validation_data["checks_failed"].append("drawdown_limit")
return False, reason, validation_data

# Check 2: Cycle repetition limit
if not self._check_cycle_repetition(tag):
reason = "Cycle repetition limit exceeded"
validation_data["checks_failed"].append("cycle_repetition")
return False, reason, validation_data

# Check 3: Profit threshold
if not self._check_profit_threshold(tag):
reason = "Profit threshold not met"
validation_data["checks_failed"].append("profit_threshold")
return False, reason, validation_data

# Check 4: Thermal pressure
if not self._check_thermal_pressure():
reason = "Thermal pressure too high"
validation_data["checks_failed"].append("thermal_pressure")
return False, reason, validation_data

# Check 5: Memory usage
if not self._check_memory_usage():
reason = "Memory usage too high"
validation_data["checks_failed"].append("memory_usage")
return False, reason, validation_data

# Check 6: Strategy confidence
if not self._check_strategy_confidence(tag):
reason = "Strategy confidence too low"
validation_data["checks_failed"].append("strategy_confidence")
return False, reason, validation_data

# Check 7: Volume multiplier
if not self._check_volume_multiplier(tag, strategy_data):
reason = "Volume multiplier too high"
validation_data["checks_failed"].append("volume_multiplier")
return False, reason, validation_data

# All checks passed
validation_data["checks_passed"] = [
"drawdown_limit", "cycle_repetition", "profit_threshold",
"thermal_pressure", "memory_usage", "strategy_confidence", "volume_multiplier"
]

self.successful_executions += 1
self.logger.info(f"✅ Strategy '{tag}' validated successfully")
return True, "All checks passed", validation_data

except Exception as e:
self.logger.error(f"❌ Error validating strategy '{tag}': {e}")
return False, f"Validation error: {str(e)}", {"error": str(e)}

def _check_drawdown_limit(self, tag: str, strategy_data: Dict[str, Any]) -> bool:
"""
Check if strategy execution would exceed drawdown limit.

Args:
tag: Strategy tag identifier
strategy_data: Strategy execution data

Returns:
True if drawdown limit is acceptable
"""
try:
if not self.echo:
return True  # No echo cache available, allow execution

# Get recent profits for tag
recent_profits = self.echo.get_recent_profits(tag)
if not recent_profits:
return True  # No history, allow execution

# Calculate potential drawdown
avg_profit = self.echo.average_profit(tag)
max_loss = strategy_data.get("max_loss", 0.02)  # Default 2% max loss

# Check if potential loss exceeds drawdown limit
if max_loss > self.limits["max_drawdown"]:
self.logger.warning(f"⚠️ Drawdown limit exceeded for '{tag}': {max_loss:.3f} > {self.limits['max_drawdown']:.3f}")
return False

return True

except Exception as e:
self.logger.error(f"❌ Error checking drawdown limit: {e}")
return True  # Allow execution on error

def _check_cycle_repetition(self, tag: str) -> bool:
"""
Check if strategy has exceeded cycle repetition limit.

Args:
tag: Strategy tag identifier

Returns:
True if cycle repetition is acceptable
"""
try:
current_time = time.time()

# Get current cycle count
cycle_count = self.cycle_counts.get(tag, 0)

# Check if we've exceeded the limit
if cycle_count >= self.limits["max_cycle_repeats"]:
self.logger.warning(f"⚠️ Cycle repetition limit exceeded for '{tag}': {cycle_count} cycles")
return False

# Update cycle count
self.cycle_counts[tag] = cycle_count + 1

return True

except Exception as e:
self.logger.error(f"❌ Error checking cycle repetition: {e}")
return True  # Allow execution on error

def _check_profit_threshold(self, tag: str) -> bool:
"""
Check if strategy meets minimum profit threshold.

Args:
tag: Strategy tag identifier

Returns:
True if profit threshold is met
"""
try:
if not self.echo:
return True  # No echo cache available, allow execution

# Get average profit for tag
avg_profit = self.echo.average_profit(tag)

# Check if average profit meets threshold
if avg_profit < self.limits["min_profit_threshold"]:
self.logger.warning(f"⚠️ Profit threshold not met for '{tag}': {avg_profit:.4f} < {self.limits['min_profit_threshold']:.4f}")
return False

return True

except Exception as e:
self.logger.error(f"❌ Error checking profit threshold: {e}")
return True  # Allow execution on error

def _check_thermal_pressure(self) -> bool:
"""
Check if thermal pressure is within acceptable limits.

Returns:
True if thermal pressure is acceptable
"""
try:
if not self.router:
return True  # No router available, allow execution

# Get thermal metrics
thermal_metrics = self.router.tee.get_metrics()

# Check thermal pressure
thermal_pressure = thermal_metrics.get("cpu_load", 0.5) + thermal_metrics.get("ram_usage", 0.5) / 2.0

if thermal_pressure > self.limits["max_thermal_pressure"]:
self.logger.warning(f"⚠️ Thermal pressure too high: {thermal_pressure:.3f} > {self.limits['max_thermal_pressure']:.3f}")
return False

return True

except Exception as e:
self.logger.error(f"❌ Error checking thermal pressure: {e}")
return True  # Allow execution on error

def _check_memory_usage(self) -> bool:
"""
Check if memory usage is within acceptable limits.

Returns:
True if memory usage is acceptable
"""
try:
import psutil

# Get memory usage
memory_usage = psutil.virtual_memory().percent / 100.0

if memory_usage > self.limits["max_memory_usage"]:
self.logger.warning(f"⚠️ Memory usage too high: {memory_usage:.3f} > {self.limits['max_memory_usage']:.3f}")
return False

return True

except Exception as e:
self.logger.error(f"❌ Error checking memory usage: {e}")
return True  # Allow execution on error

def _check_strategy_confidence(self, tag: str) -> bool:
"""
Check if strategy confidence meets minimum threshold.

Args:
tag: Strategy tag identifier

Returns:
True if strategy confidence is acceptable
"""
try:
if not self.echo:
return True  # No echo cache available, allow execution

# Get strategy confidence
confidence = self.echo.get_strategy_confidence(tag)

if confidence < self.limits["min_confidence"]:
self.logger.warning(f"⚠️ Strategy confidence too low for '{tag}': {confidence:.3f} < {self.limits['min_confidence']:.3f}")
return False

return True

except Exception as e:
self.logger.error(f"❌ Error checking strategy confidence: {e}")
return True  # Allow execution on error

def _check_volume_multiplier(self, tag: str, strategy_data: Dict[str, Any]) -> bool:
"""
Check if volume multiplier is within acceptable limits.

Args:
tag: Strategy tag identifier
strategy_data: Strategy execution data

Returns:
True if volume multiplier is acceptable
"""
try:
if not self.profiler:
return True  # No profiler available, allow execution

# Get volume multiplier from profiler
_, multiplier = self.profiler.evaluate_profit_band(tag)

if multiplier > self.limits["max_volume_multiplier"]:
self.logger.warning(f"⚠️ Volume multiplier too high for '{tag}': {multiplier:.2f} > {self.limits['max_volume_multiplier']:.2f}")
return False

return True

except Exception as e:
self.logger.error(f"❌ Error checking volume multiplier: {e}")
return True  # Allow execution on error

def recall_and_compare(self, tag: str) -> Dict[str, Any]:
"""
Recall and compare strategy performance with historical data.

Args:
tag: Strategy tag identifier

Returns:
Comparison analysis dictionary
"""
try:
if not self.echo:
return {"error": "No echo cache available"}

# Get recent profits
recent_profits = self.echo.get_recent_profits(tag)
if not recent_profits:
return {"error": "No profit history available"}

# Get profit trend
trend = self.echo.get_profit_trend(tag)

# Get strategy confidence
confidence = self.echo.get_strategy_confidence(tag)

# Calculate similarity with previous executions
similarity_score = self._calculate_similarity_score(tag, recent_profits)

comparison = {
"tag": tag,
"recent_profits": recent_profits,
"average_profit": self.echo.average_profit(tag),
"trend": trend,
"confidence": confidence,
"similarity_score": similarity_score,
"recommendation": self._generate_recommendation(trend, confidence, similarity_score),
"timestamp": datetime.utcnow().isoformat()
}

self.logger.debug(f"📊 Strategy comparison for '{tag}': confidence={confidence:.3f}, similarity={similarity_score:.3f}")
return comparison

except Exception as e:
self.logger.error(f"❌ Error recalling and comparing strategy '{tag}': {e}")
return {"error": str(e)}

def _calculate_similarity_score(self, tag: str, recent_profits: List[float]) -> float:
"""
Calculate similarity score with previous executions.

Args:
tag: Strategy tag identifier
recent_profits: Recent profit values

Returns:
Similarity score between 0.0 and 1.0
"""
try:
if len(recent_profits) < 2:
return 0.5  # Default similarity for insufficient data

# Calculate coefficient of variation (lower = more similar)
mean_profit = np.mean(recent_profits)
std_profit = np.std(recent_profits)

if mean_profit == 0:
return 0.5

cv = std_profit / abs(mean_profit)

# Convert to similarity score (lower CV = higher similarity)
similarity = max(0.0, min(1.0, 1.0 - cv))

return similarity

except Exception as e:
self.logger.error(f"❌ Error calculating similarity score: {e}")
return 0.5

def _generate_recommendation(self, trend: Dict[str, Any], confidence: float, similarity: float) -> str:
"""
Generate execution recommendation based on analysis.

Args:
trend: Profit trend analysis
confidence: Strategy confidence
similarity: Similarity score

Returns:
Recommendation string
"""
try:
# Base recommendation on trend direction
direction = trend.get("direction", "stable")

if direction == "improving" and confidence > 0.7 and similarity > 0.6:
return "EXECUTE_HIGH_CONFIDENCE"
elif direction == "stable" and confidence > 0.6:
return "EXECUTE_MODERATE_CONFIDENCE"
elif direction == "declining" or confidence < 0.5:
return "DEFER_LOW_CONFIDENCE"
else:
return "EXECUTE_WITH_CAUTION"

except Exception as e:
self.logger.error(f"❌ Error generating recommendation: {e}")
return "EXECUTE_WITH_CAUTION"

def process_strategy(self, tag: str) -> Dict[str, Any]:
"""
Process strategy through autonomic limit layer.

Args:
tag: Strategy tag identifier

Returns:
Processing result dictionary
"""
try:
# Create placeholder strategy data
strategy_data = {
"tag": tag,
"max_loss": 0.02,  # 2% max loss
"expected_profit": 0.015,  # 1.5% expected profit
"volume": 100.0,  # Base volume
"timestamp": datetime.utcnow().isoformat()
}

# Validate strategy execution
is_valid, reason, validation_data = self.validate_strategy_execution(tag, strategy_data)

if not is_valid:
self.blocked_strategies += 1
self.logger.warning(f"🚫 Strategy '{tag}' blocked: {reason}")
return {
"tag": tag,
"status": "BLOCKED",
"reason": reason,
"validation_data": validation_data,
"timestamp": datetime.utcnow().isoformat()
}

# Recall and compare with historical data
comparison = self.recall_and_compare(tag)

# Process strategy (placeholder for actual execution)
result = {
"tag": tag,
"status": "APPROVED",
"strategy_data": strategy_data,
"validation_data": validation_data,
"comparison": comparison,
"timestamp": datetime.utcnow().isoformat()
}

self.logger.info(f"✅ Strategy '{tag}' processed successfully")
return result

except Exception as e:
self.logger.error(f"❌ Error processing strategy '{tag}': {e}")
return {
"tag": tag,
"status": "ERROR",
"error": str(e),
"timestamp": datetime.utcnow().isoformat()
}

def get_layer_stats(self) -> Dict[str, Any]:
"""Get autonomic limit layer statistics."""
try:
stats = {
"total_validations": self.total_validations,
"blocked_strategies": self.blocked_strategies,
"successful_executions": self.successful_executions,
"block_rate": self.blocked_strategies / max(self.total_validations, 1),
"success_rate": self.successful_executions / max(self.total_validations, 1),
"limits": self.limits,
"infrastructure_available": INFRASTRUCTURE_AVAILABLE,
"components_available": {
"echo_cache": self.echo is not None,
"profiler": self.profiler is not None,
"router": self.router is not None
},
"timestamp": datetime.utcnow().isoformat()
}

return stats

except Exception as e:
self.logger.error(f"❌ Error getting layer stats: {e}")
return {"error": str(e)}

def reset_cycle_counts(self) -> None:
"""Reset cycle counts for all strategies."""
try:
self.cycle_counts.clear()
self.logger.info("🔄 Cycle counts reset")
except Exception as e:
self.logger.error(f"❌ Error resetting cycle counts: {e}")

def update_limits(self, new_limits: Dict[str, float]) -> None:
"""
Update autonomic limits.

Args:
new_limits: New limit values
"""
try:
for key, value in new_limits.items():
if key in self.limits:
self.limits[key] = value
self.logger.info(f"📝 Updated limit '{key}': {value}")

except Exception as e:
self.logger.error(f"❌ Error updating limits: {e}")


# Singleton instance for global access
autonomic_limit_layer = AutonomicLimitLayer()

if __name__ == "__main__":
# Test the autonomic limit layer
print("Testing Autonomic Limit Layer...")

# Test strategy processing
print(f"✅ Strategy processing: {result['status']}")

# Test recall and compare
comparison = autonomic_limit_layer.recall_and_compare("btc_usdc_snipe")
print(f"✅ Strategy comparison: {comparison}")

# Show layer stats
stats = autonomic_limit_layer.get_layer_stats()
print(f"📊 Layer stats: {stats}")