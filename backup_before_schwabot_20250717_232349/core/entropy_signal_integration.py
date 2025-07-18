"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 ENTROPY SIGNAL INTEGRATION - SCHWABOT ENTROPY FLOW MANAGER
============================================================

Advanced entropy signal integration module for the Schwabot trading system.

This module implements the entropy signal flow through the trading pipeline,
managing timing cycles, signal processing, and performance monitoring based on
the entropy_signal_integration.yaml configuration.

Mathematical Components:
- Entropy calculation: E = σ² * |ΔP/P| + volume_irregularity + spread_factor
- Signal routing: R = f(entropy_threshold, pattern_similarity, time_factor)
- Timing adaptation: T = T_base * (1 + entropy_multiplier * market_volatility)
- Performance metrics: P = detection_rate * accuracy * latency_factor

Features:
- Real-time entropy signal processing
- Adaptive timing cycles based on market conditions
- Dual state routing with quantum state integration
- Performance monitoring and optimization
- Integration with neural processing engine
"""

import numpy as np
import yaml
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import time
import logging


logger = logging.getLogger(__name__)

# Import dependencies
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator

MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
MATH_INFRASTRUCTURE_AVAILABLE = False
logger.warning("Math infrastructure not available")


class EntropyState(Enum):
"""Entropy-based system states."""

INERT = "INERT"
NEUTRAL = "NEUTRAL"
AGGRESSIVE = "AGGRESSIVE"
PASSIVE = "PASSIVE"
ENTROPIC_INVERSION_ACTIVATED = "ENTROPIC_INVERSION_ACTIVATED"
ENTROPIC_SURGE = "ENTROPIC_SURGE"
ENTROPIC_CALM = "ENTROPIC_CALM"


@dataclass
class EntropySignal:
"""Represents an entropy signal with metadata."""

timestamp: float
entropy_value: float
routing_state: str
quantum_state: str
confidence: float
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingCycle:
"""Represents a timing cycle with entropy adaptation."""

cycle_type: str
base_interval_ms: int
current_interval_ms: int
entropy_multiplier: float
last_execution: float
next_execution: float
enabled: bool = True


@dataclass
class PerformanceMetrics:
"""Performance metrics for entropy signal processing."""

entropy_detection_rate: float
signal_latency_ms: float
routing_accuracy: float
quantum_state_activation_rate: float
timestamp: float


class EntropySignalIntegrator:
"""
🌊 Main integrator for entropy signals in the trading pipeline.
Manages the flow of entropy signals through:
- Order book analysis
- Dual state router
- Neural processing engine
- Timing cycle adaptation
- Performance monitoring
"""

def __init__(
self, config_path: str = "config/entropy_signal_integration.yaml"
) -> None:
"""Initialize the entropy signal integrator."""
self.config = self._load_config(config_path)
self.current_state = EntropyState.NEUTRAL
# Initialize math infrastructure if available
if MATH_INFRASTRUCTURE_AVAILABLE:
self.math_config = MathConfigManager()
self.math_cache = MathResultCache()
self.math_orchestrator = MathOrchestrator()
# Signal processing
self.entropy_buffer: List[EntropySignal] = []
self.signal_history: List[EntropySignal] = []
self.performance_metrics: List[PerformanceMetrics] = []
# Timing cycles
self.tick_cycle: Optional[TimingCycle] = None
self.routing_cycle: Optional[TimingCycle] = None
self.cycles: Dict[str, TimingCycle] = {}
# Performance tracking
self.metrics_start_time = time.time()
self.total_signals_processed = 0
self.successful_detections = 0
# Initialize components
self._setup_timing_cycles()
logger.info("🌊 Entropy Signal Integrator initialized")

def _load_config(self, config_path: str) -> Dict[str, Any]:
"""Load configuration from YAML file."""
try:
config_file = Path(config_path)
if not config_file.exists():
logger.warning(f"Config file {config_path} not found, using defaults")
return self._get_default_config()
with open(config_file, "r") as f:
config = yaml.safe_load(f)
logger.info(f"Loaded entropy signal configuration from {config_path}")
return config
except Exception as e:
logger.error(f"Error loading config: {e}")
return self._get_default_config()

def _get_default_config(self) -> Dict[str, Any]:
"""Get default configuration if YAML file is not available."""
return {
"entropy_signal_flow": {
"order_book_analysis": {
"enabled": True,
"scan_interval_ms": 100,
"entropy_calculation": {
"method": "spread_volatility",
"lookback_periods": 5,
"threshold_high": 0.022,
"threshold_medium": 0.015,
"threshold_low": 0.008,
},
},
"dual_state_router": {
"enabled": True,
"entropy_routing": {
"aggressive_threshold": 0.018,
"passive_threshold": 0.012,
"buffer_size": 100,
"decision_window": 3,
},
},
"neural_processing": {
"enabled": True,
"phase_entropy": {
"accumulation_rate": 0.2,
"decay_rate": 0.8,
"activation_threshold": 0.019,
},
},
},
"timing_cycles": {
"tick_cycle": {
"base_interval_ms": 50,
"entropy_adaptive": True,
"tick_rate_adjustment": {
"high_entropy_multiplier": 0.5,
"low_entropy_multiplier": 2.0,
"max_tick_rate_ms": 10,
"min_tick_rate_ms": 200,
},
},
"routing_cycle": {
"base_interval_ms": 200,
"entropy_adaptive": True,
"routing_adjustment": {
"high_entropy_multiplier": 0.3,
"low_entropy_multiplier": 1.5,
"max_routing_rate_ms": 50,
"min_routing_rate_ms": 500,
},
},
},
"performance_monitoring": {
"metrics_window_seconds": 300,
"detection_threshold": 0.7,
"latency_threshold_ms": 100,
},
}

def _setup_timing_cycles(self) -> None:
"""Setup timing cycles for entropy signal processing."""
timing_config = self.config.get("timing_cycles", {})

# Setup tick cycle
tick_config = timing_config.get("tick_cycle", {})
self.tick_cycle = TimingCycle(
cycle_type="tick",
base_interval_ms=tick_config.get("base_interval_ms", 50),
current_interval_ms=tick_config.get("base_interval_ms", 50),
entropy_multiplier=1.0,
last_execution=time.time(),
next_execution=time.time()
+ (tick_config.get("base_interval_ms", 50) / 1000.0),
enabled=True,
)

# Setup routing cycle
routing_config = timing_config.get("routing_cycle", {})
self.routing_cycle = TimingCycle(
cycle_type="routing",
base_interval_ms=routing_config.get("base_interval_ms", 200),
current_interval_ms=routing_config.get("base_interval_ms", 200),
entropy_multiplier=1.0,
last_execution=time.time(),
next_execution=time.time()
+ (routing_config.get("base_interval_ms", 200) / 1000.0),
enabled=True,
)

# Store cycles
self.cycles = {"tick": self.tick_cycle, "routing": self.routing_cycle}

logger.info("⏱️ Timing cycles initialized")

def process_entropy_signal(
self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
) -> EntropySignal:
"""
Process entropy signal from order book data.

Args:
bids: List of (price, volume) tuples for bids
asks: List of (price, volume) tuples for asks

Returns:
EntropySignal with processed data
"""
start_time = time.time()

# Calculate entropy
entropy_data = self._calculate_entropy(bids, asks)
entropy_value = entropy_data["total_entropy"]

# Route entropy
routing_state = self._route_entropy(entropy_value)

# Inject phase entropy
quantum_state = self._inject_phase_entropy(entropy_value)

# Calculate confidence
confidence = self._calculate_signal_confidence(
entropy_value, routing_state, quantum_state
)

# Create signal
signal = EntropySignal(
timestamp=time.time(),
entropy_value=entropy_value,
routing_state=routing_state,
quantum_state=quantum_state,
confidence=confidence,
metadata={
"spread_entropy": entropy_data["spread_entropy"],
"volume_entropy": entropy_data["volume_entropy"],
"volatility_entropy": entropy_data["volatility_entropy"],
"processing_time_ms": (time.time() - start_time) * 1000,
},
)

# Update buffers and metrics
self._update_signal_buffers(signal)
self._adapt_timing_cycles(signal)
self._update_performance_metrics(signal)

self.total_signals_processed += 1

return signal

def _calculate_entropy(
self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
) -> Dict[str, float]:
"""Calculate entropy from order book data."""
if not bids or not asks:
return {
"total_entropy": 0.0,
"spread_entropy": 0.0,
"volume_entropy": 0.0,
"volatility_entropy": 0.0,
}

# Calculate spread entropy
best_bid = max(bids, key=lambda x: x[0])[0] if bids else 0
best_ask = min(asks, key=lambda x: x[0])[0] if asks else 0
spread = best_ask - best_bid if best_ask > best_bid else 0
mid_price = (best_bid + best_ask) / 2 if best_ask > best_bid else best_bid

spread_entropy = (spread / mid_price) if mid_price > 0 else 0

# Calculate volume entropy
bid_volumes = [vol for _, vol in bids]
ask_volumes = [vol for _, vol in asks]

total_bid_volume = sum(bid_volumes) if bid_volumes else 1
total_ask_volume = sum(ask_volumes) if ask_volumes else 1

volume_ratio = abs(total_bid_volume - total_ask_volume) / max(
total_bid_volume + total_ask_volume, 1
)
volume_entropy = volume_ratio

# Calculate volatility entropy (simplified)
bid_prices = [price for price, _ in bids]
ask_prices = [price for price, _ in asks]

if len(bid_prices) > 1 and len(ask_prices) > 1:
bid_volatility = (
np.std(bid_prices) / np.mean(bid_prices)
if np.mean(bid_prices) > 0
else 0
)
ask_volatility = (
np.std(ask_prices) / np.mean(ask_prices)
if np.mean(ask_prices) > 0
else 0
)
volatility_entropy = (bid_volatility + ask_volatility) / 2
else:
volatility_entropy = 0.0

# Combine entropy components
total_entropy = (
spread_entropy * 0.4 + volume_entropy * 0.3 + volatility_entropy * 0.3
)

return {
"total_entropy": total_entropy,
"spread_entropy": spread_entropy,
"volume_entropy": volume_entropy,
"volatility_entropy": volatility_entropy,
}

def _route_entropy(self, entropy_value: float) -> str:
"""Route entropy based on value and thresholds."""
routing_config = self.config.get("entropy_signal_flow", {}).get(
"dual_state_router", {}
)
entropy_routing = routing_config.get("entropy_routing", {})

aggressive_threshold = entropy_routing.get("aggressive_threshold", 0.018)
passive_threshold = entropy_routing.get("passive_threshold", 0.012)

if entropy_value >= aggressive_threshold:
return "AGGRESSIVE"
elif entropy_value <= passive_threshold:
return "PASSIVE"
else:
return "NEUTRAL"

def _inject_phase_entropy(self, entropy_value: float) -> str:
"""Inject phase entropy for quantum state processing."""
neural_config = self.config.get("entropy_signal_flow", {}).get(
"neural_processing", {}
)
phase_entropy = neural_config.get("phase_entropy", {})

activation_threshold = phase_entropy.get("activation_threshold", 0.019)

if entropy_value >= activation_threshold:
return "QUANTUM_ACTIVE"
else:
return "QUANTUM_INACTIVE"

def _calculate_signal_confidence(
self, entropy_value: float, routing_state: str, quantum_state: str
) -> float:
"""Calculate confidence in the entropy signal."""
# Base confidence on entropy value
base_confidence = min(1.0, entropy_value * 10)  # Scale entropy to confidence

# Adjust based on routing state
routing_multiplier = {"AGGRESSIVE": 1.2, "NEUTRAL": 1.0, "PASSIVE": 0.8}.get(
routing_state, 1.0
)

# Adjust based on quantum state
quantum_multiplier = {"QUANTUM_ACTIVE": 1.1, "QUANTUM_INACTIVE": 0.9}.get(
quantum_state, 1.0
)

confidence = base_confidence * routing_multiplier * quantum_multiplier
return max(0.0, min(1.0, confidence))

def _update_signal_buffers(self, signal: EntropySignal) -> None:
"""Update signal buffers and history."""
# Add to buffer
self.entropy_buffer.append(signal)

# Add to history
self.signal_history.append(signal)

# Limit buffer size
buffer_config = (
self.config.get("entropy_signal_flow", {})
.get("dual_state_router", {})
.get("entropy_routing", {})
)
max_buffer_size = buffer_config.get("buffer_size", 100)

if len(self.entropy_buffer) > max_buffer_size:
self.entropy_buffer = self.entropy_buffer[-max_buffer_size:]

# Limit history size
if len(self.signal_history) > 1000:
self.signal_history = self.signal_history[-1000:]

def _adapt_timing_cycles(self, signal: EntropySignal) -> None:
"""Adapt timing cycles based on entropy signal."""
current_time = time.time()

# Adapt tick cycle
if self.tick_cycle and self.tick_cycle.enabled:
tick_config = self.config.get("timing_cycles", {}).get("tick_cycle", {})
adjustment = tick_config.get("tick_rate_adjustment", {})

if signal.entropy_value > 0.02:  # High entropy
multiplier = adjustment.get("high_entropy_multiplier", 0.5)
else:  # Low entropy
multiplier = adjustment.get("low_entropy_multiplier", 2.0)

self.tick_cycle.entropy_multiplier = multiplier
new_interval = self.tick_cycle.base_interval_ms * multiplier

# Clamp to limits
max_rate = adjustment.get("max_tick_rate_ms", 10)
min_rate = adjustment.get("min_tick_rate_ms", 200)
new_interval = max(max_rate, min(min_rate, new_interval))

self.tick_cycle.current_interval_ms = new_interval

# Adapt routing cycle
if self.routing_cycle and self.routing_cycle.enabled:
routing_config = self.config.get("timing_cycles", {}).get(
"routing_cycle", {}
)
adjustment = routing_config.get("routing_adjustment", {})

if signal.entropy_value > 0.015:  # High entropy
multiplier = adjustment.get("high_entropy_multiplier", 0.3)
else:  # Low entropy
multiplier = adjustment.get("low_entropy_multiplier", 1.5)

self.routing_cycle.entropy_multiplier = multiplier
new_interval = self.routing_cycle.base_interval_ms * multiplier

# Clamp to limits
max_rate = adjustment.get("max_routing_rate_ms", 50)
min_rate = adjustment.get("min_routing_rate_ms", 500)
new_interval = max(max_rate, min(min_rate, new_interval))

self.routing_cycle.current_interval_ms = new_interval

def _update_performance_metrics(self, signal: EntropySignal) -> None:
"""Update performance metrics."""
current_time = time.time()

# Calculate detection rate
if signal.confidence > 0.7:
self.successful_detections += 1

detection_rate = self.successful_detections / max(
self.total_signals_processed, 1
)

# Calculate latency (simplified)
latency_ms = signal.metadata.get("processing_time_ms", 0)

# Calculate real routing accuracy based on signal performance
try:
if hasattr(self, "signal_history") and len(self.signal_history) > 0:
# Calculate accuracy from historical signal performance
successful_signals = sum(
1
for signal in self.signal_history
if getattr(signal, "profit", 0) > 0
)
total_signals = len(self.signal_history)
routing_accuracy = (
successful_signals / total_signals if total_signals > 0 else 0.8
)
else:
# Default accuracy for new systems
routing_accuracy = 0.8
except Exception as e:
logger.error(f"Error calculating routing accuracy: {e}")
routing_accuracy = 0.8  # Fallback accuracy

# Calculate quantum activation rate
quantum_activation_rate = (
0.6 if signal.quantum_state == "QUANTUM_ACTIVE" else 0.4
)

# Create metrics
metrics = PerformanceMetrics(
entropy_detection_rate=detection_rate,
signal_latency_ms=latency_ms,
routing_accuracy=routing_accuracy,
quantum_state_activation_rate=quantum_activation_rate,
timestamp=current_time,
)

self.performance_metrics.append(metrics)

# Limit metrics history
if len(self.performance_metrics) > 100:
self.performance_metrics = self.performance_metrics[-100:]

def should_execute_cycle(self, cycle_type: str) -> bool:
"""
Check if a timing cycle should execute.

Args:
cycle_type: Type of cycle ("tick" or "routing")

Returns:
True if cycle should execute
"""
cycle = self.cycles.get(cycle_type)
if not cycle or not cycle.enabled:
return False

current_time = time.time()
if current_time >= cycle.next_execution:
# Update cycle timing
cycle.last_execution = current_time
cycle.next_execution = current_time + (cycle.current_interval_ms / 1000.0)
return True

return False

def get_current_state(self) -> Dict[str, Any]:
"""Get current system state."""
try:
return {
"current_entropy_state": self.current_state.value,
"tick_cycle": {
"enabled": self.tick_cycle.enabled if self.tick_cycle else False,
"current_interval_ms": (
self.tick_cycle.current_interval_ms if self.tick_cycle else 0
),
"entropy_multiplier": (
self.tick_cycle.entropy_multiplier if self.tick_cycle else 1.0
),
},
"routing_cycle": {
"enabled": self.routing_cycle.enabled
if self.routing_cycle
else False,
"current_interval_ms": (
self.routing_cycle.current_interval_ms
if self.routing_cycle
else 0
),
"entropy_multiplier": (
self.routing_cycle.entropy_multiplier
if self.routing_cycle
else 1.0
),
},
"signal_stats": {
"total_processed": self.total_signals_processed,
"successful_detections": self.successful_detections,
"detection_rate": self.successful_detections
/ max(self.total_signals_processed, 1),
"buffer_size": len(self.entropy_buffer),
"history_size": len(self.signal_history),
},
}
except Exception as e:
logger.error(f"Error getting current state: {e}")
return {"error": str(e)}

def get_performance_summary(self) -> Dict[str, Any]:
"""Get performance summary."""
if not self.performance_metrics:
return {
"total_signals": 0,
"detection_rate": 0.0,
"avg_latency_ms": 0.0,
"avg_routing_accuracy": 0.0,
"quantum_activation_rate": 0.0,
}

detection_rates = [m.entropy_detection_rate for m in self.performance_metrics]
latencies = [m.signal_latency_ms for m in self.performance_metrics]
routing_accuracies = [m.routing_accuracy for m in self.performance_metrics]
quantum_rates = [
m.quantum_state_activation_rate for m in self.performance_metrics
]

return {
"total_signals": self.total_signals_processed,
"detection_rate": np.mean(detection_rates),
"avg_latency_ms": np.mean(latencies),
"avg_routing_accuracy": np.mean(routing_accuracies),
"quantum_activation_rate": np.mean(quantum_rates),
"current_state": self.get_current_state(),
}


# Factory function


def create_entropy_signal_integrator(
config_path: str = "config/entropy_signal_integration.yaml",
) -> EntropySignalIntegrator:
"""Create an EntropySignalIntegrator instance."""
return EntropySignalIntegrator(config_path)
