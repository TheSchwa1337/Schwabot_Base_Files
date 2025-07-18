"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 ENTROPY PACKAGE - Entropy-Driven Market Dynamics & Decision Engine
====================================================================

This package provides entropy-driven market analysis and decision making:
- GalileoTensorField for entropy field calculations
- Entropy-based market entry/exit/hold decisions
- Tensor drift and oscillation analysis
- Quantum-inspired entropy operations

Core Components:
- GalileoTensorField: Entropy field calculations with GPU acceleration
- EntropyDecisionEngine: Market decisions based on entropy analysis
- EntropySignalProcessor: Processing of entropy-based signals
"""

import logging
import os
import yaml
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Import the main GalileoTensorField class
from .galileo_tensor_field import (
GalileoTensorField,
TensorFieldConfig,
EntropyMetrics
)

logger = logging.getLogger(__name__)

class EntropyDecision(Enum):
"""Class for Schwabot trading functionality."""
"""Entropy-based market decision types."""
ENTER_LOW_ENTROPY = "enter_low_entropy"      # Enter when entropy is low (predictable)
ENTER_HIGH_ENTROPY = "enter_high_entropy"    # Enter when entropy is high (volatile)
EXIT_ENTROPY_SPIKE = "exit_entropy_spike"    # Exit on entropy spike
HOLD_STABLE_ENTROPY = "hold_stable_entropy"  # Hold when entropy is stable
WAIT_ENTROPY_CALM = "wait_entropy_calm"     # Wait for entropy to calm
EMERGENCY_EXIT = "emergency_exit"           # Emergency exit on extreme entropy


class EntropyState(Enum):
"""Class for Schwabot trading functionality."""
"""Entropy state classifications."""
LOW_ENTROPY = "low_entropy"           # Low entropy (predictable market)
MEDIUM_ENTROPY = "medium_entropy"     # Medium entropy (normal volatility)
HIGH_ENTROPY = "high_entropy"         # High entropy (high volatility)
EXTREME_ENTROPY = "extreme_entropy"   # Extreme entropy (chaotic market)
ENTROPY_SPIKE = "entropy_spike"       # Sudden entropy increase
ENTROPY_CALM = "entropy_calm"         # Entropy returning to normal


@dataclass
class EntropySignal:
"""Class for Schwabot trading functionality."""
"""Entropy-based market signal."""
timestamp: float
price: float
volume: float
entropy_metrics: EntropyMetrics
entropy_state: EntropyState
decision: EntropyDecision
confidence: float
risk_level: float
drift_direction: str
oscillation_strength: float
metadata: Dict[str, Any]


@dataclass
class EntropySystemConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for entropy system operations."""
# GalileoTensorField parameters
field_dimension: int = 3
precision: float = 1e-8
max_iterations: int = 1000
convergence_threshold: float = 1e-6
use_gpu: bool = True
fallback_enabled: bool = True

# Entropy decision parameters
low_entropy_threshold: float = 0.3
high_entropy_threshold: float = 0.7
extreme_entropy_threshold: float = 0.9
entropy_spike_threshold: float = 0.2  # Change in entropy

# Market decision thresholds
enter_low_entropy_confidence: float = 0.7
enter_high_entropy_confidence: float = 0.6
exit_spike_confidence: float = 0.8
hold_stable_confidence: float = 0.5

# Risk management
max_risk_level: float = 0.8
min_confidence: float = 0.3
emergency_exit_threshold: float = 0.95


class EntropyDecisionEngine:
"""Class for Schwabot trading functionality."""
"""
Entropy-based decision engine for market analysis.

Uses GalileoTensorField to analyze market entropy and make
entry/exit/hold decisions based on entropy patterns.
"""

def __init__(self, config: Optional[EntropySystemConfig] = None) -> None:
"""Initialize the entropy decision engine."""
self.config = config or EntropySystemConfig()
self.logger = logging.getLogger(__name__)

# Initialize GalileoTensorField
tensor_config = TensorFieldConfig(
dimension=self.config.field_dimension,
precision=self.config.precision,
max_iterations=self.config.max_iterations,
convergence_threshold=self.config.convergence_threshold,
use_gpu=self.config.use_gpu,
fallback_enabled=self.config.fallback_enabled
)
self.galileo_field = GalileoTensorField(tensor_config)

# State tracking
self.signal_history: List[EntropySignal] = []
self.entropy_history: List[float] = []
self.decision_history: List[EntropyDecision] = []

self.logger.info("Entropy decision engine initialized with GalileoTensorField")

def analyze_market_entropy(self, price_data: np.ndarray, volume_data: np.ndarray, -> None
current_price: float, current_volume: float) -> EntropySignal:
"""
Analyze market entropy and make decisions.

Args:
price_data: Historical price data
volume_data: Historical volume data
current_price: Current market price
current_volume: Current market volume

Returns:
EntropySignal with decision and analysis
"""
try:
# Calculate entropy metrics
entropy_metrics = self.galileo_field.calculate_entropy_field(price_data, volume_data)

# Determine entropy state
entropy_state = self._classify_entropy_state(entropy_metrics.shannon_entropy)

# Calculate drift and oscillation
drift = self.galileo_field.calculate_tensor_drift(price_data)
oscillation = self.galileo_field.tensor_oscillation(price_data)

# Make entropy-based decision
decision = self._make_entropy_decision(
entropy_metrics, entropy_state, drift, oscillation
)

# Calculate confidence and risk
confidence = self._calculate_entropy_confidence(entropy_metrics, entropy_state)
risk_level = self._calculate_entropy_risk(entropy_metrics, entropy_state)

# Determine drift direction
drift_direction = self._determine_drift_direction(drift)

# Calculate oscillation strength
oscillation_strength = np.mean(np.abs(oscillation))

# Create entropy signal
signal = EntropySignal(
timestamp=0.0,  # Will be set by caller
price=current_price,
volume=current_volume,
entropy_metrics=entropy_metrics,
entropy_state=entropy_state,
decision=decision,
confidence=confidence,
risk_level=risk_level,
drift_direction=drift_direction,
oscillation_strength=oscillation_strength,
metadata={
'shannon_entropy': entropy_metrics.shannon_entropy,
'renyi_entropy': entropy_metrics.renyi_entropy,
'tsallis_entropy': entropy_metrics.tsallis_entropy,
'tensor_entropy': entropy_metrics.tensor_entropy,
'field_strength': entropy_metrics.field_strength,
'oscillation_frequency': entropy_metrics.oscillation_frequency,
'drift_coefficient': entropy_metrics.drift_coefficient
}
)

# Update history
self.signal_history.append(signal)
self.entropy_history.append(entropy_metrics.shannon_entropy)
self.decision_history.append(decision)

return signal

except Exception as e:
self.logger.error(f"Error analyzing market entropy: {e}")
# Return default signal
return EntropySignal(
timestamp=0.0,
price=current_price,
volume=current_volume,
entropy_metrics=EntropyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
entropy_state=EntropyState.MEDIUM_ENTROPY,
decision=EntropyDecision.WAIT_ENTROPY_CALM,
confidence=0.0,
risk_level=1.0,
drift_direction="unknown",
oscillation_strength=0.0,
metadata={'error': str(e)}
)

def _classify_entropy_state(self, shannon_entropy: float) -> EntropyState:
"""Classify entropy state based on Shannon entropy."""
if shannon_entropy < self.config.low_entropy_threshold:
return EntropyState.LOW_ENTROPY
elif shannon_entropy < self.config.high_entropy_threshold:
return EntropyState.MEDIUM_ENTROPY
elif shannon_entropy < self.config.extreme_entropy_threshold:
return EntropyState.HIGH_ENTROPY
else:
return EntropyState.EXTREME_ENTROPY

def _make_entropy_decision(self, entropy_metrics: EntropyMetrics, -> None
entropy_state: EntropyState,
drift: np.ndarray, oscillation: np.ndarray) -> EntropyDecision:
"""Make entropy-based market decision."""
shannon_entropy = entropy_metrics.shannon_entropy
field_strength = entropy_metrics.field_strength
drift_coefficient = entropy_metrics.drift_coefficient

# Check for entropy spike
if len(self.entropy_history) > 1:
entropy_change = abs(shannon_entropy - self.entropy_history[-1])
if entropy_change > self.config.entropy_spike_threshold:
return EntropyDecision.EXIT_ENTROPY_SPIKE

# Check for extreme entropy (emergency exit)
if entropy_state == EntropyState.EXTREME_ENTROPY:
return EntropyDecision.EMERGENCY_EXIT

# Low entropy decisions (predictable market)
if entropy_state == EntropyState.LOW_ENTROPY:
if field_strength > 0.5 and drift_coefficient > 0:
return EntropyDecision.ENTER_LOW_ENTROPY
else:
return EntropyDecision.HOLD_STABLE_ENTROPY

# High entropy decisions (volatile market)
elif entropy_state == EntropyState.HIGH_ENTROPY:
if field_strength < 0.3:  # Weak correlation suggests opportunity
return EntropyDecision.ENTER_HIGH_ENTROPY
else:
return EntropyDecision.WAIT_ENTROPY_CALM

# Medium entropy (normal market)
else:
if field_strength > 0.6:  # Strong correlation suggests stability
return EntropyDecision.HOLD_STABLE_ENTROPY
else:
return EntropyDecision.WAIT_ENTROPY_CALM

def _calculate_entropy_confidence(self, entropy_metrics: EntropyMetrics, -> None
entropy_state: EntropyState) -> float:
"""Calculate confidence based on entropy metrics."""
field_strength = entropy_metrics.field_strength
oscillation_frequency = entropy_metrics.oscillation_frequency

# Base confidence from field strength
confidence = field_strength

# Adjust based on entropy state
if entropy_state == EntropyState.LOW_ENTROPY:
confidence *= 1.2  # Higher confidence in predictable markets
elif entropy_state == EntropyState.HIGH_ENTROPY:
confidence *= 0.8  # Lower confidence in volatile markets
elif entropy_state == EntropyState.EXTREME_ENTROPY:
confidence *= 0.5  # Very low confidence in chaotic markets

# Adjust based on oscillation frequency (lower frequency = higher confidence)
if oscillation_frequency > 0:
confidence *= (1.0 / (1.0 + oscillation_frequency))

return max(0.0, min(1.0, confidence))

def _calculate_entropy_risk(self, entropy_metrics: EntropyMetrics, -> None
entropy_state: EntropyState) -> float:
"""Calculate risk level based on entropy metrics."""
shannon_entropy = entropy_metrics.shannon_entropy
drift_coefficient = entropy_metrics.drift_coefficient

# Base risk from entropy
risk = shannon_entropy

# Adjust based on entropy state
if entropy_state == EntropyState.EXTREME_ENTROPY:
risk *= 1.5
elif entropy_state == EntropyState.HIGH_ENTROPY:
risk *= 1.2
elif entropy_state == EntropyState.LOW_ENTROPY:
risk *= 0.8

# Adjust based on drift coefficient
if abs(drift_coefficient) > 0.5:
risk *= 1.1  # Higher risk with strong drift

return max(0.0, min(1.0, risk))

def _determine_drift_direction(self, drift: np.ndarray) -> str:
"""Determine drift direction from tensor drift."""
if len(drift) == 0:
return "unknown"

recent_drift = drift[-10:] if len(drift) >= 10 else drift
mean_drift = np.mean(recent_drift)

if mean_drift > 0.01:
return "upward"
elif mean_drift < -0.01:
return "downward"
else:
return "stable"

def get_system_status(self) -> Dict[str, Any]:
"""Get entropy system status."""
return {
'galileo_field_status': self.galileo_field.get_field_status(),
'signal_count': len(self.signal_history),
'recent_decisions': self.decision_history[-10:] if self.decision_history else [],
'entropy_trend': self.entropy_history[-10:] if self.entropy_history else [],
'config': {
'low_entropy_threshold': self.config.low_entropy_threshold,
'high_entropy_threshold': self.config.high_entropy_threshold,
'extreme_entropy_threshold': self.config.extreme_entropy_threshold,
'entropy_spike_threshold': self.config.entropy_spike_threshold
}
}


class EntropySystemFactory:
"""Class for Schwabot trading functionality."""
"""Factory for creating entropy system instances."""

@staticmethod
def create_from_config(config_path: Optional[str] = None) -> EntropyDecisionEngine:
"""Create entropy system from configuration file."""
config = EntropySystemFactory._load_config(config_path)
return EntropyDecisionEngine(config)

@staticmethod
def create_with_params(**kwargs) -> EntropyDecisionEngine:
"""Create entropy system with custom parameters."""
config = EntropySystemConfig(**kwargs)
return EntropyDecisionEngine(config)

@staticmethod
def _load_config(config_path: Optional[str] = None) -> EntropySystemConfig:
"""Load configuration from file."""
if config_path is None:
# Try to find default config
default_paths = [
"config/entropy_system_config.yaml",
"config/schwabot_config.yaml"
]

for path in default_paths:
if os.path.exists(path):
config_path = path
break

if config_path and os.path.exists(config_path):
try:
with open(config_path, 'r') as f:
config_data = yaml.safe_load(f)

# Extract entropy system config
entropy_config = config_data.get('entropy_system', {})
return EntropySystemConfig(**entropy_config)

except Exception as e:
logger.warning(f"Could not load entropy system config from {config_path}: {e}")

# Return default config
return EntropySystemConfig()


# Auto-load mathematical functions registry if available
ENTROPY_FUNCTIONS_REGISTRY = {}

try:
registry_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'mathematical_functions_registry.yaml')
if os.path.exists(registry_path):
with open(registry_path, 'r') as f:
registry_data = yaml.safe_load(f)
entropy_functions = registry_data.get('mathematical_functions', {}).get('entropy_system', {})
ENTROPY_FUNCTIONS_REGISTRY.update(entropy_functions)
except Exception as e:
logger.warning(f"Could not load entropy functions registry: {e}")


# Export main classes and functions
__all__ = [
"GalileoTensorField",
"TensorFieldConfig",
"EntropyMetrics",
"EntropyDecisionEngine",
"EntropySystemConfig",
"EntropySystemFactory",
"EntropyDecision",
"EntropyState",
"EntropySignal",
"ENTROPY_FUNCTIONS_REGISTRY"
]

# Convenience functions for quick access
def create_entropy_field(*args, **kwargs) -> GalileoTensorField:
"""Factory for GalileoTensorField (entropy field)."""
return GalileoTensorField(*args, **kwargs)

def create_entropy_system(*args, **kwargs) -> EntropyDecisionEngine:
"""Factory for EntropyDecisionEngine."""
return EntropyDecisionEngine(*args, **kwargs)

def analyze_market_entropy(price_data: np.ndarray, volume_data: np.ndarray,
current_price: float, current_volume: float,
config: Optional[EntropySystemConfig] = None) -> EntropySignal:
"""Quick function to analyze market entropy."""
entropy_system = EntropyDecisionEngine(config)
return entropy_system.analyze_market_entropy(price_data, volume_data, current_price, current_volume)