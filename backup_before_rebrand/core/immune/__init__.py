"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 IMMUNE SYSTEM PACKAGE - Quantum Signal Collapse & Market Decision Engine
==========================================================================

This package provides quantum-inspired immune system functionality for market decision making:
- QSC (Quantum Symbolic Collapse) gates for signal validation
- Market entry/exit/hold decision logic
- Immune system response to market anomalies
- Quantum state analysis for trading signals

Core Components:
- QSCGate: Quantum signal collapse and phase gate activation
- ImmuneSystem: Market immune response system
- MarketDecisionEngine: Entry/exit/hold decision logic
"""

import logging
import os
import yaml
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Import the main QSCGate class
from .qsc_gate import QSCGate, CollapseState, GateType, QuantumState, CollapseResult

logger = logging.getLogger(__name__)

class MarketDecision(Enum):
"""Class for Schwabot trading functionality."""
"""Market decision types."""
ENTER_LONG = "enter_long"      # Enter long position
ENTER_SHORT = "enter_short"    # Enter short position
EXIT_LONG = "exit_long"        # Exit long position
EXIT_SHORT = "exit_short"      # Exit short position
HOLD = "hold"                  # Hold current position
WAIT = "wait"                  # Wait for better conditions


class ImmuneResponse(Enum):
"""Class for Schwabot trading functionality."""
"""Immune system response types."""
NORMAL = "normal"              # Normal market conditions
ANOMALY_DETECTED = "anomaly"   # Market anomaly detected
STRESS_RESPONSE = "stress"     # Stress response activated
RECOVERY = "recovery"          # Recovery phase
IMMUNE_BOOST = "boost"         # Immune system boost


@dataclass
class MarketSignal:
"""Class for Schwabot trading functionality."""
"""Market signal with immune system analysis."""
timestamp: float
price: float
volume: float
qsc_collapse_state: CollapseState
quantum_confidence: float
immune_response: ImmuneResponse
market_decision: MarketDecision
signal_strength: float
risk_score: float
metadata: Dict[str, Any]


@dataclass
class ImmuneSystemConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for immune system operations."""
# QSC Gate parameters
collapse_threshold: float = 0.5
phase_resolution: int = 100
decoherence_rate: float = 0.1
measurement_strength: float = 1.0

# Market decision parameters
entry_confidence_threshold: float = 0.7
exit_confidence_threshold: float = 0.6
hold_confidence_threshold: float = 0.5
anomaly_threshold: float = 0.8

# Risk management
max_risk_score: float = 0.8
min_signal_strength: float = 0.3
immune_boost_threshold: float = 0.9


class ImmuneSystem:
"""Class for Schwabot trading functionality."""
"""
Immune System for market decision making.

Uses QSC gates to analyze market signals and make entry/exit/hold decisions
based on quantum state analysis and immune response patterns.
"""

def __init__(self, config: Optional[ImmuneSystemConfig] = None) -> None:
"""Initialize the immune system."""
self.config = config or ImmuneSystemConfig()
self.logger = logging.getLogger(__name__)

# Initialize QSC gate
qsc_config = {
'collapse_threshold': self.config.collapse_threshold,
'phase_resolution': self.config.phase_resolution,
'decoherence_rate': self.config.decoherence_rate,
'measurement_strength': self.config.measurement_strength
}
self.qsc_gate = QSCGate(qsc_config)

# State tracking
self.signal_history: List[MarketSignal] = []
self.immune_responses: List[ImmuneResponse] = []
self.decision_history: List[MarketDecision] = []

self.logger.info("Immune system initialized with QSC gate")

def analyze_market_signal(self, price: float, volume: float, -> None
additional_data: Optional[Dict[str, Any]] = None) -> MarketSignal:
"""
Analyze market signal using QSC gate and immune system.

Args:
price: Current market price
volume: Current market volume
additional_data: Additional market data

Returns:
MarketSignal with decision and analysis
"""
try:
# Create quantum state from market data
market_state = self._create_market_quantum_state(price, volume, additional_data)

# Process through QSC gate
collapse_result = self.qsc_gate.process_ai_signal(
str(price), str(volume)
)

# Determine immune response
immune_response = self._determine_immune_response(collapse_result)

# Make market decision
market_decision = self._make_market_decision(
collapse_result, immune_response, price, volume
)

# Calculate signal strength and risk
signal_strength = self._calculate_signal_strength(collapse_result)
risk_score = self._calculate_risk_score(collapse_result, immune_response)

# Create market signal
signal = MarketSignal(
timestamp=collapse_result.timestamp,
price=price,
volume=volume,
qsc_collapse_state=collapse_result.collapse_state,
quantum_confidence=collapse_result.confidence,
immune_response=immune_response,
market_decision=market_decision,
signal_strength=signal_strength,
risk_score=risk_score,
metadata=collapse_result.metadata
)

# Update history
self.signal_history.append(signal)
self.immune_responses.append(immune_response)
self.decision_history.append(market_decision)

return signal

except Exception as e:
self.logger.error(f"Error analyzing market signal: {e}")
# Return default signal
return MarketSignal(
timestamp=0.0,
price=price,
volume=volume,
qsc_collapse_state=CollapseState.DECOHERENT,
quantum_confidence=0.0,
immune_response=ImmuneResponse.NORMAL,
market_decision=MarketDecision.WAIT,
signal_strength=0.0,
risk_score=1.0,
metadata={'error': str(e)}
)

def _create_market_quantum_state(self, price: float, volume: float, -> None
additional_data: Optional[Dict[str, Any]]) -> QuantumState:
"""Create quantum state from market data."""
# Normalize price and volume for quantum state
normalized_price = (price - 10000) / 100000  # Rough normalization
normalized_volume = volume / 1000000  # Rough normalization

# Create state vector
state_vector = np.array([normalized_price, normalized_volume])

# Create density matrix
density_matrix = np.outer(state_vector, state_vector.conj())

return QuantumState(
timestamp=0.0,
state_vector=state_vector,
density_matrix=density_matrix,
collapse_state=CollapseState.SUPERPOSITION,
phase=0.0,
amplitude=1.0
)

def _determine_immune_response(self, collapse_result: CollapseResult) -> ImmuneResponse:
"""Determine immune response based on collapse result."""
confidence = collapse_result.confidence
entropy = collapse_result.entropy

if confidence > self.config.immune_boost_threshold:
return ImmuneResponse.IMMUNE_BOOST
elif entropy > self.config.anomaly_threshold:
return ImmuneResponse.ANOMALY_DETECTED
elif confidence < self.config.entry_confidence_threshold:
return ImmuneResponse.STRESS_RESPONSE
elif len(self.immune_responses) > 0 and self.immune_responses[-1] == ImmuneResponse.STRESS_RESPONSE:
return ImmuneResponse.RECOVERY
else:
return ImmuneResponse.NORMAL

def _make_market_decision(self, collapse_result: CollapseResult, -> None
immune_response: ImmuneResponse,
price: float, volume: float) -> MarketDecision:
"""Make market decision based on QSC analysis and immune response."""
confidence = collapse_result.confidence
entropy = collapse_result.entropy

# Check for high-risk conditions
if entropy > self.config.anomaly_threshold:
return MarketDecision.WAIT

# Check for strong entry signals
if (confidence > self.config.entry_confidence_threshold and
immune_response in [ImmuneResponse.NORMAL, ImmuneResponse.IMMUNE_BOOST]):

# Determine long vs short based on quantum state
if collapse_result.phase_gate_output[0] > 0:
return MarketDecision.ENTER_LONG
else:
return MarketDecision.ENTER_SHORT

# Check for exit signals
elif confidence > self.config.exit_confidence_threshold:
if collapse_result.phase_gate_output[0] > 0:
return MarketDecision.EXIT_SHORT
else:
return MarketDecision.EXIT_LONG

# Check for hold signals
elif confidence > self.config.hold_confidence_threshold:
return MarketDecision.HOLD

# Default to wait
return MarketDecision.WAIT

def _calculate_signal_strength(self, collapse_result: CollapseResult) -> float:
"""Calculate signal strength from collapse result."""
confidence = collapse_result.confidence
entropy = collapse_result.entropy

# Signal strength increases with confidence and decreases with entropy
signal_strength = confidence * (1.0 - entropy)
return max(0.0, min(1.0, signal_strength))

def _calculate_risk_score(self, collapse_result: CollapseResult, -> None
immune_response: ImmuneResponse) -> float:
"""Calculate risk score from collapse result and immune response."""
entropy = collapse_result.entropy
confidence = collapse_result.confidence

# Base risk from entropy
base_risk = entropy

# Adjust based on immune response
if immune_response == ImmuneResponse.ANOMALY_DETECTED:
base_risk *= 1.5
elif immune_response == ImmuneResponse.STRESS_RESPONSE:
base_risk *= 1.2
elif immune_response == ImmuneResponse.IMMUNE_BOOST:
base_risk *= 0.8

# Adjust based on confidence
risk_score = base_risk * (1.0 - confidence)

return max(0.0, min(1.0, risk_score))

def get_system_status(self) -> Dict[str, Any]:
"""Get immune system status."""
return {
'qsc_gate_status': self.qsc_gate.get_gate_summary(),
'signal_count': len(self.signal_history),
'recent_decisions': self.decision_history[-10:] if self.decision_history else [],
'recent_responses': self.immune_responses[-10:] if self.immune_responses else [],
'config': {
'entry_threshold': self.config.entry_confidence_threshold,
'exit_threshold': self.config.exit_confidence_threshold,
'hold_threshold': self.config.hold_confidence_threshold,
'anomaly_threshold': self.config.anomaly_threshold
}
}


class ImmuneSystemFactory:
"""Class for Schwabot trading functionality."""
"""Factory for creating immune system instances."""

@staticmethod
def create_from_config(config_path: Optional[str] = None) -> ImmuneSystem:
"""Create immune system from configuration file."""
config = ImmuneSystemFactory._load_config(config_path)
return ImmuneSystem(config)

@staticmethod
def create_with_params(**kwargs) -> ImmuneSystem:
"""Create immune system with custom parameters."""
config = ImmuneSystemConfig(**kwargs)
return ImmuneSystem(config)

@staticmethod
def _load_config(config_path: Optional[str] = None) -> ImmuneSystemConfig:
"""Load configuration from file."""
if config_path is None:
# Try to find default config
default_paths = [
"config/immune_system_config.yaml",
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

# Extract immune system config
immune_config = config_data.get('immune_system', {})
return ImmuneSystemConfig(**immune_config)

except Exception as e:
logger.warning(f"Could not load immune system config from {config_path}: {e}")

# Return default config
return ImmuneSystemConfig()


# Auto-load mathematical functions registry if available
IMMUNE_FUNCTIONS_REGISTRY = {}

try:
registry_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'mathematical_functions_registry.yaml')
if os.path.exists(registry_path):
with open(registry_path, 'r') as f:
registry_data = yaml.safe_load(f)
immune_functions = registry_data.get('mathematical_functions', {}).get('immune_system', {})
IMMUNE_FUNCTIONS_REGISTRY.update(immune_functions)
except Exception as e:
logger.warning(f"Could not load immune functions registry: {e}")


# Export main classes and functions
__all__ = [
"QSCGate",
"CollapseState",
"GateType",
"QuantumState",
"CollapseResult",
"ImmuneSystem",
"ImmuneSystemConfig",
"ImmuneSystemFactory",
"MarketDecision",
"ImmuneResponse",
"MarketSignal",
"IMMUNE_FUNCTIONS_REGISTRY"
]

# Convenience functions for quick access
def create_immune_gate(*args, **kwargs) -> QSCGate:
"""Factory for QSCGate (immune system gate)."""
return QSCGate(*args, **kwargs)

def create_immune_system(*args, **kwargs) -> ImmuneSystem:
"""Factory for ImmuneSystem."""
return ImmuneSystem(*args, **kwargs)

def analyze_market_with_immune_system(price: float, volume: float,
config: Optional[ImmuneSystemConfig] = None) -> MarketSignal:
"""Quick function to analyze market with immune system."""
immune_system = ImmuneSystem(config)
return immune_system.analyze_market_signal(price, volume)