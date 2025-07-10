#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type Aliases for Schwabot
=========================

Provides type aliases for better Cursor understanding and type safety.
These aliases help Cursor recognize variable intent and improve
auto-completion and debugging capabilities.

Key Benefits:
• Cursor recognizes and tracks variable intent better
• Improves auto-completion and debugging
• Maintains type safety across the mathematical framework
• Provides clear semantic meaning for complex operations
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ============================================================================
# CORE MATHEMATICAL TYPES
# ============================================================================

# Signal and Data Types
SignalField = np.ndarray
SignalVector = np.ndarray
SignalMatrix = np.ndarray
SignalTensor = np.ndarray

# Time and Index Types
TimeIndex = int
TimeStamp = float
CycleIndex = int
EntropyIndex = int

# Mathematical Value Types
PhaseValue = float
EntropyValue = float
DriftCoefficient = float
DriftFactor = float
NoiseFactor = float
GradientValue = float

# Context Types
VaultState = str
StrategyID = str
HashValue = str
PatternTuple = Tuple

# ============================================================================
# COMPLEX MATHEMATICAL TYPES
# ============================================================================

# Quantum Types
QuantumState = np.ndarray
WaveFunction = np.ndarray
EnergyLevel = float

# Thermal Types
ThermalField = np.ndarray
Temperature = float
HeatCapacity = float

# Entropy Types
EntropyField = np.ndarray
EntropyWeight = float
EntropyGradient = np.ndarray

# ============================================================================
# PIPELINE TYPES
# ============================================================================

# Pipeline Context Types
PipelineContext = Dict[str, Any]
SymbolicContext = Dict[str, Any]
ComputationContext = Dict[str, Any]

# Configuration Types
ConfigDict = Dict[str, Any]
ConfigSection = Dict[str, Any]
ConfigValue = Union[str, int, float, bool, List, Dict]

# Result Types
ComputationResult = Dict[str, Any]
ValidationResult = Dict[str, Any]
PerformanceResult = Dict[str, Any]

# ============================================================================
# HARDWARE TYPES
# ============================================================================

# Hardware Backend Types
BackendType = str  # "cupy", "torch", "numpy"
DeviceType = str   # "cpu", "gpu"
MemorySize = int   # bytes

# Performance Types
ProcessingTime = float
MemoryUsage = int
Throughput = float

# ============================================================================
# STRATEGY TYPES
# ============================================================================

# Strategy Types
StrategyPattern = Tuple
StrategyHash = str
StrategyWeight = float
StrategyConfidence = float

# Pattern Types
PatternFrequency = int
PatternSimilarity = float
PatternBurst = float

# ============================================================================
# VALIDATION TYPES
# ============================================================================

# Validation Types
ValidationStatus = bool
ValidationError = str
ValidationWarning = str

# Error Types
ErrorCode = str
ErrorMessage = str
ErrorContext = Dict[str, Any]

# ============================================================================
# LOGGING TYPES
# ============================================================================

# Logging Types
LogLevel = str  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LogMessage = str
LogContext = Dict[str, Any]

# ============================================================================
# CACHE TYPES
# ============================================================================

# Cache Types
CacheKey = str
CacheValue = Any
CacheTTL = int  # seconds
CacheSize = int

# ============================================================================
# UTILITY TYPE FUNCTIONS
# ============================================================================

def is_signal_field(obj: Any) -> bool:
    """Check if object is a signal field."""
    return isinstance(obj, np.ndarray) and obj.ndim >= 1

def is_phase_value(obj: Any) -> bool:
    """Check if object is a phase value."""
    return isinstance(obj, (int, float)) and not isinstance(obj, bool)

def is_entropy_value(obj: Any) -> bool:
    """Check if object is an entropy value."""
    return isinstance(obj, (int, float)) and not isinstance(obj, bool)

def is_time_index(obj: Any) -> bool:
    """Check if object is a time index."""
    return isinstance(obj, int) and obj >= 0

def is_vault_state(obj: Any) -> bool:
    """Check if object is a valid vault state."""
    valid_states = {"normal", "phantom", "transition", "error"}
    return isinstance(obj, str) and obj in valid_states

def is_strategy_id(obj: Any) -> bool:
    """Check if object is a strategy ID."""
    return isinstance(obj, str) and len(obj) > 0

def is_config_dict(obj: Any) -> bool:
    """Check if object is a configuration dictionary."""
    return isinstance(obj, dict)

def is_computation_result(obj: Any) -> bool:
    """Check if object is a computation result."""
    return isinstance(obj, dict) and "success" in obj

# ============================================================================
# TYPE CONVERSION FUNCTIONS
# ============================================================================

def to_signal_field(data: Union[List, np.ndarray]) -> SignalField:
    """Convert data to signal field."""
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Cannot convert {type(data)} to SignalField")

def to_phase_value(value: Union[int, float]) -> PhaseValue:
    """Convert value to phase value."""
    return float(value)

def to_time_index(value: Union[int, float]) -> TimeIndex:
    """Convert value to time index."""
    return int(value)

def to_entropy_value(value: Union[int, float]) -> EntropyValue:
    """Convert value to entropy value."""
    return float(value)

def to_drift_coefficient(value: Union[int, float]) -> DriftCoefficient:
    """Convert value to drift coefficient."""
    return float(value)

# ============================================================================
# TYPE VALIDATION FUNCTIONS
# ============================================================================

def validate_signal_field(signal: SignalField) -> bool:
    """Validate signal field."""
    if not is_signal_field(signal):
        return False
    if signal.size == 0:
        return False
    if not np.isfinite(signal).all():
        return False
    return True

def validate_phase_value(phase: PhaseValue) -> bool:
    """Validate phase value."""
    if not is_phase_value(phase):
        return False
    if not np.isfinite(phase):
        return False
    return True

def validate_time_index(time_idx: TimeIndex) -> bool:
    """Validate time index."""
    if not is_time_index(time_idx):
        return False
    return True

def validate_vault_state(state: VaultState) -> bool:
    """Validate vault state."""
    return is_vault_state(state)

def validate_strategy_id(strategy_id: StrategyID) -> bool:
    """Validate strategy ID."""
    return is_strategy_id(strategy_id)

# ============================================================================
# TYPE DOCUMENTATION
# ============================================================================

"""
Type Usage Examples:

# Signal processing
signal_data: SignalField = np.array([1.0, 2.0, 3.0])
time_idx: TimeIndex = 5
phase: PhaseValue = 0.123

# Context creation
context: PipelineContext = {
    "cycle_id": 42,
    "vault_state": "phantom",
    "entropy_index": 5
}

# Strategy handling
strategy_id: StrategyID = "strategy_001"
pattern: StrategyPattern = ("up", "down")
confidence: StrategyConfidence = 0.85

# Configuration
config: ConfigDict = {
    "enabled": True,
    "timeout": 30.0
}

# Results
result: ComputationResult = {
    "success": True,
    "value": 0.123,
    "timestamp": 1234567890.0
}
""" 