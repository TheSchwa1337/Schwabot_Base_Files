"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pipeline Manager for Schwabot
=====================================

Provides unified management of Schwabot's mathematical pipeline:
• Coordinates all mathematical components
• Manages hardware optimization (CPU/GPU)
• Provides Cursor-friendly interfaces
• Handles configuration management
• Integrates symbolic math with 2-gram detection

This manager acts as the central coordinator for all mathematical operations,
ensuring Cursor can understand and work with the complex recursive logic
while maintaining the full power of Schwabot's mathematical framework.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

# Import core components
try:
from core.math_cache import MathResultCache
from core.math_config_manager import MathConfigManager
from core.math_orchestrator import MathOrchestrator
from core.symbolic_math_interface import (
DriftField,
EntropicGradient,
NoiseField,
PhaseOmega,
PhaseValue,
SignalField,
SignalPsi,
SymbolicContext,
SymbolicMathEngine,
TimeIndex,
)
from core.two_gram_detector import TwoGramDetector

CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
CORE_COMPONENTS_AVAILABLE = False
logger.warning(f"Core components not available: {e}")

@dataclass
class PipelineContext:
"""Class for Schwabot trading functionality."""
"""Context for pipeline operations."""
cycle_id: int
vault_state: str
entropy_index: int
phantom_layer: bool = False
timestamp: float = None


def __post_init__(self) -> None:
if self.timestamp is None:
self.timestamp = time.time()

class UnifiedPipelineManager:
"""Class for Schwabot trading functionality."""
"""
Unified Pipeline Manager for Schwabot.

Coordinates all mathematical components and provides a clean interface
for complex mathematical operations while maintaining Cursor compatibility.
"""


def __init__(self, config_path: Optional[str] = None) -> None:
self.logger = logging.getLogger(__name__)
self.config_path = config_path or "config/pipeline_config.yaml"
self.config = self._load_config()

# Component initialization
self.symbolic_math_engine = None
self.two_gram_detector = None
self.math_config_manager = None
self.math_orchestrator = None
self.math_cache = None

# State management
self.active = False
self.initialized = False
self.last_operation_time = 0.0

# Performance tracking
self.operation_count = 0
self.total_processing_time = 0.0

self._initialize_components()

def _load_config(self) -> Dict[str, Any]:
"""Load configuration from YAML file."""
try:
config_file = Path(self.config_path)
if config_file.exists():
with open(config_file, 'r') as f:
config = yaml.safe_load(f)
self.logger.info(f"✅ Configuration loaded from {self.config_path}")
return config
else:
self.logger.warning(f"Config file {self.config_path} not found, using defaults")
return self._default_config()
except Exception as e:
self.logger.error(f"❌ Error loading config: {e}")
return self._default_config()

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
'pipeline': {
'enabled': True,
'mode': 'production',
'debug': False,
'log_level': 'INFO',
'hardware': {
'preference': 'auto',
'enable_gpu_acceleration': True,
'enable_cpu_optimization': True,
'memory_limit_gb': 8,
'batch_size': 100
},
'performance': {
'timeout_seconds': 30.0,
'max_retries': 3,
'cache_enabled': True,
'cache_ttl_seconds': 3600,
'max_cache_size': 1000,
'parallel_processing': True
}
},
'symbolic_math': {
'enabled': True,
'hardware_preference': 'auto',
'enable_phantom_boost': True,
'enable_context_awareness': True,
'max_iterations': 100,
'convergence_threshold': 1e-6
},
'two_gram': {
'enabled': True,
'max_patterns': 1024,
'pattern_threshold': 0.1,
'burst_threshold': 0.5,
'similarity_threshold': 0.8,
'entropy_window': 16
}
}

def _initialize_components(self) -> None:
"""Initialize all pipeline components."""
try:
self.logger.info("Initializing UnifiedPipelineManager components")

if not CORE_COMPONENTS_AVAILABLE:
self.logger.error("❌ Core components not available")
return

# Initialize symbolic math engine
if self.config.get('symbolic_math', {}).get('enabled', True):
self.symbolic_math_engine = SymbolicMathEngine(
self.config.get('symbolic_math', {})
)
self.logger.info("✅ Symbolic math engine initialized")

# Initialize two-gram detector
if self.config.get('two_gram', {}).get('enabled', True):
self.two_gram_detector = TwoGramDetector(
self.config.get('two_gram', {})
)
self.logger.info("✅ Two-gram detector initialized")

# Initialize math infrastructure
self.math_config_manager = MathConfigManager()
self.math_orchestrator = MathOrchestrator()
self.math_cache = MathResultCache()

self.initialized = True
self.logger.info("✅ UnifiedPipelineManager initialized successfully")

except Exception as e:
self.logger.error(f"❌ Error initializing components: {e}")
self.initialized = False

def activate(self) -> bool:
"""Activate the pipeline manager."""
if not self.initialized:
self.logger.error("Pipeline manager not initialized")
return False

try:
# Activate all components
if self.symbolic_math_engine:
self.symbolic_math_engine.activate()

if self.two_gram_detector:
self.two_gram_detector.activate()

if self.math_config_manager:
self.math_config_manager.activate()

if self.math_orchestrator:
self.math_orchestrator.activate()

self.active = True
self.logger.info("✅ UnifiedPipelineManager activated")
return True

except Exception as e:
self.logger.error(f"❌ Error activating pipeline manager: {e}")
return False

def deactivate(self) -> bool:
"""Deactivate the pipeline manager."""
try:
# Deactivate all components
if self.symbolic_math_engine:
self.symbolic_math_engine.deactivate()

if self.two_gram_detector:
self.two_gram_detector.deactivate()

if self.math_config_manager:
self.math_config_manager.deactivate()

if self.math_orchestrator:
self.math_orchestrator.deactivate()

self.active = False
self.logger.info("✅ UnifiedPipelineManager deactivated")
return True

except Exception as e:
self.logger.error(f"❌ Error deactivating pipeline manager: {e}")
return False


def compute_phase_omega(self, signal_data: Union[List, SignalField], -> None
time_idx: TimeIndex, context: Optional[PipelineContext] = None) -> PhaseValue:
"""
Compute phase omega using the unified pipeline.

This is the main entry point for phase omega computation,
providing a Cursor-friendly interface to complex mathematical operations.

Args:
signal_data: Input signal data (list or numpy array)
time_idx: Time index for computation
context: Optional pipeline context

Returns:
Computed phase omega value
"""
if not self.active:
raise RuntimeError("Pipeline manager not active")

if not self.symbolic_math_engine:
raise RuntimeError("Symbolic math engine not available for phase omega computation")

start_time = time.time()

try:
# Convert context if provided
symbolic_context = None
if context:
symbolic_context = SymbolicContext(
cycle_id=context.cycle_id,
vault_state=context.vault_state,
entropy_index=context.entropy_index,
phantom_layer=context.phantom_layer
)

# Compute phase omega using symbolic math engine
omega = self.symbolic_math_engine.compute_phase_omega(
signal_data, time_idx, symbolic_context
)

# Update performance tracking
processing_time = time.time() - start_time
self.operation_count += 1
self.total_processing_time += processing_time
self.last_operation_time = time.time()

self.logger.debug(f"Phase omega computed: {omega:.6f} (took {processing_time:.4f}s)")
return omega

except Exception as e:
self.logger.error(f"Error computing phase omega: {e}")
raise


def feed_signal_to_2gram(self, signal: Union[float, int, str], -> None
context: Optional[PipelineContext] = None) -> Optional[Tuple]:
"""
Feed a signal to the 2-gram detector.

Args:
signal: Input signal value
context: Optional pipeline context

Returns:
Detected 2-gram pattern or None
"""
if not self.active:
raise RuntimeError("Pipeline manager not active")

if not self.two_gram_detector:
raise RuntimeError("2-gram detector not available")

try:
# Convert context if provided
symbolic_context = None
if context:
symbolic_context = SymbolicContext(
cycle_id=context.cycle_id,
vault_state=context.vault_state,
entropy_index=context.entropy_index,
phantom_layer=context.phantom_layer
)

# Feed signal to 2-gram detector
pattern = self.two_gram_detector.feed_signal(signal, symbolic_context)
return pattern

except Exception as e:
self.logger.error(f"Error feeding signal to 2-gram: {e}")
raise

def get_2gram_statistics(self) -> Dict[str, Any]:
"""Get 2-gram detector statistics."""
if not self.two_gram_detector:
raise RuntimeError("2-gram detector not available")

try:
return self.two_gram_detector.get_pattern_statistics()
except Exception as e:
self.logger.error(f"Error getting 2-gram statistics: {e}")
raise

def get_top_2gram_patterns(self, n: int = 10) -> List[Tuple[Tuple, int]]:
"""Get top N 2-gram patterns."""
if not self.two_gram_detector:
raise RuntimeError("2-gram detector not available")

try:
return self.two_gram_detector.get_top_patterns(n)
except Exception as e:
self.logger.error(f"Error getting top 2-gram patterns: {e}")
raise

def get_burst_patterns(self) -> List[Tuple[Tuple, float]]:
"""Get current burst patterns."""
if not self.two_gram_detector:
raise RuntimeError("2-gram detector not available")

try:
return self.two_gram_detector.get_burst_patterns()
except Exception as e:
self.logger.error(f"Error getting burst patterns: {e}")
raise

def calculate_entropy(self, window_size: Optional[int] = None) -> float:
"""Calculate Shannon entropy of recent patterns."""
if not self.two_gram_detector:
raise RuntimeError("2-gram detector not available")

try:
return self.two_gram_detector.calculate_shannon_entropy(window_size)
except Exception as e:
self.logger.error(f"Error calculating entropy: {e}")
raise

def get_hardware_info(self) -> Dict[str, Any]:
"""Get hardware information and capabilities."""
if not self.math_orchestrator:
raise RuntimeError("Math orchestrator not available")

try:
return self.math_orchestrator.get_hardware_info()
except Exception as e:
self.logger.error(f"Error getting hardware info: {e}")
raise

def get_performance_stats(self) -> Dict[str, Any]:
"""Get performance statistics."""
return {
'operation_count': self.operation_count,
'total_processing_time': self.total_processing_time,
'average_processing_time': (
self.total_processing_time / max(self.operation_count, 1)
),
'last_operation_time': self.last_operation_time,
'active': self.active,
'initialized': self.initialized,
}

def get_status(self) -> Dict[str, Any]:
"""Get comprehensive system status."""
status = {
'active': self.active,
'initialized': self.initialized,
'config_loaded': bool(self.config),
'components_available': CORE_COMPONENTS_AVAILABLE,
'performance_stats': self.get_performance_stats(),
}

# Add component statuses
if self.symbolic_math_engine:
status['symbolic_math_engine'] = self.symbolic_math_engine.get_status()

if self.two_gram_detector:
status['two_gram_detector'] = self.two_gram_detector.get_status()

if self.math_config_manager:
status['math_config_manager'] = self.math_config_manager.get_status()

if self.math_orchestrator:
status['math_orchestrator'] = self.math_orchestrator.get_status()

return status

def reload_config(self) -> bool:
"""Reload configuration from file."""
try:
self.config = self._load_config()
self.logger.info("✅ Configuration reloaded")
return True
except Exception as e:
self.logger.error(f"❌ Error reloading config: {e}")
return False

def clear_caches(self) -> bool:
"""Clear all caches."""
try:
if self.math_cache:
self.math_cache.clear()

if self.two_gram_detector:
self.two_gram_detector.clear_patterns()

self.logger.info("✅ All caches cleared")
return True
except Exception as e:
self.logger.error(f"❌ Error clearing caches: {e}")
return False

# Factory function for easy instantiation

def create_unified_pipeline_manager(config_path: Optional[str] = None) -> UnifiedPipelineManager:
"""Create a unified pipeline manager instance."""
return UnifiedPipelineManager(config_path)

# Global instance for easy access
unified_pipeline_manager = UnifiedPipelineManager()
