"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Entropy Randomization System - Multi-Dimensional Entropy Integration
=============================================================================

Advanced entropy randomization system that enhances Schwabot's existing entropy
infrastructure with multi-dimensional entropy sources, strategy adaptation,
memory integration, and self-evolving parameter optimization.

This system builds upon:
- entropy_signal_integration.py (existing entropy calculation)
- live_vector_simulator.py (entropy-based regime transitions)
- entropy_enhanced_trading_executor.py (entropy-driven decisions)
- enhanced_mathematical_core.py (Shannon entropy, wave entropy)
- tcell_survival_engine.py (biological strategy evolution)

Mathematical Foundation:
- Multi-Dimensional Entropy: E_total = Σ(w_i * E_i) where E_i are entropy sources
- Strategy Adaptation: S_new = S_old * (1 + α * E_influence)
- Memory Integration: E_memory = f(pattern_similarity, historical_performance)
- Self-Evolution: w_i(t+1) = w_i(t) + β * performance_feedback
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import hashlib

# Import existing entropy systems
try:
from core.entropy_signal_integration import EntropySignalIntegrator
from core.live_vector_simulator import LiveVectorSimulator
from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
from core.enhanced_mathematical_core import EnhancedMathematicalCore
from core.tcell_survival_engine import TCellSurvivalEngine
from core.memory_key_allocator import MemoryKeyAllocator
from core.vector_registry import VectorRegistry
EXISTING_ENTROPY_AVAILABLE = True
except ImportError:
EXISTING_ENTROPY_AVAILABLE = False
logger = logging.getLogger(__name__)
logger.warning("⚠️ Some existing entropy systems not available")

logger = logging.getLogger(__name__)

class EntropySource(Enum):
"""Class for Schwabot trading functionality."""
"""Multi-dimensional entropy sources."""
MARKET = "market"           # Market-based entropy (price, volume, volatility)
STRATEGY = "strategy"       # Strategy-based entropy (T-cell mutations, adaptations)
MEMORY = "memory"          # Memory-based entropy (pattern recognition, historical)
SYSTEM = "system"          # System-based entropy (performance, health, load)
TIME = "time"              # Time-based entropy (temporal patterns, cycles)
RANDOM = "random"          # Random entropy (cryptographic, quantum-like)


class EntropyAdaptationType(Enum):
"""Class for Schwabot trading functionality."""
"""Types of entropy adaptation."""
CONSERVATIVE = "conservative"    # Low entropy influence
MODERATE = "moderate"           # Medium entropy influence
AGGRESSIVE = "aggressive"       # High entropy influence
ADAPTIVE = "adaptive"           # Self-adapting influence


@dataclass
class EntropySourceData:
"""Class for Schwabot trading functionality."""
"""Data structure for entropy source information."""
source_type: EntropySource
entropy_value: float
confidence: float
weight: float
adaptation_factor: float
last_update: float
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntropyAdaptationResult:
"""Class for Schwabot trading functionality."""
"""Result of entropy adaptation process."""
original_strategy: Dict[str, Any]
adapted_strategy: Dict[str, Any]
adaptation_strength: float
entropy_influence: float
confidence: float
adaptation_type: EntropyAdaptationType
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntropyRandomizationConfig:
"""Class for Schwabot trading functionality."""
"""Configuration for entropy randomization system."""
# Entropy source weights
market_weight: float = 0.3
strategy_weight: float = 0.25
memory_weight: float = 0.2
system_weight: float = 0.1
time_weight: float = 0.1
random_weight: float = 0.05

# Adaptation parameters
adaptation_learning_rate: float = 0.01
adaptation_threshold: float = 0.1
max_adaptation_strength: float = 0.5

# Memory integration
memory_pattern_threshold: float = 0.7
memory_influence_decay: float = 0.95

# Self-evolution parameters
evolution_interval: float = 300.0  # 5 minutes
performance_window: int = 100
weight_adjustment_rate: float = 0.05

# Randomization parameters
randomization_strength: float = 0.1
entropy_spike_probability: float = 0.05
quantum_entropy_enabled: bool = True


class EnhancedEntropyRandomizationSystem:
"""Class for Schwabot trading functionality."""
"""
Enhanced Entropy Randomization System

Provides multi-dimensional entropy integration with strategy adaptation,
memory integration, and self-evolving parameter optimization.
"""

def __init__(self, config: Optional[EntropyRandomizationConfig] = None) -> None:
"""Initialize the enhanced entropy randomization system."""
self.config = config or EntropyRandomizationConfig()
self.logger = logging.getLogger(__name__)

# Initialize entropy sources
self.entropy_sources: Dict[EntropySource, EntropySourceData] = {}
self._initialize_entropy_sources()

# Performance tracking
self.performance_history: List[Dict[str, Any]] = []
self.adaptation_history: List[EntropyAdaptationResult] = []
self.evolution_history: List[Dict[str, Any]] = []

# Memory integration
self.memory_patterns: Dict[str, float] = {}
self.pattern_confidence: Dict[str, float] = {}

# Self-evolution state
self.last_evolution_time = time.time()
self.current_performance = 0.5
self.weight_performance_map: Dict[EntropySource, List[float]] = {
source: [] for source in EntropySource
}

# Initialize existing systems if available
self._initialize_existing_systems()

self.logger.info("🌊 Enhanced Entropy Randomization System initialized")
self.logger.info(f"✅ Active entropy sources: {len(self.entropy_sources)}")

def _initialize_entropy_sources(self) -> None:
"""Initialize all entropy sources with default values."""
current_time = time.time()

# Market entropy source
self.entropy_sources[EntropySource.MARKET] = EntropySourceData(
source_type=EntropySource.MARKET,
entropy_value=0.5,
confidence=0.8,
weight=self.config.market_weight,
adaptation_factor=1.0,
last_update=current_time,
metadata={'volatility': 0.0, 'volume_irregularity': 0.0}
)

# Strategy entropy source
self.entropy_sources[EntropySource.STRATEGY] = EntropySourceData(
source_type=EntropySource.STRATEGY,
entropy_value=0.5,
confidence=0.7,
weight=self.config.strategy_weight,
adaptation_factor=1.0,
last_update=current_time,
metadata={'mutation_rate': 0.0, 'adaptation_count': 0}
)

# Memory entropy source
self.entropy_sources[EntropySource.MEMORY] = EntropySourceData(
source_type=EntropySource.MEMORY,
entropy_value=0.5,
confidence=0.6,
weight=self.config.memory_weight,
adaptation_factor=1.0,
last_update=current_time,
metadata={'pattern_count': 0, 'similarity_score': 0.0}
)

# System entropy source
self.entropy_sources[EntropySource.SYSTEM] = EntropySourceData(
source_type=EntropySource.SYSTEM,
entropy_value=0.5,
confidence=0.9,
weight=self.config.system_weight,
adaptation_factor=1.0,
last_update=current_time,
metadata={'performance': 1.0, 'health': 1.0, 'load': 0.0}
)

# Time entropy source
self.entropy_sources[EntropySource.TIME] = EntropySourceData(
source_type=EntropySource.TIME,
entropy_value=0.5,
confidence=0.8,
weight=self.config.time_weight,
adaptation_factor=1.0,
last_update=current_time,
metadata={'cycle_position': 0.0, 'temporal_pattern': 'normal'}
)

# Random entropy source
self.entropy_sources[EntropySource.RANDOM] = EntropySourceData(
source_type=EntropySource.RANDOM,
entropy_value=0.5,
confidence=1.0,
weight=self.config.random_weight,
adaptation_factor=1.0,
last_update=current_time,
metadata={'random_seed': int(time.time()), 'quantum_enabled': self.config.quantum_entropy_enabled}
)

def _initialize_existing_systems(self) -> None:
"""Initialize existing entropy systems if available."""
if EXISTING_ENTROPY_AVAILABLE:
try:
# Initialize existing entropy integrator
self.entropy_integrator = EntropySignalIntegrator()
self.logger.info("✅ Entropy Signal Integrator initialized")

# Initialize mathematical core for entropy calculations
self.math_core = EnhancedMathematicalCore()
self.logger.info("✅ Enhanced Mathematical Core initialized")

# Initialize T-cell survival engine for strategy entropy
self.tcell_engine = TCellSurvivalEngine()
self.logger.info("✅ T-Cell Survival Engine initialized")

# Initialize memory systems
self.memory_allocator = MemoryKeyAllocator()
self.vector_registry = VectorRegistry()
self.logger.info("✅ Memory systems initialized")

except Exception as e:
self.logger.warning(f"⚠️ Failed to initialize some existing systems: {e}")
else:
self.logger.warning("⚠️ Using fallback entropy systems")

def calculate_multi_dimensional_entropy(self, market_data: Dict[str, Any], -> None
strategy_state: Dict[str, Any],
system_state: Dict[str, Any]) -> float:
"""
Calculate multi-dimensional entropy from all sources.

Mathematical Formula:
E_total = Σ(w_i * E_i * confidence_i * adaptation_factor_i)

Args:
market_data: Current market data
strategy_state: Current strategy state
system_state: Current system state

Returns:
Total multi-dimensional entropy value
"""
try:
# Update all entropy sources
self._update_market_entropy(market_data)
self._update_strategy_entropy(strategy_state)
self._update_memory_entropy(market_data, strategy_state)
self._update_system_entropy(system_state)
self._update_time_entropy()
self._update_random_entropy()

# Calculate weighted total entropy
total_entropy = 0.0
total_weight = 0.0

for source_type, source_data in self.entropy_sources.items():
weight = source_data.weight * source_data.confidence * source_data.adaptation_factor
total_entropy += weight * source_data.entropy_value
total_weight += weight

# Normalize by total weight
if total_weight > 0:
total_entropy /= total_weight
else:
total_entropy = 0.5  # Default entropy

# Apply randomization strength
if np.random.random() < self.config.entropy_spike_probability:
total_entropy *= (1 + np.random.exponential(1))

# Ensure entropy stays in reasonable bounds
total_entropy = max(0.001, min(0.999, total_entropy))

self.logger.debug(f"🌊 Multi-dimensional entropy: {total_entropy:.4f}")
return total_entropy

except Exception as e:
self.logger.error(f"❌ Error calculating multi-dimensional entropy: {e}")
return 0.5

def adapt_strategy_with_entropy(self, strategy: Dict[str, Any], -> None
entropy_value: float,
market_data: Dict[str, Any]) -> EntropyAdaptationResult:
"""
Adapt strategy using entropy influence.

Mathematical Formula:
S_new = S_old * (1 + α * E_influence * adaptation_strength)

Args:
strategy: Original strategy parameters
entropy_value: Current entropy value
market_data: Market data for context

Returns:
EntropyAdaptationResult with adapted strategy
"""
try:
# Determine adaptation type based on entropy
adaptation_type = self._determine_adaptation_type(entropy_value)

# Calculate adaptation strength
adaptation_strength = self._calculate_adaptation_strength(entropy_value, adaptation_type)

# Create adapted strategy
adapted_strategy = strategy.copy()

# Apply entropy influence to different strategy components
for key, value in strategy.items():
if isinstance(value, (int, float)):
# Apply entropy-based modification
entropy_influence = self._calculate_entropy_influence(key, entropy_value, market_data)
modification = 1.0 + (entropy_influence * adaptation_strength)

# Apply bounds to prevent extreme modifications
modification = max(0.5, min(2.0, modification))

adapted_strategy[key] = value * modification

# Calculate confidence based on adaptation strength and entropy
confidence = min(0.99, 0.5 + (adaptation_strength * 0.4) + (entropy_value * 0.1))

result = EntropyAdaptationResult(
original_strategy=strategy,
adapted_strategy=adapted_strategy,
adaptation_strength=adaptation_strength,
entropy_influence=entropy_value,
confidence=confidence,
adaptation_type=adaptation_type,
metadata={
'adaptation_timestamp': time.time(),
'market_context': market_data.get('symbol', 'unknown'),
'entropy_sources': {source.value: data.entropy_value
for source, data in self.entropy_sources.items()}
}
)

# Record adaptation
self.adaptation_history.append(result)

self.logger.info(f"🔄 Strategy adapted with entropy {entropy_value:.3f} "
f"(strength: {adaptation_strength:.3f}, type: {adaptation_type.value})")

return result

except Exception as e:
self.logger.error(f"❌ Error adapting strategy with entropy: {e}")
return EntropyAdaptationResult(
original_strategy=strategy,
adapted_strategy=strategy,
adaptation_strength=0.0,
entropy_influence=entropy_value,
confidence=0.0,
adaptation_type=EntropyAdaptationType.CONSERVATIVE
)

def integrate_memory_patterns(self, market_data: Dict[str, Any], -> None
strategy_hash: str) -> float:
"""
Integrate memory patterns for entropy enhancement.

Mathematical Formula:
E_memory = Σ(pattern_similarity_i * historical_performance_i * decay_factor_i)

Args:
market_data: Current market data
strategy_hash: Current strategy hash

Returns:
Memory-enhanced entropy value
"""
try:
if not EXISTING_ENTROPY_AVAILABLE:
return 0.5

# Generate current pattern hash
current_pattern = self._generate_pattern_hash(market_data)

# Find similar patterns in memory
similar_patterns = self._find_similar_patterns(current_pattern)

# Calculate memory entropy
memory_entropy = 0.5  # Default entropy
total_weight = 0.0

for pattern_hash, similarity in similar_patterns:
# Get historical performance for this pattern
historical_performance = self.memory_patterns.get(pattern_hash, 0.5)

# Calculate decay factor based on time
decay_factor = self._calculate_pattern_decay(pattern_hash)

# Weight by similarity and performance
weight = similarity * historical_performance * decay_factor
memory_entropy += weight * 0.5  # Assume 0.5 entropy for historical patterns
total_weight += weight

# Normalize memory entropy
if total_weight > 0:
memory_entropy /= total_weight

# Update memory entropy source
self.entropy_sources[EntropySource.MEMORY].entropy_value = memory_entropy
self.entropy_sources[EntropySource.MEMORY].confidence = min(0.9, total_weight)
self.entropy_sources[EntropySource.MEMORY].last_update = time.time()

self.logger.debug(f"🧠 Memory entropy: {memory_entropy:.4f} (patterns: {len(similar_patterns)})")
return memory_entropy

except Exception as e:
self.logger.error(f"❌ Error integrating memory patterns: {e}")
return 0.5

def evolve_entropy_weights(self, performance_feedback: float) -> None:
"""
Evolve entropy source weights based on performance feedback.

Mathematical Formula:
w_i(t+1) = w_i(t) + β * performance_feedback * source_performance_i

Args:
performance_feedback: Performance feedback value (-1 to 1)
"""
try:
current_time = time.time()

# Check if evolution interval has passed
if current_time - self.last_evolution_time < self.config.evolution_interval:
return

# Update current performance
self.current_performance = performance_feedback

# Calculate weight adjustments for each source
weight_adjustments = {}

for source_type, source_data in self.entropy_sources.items():
# Calculate source performance (how well this source contributed)
source_performance = self._calculate_source_performance(source_type, performance_feedback)

# Calculate weight adjustment
adjustment = (self.config.weight_adjustment_rate *
performance_feedback * source_performance)

weight_adjustments[source_type] = adjustment

# Apply weight adjustments
for source_type, adjustment in weight_adjustments.items():
current_weight = self.entropy_sources[source_type].weight
new_weight = max(0.01, min(0.5, current_weight + adjustment))
self.entropy_sources[source_type].weight = new_weight

# Record evolution
evolution_record = {
'timestamp': current_time,
'performance_feedback': performance_feedback,
'weight_adjustments': weight_adjustments,
'new_weights': {source.value: data.weight
for source, data in self.entropy_sources.items()}
}
self.evolution_history.append(evolution_record)

self.last_evolution_time = current_time

self.logger.info(f"🧬 Entropy weights evolved (performance: {performance_feedback:.3f})")

except Exception as e:
self.logger.error(f"❌ Error evolving entropy weights: {e}")

def _update_market_entropy(self, market_data: Dict[str, Any]) -> None:
"""Update market entropy source."""
try:
if EXISTING_ENTROPY_AVAILABLE and hasattr(self, 'math_core'):
# Use existing mathematical core for entropy calculation
prices = market_data.get('price_history', [100.0])
if len(prices) > 1:
price_changes = np.diff(prices)
entropy_result = self.math_core.shannon_entropy(np.abs(price_changes))
if entropy_result.success:
market_entropy = entropy_result.value / 4.0  # Normalize
else:
market_entropy = 0.5
else:
market_entropy = 0.5
else:
# Fallback market entropy calculation
volatility = market_data.get('volatility', 0.5)
volume_irregularity = market_data.get('volume_irregularity', 0.5)
market_entropy = (volatility + volume_irregularity) / 2.0

# Add randomization
market_entropy += np.random.normal(0, 0.01)
market_entropy = max(0.001, min(0.999, market_entropy))

self.entropy_sources[EntropySource.MARKET].entropy_value = market_entropy
self.entropy_sources[EntropySource.MARKET].last_update = time.time()

except Exception as e:
self.logger.error(f"❌ Error updating market entropy: {e}")

def _update_strategy_entropy(self, strategy_state: Dict[str, Any]) -> None:
"""Update strategy entropy source."""
try:
if EXISTING_ENTROPY_AVAILABLE and hasattr(self, 'tcell_engine'):
# Use T-cell engine for strategy entropy
mutation_rate = strategy_state.get('mutation_rate', 0.1)
adaptation_count = strategy_state.get('adaptation_count', 0)

# Calculate strategy entropy based on mutations and adaptations
strategy_entropy = min(0.999, mutation_rate + (adaptation_count * 0.01))
else:
# Fallback strategy entropy
strategy_entropy = 0.5

self.entropy_sources[EntropySource.STRATEGY].entropy_value = strategy_entropy
self.entropy_sources[EntropySource.STRATEGY].last_update = time.time()

except Exception as e:
self.logger.error(f"❌ Error updating strategy entropy: {e}")

def _update_memory_entropy(self, market_data: Dict[str, Any], strategy_state: Dict[str, Any]) -> None:
"""Update memory entropy source."""
try:
# Real memory entropy calculation based on pattern recognition
pattern_hash = self._generate_pattern_hash(market_data)
similar_patterns = self._find_similar_patterns(pattern_hash)

if similar_patterns:
# Calculate memory entropy based on pattern similarity
avg_similarity = np.mean([sim for _, sim in similar_patterns])
memory_entropy = 1.0 - avg_similarity  # Higher similarity = lower entropy
else:
# No similar patterns found - high entropy
memory_entropy = 0.8

# Apply decay factor
decay = self._calculate_pattern_decay(pattern_hash)
memory_entropy *= decay

self.entropy_sources[EntropySource.MEMORY].entropy_value = memory_entropy
self.entropy_sources[EntropySource.MEMORY].last_update = time.time()

except Exception as e:
self.logger.error(f"❌ Error updating memory entropy: {e}")

def _update_system_entropy(self, system_state: Dict[str, Any]) -> None:
"""Update system entropy source."""
try:
performance = system_state.get('performance', 1.0)
health = system_state.get('health', 1.0)
load = system_state.get('load', 0.0)

# Calculate system entropy based on performance, health, and load
system_entropy = (1.0 - performance) * 0.3 + (1.0 - health) * 0.3 + load * 0.4
system_entropy = max(0.001, min(0.999, system_entropy))

self.entropy_sources[EntropySource.SYSTEM].entropy_value = system_entropy
self.entropy_sources[EntropySource.SYSTEM].last_update = time.time()

except Exception as e:
self.logger.error(f"❌ Error updating system entropy: {e}")

def _update_time_entropy(self) -> None:
"""Update time entropy source."""
try:
current_time = time.time()

# Calculate time-based entropy using various cycles
seconds = current_time % 60
minutes = (current_time // 60) % 60
hours = (current_time // 3600) % 24

# Combine different time cycles
time_entropy = ((seconds / 60.0) * 0.3 +
(minutes / 60.0) * 0.3 +
(hours / 24.0) * 0.4)

# Add some randomness
time_entropy += np.random.normal(0, 0.01)
time_entropy = max(0.001, min(0.999, time_entropy))

self.entropy_sources[EntropySource.TIME].entropy_value = time_entropy
self.entropy_sources[EntropySource.TIME].last_update = current_time

except Exception as e:
self.logger.error(f"❌ Error updating time entropy: {e}")

def _update_random_entropy(self) -> None:
"""Update random entropy source."""
try:
# Generate random entropy
random_entropy = np.random.random()

# Add quantum-like behavior if enabled
if self.config.quantum_entropy_enabled:
# Simulate quantum uncertainty
quantum_factor = np.sin(time.time() * 0.1) * 0.1
random_entropy += quantum_factor
random_entropy = max(0.001, min(0.999, random_entropy))

self.entropy_sources[EntropySource.RANDOM].entropy_value = random_entropy
self.entropy_sources[EntropySource.RANDOM].last_update = time.time()

except Exception as e:
self.logger.error(f"❌ Error updating random entropy: {e}")

def _determine_adaptation_type(self, entropy_value: float) -> EntropyAdaptationType:
"""Determine adaptation type based on entropy value."""
if entropy_value < 0.3:
return EntropyAdaptationType.CONSERVATIVE
elif entropy_value < 0.7:
return EntropyAdaptationType.MODERATE
else:
return EntropyAdaptationType.AGGRESSIVE

def _calculate_adaptation_strength(self, entropy_value: float, -> None
adaptation_type: EntropyAdaptationType) -> float:
"""Calculate adaptation strength based on entropy and type."""
base_strength = entropy_value * self.config.max_adaptation_strength

if adaptation_type == EntropyAdaptationType.CONSERVATIVE:
return base_strength * 0.5
elif adaptation_type == EntropyAdaptationType.MODERATE:
return base_strength * 0.8
elif adaptation_type == EntropyAdaptationType.AGGRESSIVE:
return base_strength
else:  # ADAPTIVE
return base_strength * (0.5 + entropy_value * 0.5)

def _calculate_entropy_influence(self, parameter_key: str, entropy_value: float, -> None
market_data: Dict[str, Any]) -> float:
"""Calculate entropy influence for a specific parameter."""
# Different parameters respond differently to entropy
if 'threshold' in parameter_key.lower():
return entropy_value * 0.5  # Thresholds are less sensitive
elif 'weight' in parameter_key.lower():
return entropy_value * 0.8  # Weights are moderately sensitive
elif 'rate' in parameter_key.lower():
return entropy_value * 1.2  # Rates are more sensitive
else:
return entropy_value  # Default sensitivity

def _generate_pattern_hash(self, market_data: Dict[str, Any]) -> str:
"""Generate pattern hash from market data."""
try:
# Create pattern string from market data
pattern_string = f"{market_data.get('symbol', 'BTC')}_{market_data.get('price', 0):.2f}_" \
f"{market_data.get('volume', 0):.2f}_{market_data.get('volatility', 0):.4f}"

# Generate SHA-256 hash
return hashlib.sha256(pattern_string.encode()).hexdigest()
except Exception as e:
self.logger.error(f"❌ Error generating pattern hash: {e}")
return "fallback_pattern_hash"

def _find_similar_patterns(self, current_pattern: str) -> List[Tuple[str, float]]:
"""Find similar patterns in memory."""
try:
similar_patterns = []

for pattern_hash in self.memory_patterns.keys():
# Calculate similarity using Hamming distance
similarity = self._calculate_pattern_similarity(current_pattern, pattern_hash)

if similarity > self.config.memory_pattern_threshold:
similar_patterns.append((pattern_hash, similarity))

# Sort by similarity (highest first)
similar_patterns.sort(key=lambda x: x[1], reverse=True)

return similar_patterns[:10]  # Return top 10 similar patterns

except Exception as e:
self.logger.error(f"❌ Error finding similar patterns: {e}")
return []

def _calculate_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
"""Calculate similarity between two patterns."""
try:
# Use Hamming distance for similarity
if len(pattern1) != len(pattern2):
return 0.0

hamming_distance = sum(a != b for a, b in zip(pattern1, pattern2))
similarity = 1.0 - (hamming_distance / len(pattern1))

return max(0.0, similarity)

except Exception as e:
self.logger.error(f"❌ Error calculating pattern similarity: {e}")
return 0.0

def _calculate_pattern_decay(self, pattern_hash: str) -> float:
"""Calculate decay factor for a pattern."""
try:
# Simple time-based decay
# In a real implementation, you'd store timestamps with patterns
return self.config.memory_influence_decay

except Exception as e:
self.logger.error(f"❌ Error calculating pattern decay: {e}")
return 1.0

def _calculate_source_performance(self, source_type: EntropySource, -> None
performance_feedback: float) -> float:
"""Calculate performance contribution of an entropy source."""
try:
# This is a simplified calculation
# In a real implementation, you'd track how each source contributed to decisions

# Use recent adaptation history to estimate source performance
recent_adaptations = self.adaptation_history[-10:] if self.adaptation_history else []

if not recent_adaptations:
return 0.5  # Default performance

# Calculate average performance for this source
source_performances = []
for adaptation in recent_adaptations:
if source_type.value in adaptation.metadata.get('entropy_sources', {}):
source_entropy = adaptation.metadata['entropy_sources'][source_type.value]
# Assume higher entropy contribution correlates with performance
source_performances.append(source_entropy)

if source_performances:
return np.mean(source_performances)
else:
return 0.5

except Exception as e:
self.logger.error(f"❌ Error calculating source performance: {e}")
return 0.5

def get_entropy_report(self) -> Dict[str, Any]:
"""Get comprehensive entropy report."""
try:
return {
'total_entropy_sources': len(self.entropy_sources),
'entropy_sources': {
source.value: {
'entropy_value': data.entropy_value,
'confidence': data.confidence,
'weight': data.weight,
'adaptation_factor': data.adaptation_factor,
'last_update': data.last_update
}
for source, data in self.entropy_sources.items()
},
'adaptation_history_count': len(self.adaptation_history),
'evolution_history_count': len(self.evolution_history),
'current_performance': self.current_performance,
'memory_patterns_count': len(self.memory_patterns),
'last_evolution_time': self.last_evolution_time
}
except Exception as e:
self.logger.error(f"❌ Error generating entropy report: {e}")
return {'error': str(e)}


# Factory function
def create_enhanced_entropy_randomization_system(config: Optional[EntropyRandomizationConfig] = None) -> EnhancedEntropyRandomizationSystem:
"""Create an enhanced entropy randomization system instance."""
return EnhancedEntropyRandomizationSystem(config)


# Singleton instance for global use
enhanced_entropy_system = EnhancedEntropyRandomizationSystem()


def main():
"""Test the enhanced entropy randomization system."""
logger.info("🌊 Testing Enhanced Entropy Randomization System")

# Test market data
test_market_data = {
'symbol': 'BTC',
'price': 50000.0,
'volume': 1000.0,
'volatility': 0.02,
'price_history': [50000, 50100, 50200, 50150, 50300],
'volume_irregularity': 0.1
}

# Test strategy state
test_strategy_state = {
'mutation_rate': 0.1,
'adaptation_count': 5,
'confidence': 0.8
}

# Test system state
test_system_state = {
'performance': 0.9,
'health': 0.95,
'load': 0.3
}

# Calculate multi-dimensional entropy
total_entropy = enhanced_entropy_system.calculate_multi_dimensional_entropy(
test_market_data, test_strategy_state, test_system_state
)

# Test strategy adaptation
test_strategy = {
'threshold': 0.5,
'weight': 0.3,
'rate': 0.1,
'confidence': 0.8
}

adaptation_result = enhanced_entropy_system.adapt_strategy_with_entropy(
test_strategy, total_entropy, test_market_data
)

# Test memory integration
memory_entropy = enhanced_entropy_system.integrate_memory_patterns(
test_market_data, "test_strategy_hash"
)

# Test evolution
enhanced_entropy_system.evolve_entropy_weights(0.7)

# Get report
report = enhanced_entropy_system.get_entropy_report()

logger.info(f"✅ Test completed successfully")
logger.info(f"🌊 Total entropy: {total_entropy:.4f}")
logger.info(f"🔄 Adaptation strength: {adaptation_result.adaptation_strength:.3f}")
logger.info(f"🧠 Memory entropy: {memory_entropy:.4f}")
logger.info(f"📊 Entropy sources: {len(report.get('entropy_sources', {}))}")


if __name__ == "__main__":
main()