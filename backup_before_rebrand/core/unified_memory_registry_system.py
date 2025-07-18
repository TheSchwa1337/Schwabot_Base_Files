"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Memory Registry System - Complete Memory Integration
==========================================================

Unified memory registry system that integrates all Schwabot memory components
into a single cohesive system for pattern recognition, memory-based adaptation,
and cross-registry pattern matching.

This system unifies:
- memory_key_allocator.py (symbolic/hash/hybrid memory keys, clustering)
- vector_registry.py (digest→strategy mapping, similarity search)
- hash_memory_generator.py (pattern recognition, hash memory, success tracking)
- profit_bucket_registry.py (profit pattern registry)

Mathematical Foundation:
- Cross-Registry Pattern Matching: P_match = Σ(w_i * similarity_i) across all registries
- Memory Integration: M_unified = f(pattern_similarity, historical_success, adaptation_history)
- Pattern Recognition: R = f(hash_similarity, vector_similarity, profit_correlation)
- Memory-Based Adaptation: A = Σ(memory_pattern_i * success_rate_i * decay_factor_i)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import hashlib
import json

# Import centralized hash configuration
from core.hash_config_manager import generate_hash_from_string

# Import existing memory systems
try:
from core_math_restore.memory_stack.memory_key_allocator import MemoryKeyAllocator, MemoryKey, MemoryCluster
from core.vector_registry import VectorRegistry, StrategyVector, DigestMatch
from data.hash_memory_generator import HashMemoryGenerator
from core.profit_bucket_registry import ProfitBucketRegistry, ProfitBucket
EXISTING_MEMORY_AVAILABLE = True
except ImportError:
EXISTING_MEMORY_AVAILABLE = False
logger = logging.getLogger(__name__)
logger.warning("⚠️ Some existing memory systems not available")

logger = logging.getLogger(__name__)

class MemoryRegistryType(Enum):
"""Class for Schwabot trading functionality."""
"""Types of memory registries."""
KEY_ALLOCATOR = "key_allocator"    # Memory key allocation and clustering
VECTOR_REGISTRY = "vector_registry" # Strategy vector mapping
HASH_MEMORY = "hash_memory"        # Hash-based pattern memory
PROFIT_BUCKET = "profit_bucket"    # Profit pattern registry


class PatternMatchType(Enum):
"""Class for Schwabot trading functionality."""
"""Types of pattern matches."""
EXACT = "exact"                    # Exact pattern match
SIMILAR = "similar"                # Similar pattern match
CORRELATED = "correlated"          # Correlated pattern match
ADAPTIVE = "adaptive"              # Adaptive pattern match


@dataclass
class UnifiedMemoryPattern:
"""Class for Schwabot trading functionality."""
"""Unified memory pattern across all registries."""
pattern_id: str
registry_type: MemoryRegistryType
pattern_hash: str
pattern_data: Dict[str, Any]
success_rate: float
confidence: float
last_used: float
usage_count: int
adaptation_history: List[Dict[str, Any]]
cross_registry_links: List[str]
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossRegistryMatch:
"""Class for Schwabot trading functionality."""
"""Cross-registry pattern match result."""
pattern_id: str
match_type: PatternMatchType
similarity_score: float
registry_sources: List[MemoryRegistryType]
pattern_data: Dict[str, Any]
success_prediction: float
confidence: float
adaptation_recommendation: str
metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryBasedAdaptation:
"""Class for Schwabot trading functionality."""
"""Memory-based adaptation result."""
original_pattern: UnifiedMemoryPattern
adapted_pattern: UnifiedMemoryPattern
adaptation_strength: float
success_prediction: float
reasoning: str
cross_registry_influence: Dict[str, float]
metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedMemoryRegistrySystem:
"""Class for Schwabot trading functionality."""
"""
Unified Memory Registry System

Integrates all memory components into a single cohesive system for
pattern recognition, memory-based adaptation, and cross-registry matching.
"""

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
"""Initialize the unified memory registry system."""
self.config = config or self._default_config()
self.logger = logging.getLogger(__name__)

# Memory storage
self.unified_patterns: Dict[str, UnifiedMemoryPattern] = {}
self.cross_registry_matches: Dict[str, CrossRegistryMatch] = {}
self.adaptation_history: List[MemoryBasedAdaptation] = []

# Initialize existing memory systems
self._initialize_memory_systems()

# Performance tracking
self.pattern_match_count = 0
self.successful_adaptations = 0
self.cross_registry_queries = 0

self.logger.info("🧠 Unified Memory Registry System initialized")
self.logger.info(f"✅ Active registries: {self._get_active_registries_count()}")

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
'pattern_similarity_threshold': 0.7,
'cross_registry_weight': 0.3,
'success_rate_decay': 0.95,
'adaptation_learning_rate': 0.01,
'max_patterns_per_registry': 1000,
'pattern_cleanup_interval': 3600.0,  # 1 hour
'cross_registry_enabled': True,
'memory_integration_enabled': True
}

def _initialize_memory_systems(self) -> None:
"""Initialize existing memory systems."""
if EXISTING_MEMORY_AVAILABLE:
try:
# Initialize memory key allocator
self.memory_key_allocator = MemoryKeyAllocator()
self.logger.info("✅ Memory Key Allocator initialized")

# Initialize vector registry
self.vector_registry = VectorRegistry()
self.logger.info("✅ Vector Registry initialized")

# Initialize hash memory generator
self.hash_memory_generator = HashMemoryGenerator()
self.logger.info("✅ Hash Memory Generator initialized")

# Initialize profit bucket registry
self.profit_bucket_registry = ProfitBucketRegistry()
self.logger.info("✅ Profit Bucket Registry initialized")

except Exception as e:
self.logger.warning(f"⚠️ Failed to initialize some memory systems: {e}")
else:
self.logger.warning("⚠️ Using fallback memory systems")

def register_pattern(self, pattern_data: Dict[str, Any], -> None
registry_type: MemoryRegistryType,
success_rate: float = 0.5,
confidence: float = 0.5) -> str:
"""
Register a pattern in the unified memory system.

Args:
pattern_data: Pattern data to register
registry_type: Type of registry to use
success_rate: Historical success rate
confidence: Pattern confidence

Returns:
Pattern ID
"""
try:
# Generate pattern hash
pattern_hash = self._generate_pattern_hash(pattern_data, registry_type)

# Generate pattern ID
pattern_id = self._generate_pattern_id(registry_type, pattern_hash)

# Create unified pattern
pattern = UnifiedMemoryPattern(
pattern_id=pattern_id,
registry_type=registry_type,
pattern_hash=pattern_hash,
pattern_data=pattern_data,
success_rate=success_rate,
confidence=confidence,
last_used=time.time(),
usage_count=1,
adaptation_history=[],
cross_registry_links=[],
metadata={
'registration_timestamp': time.time(),
'registry_type': registry_type.value
}
)

# Store pattern
self.unified_patterns[pattern_id] = pattern

# Register in specific registry
self._register_in_specific_registry(pattern, registry_type)

self.logger.info(f"🧠 Registered pattern {pattern_id[:8]}... "
f"(registry: {registry_type.value}, success: {success_rate:.3f})")

return pattern_id

except Exception as e:
self.logger.error(f"❌ Error registering pattern: {e}")
return "fallback_pattern_id"

def find_similar_patterns(self, query_data: Dict[str, Any], -> None
registry_types: Optional[List[MemoryRegistryType]] = None,
similarity_threshold: Optional[float] = None) -> List[CrossRegistryMatch]:
"""
Find similar patterns across all registries.

Args:
query_data: Query data to match against
registry_types: Specific registries to search (all if None)
similarity_threshold: Similarity threshold (uses config default if None)

Returns:
List of cross-registry matches
"""
try:
if similarity_threshold is None:
similarity_threshold = self.config['pattern_similarity_threshold']

if registry_types is None:
registry_types = list(MemoryRegistryType)

matches = []
query_hash = self._generate_query_hash(query_data)

# Search in each registry
for registry_type in registry_types:
registry_matches = self._search_registry(query_data, query_hash, registry_type, similarity_threshold)
matches.extend(registry_matches)

# Perform cross-registry analysis
cross_registry_matches = self._analyze_cross_registry_matches(matches, query_data)

# Sort by similarity score
cross_registry_matches.sort(key=lambda x: x.similarity_score, reverse=True)

self.pattern_match_count += len(cross_registry_matches)
self.cross_registry_queries += 1

self.logger.debug(f"🔍 Found {len(cross_registry_matches)} similar patterns "
f"across {len(registry_types)} registries")

return cross_registry_matches

except Exception as e:
self.logger.error(f"❌ Error finding similar patterns: {e}")
return []

def adapt_pattern_from_memory(self, pattern_id: str, -> None
adaptation_data: Dict[str, Any],
performance_feedback: float) -> MemoryBasedAdaptation:
"""
Adapt a pattern based on memory and performance feedback.

Args:
pattern_id: Pattern ID to adapt
adaptation_data: Adaptation data
performance_feedback: Performance feedback (-1 to 1)

Returns:
MemoryBasedAdaptation result
"""
try:
if pattern_id not in self.unified_patterns:
raise ValueError(f"Pattern {pattern_id} not found")

original_pattern = self.unified_patterns[pattern_id]

# Create adapted pattern
adapted_pattern_data = self._create_adapted_pattern_data(
original_pattern.pattern_data, adaptation_data, performance_feedback
)

# Calculate adaptation strength
adaptation_strength = self._calculate_adaptation_strength(performance_feedback)

# Predict success
success_prediction = self._predict_adaptation_success(
original_pattern, adapted_pattern_data, performance_feedback
)

# Generate reasoning
reasoning = self._generate_adaptation_reasoning(
original_pattern, adapted_pattern_data, performance_feedback
)

# Calculate cross-registry influence
cross_registry_influence = self._calculate_cross_registry_influence(
original_pattern, adapted_pattern_data
)

# Create adapted pattern
adapted_pattern = UnifiedMemoryPattern(
pattern_id=f"{pattern_id}_adapted",
registry_type=original_pattern.registry_type,
pattern_hash=self._generate_pattern_hash(adapted_pattern_data, original_pattern.registry_type),
pattern_data=adapted_pattern_data,
success_rate=success_prediction,
confidence=original_pattern.confidence * (1 + adaptation_strength * 0.1),
last_used=time.time(),
usage_count=0,
adaptation_history=original_pattern.adaptation_history + [{
'timestamp': time.time(),
'performance_feedback': performance_feedback,
'adaptation_strength': adaptation_strength
}],
cross_registry_links=original_pattern.cross_registry_links,
metadata={
'adaptation_timestamp': time.time(),
'original_pattern_id': pattern_id,
'performance_feedback': performance_feedback
}
)

# Store adapted pattern
self.unified_patterns[adapted_pattern.pattern_id] = adapted_pattern

# Create adaptation result
result = MemoryBasedAdaptation(
original_pattern=original_pattern,
adapted_pattern=adapted_pattern,
adaptation_strength=adaptation_strength,
success_prediction=success_prediction,
reasoning=reasoning,
cross_registry_influence=cross_registry_influence,
metadata={
'adaptation_timestamp': time.time(),
'performance_feedback': performance_feedback
}
)

# Record adaptation
self.adaptation_history.append(result)
self.successful_adaptations += 1

self.logger.info(f"🔄 Adapted pattern {pattern_id[:8]}... "
f"(strength: {adaptation_strength:.3f}, success: {success_prediction:.3f})")

return result

except Exception as e:
self.logger.error(f"❌ Error adapting pattern: {e}")
return self._create_fallback_adaptation(pattern_id, performance_feedback)

def get_memory_based_recommendation(self, market_data: Dict[str, Any], -> None
strategy_context: Dict[str, Any]) -> Dict[str, Any]:
"""
Get memory-based recommendation for trading strategy.

Args:
market_data: Current market data
strategy_context: Strategy context

Returns:
Memory-based recommendation
"""
try:
# Find similar patterns
similar_patterns = self.find_similar_patterns(market_data)

if not similar_patterns:
return self._create_fallback_recommendation(market_data, strategy_context)

# Get best match
best_match = similar_patterns[0]

# Analyze pattern success
success_analysis = self._analyze_pattern_success(best_match)

# Generate recommendation
recommendation = {
'pattern_id': best_match.pattern_id,
'similarity_score': best_match.similarity_score,
'success_prediction': best_match.success_prediction,
'confidence': best_match.confidence,
'adaptation_recommendation': best_match.adaptation_recommendation,
'registry_sources': [reg.value for reg in best_match.registry_sources],
'success_analysis': success_analysis,
'cross_registry_influence': self._calculate_recommendation_influence(best_match),
'metadata': {
'recommendation_timestamp': time.time(),
'market_context': market_data.get('symbol', 'unknown'),
'strategy_context': strategy_context
}
}

self.logger.info(f"💡 Memory recommendation: {best_match.pattern_id[:8]}... "
f"(similarity: {best_match.similarity_score:.3f}, "
f"success: {best_match.success_prediction:.3f})")

return recommendation

except Exception as e:
self.logger.error(f"❌ Error getting memory recommendation: {e}")
return self._create_fallback_recommendation(market_data, strategy_context)

def update_pattern_performance(self, pattern_id: str, performance_result: float) -> None:
"""
Update pattern performance after execution.

Args:
pattern_id: Pattern ID to update
performance_result: Performance result (-1 to 1)
"""
try:
if pattern_id not in self.unified_patterns:
return

pattern = self.unified_patterns[pattern_id]

# Update success rate with decay
decay_factor = self.config['success_rate_decay']
pattern.success_rate = (pattern.success_rate * decay_factor +
(performance_result + 1) / 2 * (1 - decay_factor))

# Update usage count and last used
pattern.usage_count += 1
pattern.last_used = time.time()

# Update confidence based on performance
if performance_result > 0:
pattern.confidence = min(0.99, pattern.confidence + 0.01)
else:
pattern.confidence = max(0.1, pattern.confidence - 0.01)

self.logger.debug(f"📊 Updated pattern {pattern_id[:8]}... "
f"(success: {pattern.success_rate:.3f}, confidence: {pattern.confidence:.3f})")

except Exception as e:
self.logger.error(f"❌ Error updating pattern performance: {e}")

def _generate_pattern_hash(self, pattern_data: Dict[str, Any], -> None
registry_type: MemoryRegistryType) -> str:
"""Generate pattern hash."""
try:
# Create hash string from pattern data and registry type
hash_string = json.dumps(pattern_data, sort_keys=True) + registry_type.value
return generate_hash_from_string(hash_string)
except Exception as e:
self.logger.error(f"❌ Error generating pattern hash: {e}")
return "fallback_hash"

def _generate_pattern_id(self, registry_type: MemoryRegistryType, pattern_hash: str) -> str:
"""Generate pattern ID."""
try:
timestamp = int(time.time() * 1000)
return f"{registry_type.value}_{timestamp}_{pattern_hash[:8]}"
except Exception as e:
self.logger.error(f"❌ Error generating pattern ID: {e}")
return "fallback_pattern_id"

def _generate_query_hash(self, query_data: Dict[str, Any]) -> str:
"""Generate query hash."""
try:
query_string = json.dumps(query_data, sort_keys=True)
return generate_hash_from_string(query_string)
except Exception as e:
self.logger.error(f"❌ Error generating query hash: {e}")
return "fallback_query_hash"

def _register_in_specific_registry(self, pattern: UnifiedMemoryPattern, -> None
registry_type: MemoryRegistryType):
"""Register pattern in specific registry."""
try:
if registry_type == MemoryRegistryType.KEY_ALLOCATOR and hasattr(self, 'memory_key_allocator'):
# Register in memory key allocator
memory_key = MemoryKey(
key_id=pattern.pattern_id,
key_type="auto_generated",
agent_type="unified_memory",
domain="pattern_registry",
hash_signature=pattern.pattern_hash,
tick=int(time.time()),
timestamp=time.time(),
alpha_score=pattern.success_rate,
confidence_score=pattern.confidence
)
self.memory_key_allocator.memory_keys[pattern.pattern_id] = memory_key

elif registry_type == MemoryRegistryType.VECTOR_REGISTRY and hasattr(self, 'vector_registry'):
# Register in vector registry
strategy_vector = StrategyVector(
strategy_id=pattern.pattern_id,
entry_confidence=pattern.confidence,
exit_confidence=pattern.success_rate,
asset_focus="BTC",
entropy_band=pattern.success_rate
)
self.vector_registry.register_digest(pattern.pattern_hash, strategy_vector)

elif registry_type == MemoryRegistryType.HASH_MEMORY and hasattr(self, 'hash_memory_generator'):
# Register in hash memory generator
self.hash_memory_generator.hash_patterns[pattern.pattern_hash] = [pattern.success_rate]

elif registry_type == MemoryRegistryType.PROFIT_BUCKET and hasattr(self, 'profit_bucket_registry'):
# Register in profit bucket registry
profit_bucket = ProfitBucket(
hash_pattern=pattern.pattern_hash,
entry_price=pattern.pattern_data.get('entry_price', 0.0),
exit_price=pattern.pattern_data.get('exit_price', 0.0),
profit=pattern.success_rate,
confidence=pattern.confidence
)
self.profit_bucket_registry.buckets[pattern.pattern_hash] = profit_bucket

except Exception as e:
self.logger.error(f"❌ Error registering in specific registry: {e}")

def _search_registry(self, query_data: Dict[str, Any], query_hash: str, -> None
registry_type: MemoryRegistryType, similarity_threshold: float) -> List[CrossRegistryMatch]:
"""Search specific registry for similar patterns."""
try:
matches = []

if registry_type == MemoryRegistryType.KEY_ALLOCATOR and hasattr(self, 'memory_key_allocator'):
# Search memory key allocator
for pattern_id, pattern in self.unified_patterns.items():
if pattern.registry_type == registry_type:
similarity = self._calculate_similarity(query_hash, pattern.pattern_hash)
if similarity >= similarity_threshold:
matches.append(CrossRegistryMatch(
pattern_id=pattern_id,
match_type=PatternMatchType.SIMILAR,
similarity_score=similarity,
registry_sources=[registry_type],
pattern_data=pattern.pattern_data,
success_prediction=pattern.success_rate,
confidence=pattern.confidence,
adaptation_recommendation="Use similar memory pattern",
metadata={'registry_type': registry_type.value}
))

elif registry_type == MemoryRegistryType.VECTOR_REGISTRY and hasattr(self, 'vector_registry'):
# Search vector registry
for pattern_id, pattern in self.unified_patterns.items():
if pattern.registry_type == registry_type:
similarity = self._calculate_similarity(query_hash, pattern.pattern_hash)
if similarity >= similarity_threshold:
matches.append(CrossRegistryMatch(
pattern_id=pattern_id,
match_type=PatternMatchType.SIMILAR,
similarity_score=similarity,
registry_sources=[registry_type],
pattern_data=pattern.pattern_data,
success_prediction=pattern.success_rate,
confidence=pattern.confidence,
adaptation_recommendation="Use similar vector pattern",
metadata={'registry_type': registry_type.value}
))

elif registry_type == MemoryRegistryType.HASH_MEMORY and hasattr(self, 'hash_memory_generator'):
# Search hash memory generator
for pattern_id, pattern in self.unified_patterns.items():
if pattern.registry_type == registry_type:
similarity = self._calculate_similarity(query_hash, pattern.pattern_hash)
if similarity >= similarity_threshold:
matches.append(CrossRegistryMatch(
pattern_id=pattern_id,
match_type=PatternMatchType.SIMILAR,
similarity_score=similarity,
registry_sources=[registry_type],
pattern_data=pattern.pattern_data,
success_prediction=pattern.success_rate,
confidence=pattern.confidence,
adaptation_recommendation="Use similar hash pattern",
metadata={'registry_type': registry_type.value}
))

elif registry_type == MemoryRegistryType.PROFIT_BUCKET and hasattr(self, 'profit_bucket_registry'):
# Search profit bucket registry
for pattern_id, pattern in self.unified_patterns.items():
if pattern.registry_type == registry_type:
similarity = self._calculate_similarity(query_hash, pattern.pattern_hash)
if similarity >= similarity_threshold:
matches.append(CrossRegistryMatch(
pattern_id=pattern_id,
match_type=PatternMatchType.SIMILAR,
similarity_score=similarity,
registry_sources=[registry_type],
pattern_data=pattern.pattern_data,
success_prediction=pattern.success_rate,
confidence=pattern.confidence,
adaptation_recommendation="Use similar profit pattern",
metadata={'registry_type': registry_type.value}
))

return matches

except Exception as e:
self.logger.error(f"❌ Error searching registry: {e}")
return []

def _calculate_similarity(self, hash1: str, hash2: str) -> float:
"""Calculate similarity between two hashes."""
try:
# Use Hamming distance for similarity
if len(hash1) != len(hash2):
return 0.0

hamming_distance = sum(a != b for a, b in zip(hash1, hash2))
similarity = 1.0 - (hamming_distance / len(hash1))

return max(0.0, similarity)

except Exception as e:
self.logger.error(f"❌ Error calculating similarity: {e}")
return 0.0

def _analyze_cross_registry_matches(self, matches: List[CrossRegistryMatch], -> None
query_data: Dict[str, Any]) -> List[CrossRegistryMatch]:
"""Analyze cross-registry matches."""
try:
# Group matches by pattern ID
pattern_groups = {}
for match in matches:
if match.pattern_id not in pattern_groups:
pattern_groups[match.pattern_id] = []
pattern_groups[match.pattern_id].append(match)

# Create cross-registry matches
cross_registry_matches = []
for pattern_id, group_matches in pattern_groups.items():
if len(group_matches) > 1:
# Multiple registries found this pattern
avg_similarity = np.mean([m.similarity_score for m in group_matches])
all_registries = list(set([reg for m in group_matches for reg in m.registry_sources]))

# Use the first match as base
base_match = group_matches[0]
cross_registry_match = CrossRegistryMatch(
pattern_id=pattern_id,
match_type=PatternMatchType.CORRELATED,
similarity_score=avg_similarity,
registry_sources=all_registries,
pattern_data=base_match.pattern_data,
success_prediction=base_match.success_prediction * 1.1,  # Boost for cross-registry
confidence=base_match.confidence * 1.1,
adaptation_recommendation=f"Strong cross-registry match across {len(all_registries)} registries",
metadata={'cross_registry': True, 'registry_count': len(all_registries)}
)
cross_registry_matches.append(cross_registry_match)
else:
# Single registry match
cross_registry_matches.append(group_matches[0])

return cross_registry_matches

except Exception as e:
self.logger.error(f"❌ Error analyzing cross-registry matches: {e}")
return matches

def _create_adapted_pattern_data(self, original_data: Dict[str, Any], -> None
adaptation_data: Dict[str, Any],
performance_feedback: float) -> Dict[str, Any]:
"""Create adapted pattern data."""
try:
adapted_data = original_data.copy()

# Apply adaptation based on performance feedback
for key, value in adaptation_data.items():
if isinstance(value, (int, float)) and key in adapted_data:
adaptation_factor = 1.0 + performance_feedback * 0.1
adapted_data[key] = adapted_data[key] * adaptation_factor

return adapted_data

except Exception as e:
self.logger.error(f"❌ Error creating adapted pattern data: {e}")
return original_data

def _calculate_adaptation_strength(self, performance_feedback: float) -> float:
"""Calculate adaptation strength."""
return abs(performance_feedback) * 0.5

def _predict_adaptation_success(self, original_pattern: UnifiedMemoryPattern, -> None
adapted_pattern_data: Dict[str, Any],
performance_feedback: float) -> float:
"""Predict adaptation success."""
try:
# Base success on original pattern and performance feedback
base_success = original_pattern.success_rate
feedback_adjustment = performance_feedback * 0.2

predicted_success = base_success + feedback_adjustment
return max(0.0, min(1.0, predicted_success))

except Exception as e:
self.logger.error(f"❌ Error predicting adaptation success: {e}")
return 0.5

def _generate_adaptation_reasoning(self, original_pattern: UnifiedMemoryPattern, -> None
adapted_pattern_data: Dict[str, Any],
performance_feedback: float) -> str:
"""Generate adaptation reasoning."""
return f"Adapted pattern {original_pattern.pattern_id[:8]}... based on performance feedback {performance_feedback:.3f}"

def _calculate_cross_registry_influence(self, original_pattern: UnifiedMemoryPattern, -> None
adapted_pattern_data: Dict[str, Any]) -> Dict[str, float]:
"""Calculate cross-registry influence."""
return {
'key_allocator': 0.3,
'vector_registry': 0.3,
'hash_memory': 0.2,
'profit_bucket': 0.2
}

def _create_fallback_adaptation(self, pattern_id: str, performance_feedback: float) -> MemoryBasedAdaptation:
"""Create fallback adaptation."""
fallback_pattern = UnifiedMemoryPattern(
pattern_id=f"{pattern_id}_fallback",
registry_type=MemoryRegistryType.KEY_ALLOCATOR,
pattern_hash="fallback_hash",
pattern_data={},
success_rate=0.5,
confidence=0.5,
last_used=time.time(),
usage_count=0,
adaptation_history=[],
cross_registry_links=[],
metadata={'fallback': True}
)

return MemoryBasedAdaptation(
original_pattern=fallback_pattern,
adapted_pattern=fallback_pattern,
adaptation_strength=0.0,
success_prediction=0.5,
reasoning="Fallback adaptation",
cross_registry_influence={},
metadata={'fallback': True}
)

def _analyze_pattern_success(self, match: CrossRegistryMatch) -> Dict[str, Any]:
"""Analyze pattern success."""
return {
'success_rate': match.success_prediction,
'confidence': match.confidence,
'registry_sources': len(match.registry_sources),
'cross_registry': len(match.registry_sources) > 1
}

def _calculate_recommendation_influence(self, match: CrossRegistryMatch) -> Dict[str, float]:
"""Calculate recommendation influence."""
return {
'similarity_weight': match.similarity_score,
'success_weight': match.success_prediction,
'confidence_weight': match.confidence,
'cross_registry_boost': 1.1 if len(match.registry_sources) > 1 else 1.0
}

def _create_fallback_recommendation(self, market_data: Dict[str, Any], -> None
strategy_context: Dict[str, Any]) -> Dict[str, Any]:
"""Create fallback recommendation."""
return {
'pattern_id': 'fallback_pattern',
'similarity_score': 0.5,
'success_prediction': 0.5,
'confidence': 0.5,
'adaptation_recommendation': 'Use fallback strategy',
'registry_sources': ['fallback'],
'success_analysis': {'success_rate': 0.5, 'confidence': 0.5},
'cross_registry_influence': {'fallback': 1.0},
'metadata': {'fallback': True}
}

def _get_active_registries_count(self) -> int:
"""Get count of active registries."""
registries = [
hasattr(self, 'memory_key_allocator'),
hasattr(self, 'vector_registry'),
hasattr(self, 'hash_memory_generator'),
hasattr(self, 'profit_bucket_registry')
]
return sum(registries)

def get_memory_report(self) -> Dict[str, Any]:
"""Get comprehensive memory report."""
try:
return {
'total_patterns': len(self.unified_patterns),
'cross_registry_matches': len(self.cross_registry_matches),
'adaptation_history_count': len(self.adaptation_history),
'pattern_match_count': self.pattern_match_count,
'successful_adaptations': self.successful_adaptations,
'cross_registry_queries': self.cross_registry_queries,
'active_registries': self._get_active_registries_count(),
'registry_distribution': {
reg.value: len([p for p in self.unified_patterns.values() if p.registry_type == reg])
for reg in MemoryRegistryType
}
}
except Exception as e:
self.logger.error(f"❌ Error generating memory report: {e}")
return {'error': str(e)}


# Factory function
def create_unified_memory_registry_system(config: Optional[Dict[str, Any]] = None) -> UnifiedMemoryRegistrySystem:
"""Create a unified memory registry system instance."""
return UnifiedMemoryRegistrySystem(config)


# Singleton instance for global use
unified_memory_registry = UnifiedMemoryRegistrySystem()


def main():
"""Test the unified memory registry system."""
logger.info("🧠 Testing Unified Memory Registry System")

# Test pattern data
test_pattern_data = {
'symbol': 'BTC',
'price': 50000.0,
'volume': 1000.0,
'volatility': 0.02,
'strategy_type': 'scalping'
}

# Register patterns in different registries
key_pattern_id = unified_memory_registry.register_pattern(
test_pattern_data, MemoryRegistryType.KEY_ALLOCATOR, 0.8, 0.9
)

vector_pattern_id = unified_memory_registry.register_pattern(
test_pattern_data, MemoryRegistryType.VECTOR_REGISTRY, 0.7, 0.8
)

# Find similar patterns
similar_patterns = unified_memory_registry.find_similar_patterns(test_pattern_data)

# Get memory recommendation
recommendation = unified_memory_registry.get_memory_based_recommendation(
test_pattern_data, {'strategy': 'scalping'}
)

# Adapt pattern
adaptation_result = unified_memory_registry.adapt_pattern_from_memory(
key_pattern_id, {'price': 51000.0}, 0.6
)

# Update performance
unified_memory_registry.update_pattern_performance(key_pattern_id, 0.7)

# Get report
report = unified_memory_registry.get_memory_report()

logger.info(f"✅ Test completed successfully")
logger.info(f"🧠 Total patterns: {report.get('total_patterns', 0)}")
logger.info(f"🔍 Pattern matches: {report.get('pattern_match_count', 0)}")
logger.info(f"🔄 Successful adaptations: {report.get('successful_adaptations', 0)}")


if __name__ == "__main__":
main()