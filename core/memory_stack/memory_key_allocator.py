# -*- coding: utf-8 -*-\nfrom __future__ import annotations

# #!/usr/bin/env python3
"""
Memory Key Allocator - Strategy Memory Management System
=======================================================

Assigns logic memory keys to strategies for tracking and lookup.
Provides intelligent memory key allocation for the Schwabot trading system.
"""


import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from core.utils.windows_cli_compatibility import (
    safe_print, safe_format_error, log_safe, WindowsCliCompatibilityHandler


logger=logging.getLogger(__name__)


class KeyType(Enum):


    """Memory key types for different strategies."""
MOMENTUM="momentum"
VOLATILITY="volatility"
ENTROPY="entropy"
RESONANCE="resonance"
PHASE="phase"
GHOST="ghost"
TENSOR="tensor"
PROFIT="profit"
RISK="risk"
EXECUTION="execution"


@ dataclass
class MemoryKey:


    """Represents a memory key for strategy tracking."""
key_id: str
strategy_name: str
key_type: KeyType
hash_signature: str
confidence_score: float
allocation_timestamp: datetime
last_accessed: datetime
access_count: int=0
metadata: Dict[str, Any]=field(default_factory=dict)
    performance_metrics: Dict[str, float]=field(default_factory=dict)


@ dataclass
class StrategyAllocation:


    """Represents a strategy allocation with memory keys."""
allocation_id: str
strategy_name: str
allocated_keys: List[MemoryKey]
allocation_score: float
timestamp: datetime
status: str="active"
metadata: Dict[str, Any]=field(default_factory=dict)


class MemoryKeyAllocator:


    """
Memory Key Allocator for Strategy Management.

This allocator assigns intelligent memory keys to strategies based on
their characteristics, performance, and requirements for optimal tracking.
"""

def __init__(self):


    pass
    pass
        """Initialize the memory key allocator."""
self.memory_keys: Dict[str, MemoryKey]={}
self.strategy_allocations: Dict[str, StrategyAllocation]={}
self.key_type_allocations: Dict[KeyType, List[str]]={key_type: [] for key_type in KeyType}

        # Allocation parameters
self.max_keys_per_strategy=5
self.key_confidence_threshold=0.6
self.allocation_score_threshold=0.7
self.key_expiry_days=30

        # Performance tracking
self.total_allocations=0
self.successful_allocations=0
self.allocation_success_rate=0.0

        # CLI compatibility
self.cli_handler=WindowsCliCompatibilityHandler()

logger.info("Memory Key Allocator initialized")

def assign(self, strategy: str) -> str:


    pass
    pass
        """
Assign memory keys to a strategy.

Args:
strategy: Strategy name

Returns:
Allocation ID
"""
        try:
    pass
    pass
start_time=time.time()

            # Generate strategy hash
strategy_hash=self._generate_strategy_hash(strategy)

            # Determine key types for strategy
key_types=self._determine_key_types(strategy, strategy_hash)

            # Allocate memory keys
allocated_keys=[]
            for key_type in key_types:
memory_key=self._create_memory_key(strategy, key_type, strategy_hash)
                if memory_key:
allocated_keys.append(memory_key)
                    self.memory_keys[memory_key.key_id]=memory_key
self.key_type_allocations[key_type].append(memory_key.key_id)

            # Calculate allocation score
allocation_score=self._calculate_allocation_score(allocated_keys, strategy_hash)

            # Create strategy allocation
allocation=StrategyAllocation(
                allocation_id=self._generate_allocation_id(strategy),
                strategy_name=strategy,
allocated_keys=allocated_keys,
allocation_score=allocation_score,
timestamp=datetime.now()


self.strategy_allocations[allocation.allocation_id]=allocation

            # Update performance metrics
self.total_allocations += 1
            if allocation_score >= self.allocation_score_threshold:
self.successful_allocations += 1
self.allocation_success_rate=self.successful_allocations / self.total_allocations

execution_time=time.time() - start_time
            logger.info(f"Allocation completed in {execution_time:.3f}s - Score: {allocation_score:.3f}")

            return allocation.allocation_id

        except Exception as e:
error_msg=safe_format_error(e, "MemoryKeyAllocator.assign")
            logger.error(error_msg)
            return f"failed_allocation_{int(time.time())}"

def _generate_strategy_hash(self, strategy: str) -> str:


    pass
    pass
        """
Generate hash for strategy.

Args:
strategy: Strategy name

Returns:
Strategy hash
"""
        try:
    pass
    pass
            # Create hash input with timestamp for uniqueness
hash_input=f"{strategy}_{int(time.time())}"
            hash_result=hashlib.sha256(hash_input.encode()).hexdigest()
            return hash_result[:16]  # Return first 16 characters

        except Exception as e:
logger.error(f"Strategy hash generation failed: {e}")
            return "0000000000000000"

def _determine_key_types(self, strategy: str, strategy_hash: str) -> List[KeyType]:


    pass
    pass
        """
Determine appropriate key types for strategy.

Args:
strategy: Strategy name
strategy_hash: Strategy hash

Returns:
List of key types
"""
        try:
    pass
    pass
key_types=[]

            # Convert hash to numeric values for analysis
hash_bytes=bytes.fromhex(strategy_hash)
            hash_array=np.frombuffer(hash_bytes, dtype=np.uint8)

            # Analyze strategy characteristics
entropy=self._calculate_hash_entropy(hash_array)
            frequency=self._calculate_hash_frequency(hash_array)
            phase=self._calculate_hash_phase(hash_array)

            # Determine key types based on characteristics
            if entropy > 0.7:
key_types.append(KeyType.ENTROPY)

            if frequency > 0.5:
key_types.append(KeyType.MOMENTUM)

            if phase > 0.6:
key_types.append(KeyType.PHASE)

            # Add core key types
key_types.extend([KeyType.RISK, KeyType.EXECUTION])

            # Add specialized key types based on strategy name
strategy_lower=strategy.lower()
            if "ghost" in strategy_lower:
key_types.append(KeyType.GHOST)
            if "tensor" in strategy_lower:
key_types.append(KeyType.TENSOR)
            if "profit" in strategy_lower:
key_types.append(KeyType.PROFIT)
            if "volatility" in strategy_lower:
key_types.append(KeyType.VOLATILITY)
            if "resonance" in strategy_lower:
key_types.append(KeyType.RESONANCE)

            # Limit number of key types
            if len(key_types) > self.max_keys_per_strategy:
                key_types=key_types[:self.max_keys_per_strategy]

            # Ensure unique key types
key_types=list(set(key_types))

            return key_types

        except Exception as e:
logger.error(f"Key type determination failed: {e}")
            return [KeyType.RISK, KeyType.EXECUTION]

def _calculate_hash_entropy(self, hash_array: NDArray) -> float:


    pass
    pass
        """Calculate entropy of hash array."""
        try:
    pass
    pass
unique_values=np.unique(hash_array)
            if len(unique_values) == 1:
                return 0.0

            # Calculate normalized entropy
entropy = -np.sum(np.bincount(hash_array) / len(hash_array) *)
                            np.log2(np.bincount(hash_array) / len(hash_array) + 1e-10))
            max_entropy=np.log2(len(unique_values))

            return float(entropy / max_entropy) if max_entropy > 0 else 0.0
        except Exception:
            return 0.5

def _calculate_hash_frequency(self, hash_array: NDArray) -> float:


    pass
    pass
        """Calculate frequency characteristic of hash array."""
        try:
    pass
    pass
            # Use FFT to find dominant frequency
fft_result=np.fft.fft(hash_array)
            frequencies=np.abs(fft_result)

            # Find dominant frequency
dominant_freq_idx=np.argmax(frequencies[1:]) + 1
            dominant_freq=dominant_freq_idx / len(hash_array)

            return float(dominant_freq)
        except Exception:
            return 0.5

def _calculate_hash_phase(self, hash_array: NDArray) -> float:


    pass
    pass
        """Calculate phase characteristic of hash array."""
        try:
    pass
    pass
            # Use circular statistics for phase
angles=2 * np.pi * hash_array / 256
mean_angle=np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

            # Normalize to [0, 2π]
phase=(mean_angle + 2 * np.pi) % (2 * np.pi)
            return float(phase / (2 * np.pi))
        except Exception:
            return 0.5

def _create_memory_key(self, strategy: str, key_type: KeyType, strategy_hash: str) -> Optional[MemoryKey]:


    pass
    pass
        """
Create a memory key for strategy and key type.

Args:
strategy: Strategy name
key_type: Key type
strategy_hash: Strategy hash

Returns:
MemoryKey object or None
"""
        try:
    pass
    pass
            # Generate key-specific hash
key_hash=self._generate_key_hash(strategy, key_type, strategy_hash)

            # Calculate confidence score
confidence_score=self._calculate_key_confidence(strategy, key_type, key_hash)

            # Only create key if confidence is sufficient
            if confidence_score < self.key_confidence_threshold:
                return None

            # Create memory key
memory_key=MemoryKey(
                key_id=self._generate_key_id(strategy, key_type),
                strategy_name=strategy,
key_type=key_type,
hash_signature=key_hash,
confidence_score=confidence_score,
allocation_timestamp=datetime.now(),
                last_accessed=datetime.now(),
                metadata={
'strategy_hash': strategy_hash,
'key_type_characteristics': self._get_key_type_characteristics(key_type)
                }


            return memory_key

        except Exception as e:
logger.error(f"Memory key creation failed: {e}")
            return None

def _generate_key_hash(self, strategy: str, key_type: KeyType, strategy_hash: str) -> str:


    pass
    pass
        """Generate hash for specific key type."""
        try:
    pass
    pass
hash_input=f"{strategy}_{key_type.value}_{strategy_hash}"
hash_result=hashlib.sha256(hash_input.encode()).hexdigest()
            return hash_result[:16]
        except Exception:
            return "0000000000000000"

def _calculate_key_confidence(self, strategy: str, key_type: KeyType, key_hash: str) -> float:


    pass
    pass
        """Calculate confidence score for memory key."""
        try:
    pass
    pass
confidence_factors=[]

            # Strategy complexity factor
strategy_complexity=len(strategy) / 100.0  # Normalize by expected max length
            confidence_factors.append(min(strategy_complexity, 1.0))

            # Key type relevance factor
key_relevance=self._calculate_key_type_relevance(strategy, key_type)
            confidence_factors.append(key_relevance)

            # Hash quality factor
hash_quality=self._calculate_hash_quality(key_hash)
            confidence_factors.append(hash_quality)

            # Calculate weighted confidence
weights=[0.3, 0.4, 0.3]  # Strategy complexity, key relevance, hash quality
confidence=sum(factor * weight for factor, weight in zip(confidence_factors, weights))

            return float(confidence)

        except Exception:
            return 0.5

def _calculate_key_type_relevance(self, strategy: str, key_type: KeyType) -> float:


    pass
    pass
        """Calculate relevance of key type to strategy."""
        try:
    pass
    pass
strategy_lower=strategy.lower()
            key_type_lower=key_type.value.lower()

            # Check if key type is mentioned in strategy name
            if key_type_lower in strategy_lower:
                return 1.0

            # Check for related terms
related_terms={
KeyType.MOMENTUM: ['momentum', 'trend', 'velocity'],
KeyType.VOLATILITY: ['volatility', 'vol', 'variance'],
KeyType.ENTROPY: ['entropy', 'random', 'chaos'],
KeyType.RESONANCE: ['resonance', 'frequency', 'oscillation'],
KeyType.PHASE: ['phase', 'cycle', 'period'],
KeyType.GHOST: ['ghost', 'phantom', 'shadow'],
KeyType.TENSOR: ['tensor', 'matrix', 'vector'],
KeyType.PROFIT: ['profit', 'gain', 'return'],
KeyType.RISK: ['risk', 'danger', 'uncertainty'],
KeyType.EXECUTION: ['execution', 'trade', 'order']
}

terms=related_terms.get(key_type, [])
            for term in terms:
                if term in strategy_lower:
                    return 0.8

            # Default relevance for core types
            if key_type in [KeyType.RISK, KeyType.EXECUTION]:
                return 0.6

            return 0.3

        except Exception:
            return 0.5

def _calculate_hash_quality(self, key_hash: str) -> float:


    pass
    pass
        """Calculate quality of hash."""
        try:
    pass
    pass
            # Check hash diversity
unique_chars=len(set(key_hash))
            diversity_score=unique_chars / len(key_hash)

            # Check hash balance
char_counts={}
            for char in key_hash:
char_counts[char]=char_counts.get(char, 0) + 1

max_count=max(char_counts.values())
            min_count=min(char_counts.values())
            balance_score=min_count / max_count if max_count > 0 else 0.0

            # Combined quality score
quality=(diversity_score + balance_score) / 2.0
            return float(quality)

        except Exception:
            return 0.5

def _get_key_type_characteristics(self, key_type: KeyType) -> Dict[str, Any]:


    pass
    pass
        """Get characteristics for key type."""
        try:
    pass
    pass
characteristics={
KeyType.MOMENTUM: {
'description': 'Momentum-based strategy tracking',
'update_frequency': 'high',
'memory_requirements': 'medium',
'performance_impact': 'low'
},
KeyType.VOLATILITY: {
'description': 'Volatility-based strategy tracking',
'update_frequency': 'medium',
'memory_requirements': 'high',
'performance_impact': 'medium'
},
KeyType.ENTROPY: {
'description': 'Entropy-based strategy tracking',
'update_frequency': 'low',
'memory_requirements': 'low',
'performance_impact': 'low'
},
KeyType.RESONANCE: {
'description': 'Resonance-based strategy tracking',
'update_frequency': 'high',
'memory_requirements': 'high',
'performance_impact': 'high'
},
KeyType.PHASE: {
'description': 'Phase-based strategy tracking',
'update_frequency': 'medium',
'memory_requirements': 'medium',
'performance_impact': 'medium'
},
KeyType.GHOST: {
'description': 'Ghost-based strategy tracking',
'update_frequency': 'very_high',
'memory_requirements': 'very_high',
'performance_impact': 'very_high'
},
KeyType.TENSOR: {
'description': 'Tensor-based strategy tracking',
'update_frequency': 'high',
'memory_requirements': 'very_high',
'performance_impact': 'high'
},
KeyType.PROFIT: {
'description': 'Profit-based strategy tracking',
'update_frequency': 'medium',
'memory_requirements': 'medium',
'performance_impact': 'low'
},
KeyType.RISK: {
'description': 'Risk-based strategy tracking',
'update_frequency': 'high',
'memory_requirements': 'medium',
'performance_impact': 'medium'
},
KeyType.EXECUTION: {
'description': 'Execution-based strategy tracking',
'update_frequency': 'very_high',
'memory_requirements': 'low',
'performance_impact': 'low'
}
}

            return characteristics.get(key_type, {})

        except Exception:
            return {}

def _calculate_allocation_score(self, allocated_keys: List[MemoryKey], strategy_hash: str) -> float:


    pass
    pass
        """Calculate overall allocation score."""
        try:
    pass
    pass
            if not allocated_keys:
                return 0.0

            # Calculate average confidence
avg_confidence=np.mean([key.confidence_score for key in allocated_keys])

            # Calculate key diversity
key_types=[key.key_type for key in allocated_keys]
diversity_score=len(set(key_types)) / len(key_types)

            # Calculate hash quality
hash_quality=self._calculate_hash_quality(strategy_hash)

            # Weighted combination
weights=[0.5, 0.3, 0.2]  # Confidence, diversity, hash quality
allocation_score=(
                avg_confidence * weights[0] +
diversity_score * weights[1] +
hash_quality * weights[2]


            return float(allocation_score)

        except Exception:
            return 0.5

def _generate_key_id(self, strategy: str, key_type: KeyType) -> str:


    pass
    pass
        """Generate unique key ID."""
        try:
    pass
    pass
timestamp=datetime.now().isoformat()
            return f"key_{strategy}_{key_type.value}_{timestamp}"
        except Exception:
            return f"key_{int(time.time())}"

def _generate_allocation_id(self, strategy: str) -> str:


    pass
    pass
        """Generate unique allocation ID."""
        try:
    pass
    pass
timestamp=datetime.now().isoformat()
            return f"alloc_{strategy}_{timestamp}"
        except Exception:
            return f"alloc_{int(time.time())}"

def get_memory_key(self, key_id: str) -> Optional[MemoryKey]:


    pass
    pass
        """Get memory key by ID."""
        try:
    pass
    pass
memory_key=self.memory_keys.get(key_id)
            if memory_key:
                # Update access statistics
memory_key.last_accessed=datetime.now()
                memory_key.access_count += 1

            return memory_key
        except Exception:
            return None

def get_strategy_allocation(self, allocation_id: str) -> Optional[StrategyAllocation]:


    pass
    pass
        """Get strategy allocation by ID."""
        return self.strategy_allocations.get(allocation_id)

def get_keys_by_type(self, key_type: KeyType) -> List[MemoryKey]:


    pass
    pass
        """Get all memory keys of a specific type."""
        try:
    pass
    pass
key_ids=self.key_type_allocations.get(key_type, [])
            return [self.memory_keys[key_id] for key_id in key_ids if key_id in self.memory_keys]
        except Exception:
            return []

def update_key_performance(self, key_id: str, performance_metrics: Dict[str, float]) -> bool:


    pass
    pass
        """Update performance metrics for a memory key."""
        try:
    pass
    pass
memory_key=self.memory_keys.get(key_id)
            if not memory_key:
                return False

memory_key.performance_metrics.update(performance_metrics)
            memory_key.last_accessed=datetime.now()

            return True
        except Exception:
            return False

def cleanup_expired_keys(self) -> int:


    pass
    pass
        """Clean up expired memory keys."""
        try:
    pass
    pass
cutoff_time=datetime.now()
            expired_keys=[]

            for key_id, memory_key in self.memory_keys.items():
                days_since_allocation=(cutoff_time - memory_key.allocation_timestamp).days
                if days_since_allocation > self.key_expiry_days:
expired_keys.append(key_id)

            # Remove expired keys
            for key_id in expired_keys:
                del self.memory_keys[key_id]

            # Clean up allocations
expired_allocations=[]
            for alloc_id, allocation in self.strategy_allocations.items():
                if not allocation.allocated_keys:  # No keys left
expired_allocations.append(alloc_id)

            for alloc_id in expired_allocations:
                del self.strategy_allocations[alloc_id]

logger.info(f"Cleaned up {len(expired_keys)} expired keys and {len(expired_allocations)} expired allocations")
            return len(expired_keys)

        except Exception as e:
logger.error(f"Cleanup failed: {e}")
            return 0

def get_allocation_statistics(self) -> Dict[str, Any]:


    pass
    pass
        """Get allocation execution statistics."""
        try:
    pass
    pass
            return {
"total_allocations": self.total_allocations,
"successful_allocations": self.successful_allocations,
"success_rate": self.allocation_success_rate,
"total_memory_keys": len(self.memory_keys),
                "total_strategy_allocations": len(self.strategy_allocations),
                "keys_by_type": {key_type.value: len(keys) for key_type, keys in self.key_type_allocations.items()},
                "average_confidence": np.mean([key.confidence_score for key in self.memory_keys.values()]) if self.memory_keys else 0.0,
                "average_allocation_score": np.mean([alloc.allocation_score for alloc in self.strategy_allocations.values()]) if self.strategy_allocations else 0.0
            }
        except Exception:
            return {
"total_allocations": 0,
"successful_allocations": 0,
"success_rate": 0.0,
"total_memory_keys": 0,
"total_strategy_allocations": 0,
"keys_by_type": {},
"average_confidence": 0.0,
"average_allocation_score": 0.0
}


# Convenience functions
def assign_memory_keys(strategy: str) -> str:


    pass
    pass
    """Convenience function to assign memory keys to strategy."""
allocator=MemoryKeyAllocator()
    return allocator.assign(strategy)


def get_memory_key(key_id: str) -> Optional[MemoryKey]:


    pass
    pass
    """Convenience function to get memory key."""
allocator=MemoryKeyAllocator()
    return allocator.get_memory_key(key_id)


if __name__ == "__main__":
    pass
    pass
    # Test the memory key allocator
test_strategies=[
"ghost_momentum_strategy",
"tensor_entropy_resonance",
"volatility_profit_risk",
"phase_execution_ghost",
"simple_risk_management"
]

allocator=MemoryKeyAllocator()

    for strategy in test_strategies:
safe_print(f"\nTesting strategy: {strategy}")

        # Assign memory keys
allocation_id=allocator.assign(strategy)
        safe_print(f"Allocation ID: {allocation_id}")

        # Get allocation details
allocation=allocator.get_strategy_allocation(allocation_id)
        if allocation:
safe_print(f"Allocated keys: {len(allocation.allocated_keys)}")
            safe_print(f"Allocation score: {allocation.allocation_score:.3f}")

            for key in allocation.allocated_keys:
safe_print(f"  - {key.key_type.value}: {key.confidence_score:.3f}")

    # Print statistics
stats=allocator.get_allocation_statistics()
    safe_print(f"\nAllocator Statistics: {stats}")

    # Test cleanup
expired_count=allocator.cleanup_expired_keys()
    safe_print(f"Cleaned up {expired_count} expired keys")
