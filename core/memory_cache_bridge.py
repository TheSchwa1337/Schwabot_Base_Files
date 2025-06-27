# -*- coding: utf-8 -*-
"""
Memory Cache Bridge - Unified Memory Interface for Schwabot System
=================================================================

Provides infinite-reactive memory cache bridge connected to visualized logic cycling,
under SHA-patterned symbolic states. This system measures logic, not time, using
pattern-based switching across cache-volatility resolution.

Key Features:
- Pattern-based timing instead of clock-based switching
- SHA-256 entropy alignment with 2-bit gating + 256-block overlays
- Linguistic syntactic triggers via news/API integration
- Vault bridging conditions with synth-based observability windows
- Stack-layer profit awareness across cache tiers

Memory Architecture:
- Short-term: Fast access, symbolic triggers, API flows
- Mid-term: Strategy mapping, profit vectorization, phase states
- Long-term: Historical patterns, fractal recursion, vault states
"""

import logging
import hashlib
import time
import threading
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Memory tier enumeration for cache management."""
    SHORT = "short"     # Fast access, symbolic triggers
    MID = "mid"         # Strategy mapping, profit vectors
    LONG = "long"       # Historical patterns, vault states
    VAULT = "vault"     # Persistent vault memory
    PATTERN = "pattern" # SHA-256 pattern triggers

class PatternType(Enum):
    """Pattern types for logic-based measurement."""
    SHA_GATE = "sha_gate"           # SHA-256 gate patterns
    SYMBOLIC_TRIGGER = "symbolic"   # Unicode/emoji triggers
    ENTROPY_FLOW = "entropy"        # Entropy-based patterns
    PROFIT_VECTOR = "profit"        # Profit vectorization patterns
    SYNTACTIC_API = "syntactic"     # Language/news API patterns
    FRACTAL_RECURSION = "fractal"   # Fractal pattern states

@dataclass
class MemoryEntry:
    """Memory entry with pattern-based metadata."""
    key: str
    value: Any
    tier: MemoryTier
    pattern_type: PatternType
    sha_signature: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    pattern_strength: float = 1.0
    entropy_level: float = 0.0
    profit_correlation: float = 0.0
    vault_bridge_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternGate:
    """SHA-256 pattern gate for logic switching."""
    pattern_hash: str
    gate_type: str  # "2bit", "8bit", "256bit"
    activation_threshold: float
    current_strength: float
    last_triggered: float
    trigger_count: int = 0
    associated_keys: List[str] = field(default_factory=list)

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    total_entries: int
    hit_rate: float
    pattern_activations: int
    vault_bridges_active: int
    entropy_flow_rate: float
    profit_correlation_avg: float
    memory_efficiency: float
    last_update: float

class MemoryCacheBridge:
    """
    Unified memory interface measuring logic patterns instead of time.
    
    Core Philosophy:
    - Measures logic, not time
    - Pattern-based switching over SHA-256 entropy alignment
    - Vault bridging conditions with observability windows
    - Stack-layer profit awareness across memory tiers
    """
    
    def __init__(self, max_entries_per_tier: int = 10000):
        # Memory storage by tier
        self.short_term: Dict[str, MemoryEntry] = {}
        self.mid_term: Dict[str, MemoryEntry] = {}
        self.long_term: Dict[str, MemoryEntry] = {}
        self.vault_memory: Dict[str, MemoryEntry] = {}
        
        # Pattern gates for logic switching
        self.pattern_gates: Dict[str, PatternGate] = {}
        self.sha_gate_store: Dict[str, Any] = {}
        
        # Global timing map (pattern-indexed, not time-indexed)
        self.global_pattern_map: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_entries_per_tier = max_entries_per_tier
        
        # Threading and performance
        self.lock = threading.RLock()
        self.metrics = CacheMetrics(
            total_entries=0,
            hit_rate=0.0,
            pattern_activations=0,
            vault_bridges_active=0,
            entropy_flow_rate=0.0,
            profit_correlation_avg=0.0,
            memory_efficiency=0.0,
            last_update=time.time()
        )
        
        # Initialize core pattern gates
        self._initialize_pattern_gates()
        
        logger.info("Memory Cache Bridge initialized with pattern-based logic measurement")

    def _initialize_pattern_gates(self):
        """Initialize core SHA-256 pattern gates for logic switching."""
        
        # 2-bit logic gates (primary atomization)
        self.pattern_gates["2bit_primary"] = PatternGate(
            pattern_hash="2bit_gate",
            gate_type="2bit",
            activation_threshold=0.5,
            current_strength=0.0,
            last_triggered=0.0
        )
        
        # 8-bit memory register patterns
        self.pattern_gates["8bit_register"] = PatternGate(
            pattern_hash="8bit_gate", 
            gate_type="8bit",
            activation_threshold=0.65,
            current_strength=0.0,
            last_triggered=0.0
        )
        
        # 256-bit SHA encrypted identity
        self.pattern_gates["256bit_sha"] = PatternGate(
            pattern_hash="256bit_gate",
            gate_type="256bit", 
            activation_threshold=0.75,
            current_strength=0.0,
            last_triggered=0.0
        )

    def update_cache(self, key: str, value: Any, tier: Union[str, MemoryTier] = MemoryTier.MID,
                    pattern_type: PatternType = PatternType.SHA_GATE,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update cache with pattern-based logic measurement.
        
        Args:
            key: Cache key
            value: Value to store
            tier: Memory tier (short/mid/long/vault)
            pattern_type: Type of pattern for logic measurement
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                # Convert string tier to enum
                if isinstance(tier, str):
                    tier = MemoryTier(tier.lower())
                
                # Generate SHA signature for pattern matching
                sha_signature = self._generate_sha_signature(key, value, pattern_type)
                
                # Calculate pattern strength and entropy
                pattern_strength = self._calculate_pattern_strength(value, pattern_type)
                entropy_level = self._calculate_entropy_level(value)
                profit_correlation = self._calculate_profit_correlation(value, metadata)
                
                # Create memory entry
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    tier=tier,
                    pattern_type=pattern_type,
                    sha_signature=sha_signature,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    pattern_strength=pattern_strength,
                    entropy_level=entropy_level,
                    profit_correlation=profit_correlation,
                    vault_bridge_active=self._should_activate_vault_bridge(pattern_strength, entropy_level),
                    metadata=metadata or {}
                )
                
                # Store in appropriate tier
                target_store = self._get_tier_store(tier)
                target_store[key] = entry
                
                # Update pattern gates
                self._update_pattern_gates(sha_signature, pattern_strength, pattern_type)
                
                # Update global pattern map
                self._update_global_pattern_map(key, sha_signature, pattern_type, tier)
                
                # Maintain cache size limits
                self._maintain_cache_limits(tier)
                
                # Update metrics
                self._update_metrics()
                
                logger.debug(f"Cache updated: {key} -> {tier.value} (pattern: {pattern_type.value})")
                return True
                
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
            return False

    def fetch_pattern(self, hash_key: str, pattern_type: Optional[PatternType] = None) -> Optional[Any]:
        """
        Fetch data based on SHA-256 hash pattern matching.
        
        Args:
            hash_key: SHA hash key or pattern
            pattern_type: Optional pattern type filter
            
        Returns:
            Matched value or None
        """
        try:
            with self.lock:
                # Try direct hash lookup first
                direct_match = self.sha_gate_store.get(hash_key[:8])
                if direct_match:
                    return direct_match
                
                # Search across all memory tiers
                all_stores = [self.short_term, self.mid_term, self.long_term, self.vault_memory]
                
                for store in all_stores:
                    for entry in store.values():
                        # Check SHA signature match
                        if entry.sha_signature.startswith(hash_key[:8]):
                            if pattern_type is None or entry.pattern_type == pattern_type:
                                entry.last_accessed = time.time()
                                entry.access_count += 1
                                return entry.value
                
                # Pattern-based fuzzy matching
                best_match = self._find_pattern_match(hash_key, pattern_type)
                if best_match:
                    return best_match.value
                
                return None
                
        except Exception as e:
            logger.error(f"Error fetching pattern: {e}")
            return None

    def resolve_visualization_payload(self) -> Dict[str, Any]:
        """
        Compile waveform-ready payload for GUI state update.
        
        Returns:
            Visualization payload for GUI rendering
        """
        try:
            with self.lock:
                # Calculate current system state
                entropy_drift = self._calculate_system_entropy()
                profit_score = self._calculate_system_profit_score()
                phase_state = self._determine_system_phase()
                
                # Get active pattern gates
                active_gates = [gate for gate in self.pattern_gates.values() 
                              if gate.current_strength > gate.activation_threshold]
                
                # Compile payload
                payload = {
                    'entropy_drift': entropy_drift,
                    'profit_score': profit_score,
                    'phase_state': phase_state,
                    'active_pattern_gates': len(active_gates),
                    'memory_tiers': {
                        'short': len(self.short_term),
                        'mid': len(self.mid_term), 
                        'long': len(self.long_term),
                        'vault': len(self.vault_memory)
                    },
                    'pattern_activations': self.metrics.pattern_activations,
                    'vault_bridges_active': self.metrics.vault_bridges_active,
                    'hash_block': list(self.sha_gate_store.keys())[-1] if self.sha_gate_store else 'N/A',
                    'last_pattern_hash': self._get_latest_pattern_hash(),
                    'system_coherence': self._calculate_system_coherence(),
                    'timestamp': time.time()
                }
                
                return payload
                
        except Exception as e:
            logger.error(f"Error resolving visualization payload: {e}")
            return {'error': str(e), 'timestamp': time.time()}

    def activate_vault_bridge(self, key: str, bridge_type: str = "profit_correlation") -> bool:
        """
        Activate vault bridge for memory entry.
        
        Args:
            key: Memory key to bridge
            bridge_type: Type of vault bridge
            
        Returns:
            Bridge activation success
        """
        try:
            with self.lock:
                # Search for entry across all tiers
                entry = self._find_entry_by_key(key)
                if not entry:
                    return False
                
                # Activate vault bridge
                entry.vault_bridge_active = True
                entry.metadata['bridge_type'] = bridge_type
                entry.metadata['bridge_activated'] = time.time()
                
                # Move to vault memory if high correlation
                if entry.profit_correlation > 0.75:
                    self.vault_memory[key] = entry
                    self._remove_from_tier(key, entry.tier)
                
                self.metrics.vault_bridges_active += 1
                
                logger.info(f"Vault bridge activated for {key} ({bridge_type})")
                return True
                
        except Exception as e:
            logger.error(f"Error activating vault bridge: {e}")
            return False

    def get_pattern_gate_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pattern gates."""
        return {
            gate_id: {
                'pattern_hash': gate.pattern_hash,
                'gate_type': gate.gate_type,
                'current_strength': gate.current_strength,
                'activation_threshold': gate.activation_threshold,
                'is_active': gate.current_strength > gate.activation_threshold,
                'trigger_count': gate.trigger_count,
                'last_triggered': gate.last_triggered,
                'associated_keys_count': len(gate.associated_keys)
            }
            for gate_id, gate in self.pattern_gates.items()
        }

    def inject_syntactic_trigger(self, trigger_text: str, api_source: str = "news") -> str:
        """
        Inject syntactic trigger from news/API for pattern activation.
        
        Args:
            trigger_text: Text trigger from API
            api_source: Source of the trigger
            
        Returns:
            Generated pattern hash
        """
        try:
            # Generate pattern hash from syntactic content
            pattern_hash = hashlib.sha256(f"{trigger_text}_{api_source}_{time.time()}".encode()).hexdigest()
            
            # Store in short-term memory for rapid access
            self.update_cache(
                key=f"syntactic_{pattern_hash[:8]}",
                value={
                    'trigger_text': trigger_text,
                    'api_source': api_source,
                    'pattern_hash': pattern_hash,
                    'injection_time': time.time()
                },
                tier=MemoryTier.SHORT,
                pattern_type=PatternType.SYNTACTIC_API,
                metadata={'source': api_source, 'trigger_type': 'syntactic'}
            )
            
            # Update pattern gates based on syntactic content
            self._process_syntactic_patterns(trigger_text, pattern_hash)
            
            logger.info(f"Syntactic trigger injected: {pattern_hash[:8]} from {api_source}")
            return pattern_hash
            
        except Exception as e:
            logger.error(f"Error injecting syntactic trigger: {e}")
            return ""

    def get_system_metrics(self) -> CacheMetrics:
        """Get comprehensive system metrics."""
        with self.lock:
            self._update_metrics()
            return self.metrics

    # Private helper methods
    def _generate_sha_signature(self, key: str, value: Any, pattern_type: PatternType) -> str:
        """Generate SHA signature for pattern matching."""
        try:
            # Create signature data
            signature_data = f"{key}_{str(value)[:100]}_{pattern_type.value}_{time.time()}"
            return hashlib.sha256(signature_data.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(f"fallback_{key}_{time.time()}".encode()).hexdigest()

    def _calculate_pattern_strength(self, value: Any, pattern_type: PatternType) -> float:
        """Calculate pattern strength for logic measurement."""
        try:
            if pattern_type == PatternType.SHA_GATE:
                return 0.8 + (hash(str(value)) % 100) / 500.0
            elif pattern_type == PatternType.PROFIT_VECTOR:
                if isinstance(value, (int, float)):
                    return min(1.0, abs(value) / 100.0)
                return 0.6
            elif pattern_type == PatternType.ENTROPY_FLOW:
                return 0.7 + (len(str(value)) % 50) / 200.0
            else:
                return 0.5 + (hash(str(value)) % 100) / 200.0
        except Exception:
            return 0.5

    def _calculate_entropy_level(self, value: Any) -> float:
        """Calculate entropy level for the value."""
        try:
            value_str = str(value)
            if len(value_str) == 0:
                return 0.0
            
            # Calculate character frequency entropy
            char_counts = defaultdict(int)
            for char in value_str:
                char_counts[char] += 1
            
            total_chars = len(value_str)
            entropy = 0.0
            for count in char_counts.values():
                prob = count / total_chars
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            
            return min(1.0, entropy / 8.0)  # Normalize to [0, 1]
        except Exception:
            return 0.5

    def _calculate_profit_correlation(self, value: Any, metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate profit correlation for the value."""
        try:
            if metadata and 'profit_score' in metadata:
                return min(1.0, abs(metadata['profit_score']) / 100.0)
            
            if isinstance(value, dict) and 'profit' in str(value).lower():
                return 0.7
            
            if isinstance(value, (int, float)) and value > 0:
                return min(1.0, value / 1000.0)
            
            return 0.3
        except Exception:
            return 0.3

    def _should_activate_vault_bridge(self, pattern_strength: float, entropy_level: float) -> bool:
        """Determine if vault bridge should be activated."""
        return pattern_strength > 0.7 and entropy_level > 0.6

    def _get_tier_store(self, tier: MemoryTier) -> Dict[str, MemoryEntry]:
        """Get the appropriate memory store for tier."""
        if tier == MemoryTier.SHORT:
            return self.short_term
        elif tier == MemoryTier.MID:
            return self.mid_term
        elif tier == MemoryTier.LONG:
            return self.long_term
        elif tier == MemoryTier.VAULT:
            return self.vault_memory
        else:
            return self.mid_term

    def _update_pattern_gates(self, sha_signature: str, pattern_strength: float, pattern_type: PatternType):
        """Update pattern gates based on new entry."""
        try:
            # Determine which gates to update based on pattern type
            gates_to_update = []
            
            if pattern_type in [PatternType.SHA_GATE, PatternType.SYMBOLIC_TRIGGER]:
                gates_to_update.append("2bit_primary")
            
            if pattern_strength > 0.6:
                gates_to_update.append("8bit_register")
            
            if pattern_strength > 0.8:
                gates_to_update.append("256bit_sha")
            
            # Update gate strengths
            for gate_id in gates_to_update:
                if gate_id in self.pattern_gates:
                    gate = self.pattern_gates[gate_id]
                    gate.current_strength = min(1.0, gate.current_strength + pattern_strength * 0.1)
                    
                    if gate.current_strength > gate.activation_threshold:
                        gate.last_triggered = time.time()
                        gate.trigger_count += 1
                        self.metrics.pattern_activations += 1
            
            # Store SHA pattern for quick lookup
            self.sha_gate_store[sha_signature[:8]] = pattern_strength
            
        except Exception as e:
            logger.error(f"Error updating pattern gates: {e}")

    def _update_global_pattern_map(self, key: str, sha_signature: str, pattern_type: PatternType, tier: MemoryTier):
        """Update global pattern map with new entry."""
        pattern_key = f"{pattern_type.value}_{tier.value}"
        
        if pattern_key not in self.global_pattern_map:
            self.global_pattern_map[pattern_key] = {}
        
        self.global_pattern_map[pattern_key][key] = {
            'sha_signature': sha_signature,
            'created_at': time.time(),
            'tier': tier.value
        }

    def _maintain_cache_limits(self, tier: MemoryTier):
        """Maintain cache size limits by removing oldest entries."""
        try:
            store = self._get_tier_store(tier)
            
            if len(store) > self.max_entries_per_tier:
                # Sort by last accessed time and remove oldest
                sorted_entries = sorted(store.items(), key=lambda x: x[1].last_accessed)
                entries_to_remove = len(store) - self.max_entries_per_tier
                
                for i in range(entries_to_remove):
                    key_to_remove = sorted_entries[i][0]
                    del store[key_to_remove]
                    
        except Exception as e:
            logger.error(f"Error maintaining cache limits: {e}")

    def _update_metrics(self):
        """Update cache performance metrics."""
        try:
            total_entries = (len(self.short_term) + len(self.mid_term) + 
                           len(self.long_term) + len(self.vault_memory))
            
            # Calculate hit rate (simplified)
            vault_bridges = sum(1 for store in [self.short_term, self.mid_term, self.long_term, self.vault_memory]
                              for entry in store.values() if entry.vault_bridge_active)
            
            # Calculate average profit correlation
            all_entries = []
            for store in [self.short_term, self.mid_term, self.long_term, self.vault_memory]:
                all_entries.extend(store.values())
            
            avg_profit_correlation = (sum(entry.profit_correlation for entry in all_entries) / 
                                    len(all_entries)) if all_entries else 0.0
            
            # Update metrics
            self.metrics.total_entries = total_entries
            self.metrics.vault_bridges_active = vault_bridges
            self.metrics.profit_correlation_avg = avg_profit_correlation
            self.metrics.memory_efficiency = min(1.0, total_entries / (self.max_entries_per_tier * 4))
            self.metrics.last_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _calculate_system_entropy(self) -> float:
        """Calculate overall system entropy."""
        try:
            all_entries = []
            for store in [self.short_term, self.mid_term, self.long_term, self.vault_memory]:
                all_entries.extend(store.values())
            
            if not all_entries:
                return 0.0
            
            return sum(entry.entropy_level for entry in all_entries) / len(all_entries)
        except Exception:
            return 0.5

    def _calculate_system_profit_score(self) -> float:
        """Calculate overall system profit score."""
        try:
            all_entries = []
            for store in [self.short_term, self.mid_term, self.long_term, self.vault_memory]:
                all_entries.extend(store.values())
            
            if not all_entries:
                return 0.0
            
            return sum(entry.profit_correlation for entry in all_entries) / len(all_entries)
        except Exception:
            return 0.3

    def _determine_system_phase(self) -> str:
        """Determine current system phase based on patterns."""
        try:
            active_gates = sum(1 for gate in self.pattern_gates.values() 
                             if gate.current_strength > gate.activation_threshold)
            
            entropy = self._calculate_system_entropy()
            profit = self._calculate_system_profit_score()
            
            if active_gates >= 3 and profit > 0.8:
                return "HIGH-YIELD"
            elif entropy > 0.7:
                return "CHAOS"
            elif active_gates == 0 and profit < 0.3:
                return "STASIS"
            else:
                return "FLOW"
        except Exception:
            return "UNKNOWN"

    def _get_latest_pattern_hash(self) -> str:
        """Get the latest pattern hash."""
        try:
            if self.sha_gate_store:
                return list(self.sha_gate_store.keys())[-1]
            return "N/A"
        except Exception:
            return "N/A"

    def _calculate_system_coherence(self) -> float:
        """Calculate system coherence score."""
        try:
            active_gates = sum(1 for gate in self.pattern_gates.values() 
                             if gate.current_strength > gate.activation_threshold)
            
            total_gates = len(self.pattern_gates)
            gate_coherence = active_gates / total_gates if total_gates > 0 else 0.0
            
            memory_coherence = self.metrics.memory_efficiency
            profit_coherence = self.metrics.profit_correlation_avg
            
            return (gate_coherence + memory_coherence + profit_coherence) / 3.0
        except Exception:
            return 0.5

    def _find_entry_by_key(self, key: str) -> Optional[MemoryEntry]:
        """Find memory entry by key across all tiers."""
        for store in [self.short_term, self.mid_term, self.long_term, self.vault_memory]:
            if key in store:
                return store[key]
        return None

    def _remove_from_tier(self, key: str, tier: MemoryTier):
        """Remove entry from specific tier."""
        store = self._get_tier_store(tier)
        if key in store:
            del store[key]

    def _find_pattern_match(self, hash_key: str, pattern_type: Optional[PatternType]) -> Optional[MemoryEntry]:
        """Find best pattern match using fuzzy matching."""
        try:
            best_match = None
            best_score = 0.0
            
            all_stores = [self.short_term, self.mid_term, self.long_term, self.vault_memory]
            
            for store in all_stores:
                for entry in store.values():
                    if pattern_type and entry.pattern_type != pattern_type:
                        continue
                    
                    # Calculate similarity score
                    signature_similarity = self._calculate_hash_similarity(hash_key, entry.sha_signature)
                    pattern_strength_bonus = entry.pattern_strength * 0.2
                    
                    total_score = signature_similarity + pattern_strength_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_match = entry
            
            return best_match if best_score > 0.5 else None
            
        except Exception as e:
            logger.error(f"Error finding pattern match: {e}")
            return None

    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes."""
        try:
            if len(hash1) != len(hash2):
                min_len = min(len(hash1), len(hash2))
                hash1 = hash1[:min_len]
                hash2 = hash2[:min_len]
            
            if len(hash1) == 0:
                return 0.0
            
            matches = sum(1 for c1, c2 in zip(hash1, hash2) if c1 == c2)
            return matches / len(hash1)
        except Exception:
            return 0.0

    def _process_syntactic_patterns(self, trigger_text: str, pattern_hash: str):
        """Process syntactic patterns for gate activation."""
        try:
            # Analyze text for profit-related keywords
            profit_keywords = ['profit', 'gain', 'bull', 'surge', 'pump', 'moon', 'rocket']
            chaos_keywords = ['crash', 'dump', 'bear', 'decline', 'fall', 'panic']
            
            text_lower = trigger_text.lower()
            
            profit_score = sum(1 for keyword in profit_keywords if keyword in text_lower)
            chaos_score = sum(1 for keyword in chaos_keywords if keyword in text_lower)
            
            # Update gates based on syntactic analysis
            if profit_score > 0:
                if "2bit_primary" in self.pattern_gates:
                    self.pattern_gates["2bit_primary"].current_strength += 0.1 * profit_score
            
            if chaos_score > 0:
                if "8bit_register" in self.pattern_gates:
                    self.pattern_gates["8bit_register"].current_strength += 0.15 * chaos_score
            
            # Store pattern for future reference
            self.sha_gate_store[pattern_hash[:8]] = max(profit_score, chaos_score) / 10.0
            
        except Exception as e:
            logger.error(f"Error processing syntactic patterns: {e}")


# Global memory cache bridge instance
_memory_bridge = None

def get_memory_bridge() -> MemoryCacheBridge:
    """Get global memory cache bridge instance."""
    global _memory_bridge
    if _memory_bridge is None:
        _memory_bridge = MemoryCacheBridge()
    return _memory_bridge

def initialize_memory_bridge() -> MemoryCacheBridge:
    """Initialize and return memory cache bridge."""
    global _memory_bridge
    _memory_bridge = MemoryCacheBridge()
    return _memory_bridge

def main():
    """Test memory cache bridge functionality."""
    print("🧠 Memory Cache Bridge Test")
    print("-" * 50)
    
    bridge = MemoryCacheBridge()
    
    # Test pattern-based caching
    bridge.update_cache("test_profit", 85.5, MemoryTier.MID, PatternType.PROFIT_VECTOR)
    bridge.update_cache("test_entropy", 0.75, MemoryTier.SHORT, PatternType.ENTROPY_FLOW)
    bridge.update_cache("test_sha", "abc123def456", MemoryTier.LONG, PatternType.SHA_GATE)
    
    # Test pattern fetching
    profit_data = bridge.fetch_pattern("test_profit")
    print(f"💰 Profit data: {profit_data}")
    
    # Test syntactic trigger
    pattern_hash = bridge.inject_syntactic_trigger("Bitcoin surges to new highs with major profit gains", "news_api")
    print(f"📰 Syntactic pattern: {pattern_hash[:8]}")
    
    # Test visualization payload
    payload = bridge.resolve_visualization_payload()
    print(f"📊 Visualization payload: {payload}")
    
    # Test pattern gate status
    gate_status = bridge.get_pattern_gate_status()
    print(f"🚪 Pattern gates: {gate_status}")
    
    # Test metrics
    metrics = bridge.get_system_metrics()
    print(f"📈 System metrics: {metrics}")
    
    print("\n✅ Memory Cache Bridge Test Complete")

if __name__ == "__main__":
    main() 