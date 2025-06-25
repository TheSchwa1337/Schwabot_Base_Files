from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Memory Key Allocator - Symbolic Memory Management System.

This module generates symbolic or hash-based memory keys and links them to
hash, matrix, curve, and profit data for Schwabot's recursive memory system.

Mathematical Foundation:
- Memory Key: MK = f(agent, hash, tick, α, matrix_id)
- Hash Similarity: S = Σ(1 for a==b in zip(h1, h2)) / len(h1)
- Symbolic Key: SK = AgentType + Domain + Date + HashSuffix
- Link Strength: L = α * confidence * time_decay
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from core.unified_math_system import unified_math

# Import centralized CLI handler
try:
    from core.utils.windows_cli_compatibility import (
        WindowsCliCompatibilityHandler,
        safe_print,
        safe_format_error,
        log_safe,
        cli_handler,
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)
    cli_handler = None

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Enumeration of memory key types."""
    SYMBOLIC = "symbolic"
    HASH_BASED = "hash_based"
    HYBRID = "hybrid"
    AUTO_GENERATED = "auto_generated"


class LinkStrength(Enum):
    """Enumeration of link strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


@dataclass
class MemoryKey:
    """Memory key structure."""
    key_id: str
    key_type: KeyType
    agent_type: str
    domain: str
    hash_signature: str
    tick: int
    timestamp: datetime
    alpha_score: float = 0.0
    matrix_id: Optional[str] = None
    curve_id: Optional[str] = None
    profit_delta: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.metadata:
            self.metadata = {}


@dataclass
class MemoryLink:
    """Memory link structure."""
    link_id: str
    source_key: str
    target_key: str
    link_type: str
    strength: LinkStrength
    alpha_correlation: float = 0.0
    time_decay: float = 1.0
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.metadata:
            self.metadata = {}


@dataclass
class MemoryCluster:
    """Memory cluster structure."""
    cluster_id: str
    cluster_type: str
    memory_keys: List[str]
    center_key: str
    similarity_threshold: float
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.metadata:
            self.metadata = {}


class MemoryKeyAllocator:
    """
    Memory Key Allocator - Symbolic Memory Management System.
    
    This class manages the generation and linking of memory keys for
    Schwabot's recursive memory system.
    """
    
    def __init__(self, memory_file: str = "memory_stack/memory_keys.json"):
        """Initialize the memory key allocator."""
        self.memory_file = memory_file
        self.logger = logging.getLogger("memory_key_allocator")
        self.logger.setLevel(logging.INFO)
        
        # Memory storage
        self.memory_keys: Dict[str, MemoryKey] = {}
        self.memory_links: Dict[str, MemoryLink] = {}
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        
        # Configuration parameters
        self.similarity_threshold = 0.85
        self.max_cluster_size = 50
        self.time_decay_factor = 0.95
        self.auto_clustering = True
        
        # Performance tracking
        self.total_keys_allocated = 0
        self.total_links_created = 0
        self.total_clusters_formed = 0
        self.average_similarity_score = 0.0
        
        # Load existing memory
        self._load_memory_keys()
        
        safe_safe_print("🔑 Memory Key Allocator initialized - Symbolic memory active")
    
    def _load_memory_keys(self) -> None:
        """Load existing memory keys from file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                # Load memory keys
                for key_data in memory_data.get('memory_keys', []):
                    memory_key = MemoryKey(
                        key_id=key_data['key_id'],
                        key_type=KeyType(key_data['key_type']),
                        agent_type=key_data['agent_type'],
                        domain=key_data['domain'],
                        hash_signature=key_data['hash_signature'],
                        tick=key_data['tick'],
                        timestamp=datetime.fromisoformat(key_data['timestamp']),
                        alpha_score=key_data.get('alpha_score', 0.0),
                        matrix_id=key_data.get('matrix_id'),
                        curve_id=key_data.get('curve_id'),
                        profit_delta=key_data.get('profit_delta', 0.0),
                        confidence_score=key_data.get('confidence_score', 0.0),
                        metadata=key_data.get('metadata', {})
                    )
                    self.memory_keys[memory_key.key_id] = memory_key
                
                # Load memory links
                for link_data in memory_data.get('memory_links', []):
                    memory_link = MemoryLink(
                        link_id=link_data['link_id'],
                        source_key=link_data['source_key'],
                        target_key=link_data['target_key'],
                        link_type=link_data['link_type'],
                        strength=LinkStrength(link_data['strength']),
                        alpha_correlation=link_data.get('alpha_correlation', 0.0),
                        time_decay=link_data.get('time_decay', 1.0),
                        confidence=link_data.get('confidence', 0.0),
                        created_at=datetime.fromisoformat(link_data['created_at']),
                        metadata=link_data.get('metadata', {})
                    )
                    self.memory_links[memory_link.link_id] = memory_link
                
                # Load memory clusters
                for cluster_data in memory_data.get('memory_clusters', []):
                    memory_cluster = MemoryCluster(
                        cluster_id=cluster_data['cluster_id'],
                        cluster_type=cluster_data['cluster_type'],
                        memory_keys=cluster_data['memory_keys'],
                        center_key=cluster_data['center_key'],
                        similarity_threshold=cluster_data['similarity_threshold'],
                        created_at=datetime.fromisoformat(cluster_data['created_at']),
                        last_updated=datetime.fromisoformat(cluster_data['last_updated']),
                        metadata=cluster_data.get('metadata', {})
                    )
                    self.memory_clusters[memory_cluster.cluster_id] = memory_cluster
                
                safe_safe_print(f"🔑 Loaded {len(self.memory_keys)} memory keys, {len(self.memory_links)} links, {len(self.memory_clusters)} clusters")
                
        except Exception as e:
            error_msg = safe_format_error(e, "load_memory_keys")
            safe_safe_print(f"⚠️ Failed to load memory keys: {error_msg}")
    
    def _save_memory_keys(self) -> None:
        """Save memory keys to file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            memory_data = {
                'memory_keys': [],
                'memory_links': [],
                'memory_clusters': [],
                'last_updated': datetime.now().isoformat(),
                'total_keys': len(self.memory_keys),
                'total_links': len(self.memory_links),
                'total_clusters': len(self.memory_clusters)
            }
            
            # Save memory keys
            for key in self.memory_keys.values():
                key_data = asdict(key)
                key_data['timestamp'] = key.timestamp.isoformat()
                key_data['key_type'] = key.key_type.value
                memory_data['memory_keys'].append(key_data)
            
            # Save memory links
            for link in self.memory_links.values():
                link_data = asdict(link)
                link_data['created_at'] = link.created_at.isoformat()
                link_data['strength'] = link.strength.value
                memory_data['memory_links'].append(link_data)
            
            # Save memory clusters
            for cluster in self.memory_clusters.values():
                cluster_data = asdict(cluster)
                cluster_data['created_at'] = cluster.created_at.isoformat()
                cluster_data['last_updated'] = cluster.last_updated.isoformat()
                memory_data['memory_clusters'].append(cluster_data)
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
        except Exception as e:
            error_msg = safe_format_error(e, "save_memory_keys")
            safe_safe_print(f"⚠️ Failed to save memory keys: {error_msg}")
    
    def allocate_memory_key(
        self,
        agent_type: str,
        domain: str,
        hash_signature: str,
        tick: int,
        key_type: KeyType = KeyType.AUTO_GENERATED,
        alpha_score: float = 0.0,
        matrix_id: Optional[str] = None,
        curve_id: Optional[str] = None,
        profit_delta: float = 0.0,
        confidence_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryKey:
        """
        Allocate a new memory key.
        
        Args:
            agent_type: Type of AI agent
            domain: Command domain
            hash_signature: Hash signature
            tick: Current tick
            key_type: Type of memory key to generate
            alpha_score: Alpha score for profit alignment
            matrix_id: Optional matrix ID
            curve_id: Optional Prophet curve ID
            profit_delta: Profit delta achieved
            confidence_score: Confidence score
            metadata: Additional metadata
            
        Returns:
            MemoryKey object
        """
        try:
            # Generate key ID based on type
            if key_type == KeyType.SYMBOLIC:
                key_id = self._generate_symbolic_key(agent_type, domain, tick)
            elif key_type == KeyType.HASH_BASED:
                key_id = self._generate_hash_based_key(hash_signature, tick)
            elif key_type == KeyType.HYBRID:
                key_id = self._generate_hybrid_key(agent_type, domain, hash_signature, tick)
            else:  # AUTO_GENERATED
                key_id = self._generate_auto_key(agent_type, domain, hash_signature, tick, alpha_score)
            
            # Create memory key
            memory_key = MemoryKey(
                key_id=key_id,
                key_type=key_type,
                agent_type=agent_type,
                domain=domain,
                hash_signature=hash_signature,
                tick=tick,
                timestamp=datetime.now(),
                alpha_score=alpha_score,
                matrix_id=matrix_id,
                curve_id=curve_id,
                profit_delta=profit_delta,
                confidence_score=confidence_score,
                metadata=metadata or {}
            )
            
            # Store memory key
            self.memory_keys[key_id] = memory_key
            self.total_keys_allocated += 1
            
            # Auto-clustering if enabled
            if self.auto_clustering:
                self._attempt_clustering(memory_key)
            
            # Save to file
            self._save_memory_keys()
            
            safe_safe_print(f"🔑 Memory key allocated: {key_id} ({key_type.value})")
            return memory_key
            
        except Exception as e:
            error_msg = safe_format_error(e, "allocate_memory_key")
            safe_safe_print(f"❌ Memory key allocation failed: {error_msg}")
            
            # Return safe fallback key
            return MemoryKey(
                key_id=f"fallback_{int(time.time())}",
                key_type=KeyType.AUTO_GENERATED,
                agent_type=agent_type,
                domain=domain,
                hash_signature=hash_signature,
                tick=tick,
                timestamp=datetime.now(),
                metadata={'error': error_msg}
            )
    
    def create_memory_link(
        self,
        source_key: str,
        target_key: str,
        link_type: str,
        alpha_correlation: float = 0.0,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MemoryLink]:
        """
        Create a memory link between two keys.
        
        Args:
            source_key: Source memory key ID
            target_key: Target memory key ID
            link_type: Type of link
            alpha_correlation: Alpha correlation between keys
            confidence: Confidence in the link
            metadata: Additional metadata
            
        Returns:
            MemoryLink object or None if failed
        """
        try:
            # Validate keys exist
            if source_key not in self.memory_keys or target_key not in self.memory_keys:
                safe_safe_print(f"⚠️ Invalid memory keys for link: {source_key} -> {target_key}")
                return None
            
            # Generate link ID
            link_id = f"LINK_{source_key}_{target_key}_{int(time.time())}"
            
            # Determine link strength
            strength = self._determine_link_strength(alpha_correlation, confidence)
            
            # Calculate time decay
            time_decay = self._calculate_time_decay(source_key, target_key)
            
            # Create memory link
            memory_link = MemoryLink(
                link_id=link_id,
                source_key=source_key,
                target_key=target_key,
                link_type=link_type,
                strength=strength,
                alpha_correlation=alpha_correlation,
                time_decay=time_decay,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            # Store memory link
            self.memory_links[link_id] = memory_link
            self.total_links_created += 1
            
            # Save to file
            self._save_memory_keys()
            
            safe_safe_print(f"🔗 Memory link created: {link_id} ({strength.value})")
            return memory_link
            
        except Exception as e:
            error_msg = safe_format_error(e, "create_memory_link")
            safe_safe_print(f"❌ Memory link creation failed: {error_msg}")
            return None
    
    def find_similar_keys(
        self,
        target_key: str,
        similarity_threshold: Optional[float] = None,
        max_results: int = 10
    ) -> List[Tuple[MemoryKey, float]]:
        """
        Find memory keys similar to target key.
        
        Args:
            target_key: Target memory key ID
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of (MemoryKey, similarity_score) tuples
        """
        try:
            if target_key not in self.memory_keys:
                return []
            
            target_memory_key = self.memory_keys[target_key]
            threshold = similarity_threshold or self.similarity_threshold
            
            similar_keys = []
            
            for key_id, memory_key in self.memory_keys.items():
                if key_id == target_key:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_key_similarity(target_memory_key, memory_key)
                
                if similarity >= threshold:
                    similar_keys.append((memory_key, similarity))
            
            # Sort by similarity and limit results
            similar_keys.sort(key=lambda x: x[1], reverse=True)
            return similar_keys[:max_results]
            
        except Exception as e:
            error_msg = safe_format_error(e, "find_similar_keys")
            safe_safe_print(f"❌ Similar key search failed: {error_msg}")
            return []
    
    def get_memory_cluster(self, key_id: str) -> Optional[MemoryCluster]:
        """Get memory cluster containing a specific key."""
        for cluster in self.memory_clusters.values():
            if key_id in cluster.memory_keys:
                return cluster
        return None
    
    def _generate_symbolic_key(self, agent_type: str, domain: str, tick: int) -> str:
        """Generate symbolic memory key."""
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{agent_type.upper()}{domain.upper()}_{date_str}_T{tick}"
    
    def _generate_hash_based_key(self, hash_signature: str, tick: int) -> str:
        """Generate hash-based memory key."""
        hash_suffix = hash_signature[:12]
        return f"HASH_{hash_suffix}_T{tick}"
    
    def _generate_hybrid_key(self, agent_type: str, domain: str, hash_signature: str, tick: int) -> str:
        """Generate hybrid memory key."""
        agent_code = agent_type.upper()[:3]
        domain_code = domain.upper()[:3]
        hash_suffix = hash_signature[:8]
        return f"{agent_code}{domain_code}_{hash_suffix}_T{tick}"
    
    def _generate_auto_key(self, agent_type: str, domain: str, hash_signature: str, tick: int, alpha_score: float) -> str:
        """Generate auto memory key based on context."""
        # Use hybrid approach with alpha score influence
        base_key = self._generate_hybrid_key(agent_type, domain, hash_signature, tick)
        
        # Add alpha score indicator
        if alpha_score > 0.05:
            alpha_indicator = "POS"
        elif alpha_score < -0.05:
            alpha_indicator = "NEG"
        else:
            alpha_indicator = "NEU"
        
        return f"{base_key}_{alpha_indicator}"
    
    def _calculate_key_similarity(self, key1: MemoryKey, key2: MemoryKey) -> float:
        """Calculate similarity between two memory keys."""
        try:
            # Hash similarity
            hash_similarity = self._calculate_hash_similarity(key1.hash_signature, key2.hash_signature)
            
            # Domain similarity
            domain_similarity = 1.0 if key1.domain == key2.domain else 0.0
            
            # Agent similarity
            agent_similarity = 1.0 if key1.agent_type == key2.agent_type else 0.0
            
            # Alpha score similarity (normalized)
            alpha_diff = unified_math.abs(key1.alpha_score - key2.alpha_score)
            alpha_similarity = unified_math.max(0.0, 1.0 - alpha_diff)
            
            # Tick proximity (closer ticks = higher similarity)
            tick_diff = unified_math.abs(key1.tick - key2.tick)
            tick_similarity = unified_math.max(0.0, 1.0 - (tick_diff / 1000))  # Normalize to 1000 ticks
            
            # Weighted combination
            similarity = (
                hash_similarity * 0.4 +
                domain_similarity * 0.2 +
                agent_similarity * 0.1 +
                alpha_similarity * 0.2 +
                tick_similarity * 0.1
            )
            
            return similarity
            
        except Exception as e:
            safe_safe_print(f"⚠️ Similarity calculation failed: {safe_format_error(e, 'similarity')}")
            return 0.0
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hash signatures."""
        try:
            if len(hash1) != len(hash2):
                return 0.0
            
            # Calculate character-wise similarity
            matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
            similarity = matches / len(hash1)
            
            return similarity
            
        except Exception as e:
            safe_safe_print(f"⚠️ Hash similarity calculation failed: {safe_format_error(e, 'hash_similarity')}")
            return 0.0
    
    def _determine_link_strength(self, alpha_correlation: float, confidence: float) -> LinkStrength:
        """Determine link strength based on correlation and confidence."""
        combined_score = (alpha_correlation + confidence) / 2.0
        
        if combined_score >= 0.8:
            return LinkStrength.CRITICAL
        elif combined_score >= 0.6:
            return LinkStrength.STRONG
        elif combined_score >= 0.4:
            return LinkStrength.MODERATE
        else:
            return LinkStrength.WEAK
    
    def _calculate_time_decay(self, source_key: str, target_key: str) -> float:
        """Calculate time decay factor for memory link."""
        try:
            source_time = self.memory_keys[source_key].timestamp
            target_time = self.memory_keys[target_key].timestamp
            
            time_diff = abs((source_time - target_time).total_seconds())
            decay_factor = unified_math.exp(-time_diff / (24 * 3600))  # Decay over 24 hours
            
            return unified_math.max(0.1, decay_factor)  # Minimum decay of 0.1
            
        except Exception as e:
            safe_safe_print(f"⚠️ Time decay calculation failed: {safe_format_error(e, 'time_decay')}")
            return 1.0
    
    def _attempt_clustering(self, new_key: MemoryKey) -> None:
        """Attempt to add new key to existing clusters or create new cluster."""
        try:
            # Find best matching cluster
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in self.memory_clusters.values():
                if len(cluster.memory_keys) >= self.max_cluster_size:
                    continue
                
                # Calculate average similarity to cluster center
                center_key = self.memory_keys.get(cluster.center_key)
                if center_key:
                    similarity = self._calculate_key_similarity(new_key, center_key)
                    if similarity > best_similarity and similarity >= cluster.similarity_threshold:
                        best_similarity = similarity
                        best_cluster = cluster
            
            if best_cluster:
                # Add to existing cluster
                best_cluster.memory_keys.append(new_key.key_id)
                best_cluster.last_updated = datetime.now()
                safe_safe_print(f"🔗 Added key {new_key.key_id} to cluster {best_cluster.cluster_id}")
            else:
                # Create new cluster
                cluster_id = f"CLUSTER_{new_key.agent_type}_{new_key.domain}_{int(time.time())}"
                new_cluster = MemoryCluster(
                    cluster_id=cluster_id,
                    cluster_type=f"{new_key.agent_type}_{new_key.domain}",
                    memory_keys=[new_key.key_id],
                    center_key=new_key.key_id,
                    similarity_threshold=self.similarity_threshold
                )
                self.memory_clusters[cluster_id] = new_cluster
                self.total_clusters_formed += 1
                safe_safe_print(f"🔗 Created new cluster {cluster_id} for key {new_key.key_id}")
                
        except Exception as e:
            safe_safe_print(f"⚠️ Clustering failed: {safe_format_error(e, 'clustering')}")
    
    def get_memory_key(self, key_id: str) -> Optional[MemoryKey]:
        """Get memory key by ID."""
        return self.memory_keys.get(key_id)
    
    def get_memory_links(self, key_id: str) -> List[MemoryLink]:
        """Get all memory links for a specific key."""
        return [link for link in self.memory_links.values() 
                if link.source_key == key_id or link.target_key == key_id]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'total_keys_allocated': self.total_keys_allocated,
            'total_links_created': self.total_links_created,
            'total_clusters_formed': self.total_clusters_formed,
            'average_similarity_score': self.average_similarity_score,
            'memory_keys': len(self.memory_keys),
            'memory_links': len(self.memory_links),
            'memory_clusters': len(self.memory_clusters),
            'key_types_distribution': {
                key_type.value: len([k for k in self.memory_keys.values() if k.key_type == key_type])
                for key_type in KeyType
            }
        }
    
    def cleanup_old_data(self, max_age_days: int = 30) -> None:
        """Clean up old memory data."""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            # Remove old memory keys
            old_keys = [key_id for key_id, key in self.memory_keys.items() 
                       if key.timestamp < cutoff_time]
            for key_id in old_keys:
                del self.memory_keys[key_id]
            
            # Remove old memory links
            old_links = [link_id for link_id, link in self.memory_links.items() 
                        if link.created_at < cutoff_time]
            for link_id in old_links:
                del self.memory_links[link_id]
            
            # Update clusters
            for cluster in self.memory_clusters.values():
                cluster.memory_keys = [key_id for key_id in cluster.memory_keys 
                                     if key_id in self.memory_keys]
            
            # Remove empty clusters
            empty_clusters = [cluster_id for cluster_id, cluster in self.memory_clusters.items() 
                            if not cluster.memory_keys]
            for cluster_id in empty_clusters:
                del self.memory_clusters[cluster_id]
            
            safe_safe_print(f"🧹 Cleaned up {len(old_keys)} old keys, {len(old_links)} old links, {len(empty_clusters)} empty clusters")
            
        except Exception as e:
            safe_safe_print(f"⚠️ Cleanup failed: {safe_format_error(e, 'cleanup')}")


# Global instance for easy access
memory_key_allocator = MemoryKeyAllocator()


# Convenience functions for external access
def allocate_memory_key(
    agent_type: str,
    domain: str,
    hash_signature: str,
    tick: int,
    key_type: KeyType = KeyType.AUTO_GENERATED,
    alpha_score: float = 0.0,
    matrix_id: Optional[str] = None,
    curve_id: Optional[str] = None,
    profit_delta: float = 0.0,
    confidence_score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> MemoryKey:
    """Allocate memory key using global allocator."""
    return memory_key_allocator.allocate_memory_key(
        agent_type, domain, hash_signature, tick, key_type, alpha_score,
        matrix_id, curve_id, profit_delta, confidence_score, metadata
    )


def create_memory_link(
    source_key: str,
    target_key: str,
    link_type: str,
    alpha_correlation: float = 0.0,
    confidence: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[MemoryLink]:
    """Create memory link using global allocator."""
    return memory_key_allocator.create_memory_link(
        source_key, target_key, link_type, alpha_correlation, confidence, metadata
    )


def find_similar_memory_keys(
    target_key: str,
    similarity_threshold: Optional[float] = None,
    max_results: int = 10
) -> List[Tuple[MemoryKey, float]]:
    """Find similar memory keys using global allocator."""
    return memory_key_allocator.find_similar_keys(target_key, similarity_threshold, max_results)


# Example usage
if __name__ == "__main__":
    # Test memory key allocator functionality
    safe_safe_print("🔑 Testing Memory Key Allocator...")
    
    # Allocate test memory keys
    key1 = allocate_memory_key(
        agent_type="gpt",
        domain="strategy",
        hash_signature="abc123def456",
        tick=1000,
        key_type=KeyType.SYMBOLIC,
        alpha_score=0.05,
        profit_delta=50.0
    )
    
    key2 = allocate_memory_key(
        agent_type="claude",
        domain="profit",
        hash_signature="def456ghi789",
        tick=1001,
        key_type=KeyType.HASH_BASED,
        alpha_score=0.03,
        profit_delta=30.0
    )
    
    # Create memory link
    link = create_memory_link(
        source_key=key1.key_id,
        target_key=key2.key_id,
        link_type="profit_correlation",
        alpha_correlation=0.7,
        confidence=0.8
    )
    
    # Find similar keys
    similar_keys = find_similar_memory_keys(key1.key_id, max_results=5)
    
    # Get performance metrics
    metrics = memory_key_allocator.get_performance_metrics()
    
    safe_safe_print(f"✅ Test completed - Keys: {len(similar_keys)} similar, Metrics: {metrics}") 