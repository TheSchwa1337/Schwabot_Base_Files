"""
Memory Key Allocator - Symbolic Memory Management System.

This class manages the generation and linking of memory keys for
Schwabot's recursive memory system.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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

        logger.info("🔑 Memory Key Allocator initialized - Symbolic memory active")

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

                logger.info(f"🔑 Loaded {len(self.memory_keys)} memory keys, {len(self.memory_links)} links, {len(self.memory_clusters)} clusters")

        except Exception as e:
            logger.error(f"⚠️ Failed to load memory keys: {e}")

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
            logger.error(f"⚠️ Failed to save memory keys: {e}")

    def allocate_memory_key(self,
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

            # Attempt clustering if enabled
            if self.auto_clustering:
                self._attempt_clustering()

            # Save to file
            self._save_memory_keys()

            logger.info(f"🔑 Allocated memory key: {key_id} (type: {key_type.value})")
            return memory_key

        except Exception as e:
            logger.error(f"❌ Failed to allocate memory key: {e}")
            return None

    def create_memory_link(self,
                          source_key: str,
                          target_key: str,
                          link_type: str,
                          strength: LinkStrength = LinkStrength.MODERATE,
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
            strength: Link strength
            alpha_correlation: Alpha correlation
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            MemoryLink object or None if failed
        """
        try:
            # Validate keys exist
            if source_key not in self.memory_keys or target_key not in self.memory_keys:
                logger.warning(f"⚠️ Invalid memory keys for link: {source_key} -> {target_key}")
                return None

            # Generate link ID
            link_id = f"link_{source_key}_{target_key}_{int(datetime.now().timestamp())}"

            # Calculate time decay
            time_decay = self._calculate_time_decay()

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

            logger.info(f"🔗 Created memory link: {link_id} ({strength.value})")
            return memory_link

        except Exception as e:
            logger.error(f"❌ Failed to create memory link: {e}")
            return None

    def find_similar_keys(self,
                         target_key: str,
                         similarity_threshold: Optional[float] = None,
                         max_results: int = 10
                         ) -> List[MemoryKey]:
        """
        Find similar memory keys.

        Args:
            target_key: Target key ID
            similarity_threshold: Similarity threshold
            max_results: Maximum number of results

        Returns:
            List of similar memory keys
        """
        try:
            if target_key not in self.memory_keys:
                logger.warning(f"⚠️ Target key not found: {target_key}")
                return []

            target_memory_key = self.memory_keys[target_key]
            threshold = similarity_threshold or self.similarity_threshold

            similar_keys = []
            for key in self.memory_keys.values():
                if key.key_id == target_key:
                    continue

                # Calculate similarity
                similarity = self._calculate_key_similarity(target_memory_key, key)
                if similarity >= threshold:
                    similar_keys.append((key, similarity))

            # Sort by similarity and return top results
            similar_keys.sort(key=lambda x: x[1], reverse=True)
            return [key for key, _ in similar_keys[:max_results]]

        except Exception as e:
            logger.error(f"❌ Failed to find similar keys: {e}")
            return []

    def get_memory_cluster(self) -> Optional[MemoryCluster]:
        """Get memory cluster information."""
        return None

    def _generate_symbolic_key(self, agent_type: str, domain: str, tick: int) -> str:
        """Generate symbolic memory key."""
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{agent_type.upper()}{domain.upper()}_{date_str}_T{tick}"

    def _generate_hash_based_key(self, hash_signature: str, tick: int) -> str:
        """Generate hash-based memory key."""
        hash_suffix = hash_signature[:8]
        return f"HASH_{hash_suffix}_T{tick}"

    def _generate_hybrid_key(self, agent_type: str, domain: str, hash_signature: str, tick: int) -> str:
        """Generate hybrid memory key."""
        agent_code = agent_type[:3].upper()
        domain_code = domain[:3].upper()
        hash_suffix = hash_signature[:6]
        return f"{agent_code}{domain_code}_{hash_suffix}_T{tick}"

    def _generate_auto_key(self, agent_type: str, domain: str, hash_signature: str, tick: int, alpha_score: float) -> str:
        """Generate auto memory key."""
        base_key = self._generate_hybrid_key(agent_type, domain, hash_signature, tick)
        alpha_indicator = "A" if alpha_score > 0.5 else "B"
        return f"{base_key}_{alpha_indicator}"

    def _calculate_key_similarity(self, key1: MemoryKey, key2: MemoryKey) -> float:
        """Calculate similarity between two memory keys."""
        try:
            # Calculate hash similarity
            hash_similarity = self._calculate_hash_similarity(key1.hash_signature, key2.hash_signature)

            # Calculate domain similarity
            domain_similarity = 1.0 if key1.domain == key2.domain else 0.0

            # Calculate agent similarity
            agent_similarity = 1.0 if key1.agent_type == key2.agent_type else 0.0

            # Weighted combination
            similarity = (hash_similarity * 0.5 + domain_similarity * 0.3 + agent_similarity * 0.2)
            return min(1.0, max(0.0, similarity))

        except Exception as e:
            logger.error(f"❌ Failed to calculate key similarity: {e}")
            return 0.0

    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes."""
        try:
            if len(hash1) != len(hash2):
                return 0.0

            # Calculate Hamming distance
            hamming_distance = sum(a != b for a, b in zip(hash1, hash2))
            similarity = 1.0 - (hamming_distance / len(hash1))

            return max(0.0, similarity)

        except Exception as e:
            logger.error(f"❌ Failed to calculate hash similarity: {e}")
            return 0.0

    def _determine_link_strength(self, similarity: float, alpha_correlation: float) -> LinkStrength:
        """Determine link strength based on similarity and alpha correlation."""
        try:
            combined_score = (similarity * 0.7 + alpha_correlation * 0.3)

            if combined_score >= 0.9:
                return LinkStrength.CRITICAL
            elif combined_score >= 0.7:
                return LinkStrength.STRONG
            elif combined_score >= 0.5:
                return LinkStrength.MODERATE
            else:
                return LinkStrength.WEAK

        except Exception as e:
            logger.error(f"❌ Failed to determine link strength: {e}")
            return LinkStrength.WEAK

    def _calculate_time_decay(self) -> float:
        """Calculate time decay factor."""
        try:
            # Simple time decay - could be enhanced with more sophisticated algorithms
            return 1.0
        except Exception as e:
            logger.error(f"❌ Failed to calculate time decay: {e}")
            return 1.0

    def _attempt_clustering(self) -> None:
        """Attempt to form memory clusters."""
        try:
            # Simple clustering implementation
            # In practice, this would use more sophisticated clustering algorithms
            pass
        except Exception as e:
            logger.error(f"⚠️ Clustering failed: {e}")

    def get_memory_key(self, key_id: str) -> Optional[MemoryKey]:
        """Get memory key by ID."""
        return self.memory_keys.get(key_id)

    def get_memory_links(self, key_id: str) -> List[MemoryLink]:
        """Get memory links for a key."""
        return [link for link in self.memory_links.values()
                if link.source_key == key_id or link.target_key == key_id]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'total_keys': len(self.memory_keys),
            'total_links': len(self.memory_links),
            'total_clusters': len(self.memory_clusters),
            'keys_allocated': self.total_keys_allocated,
            'links_created': self.total_links_created,
            'clusters_formed': self.total_clusters_formed,
            'average_similarity': self.average_similarity_score,
            'key_distribution': {
                key_type.value: len([k for k in self.memory_keys.values() if k.key_type == key_type])
                for key_type in KeyType
            }
        }

    def cleanup_old_data(self) -> None:
        """Clean up old memory data."""
        try:
            # Implementation for cleaning up old data
            pass
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")


# Convenience functions for external access
def allocate_memory_key(agent_type: str,
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
                       ) -> Optional[MemoryKey]:
    """Convenience function to allocate memory key."""
    allocator = MemoryKeyAllocator()
    return allocator.allocate_memory_key(
        agent_type, domain, hash_signature, tick, key_type,
        alpha_score, matrix_id, curve_id, profit_delta, confidence_score, metadata
    )


def create_memory_link(source_key: str,
                      target_key: str,
                      link_type: str,
                      strength: LinkStrength = LinkStrength.MODERATE,
                      alpha_correlation: float = 0.0,
                      confidence: float = 0.0,
                      metadata: Optional[Dict[str, Any]] = None
                      ) -> Optional[MemoryLink]:
    """Convenience function to create memory link."""
    allocator = MemoryKeyAllocator()
    return allocator.create_memory_link(
        source_key, target_key, link_type, strength,
        alpha_correlation, confidence, metadata
    )


def find_similar_memory_keys(target_key: str,
                           similarity_threshold: Optional[float] = None,
                           max_results: int = 10
                           ) -> List[MemoryKey]:
    """Convenience function to find similar memory keys."""
    allocator = MemoryKeyAllocator()
    return allocator.find_similar_keys(target_key, similarity_threshold, max_results)


if __name__ == "__main__":
    # Test the memory key allocator
    allocator = MemoryKeyAllocator()
    
    # Allocate a test key
    test_key = allocator.allocate_memory_key(
        agent_type="trading_bot",
        domain="market_analysis",
        hash_signature="abc123def456",
        tick=1000,
        alpha_score=0.8
    )
    
    if test_key:
        print(f"✅ Test key allocated: {test_key.key_id}")
        
        # Get performance metrics
        metrics = allocator.get_performance_metrics()
        print(f"📊 Performance metrics: {metrics}")
    else:
        print("❌ Failed to allocate test key")
