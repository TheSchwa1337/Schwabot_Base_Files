"""Module for Schwabot trading system."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np

from .entropy_math import bit_entropy, get_backend_info, hamming_distance, shannon_entropy, vector_similarity

#!/usr/bin/env python3
"""Vector Registry 🗂️"

    Maps SHA-256 digests to compressed strategy vectors for:
    • Digest → Strategy matching
    • Historical profit pattern recognition
    • Multi-indicator confidence scoring
    • Asset basket allocation

        CUDA Integration:
        - GPU-accelerated vector similarity search with automatic CPU fallback
        - Performance monitoring and optimization
        - Cross-platform compatibility (Windows, macOS, Linux)

            Core Functions:
            * register_digest(digest, strategy, outcome)  – store new digest→strategy mapping
            * find_similar_digests(digest, threshold)     – find similar historical patterns
            * get_confidence_score(digest, strategy)     – calculate confidence based on history
            * update_profit_outcome(digest, pnl)         – reinforce successful patterns
            """

            # CUDA Integration with Fallback
                try:
                USING_CUDA = True
                _backend = 'cupy (GPU)'
                xp = cp
                    except ImportError:
                    USING_CUDA = False
                    _backend = 'numpy (CPU)'
                    xp = np

                    logger = logging.getLogger(__name__)
                        if USING_CUDA:
                        logger.info("⚡ VectorRegistry using GPU acceleration: {0}".format(_backend))
                            else:
                            logger.info("🔄 VectorRegistry using CPU fallback: {0}".format(_backend))

                            # ---------------------------------------------------------------------------
                            # Data structures
                            # ---------------------------------------------------------------------------


                            @dataclass
                                class StrategyVector:
    """Class for Schwabot trading functionality."""
                                """Class for Schwabot trading functionality."""
                                """Compressed strategy representation for a digest."""

                                digest: str  # SHA256 hex
                                strategy_id: str
                                asset_focus: str  # "BTC", "ETH", "XRP", "SOL", "USDC"
                                entry_confidence: float  # 0.0 to 1.0
                                exit_confidence: float  # 0.0 to 1.0
                                position_size: float  # 0.0 to 1.0 (fraction of, capital)
                                stop_loss_pct: float  # percentage
                                take_profit_pct: float  # percentage
                                rsi_band: int  # 0-100
                                volatility_class: int  # 0=low, 1=medium, 2=high
                                entropy_band: float  # normalized entropy
                                timestamp: float = field(default_factory=time.time)
                                usage_count: int = 0
                                success_rate: float = 0.0
                                avg_profit: float = 0.0


                                @dataclass
                                    class DigestMatch:
    """Class for Schwabot trading functionality."""
                                    """Class for Schwabot trading functionality."""
                                    """Result of digest similarity search."""

                                    digest: str
                                    similarity_score: float
                                    strategy_vector: StrategyVector
                                    hamming_distance: int
                                    entropy_diff: float


                                    # ---------------------------------------------------------------------------
                                    # Vector Registry Core
                                    # ---------------------------------------------------------------------------


                                        class VectorRegistry:
    """Class for Schwabot trading functionality."""
                                        """Class for Schwabot trading functionality."""
                                        """Maps SHA digests to strategy vectors with GPU-accelerated similarity search."""

                                            def __init__(self, registry_path: Optional[str] = None) -> None:
                                            self.registry_path = Path(registry_path) if registry_path else Path("data/vector_registry.json")
                                            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

                                            # In-memory storage
                                            self.digest_vectors: Dict[str, StrategyVector] = {}
                                            self.digest_embeddings: Dict[str, List[float]] = {}  # for similarity search

                                            # Performance tracking
                                            self.total_searches = 0
                                            self.successful_matches = 0
                                            self.avg_search_time = 0.0

                                            # Load existing registry
                                            self._load_registry()

                                            logger.info("Vector Registry initialized with {0} vectors".format(len(self.digest_vectors)))

                                                def register_digest(self, digest: bytes, strategy_vector: StrategyVector) -> bool:
                                                """Register a new digest→strategy mapping."""
                                                    try:
                                                    digest_hex = digest.hex()
                                                    strategy_vector.digest = digest_hex

                                                    # Store vector
                                                    self.digest_vectors[digest_hex] = strategy_vector

                                                    # Create embedding for similarity search
                                                    embedding = self._create_embedding(digest, strategy_vector)
                                                    self.digest_embeddings[digest_hex] = embedding

                                                    # Persist to disk
                                                    self._save_registry()

                                                    logger.debug("Registered digest {0}... with strategy {1}".format(
                                                    digest_hex[:16], strategy_vector.strategy_id))
                                                return True

                                                    except Exception as e:
                                                    logger.error("Failed to register digest: {0}".format(e))
                                                return False

                                                    def find_similar_digests(self, digest: bytes, threshold: float = 0.7, max_results: int = 10) -> List[DigestMatch]:
                                                    """Find similar historical digests using GPU-accelerated similarity search."""
                                                    start_time = time.time()

                                                        try:
                                                        digest_hex = digest.hex()
                                                        query_embedding = self._create_query_embedding(digest)

                                                        matches = []

                                                            if USING_CUDA and cp.cuda.is_available() and len(self.digest_embeddings) > 100:
                                                            # GPU-accelerated batch similarity search
                                                            matches = self._gpu_similarity_search(digest_hex, query_embedding, threshold, max_results)
                                                                else:
                                                                # CPU similarity search
                                                                matches = self._cpu_similarity_search(digest_hex, query_embedding, threshold, max_results)

                                                                # Update performance metrics
                                                                search_time = time.time() - start_time
                                                                self.total_searches += 1
                                                                self.avg_search_time = ()
                                                                self.avg_search_time * (self.total_searches - 1) + search_time
                                                                ) / self.total_searches

                                                                    if matches:
                                                                    self.successful_matches += 1

                                                                    logger.debug("Found {0} similar digests in {1:.4f}s".format())
                                                                    len(matches), search_time))
                                                                return matches

                                                                    except Exception as e:
                                                                    logger.error("Similarity search failed: {0}".format(e))
                                                                return []

                                                                    def get_confidence_score(self, digest: bytes, strategy_id: str) -> float:
                                                                    """Calculate confidence score based on historical performance."""
                                                                        try:
                                                                        digest_hex = digest.hex()

                                                                        # Find similar digests
                                                                        similar = self.find_similar_digests(digest, threshold=0.6, max_results=20)

                                                                            if not similar:
                                                                        return 0.5  # neutral confidence

                                                                        # Filter by strategy
                                                                        strategy_matches = [m for m in similar if m.strategy_vector.strategy_id == strategy_id]

                                                                            if not strategy_matches:
                                                                        return 0.3  # low confidence for new strategy

                                                                        # Calculate weighted confidence
                                                                        total_weight = 0.0
                                                                        weighted_confidence = 0.0

                                                                            for match in strategy_matches:
                                                                            weight = match.similarity_score * match.strategy_vector.success_rate
                                                                            weighted_confidence += weight * match.strategy_vector.entry_confidence
                                                                            total_weight += weight

                                                                                if total_weight == 0:
                                                                            return 0.5

                                                                            confidence = weighted_confidence / total_weight
                                                                        return max(0.0, min(1.0, confidence))

                                                                            except Exception as e:
                                                                            logger.error("Confidence calculation failed: {0}".format(e))
                                                                        return 0.5

                                                                            def update_profit_outcome(self, digest: bytes, pnl: float, success: bool=None) -> bool:
                                                                            """Update profit outcome for a digest to reinforce learning."""
                                                                                try:
                                                                                digest_hex = digest.hex()

                                                                                    if digest_hex not in self.digest_vectors:
                                                                                    logger.warning("Digest {0}... not found in registry".format(digest_hex[:16]))
                                                                                return False

                                                                                vector = self.digest_vectors[digest_hex]

                                                                                # Update success rate
                                                                                    if success is None:
                                                                                    success = pnl > 0

                                                                                    vector.usage_count += 1
                                                                                    vector.avg_profit = (vector.avg_profit * (vector.usage_count - 1) + pnl) / vector.usage_count

                                                                                        if success:
                                                                                        vector.success_rate = (vector.success_rate * (vector.usage_count - 1) + 1) / vector.usage_count
                                                                                            else:
                                                                                            vector.success_rate = (vector.success_rate * (vector.usage_count - 1)) / vector.usage_count

                                                                                            # Persist updates
                                                                                            self._save_registry()

                                                                                            logger.debug()
                                                                                            "Updated digest {0}... success_rate={1:.3f}, avg_profit={2:.4f}".format()
                                                                                            digest_hex[:16], vector.success_rate, vector.avg_profit))
                                                                                        return True

                                                                                            except Exception as e:
                                                                                            logger.error("Failed to update profit outcome: {0}".format(e))
                                                                                        return False

                                                                                            def get_registry_stats(self) -> Dict[str, Any]:
                                                                                            """Get registry statistics and performance metrics."""
                                                                                                if not self.digest_vectors:
                                                                                            return {"error": "No vectors in registry"}

                                                                                            vectors = list(self.digest_vectors.values())

                                                                                            stats = {}
                                                                                            "total_vectors": len(vectors),
                                                                                            "total_searches": self.total_searches,
                                                                                            "successful_matches": self.successful_matches,
                                                                                            "match_rate": self.successful_matches / max(self.total_searches, 1),
                                                                                            "avg_search_time": self.avg_search_time,
                                                                                            "backend_info": get_backend_info(),
                                                                                            # Strategy distribution
                                                                                            "strategy_counts": {},
                                                                                            "asset_distribution": {},
                                                                                            # Performance metrics
                                                                                            "avg_success_rate": sum(v.success_rate for v in , vectors) / len(vectors),
                                                                                            "avg_profit": sum(v.avg_profit for v in , vectors) / len(vectors),
                                                                                            "high_confidence_vectors": len([v for v in vectors if v.entry_confidence > 0.8]),
                                                                                            }

                                                                                            # Count strategies and assets
                                                                                                for vector in vectors:
                                                                                                stats["strategy_counts"][vector.strategy_id] = stats["strategy_counts"].get(vector.strategy_id, 0) + 1
                                                                                                stats["asset_distribution"][vector.asset_focus] = stats["asset_distribution"].get(vector.asset_focus, 0) + 1

                                                                                            return stats

                                                                                            # ---------------------------------------------------------------------------
                                                                                            # Internal methods
                                                                                            # ---------------------------------------------------------------------------

                                                                                                def _create_embedding(self, digest: bytes, vector: StrategyVector) -> List[float]:
                                                                                                """Create a numerical embedding for similarity search."""
                                                                                                # Convert digest to numerical features
                                                                                                digest_entropy = bit_entropy(digest)
                                                                                                hamming_wt = sum(bin(b).count('1') for b in digest)

                                                                                                # Create feature vector
                                                                                                embedding = []
                                                                                                digest_entropy,
                                                                                                hamming_wt / 256.0,  # normalized
                                                                                                vector.entry_confidence,
                                                                                                vector.exit_confidence,
                                                                                                vector.position_size,
                                                                                                vector.rsi_band / 100.0,  # normalized
                                                                                                vector.volatility_class / 2.0,  # normalized
                                                                                                vector.entropy_band,
                                                                                                vector.success_rate,
                                                                                                vector.avg_profit,
                                                                                                ]

                                                                                            return embedding

                                                                                                def _create_query_embedding(self, digest: bytes) -> List[float]:
                                                                                                """Create embedding for query digest (without strategy, info)."""
                                                                                                digest_entropy = bit_entropy(digest)
                                                                                                hamming_wt = sum(bin(b).count('1') for b in digest)

                                                                                                # Query embedding has fewer features (no strategy-specific, data)
                                                                                                embedding = []
                                                                                                digest_entropy,
                                                                                                hamming_wt / 256.0,
                                                                                                0.5,  # neutral confidence
                                                                                                0.5,  # neutral confidence
                                                                                                0.5,  # neutral position size
                                                                                                0.5,  # neutral RSI
                                                                                                1.0,  # medium volatility
                                                                                                0.5,  # neutral entropy
                                                                                                0.5,  # neutral success rate
                                                                                                0.0,  # neutral profit
                                                                                                ]

                                                                                            return embedding

                                                                                            def _gpu_similarity_search()
                                                                                            self, query_digest: str, query_embedding: List[float], threshold: float, max_results: int
                                                                                                ) -> List[DigestMatch]:
                                                                                                """GPU-accelerated similarity search."""
                                                                                                    try:
                                                                                                    # Convert to GPU arrays
                                                                                                    query_gpu = cp.asarray(query_embedding, dtype=cp.float32)

                                                                                                    # Batch all embeddings
                                                                                                    digest_hexes = list(self.digest_embeddings.keys())
                                                                                                    embeddings = [self.digest_embeddings[d] for d in digest_hexes]
                                                                                                    embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)

                                                                                                    # Calculate cosine similarities
                                                                                                    similarities = cp.zeros(len(embeddings), dtype=cp.float32)

                                                                                                        for i in range(len(embeddings)):
                                                                                                        vec_gpu = embeddings_gpu[i]
                                                                                                        dot_product = cp.dot(query_gpu, vec_gpu)
                                                                                                        norm_query = cp.linalg.norm(query_gpu)
                                                                                                        norm_vec = cp.linalg.norm(vec_gpu)

                                                                                                            if norm_query > 0 and norm_vec > 0:
                                                                                                            similarities[i] = dot_product / (norm_query * norm_vec)

                                                                                                            # Get top matches
                                                                                                            top_indices = cp.argsort(similarities)[::-1][:max_results]
                                                                                                            top_similarities = similarities[top_indices]

                                                                                                            # Convert back to CPU
                                                                                                            top_indices = cp.asnumpy(top_indices)
                                                                                                            top_similarities = cp.asnumpy(top_similarities)

                                                                                                            # Build results
                                                                                                            matches = []
                                                                                                                for idx, sim_score in zip(top_indices, top_similarities):
                                                                                                                    if sim_score >= threshold:
                                                                                                                    digest_hex = digest_hexes[idx]
                                                                                                                    vector = self.digest_vectors[digest_hex]

                                                                                                                    # Calculate Hamming distance
                                                                                                                    query_bytes = bytes.fromhex(query_digest)
                                                                                                                    stored_bytes = bytes.fromhex(digest_hex)
                                                                                                                    hamming_dist = hamming_distance(query_bytes, stored_bytes)

                                                                                                                    # Calculate entropy difference
                                                                                                                    entropy_diff = abs(bit_entropy(query_bytes) - bit_entropy(stored_bytes))

                                                                                                                    matches.append()
                                                                                                                    DigestMatch()
                                                                                                                    digest = digest_hex,
                                                                                                                    similarity_score = float(sim_score),
                                                                                                                    strategy_vector = vector,
                                                                                                                    hamming_distance = hamming_dist,
                                                                                                                    entropy_diff = entropy_diff,
                                                                                                                    )
                                                                                                                    )

                                                                                                                return matches

                                                                                                                    except Exception as e:
                                                                                                                    logger.warning("GPU similarity search failed, falling back to CPU: {0}".format(e))
                                                                                                                return self._cpu_similarity_search(query_digest, query_embedding, threshold, max_results)

                                                                                                                def _cpu_similarity_search()
                                                                                                                self, query_digest: str, query_embedding: List[float], threshold: float, max_results: int
                                                                                                                    ) -> List[DigestMatch]:
                                                                                                                    """CPU similarity search."""
                                                                                                                    matches = []

                                                                                                                        for digest_hex, stored_embedding in self.digest_embeddings.items():
                                                                                                                        # Calculate cosine similarity
                                                                                                                        similarity = vector_similarity(query_embedding, stored_embedding)

                                                                                                                            if similarity >= threshold:
                                                                                                                            vector = self.digest_vectors[digest_hex]

                                                                                                                            # Calculate Hamming distance
                                                                                                                            query_bytes = bytes.fromhex(query_digest)
                                                                                                                            stored_bytes = bytes.fromhex(digest_hex)
                                                                                                                            hamming_dist = hamming_distance(query_bytes, stored_bytes)

                                                                                                                            # Calculate entropy difference
                                                                                                                            entropy_diff = abs(bit_entropy(query_bytes) - bit_entropy(stored_bytes))

                                                                                                                            matches.append()
                                                                                                                            DigestMatch()
                                                                                                                            digest = digest_hex,
                                                                                                                            similarity_score = similarity,
                                                                                                                            strategy_vector = vector,
                                                                                                                            hamming_distance = hamming_dist,
                                                                                                                            entropy_diff = entropy_diff,
                                                                                                                            )
                                                                                                                            )

                                                                                                                            # Sort by similarity and return top results
                                                                                                                            matches.sort(key=lambda m: m.similarity_score, reverse=True)
                                                                                                                        return matches[:max_results]

                                                                                                                            def _load_registry(self) -> None:
                                                                                                                            """Load registry from disk."""
                                                                                                                                try:
                                                                                                                                    if self.registry_path.exists():
                                                                                                                                        with open(self.registry_path, 'r') as f:
                                                                                                                                        data = json.load(f)

                                                                                                                                        # Load vectors
                                                                                                                                            for digest_hex, vector_data in data.get('vectors', {}).items():
                                                                                                                                            vector = StrategyVector(**vector_data)
                                                                                                                                            self.digest_vectors[digest_hex] = vector

                                                                                                                                            # Recreate embeddings
                                                                                                                                            digest_bytes = bytes.fromhex(digest_hex)
                                                                                                                                            embedding = self._create_embedding(digest_bytes, vector)
                                                                                                                                            self.digest_embeddings[digest_hex] = embedding

                                                                                                                                            # Load performance metrics
                                                                                                                                            metrics = data.get('metrics', {})
                                                                                                                                            self.total_searches = metrics.get('total_searches', 0)
                                                                                                                                            self.successful_matches = metrics.get('successful_matches', 0)
                                                                                                                                            self.avg_search_time = metrics.get('avg_search_time', 0.0)

                                                                                                                                            logger.info("Loaded {0} vectors from registry".format(len(self.digest_vectors)))

                                                                                                                                                except Exception as e:
                                                                                                                                                logger.error("Failed to load registry: {0}".format(e))

                                                                                                                                                    def _save_registry(self) -> None:
                                                                                                                                                    """Save registry to disk."""
                                                                                                                                                        try:
                                                                                                                                                        # Prepare data
                                                                                                                                                        data = {}
                                                                                                                                                        'vectors': {},
                                                                                                                                                        'metrics': {}
                                                                                                                                                        'total_searches': self.total_searches,
                                                                                                                                                        'successful_matches': self.successful_matches,
                                                                                                                                                        'avg_search_time': self.avg_search_time,
                                                                                                                                                        },
                                                                                                                                                        }

                                                                                                                                                        # Convert vectors to dict
                                                                                                                                                            for digest_hex, vector in self.digest_vectors.items():
                                                                                                                                                            data['vectors'][digest_hex] = {}
                                                                                                                                                            'digest': vector.digest,
                                                                                                                                                            'strategy_id': vector.strategy_id,
                                                                                                                                                            'asset_focus': vector.asset_focus,
                                                                                                                                                            'entry_confidence': vector.entry_confidence,
                                                                                                                                                            'exit_confidence': vector.exit_confidence,
                                                                                                                                                            'position_size': vector.position_size,
                                                                                                                                                            'stop_loss_pct': vector.stop_loss_pct,
                                                                                                                                                            'take_profit_pct': vector.take_profit_pct,
                                                                                                                                                            'rsi_band': vector.rsi_band,
                                                                                                                                                            'volatility_class': vector.volatility_class,
                                                                                                                                                            'entropy_band': vector.entropy_band,
                                                                                                                                                            'timestamp': vector.timestamp,
                                                                                                                                                            'usage_count': vector.usage_count,
                                                                                                                                                            'success_rate': vector.success_rate,
                                                                                                                                                            'avg_profit': vector.avg_profit,
                                                                                                                                                            }

                                                                                                                                                            # Save to disk
                                                                                                                                                                with open(self.registry_path, 'w') as f:
                                                                                                                                                                json.dump(data, f, indent=2)

                                                                                                                                                                    except Exception as e:
                                                                                                                                                                    logger.error("Failed to save registry: {0}".format(e))


                                                                                                                                                                    # ---------------------------------------------------------------------------
                                                                                                                                                                    # Quick self-test
                                                                                                                                                                    # ---------------------------------------------------------------------------
                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                        # Test vector registry
                                                                                                                                                                        registry = VectorRegistry("test_registry.json")

                                                                                                                                                                        # Create test vectors
                                                                                                                                                                        test_digest = hashlib.sha256(b"test_digest").digest()
                                                                                                                                                                        test_vector = StrategyVector()
                                                                                                                                                                        digest = "",
                                                                                                                                                                        strategy_id = "test_strategy",
                                                                                                                                                                        asset_focus = "BTC",
                                                                                                                                                                        entry_confidence = 0.8,
                                                                                                                                                                        exit_confidence = 0.7,
                                                                                                                                                                        position_size = 0.5,
                                                                                                                                                                        stop_loss_pct = 2.0,
                                                                                                                                                                        take_profit_pct = 5.0,
                                                                                                                                                                        rsi_band = 50,
                                                                                                                                                                        volatility_class = 1,
                                                                                                                                                                        entropy_band = 0.6,
                                                                                                                                                                        )

                                                                                                                                                                        # Register and test
                                                                                                                                                                        registry.register_digest(test_digest, test_vector)
                                                                                                                                                                        matches = registry.find_similar_digests(test_digest, threshold=0.5)
                                                                                                                                                                        confidence = registry.get_confidence_score(test_digest, "test_strategy")

                                                                                                                                                                        print("Found {0} matches".format(len(matches)))
                                                                                                                                                                        print("Confidence))"
                                                                                                                                                                        print("Registry stats:", registry.get_registry_stats())
