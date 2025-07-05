#!/usr/bin/env python3
"""
🧬🔐 HASH-GLYPH MEMORY COMPRESSION
==================================

Compresses strategy vectors, glyphs, and AI votes into hash-based memory chunks.
Enables fast lookup and replay of successful decision paths.

Core Concept: HASH → GLYPH → VECTOR + DECISION PATH
"""

import hashlib
import json
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class GlyphMemoryChunk:
    """Compressed memory chunk containing glyph, vector, and votes"""
    hash_key: str
    glyph: str
    vector: List[float]
    votes: Dict[str, str]
    strategy_id: str
    timestamp: float
    confidence: float
    usage_count: int = 0
    last_accessed: float = 0.0

class HashGlyphCompressor:
    """
    Hash-Glyph Memory Compressor
    
    Compresses strategy decisions into hash-based memory chunks for fast lookup
    and replay of successful decision paths.
    """
    
    def __init__(self, max_memory_size: int = 1000, compression_threshold: float = 0.8):
        """
        Initialize hash-glyph compressor
        
        Args:
            max_memory_size: Maximum number of memory chunks to store
            compression_threshold: Confidence threshold for compression
        """
        self.glyph_memory: OrderedDict[str, GlyphMemoryChunk] = OrderedDict()
        self.max_memory_size = max_memory_size
        self.compression_threshold = compression_threshold
        self.stats = {
            "stored": 0,
            "retrieved": 0,
            "hits": 0,
            "misses": 0,
            "compressed": 0
        }
        
        logger.info(f"Hash-Glyph Compressor initialized (max_size: {max_memory_size}, threshold: {compression_threshold})")
    
    def store(self, strategy_id: str, q_matrix: np.ndarray, glyph: str, 
              vector: np.ndarray, votes: Dict[str, str], confidence: float = 1.0) -> str:
        """
        Store a strategy decision in compressed memory
        
        Args:
            strategy_id: Strategy identifier
            q_matrix: Qutrit matrix
            glyph: Associated glyph
            vector: Profit vector
            votes: AI agent votes
            confidence: Decision confidence
            
        Returns:
            Hash key for the stored memory
        """
        try:
            # Generate hash key
            hash_key = self._hash_key(strategy_id, q_matrix)
            
            # Create memory chunk
            memory_chunk = GlyphMemoryChunk(
                hash_key=hash_key,
                glyph=glyph,
                vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                votes=votes.copy(),
                strategy_id=strategy_id,
                timestamp=time.time(),
                confidence=confidence,
                usage_count=1,
                last_accessed=time.time()
            )
            
            # Store in memory
            self.glyph_memory[hash_key] = memory_chunk
            
            # Update stats
            self.stats["stored"] += 1
            self.stats["compressed"] += 1
            
            # Enforce memory limit
            if len(self.glyph_memory) > self.max_memory_size:
                self._cleanup_oldest()
            
            logger.debug(f"Stored glyph memory: {hash_key[:8]}... → {glyph}")
            return hash_key
            
        except Exception as e:
            logger.error(f"Error storing glyph memory: {e}")
            return ""
    
    def retrieve(self, strategy_id: str, q_matrix: np.ndarray) -> Optional[GlyphMemoryChunk]:
        """
        Retrieve a strategy decision from compressed memory
        
        Args:
            strategy_id: Strategy identifier
            q_matrix: Qutrit matrix
            
        Returns:
            Memory chunk if found, None otherwise
        """
        try:
            hash_key = self._hash_key(strategy_id, q_matrix)
            
            if hash_key in self.glyph_memory:
                # Update access statistics
                memory_chunk = self.glyph_memory[hash_key]
                memory_chunk.usage_count += 1
                memory_chunk.last_accessed = time.time()
                
                # Move to end (most recently used)
                self.glyph_memory.move_to_end(hash_key)
                
                # Update stats
                self.stats["retrieved"] += 1
                self.stats["hits"] += 1
                
                logger.debug(f"Retrieved glyph memory: {hash_key[:8]}... → {memory_chunk.glyph}")
                return memory_chunk
            else:
                self.stats["retrieved"] += 1
                self.stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving glyph memory: {e}")
            return None
    
    def _hash_key(self, strategy_id: str, q_matrix: np.ndarray) -> str:
        """
        Generate hash key from strategy ID and qutrit matrix
        
        Args:
            strategy_id: Strategy identifier
            q_matrix: Qutrit matrix
            
        Returns:
            SHA-256 hash key
        """
        try:
            # Flatten and normalize matrix
            flat_matrix = q_matrix.flatten().tolist()
            
            # Create payload
            payload = f"{strategy_id}:{json.dumps(flat_matrix, sort_keys=True)}"
            
            # Generate hash
            hash_key = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            return hash_key
            
        except Exception as e:
            logger.error(f"Error generating hash key: {e}")
            return ""
    
    def _cleanup_oldest(self) -> int:
        """
        Remove oldest memory chunks to maintain size limit
        
        Returns:
            Number of chunks removed
        """
        try:
            initial_size = len(self.glyph_memory)
            
            # Remove oldest entries until under limit
            while len(self.glyph_memory) > self.max_memory_size * 0.9:  # 90% of max
                # Remove least recently used
                oldest_key = next(iter(self.glyph_memory))
                del self.glyph_memory[oldest_key]
            
            removed = initial_size - len(self.glyph_memory)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old glyph memory chunks")
            
            return removed
            
        except Exception as e:
            logger.error(f"Error cleaning up memory: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        
        Returns:
            Dictionary of memory statistics
        """
        try:
            if not self.glyph_memory:
                return self.stats.copy()
            
            # Compute additional stats
            confidences = [chunk.confidence for chunk in self.glyph_memory.values()]
            usage_counts = [chunk.usage_count for chunk in self.glyph_memory.values()]
            
            # Get most used glyphs
            glyph_counts = {}
            for chunk in self.glyph_memory.values():
                glyph_counts[chunk.glyph] = glyph_counts.get(chunk.glyph, 0) + 1
            
            most_used_glyphs = sorted(glyph_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                **self.stats,
                "memory_size": len(self.glyph_memory),
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "max_confidence": np.max(confidences) if confidences else 0.0,
                "avg_usage_count": np.mean(usage_counts) if usage_counts else 0.0,
                "max_usage_count": np.max(usage_counts) if usage_counts else 0.0,
                "hit_rate": self.stats["hits"] / max(1, self.stats["retrieved"]),
                "most_used_glyphs": most_used_glyphs
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return self.stats.copy()
    
    def find_similar_patterns(self, strategy_id: str, q_matrix: np.ndarray, 
                            similarity_threshold: float = 0.8) -> List[GlyphMemoryChunk]:
        """
        Find similar patterns in memory
        
        Args:
            strategy_id: Strategy identifier
            q_matrix: Qutrit matrix
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar memory chunks
        """
        try:
            target_hash = self._hash_key(strategy_id, q_matrix)
            similar_chunks = []
            
            for chunk in self.glyph_memory.values():
                # Simple similarity based on hash prefix (first 8 chars)
                if chunk.hash_key[:8] == target_hash[:8]:
                    similar_chunks.append(chunk)
                elif chunk.strategy_id == strategy_id:
                    # Same strategy, consider similar
                    similar_chunks.append(chunk)
            
            # Sort by confidence and usage
            similar_chunks.sort(key=lambda x: (x.confidence, x.usage_count), reverse=True)
            
            return similar_chunks[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def export_memory(self, filepath: str) -> bool:
        """
        Export memory to JSON file
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            export_data = {
                "metadata": {
                    "version": "1.0",
                    "timestamp": time.time(),
                    "stats": self.stats
                },
                "memory": {}
            }
            
            for hash_key, chunk in self.glyph_memory.items():
                export_data["memory"][hash_key] = {
                    "glyph": chunk.glyph,
                    "vector": chunk.vector,
                    "votes": chunk.votes,
                    "strategy_id": chunk.strategy_id,
                    "timestamp": chunk.timestamp,
                    "confidence": chunk.confidence,
                    "usage_count": chunk.usage_count,
                    "last_accessed": chunk.last_accessed
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.glyph_memory)} memory chunks to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting memory: {e}")
            return False
    
    def import_memory(self, filepath: str) -> bool:
        """
        Import memory from JSON file
        
        Args:
            filepath: Input file path
            
        Returns:
            True if import successful
        """
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            for hash_key, chunk_data in import_data["memory"].items():
                memory_chunk = GlyphMemoryChunk(
                    hash_key=hash_key,
                    glyph=chunk_data["glyph"],
                    vector=chunk_data["vector"],
                    votes=chunk_data["votes"],
                    strategy_id=chunk_data["strategy_id"],
                    timestamp=chunk_data["timestamp"],
                    confidence=chunk_data["confidence"],
                    usage_count=chunk_data["usage_count"],
                    last_accessed=chunk_data["last_accessed"]
                )
                
                self.glyph_memory[hash_key] = memory_chunk
                imported_count += 1
            
            logger.info(f"Imported {imported_count} memory chunks from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing memory: {e}")
            return False


def create_hash_glyph_compressor(max_memory_size: int = 1000, 
                                compression_threshold: float = 0.8) -> HashGlyphCompressor:
    """
    Factory function to create HashGlyphCompressor
    
    Args:
        max_memory_size: Maximum number of memory chunks to store
        compression_threshold: Confidence threshold for compression
        
    Returns:
        Initialized HashGlyphCompressor instance
    """
    return HashGlyphCompressor(max_memory_size=max_memory_size, 
                              compression_threshold=compression_threshold)


def test_hash_glyph_compression():
    """Test function for hash-glyph compression"""
    print("🧬🔐 Testing Hash-Glyph Memory Compression")
    print("=" * 50)
    
    # Create compressor
    compressor = create_hash_glyph_compressor(max_memory_size=100)
    
    # Test data
    strategy_id = "test_strategy_compression"
    q_matrix = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
    glyph = "🌘"
    vector = np.array([0.1, 0.4, 0.3])
    votes = {"R1": "execute", "Claude": "recycle", "GPT-4o": "defer"}
    
    # Test 1: Store memory chunk
    print("\n💾 Test 1: Storing Memory Chunk")
    hash_key = compressor.store(strategy_id, q_matrix, glyph, vector, votes, confidence=0.9)
    print(f"  Stored with hash: {hash_key[:16]}...")
    
    # Test 2: Retrieve memory chunk
    print("\n🔍 Test 2: Retrieving Memory Chunk")
    retrieved = compressor.retrieve(strategy_id, q_matrix)
    if retrieved:
        print(f"  Retrieved: {retrieved.glyph} → {retrieved.vector}")
        print(f"  Votes: {retrieved.votes}")
        print(f"  Confidence: {retrieved.confidence}")
    else:
        print("  ❌ No memory found")
    
    # Test 3: Test with different matrix (should miss)
    print("\n❌ Test 3: Testing Cache Miss")
    different_matrix = np.array([[2, 1, 0], [1, 0, 2], [0, 2, 1]])
    missed = compressor.retrieve(strategy_id, different_matrix)
    print(f"  Cache miss: {missed is None}")
    
    # Test 4: Find similar patterns
    print("\n🔍 Test 4: Finding Similar Patterns")
    similar = compressor.find_similar_patterns(strategy_id, q_matrix)
    print(f"  Found {len(similar)} similar patterns")
    
    # Test 5: Get statistics
    print("\n📊 Test 5: Memory Statistics")
    stats = compressor.get_memory_stats()
    print(f"  Memory stats: {stats}")


if __name__ == "__main__":
    test_hash_glyph_compression() 