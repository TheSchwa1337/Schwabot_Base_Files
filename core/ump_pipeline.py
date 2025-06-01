"""
Universal Memory Pipeline (UMP)
==============================

Handles memory-safe tick ingestion, pattern storage, and strategy logic access.
Integrates with cyclic core and NCCO system for pattern recognition and profit navigation.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import hashlib
import json
from pathlib import Path
import threading
from collections import deque
from .cursor_engine import Cursor, CursorState
from .braid_pattern_engine import BraidPattern, BraidPatternEngine

@dataclass
class MemoryEntry:
    """Represents a single memory entry in the pipeline"""
    timestamp: float
    vector: np.ndarray
    triplet: str
    pattern_hash: str
    confidence: float
    metadata: Dict[str, Any]
    success_score: float = 0.0
    cursor_state: CursorState
    pattern: Optional[BraidPattern] = None

class UMPipeline:
    """Universal Memory Pipeline for pattern storage and retrieval"""
    
    def __init__(self, max_memory_size: int = 1024, pattern_cache_size: int = 512):
        self.max_memory_size = max_memory_size
        self.pattern_cache_size = pattern_cache_size
        
        # Core memory structures
        self.memory_buffer = deque(maxlen=max_memory_size)
        self.pattern_cache = {}
        self.hash_index = {}
        self.success_rates = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.total_entries = 0
        self.pattern_hits = 0
        self.pattern_misses = 0
    
    def ingest_tick(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """
        Ingest a new tick vector into memory
        
        Args:
            vector: Input vector
            metadata: Additional metadata
            
        Returns:
            Pattern hash of ingested tick
        """
        with self._lock:
            # Generate triplet and pattern hash
            triplet = self._vector_to_triplet(vector)
            pattern_hash = self._generate_pattern_hash(triplet, vector)
            
            # Create memory entry
            entry = MemoryEntry(
                timestamp=datetime.now().timestamp(),
                vector=vector,
                triplet=triplet,
                pattern_hash=pattern_hash,
                confidence=self._calculate_confidence(triplet),
                metadata=metadata or {},
                success_score=0.0,
                cursor_state=CursorState(vector),
                pattern=None
            )
            
            # Add to memory buffer
            self.memory_buffer.append(entry)
            self.hash_index[pattern_hash] = entry
            self.total_entries += 1
            
            # Update pattern cache
            self._update_pattern_cache(triplet, entry)
            
            return pattern_hash
    
    def _vector_to_triplet(self, vector: np.ndarray) -> str:
        """Convert vector to 3-digit triplet string"""
        # Scale vector to 0-999 range and convert to triplet
        scaled = abs(vector[0] * 1000) % 1000
        return str(int(scaled)).zfill(3)
    
    def _generate_pattern_hash(self, triplet: str, vector: np.ndarray) -> str:
        """Generate unique hash for pattern"""
        data = f"{triplet}_{vector.tobytes().hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_confidence(self, triplet: str) -> float:
        """Calculate confidence score for pattern"""
        if triplet in self.success_rates:
            return min(self.success_rates[triplet] * 1.2, 1.0)
        return 0.5  # Default confidence
    
    def _update_pattern_cache(self, triplet: str, entry: MemoryEntry):
        """Update pattern cache with new entry"""
        if triplet not in self.pattern_cache:
            self.pattern_cache[triplet] = []
        
        self.pattern_cache[triplet].append(entry)
        
        # Trim cache if needed
        if len(self.pattern_cache[triplet]) > self.pattern_cache_size:
            self.pattern_cache[triplet] = self.pattern_cache[triplet][-self.pattern_cache_size:]
    
    def find_similar_patterns(self, pattern_hash: str, 
                            max_distance: float = 0.1) -> List[MemoryEntry]:
        """
        Find similar patterns in memory
        
        Args:
            pattern_hash: Pattern hash to match
            max_distance: Maximum distance for similarity
            
        Returns:
            List of similar memory entries
        """
        with self._lock:
            if pattern_hash in self.hash_index:
                self.pattern_hits += 1
                return [self.hash_index[pattern_hash]]
            
            self.pattern_misses += 1
            return []
    
    def update_success_score(self, pattern_hash: str, success_score: float):
        """
        Update success score for pattern
        
        Args:
            pattern_hash: Pattern hash
            success_score: New success score
        """
        with self._lock:
            if pattern_hash in self.hash_index:
                entry = self.hash_index[pattern_hash]
                entry.success_score = success_score
                
                # Update success rates
                if entry.triplet not in self.success_rates:
                    self.success_rates[entry.triplet] = []
                self.success_rates[entry.triplet].append(success_score)
                
                # Keep only recent scores
                if len(self.success_rates[entry.triplet]) > 100:
                    self.success_rates[entry.triplet] = self.success_rates[entry.triplet][-100:]
    
    def get_pattern_history(self, triplet: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Get recent history for pattern
        
        Args:
            triplet: Pattern triplet
            limit: Maximum number of entries
            
        Returns:
            List of recent memory entries
        """
        with self._lock:
            if triplet in self.pattern_cache:
                return list(self.pattern_cache[triplet])[-limit:]
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory pipeline statistics"""
        with self._lock:
            return {
                "total_entries": self.total_entries,
                "current_memory_size": len(self.memory_buffer),
                "pattern_cache_size": len(self.pattern_cache),
                "pattern_hits": self.pattern_hits,
                "pattern_misses": self.pattern_misses,
                "hit_rate": self.pattern_hits / max(1, self.pattern_hits + self.pattern_misses)
            }
    
    def clear_memory(self):
        """Clear all memory buffers"""
        with self._lock:
            self.memory_buffer.clear()
            self.pattern_cache.clear()
            self.hash_index.clear()
            self.success_rates.clear()
            self.total_entries = 0
            self.pattern_hits = 0
            self.pattern_misses = 0
    
    def export_state(self) -> Dict[str, Any]:
        """Export current memory state"""
        with self._lock:
            return {
                "memory_entries": [
                    {
                        "timestamp": e.timestamp,
                        "vector": e.vector.tolist(),
                        "triplet": e.triplet,
                        "pattern_hash": e.pattern_hash,
                        "confidence": e.confidence,
                        "metadata": e.metadata,
                        "success_score": e.success_score,
                        "cursor_state": e.cursor_state.to_dict(),
                        "pattern": e.pattern.to_dict() if e.pattern else None
                    }
                    for e in self.memory_buffer
                ],
                "pattern_cache": {
                    triplet: [
                        {
                            "timestamp": e.timestamp,
                            "vector": e.vector.tolist(),
                            "pattern_hash": e.pattern_hash,
                            "confidence": e.confidence,
                            "success_score": e.success_score
                        }
                        for e in entries
                    ]
                    for triplet, entries in self.pattern_cache.items()
                },
                "success_rates": self.success_rates,
                "statistics": self.get_memory_stats()
            }
    
    def import_state(self, state_data: Dict[str, Any]):
        """Import memory state"""
        with self._lock:
            self.clear_memory()
            
            # Import memory entries
            for entry_data in state_data.get("memory_entries", []):
                entry = MemoryEntry(
                    timestamp=entry_data["timestamp"],
                    vector=np.array(entry_data["vector"]),
                    triplet=entry_data["triplet"],
                    pattern_hash=entry_data["pattern_hash"],
                    confidence=entry_data["confidence"],
                    metadata=entry_data["metadata"],
                    success_score=entry_data["success_score"],
                    cursor_state=CursorState.from_dict(entry_data["cursor_state"]),
                    pattern=BraidPattern.from_dict(entry_data["pattern"]) if entry_data["pattern"] else None
                )
                self.memory_buffer.append(entry)
                self.hash_index[entry.pattern_hash] = entry
            
            # Import pattern cache
            for triplet, entries_data in state_data.get("pattern_cache", {}).items():
                self.pattern_cache[triplet] = []
                for entry_data in entries_data:
                    entry = MemoryEntry(
                        timestamp=entry_data["timestamp"],
                        vector=np.array(entry_data["vector"]),
                        triplet=triplet,
                        pattern_hash=entry_data["pattern_hash"],
                        confidence=entry_data["confidence"],
                        metadata={},
                        success_score=entry_data["success_score"],
                        cursor_state=CursorState(np.array(entry_data["vector"])),
                        pattern=None
                    )
                    self.pattern_cache[triplet].append(entry)
            
            # Import success rates
            self.success_rates = state_data.get("success_rates", {})
            
            # Update statistics
            stats = state_data.get("statistics", {})
            self.total_entries = stats.get("total_entries", 0)
            self.pattern_hits = stats.get("pattern_hits", 0)
            self.pattern_misses = stats.get("pattern_misses", 0)

# Example usage
if __name__ == "__main__":
    ump = UMPipeline()
    
    # Test vector ingestion
    test_vectors = [
        np.array([0.001]),
        np.array([0.172]),
        np.array([0.345]),
        np.array([0.998])
    ]
    
    print("Testing UMPipeline:")
    print("=" * 40)
    
    for vector in test_vectors:
        pattern_hash = ump.ingest_tick(vector)
        print(f"Vector {vector[0]:.3f} -> {pattern_hash}")
    
    # Test pattern matching
    test_hash = ump.ingest_tick(np.array([0.001]))
    similar = ump.find_similar_patterns(test_hash)
    print(f"\nFound {len(similar)} similar patterns")
    
    # Show statistics
    print("\nMemory Statistics:")
    stats = ump.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}") 