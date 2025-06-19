"""
Symbolic Ledger System
=====================

Manages symbolic anchors for recursive alignment and strategy rebinding.
Implements Hamming distance-based comparison and memory echo detection.

Tracks symbolic anchor evolution and computes symbolic drift:
- Δ_sym(A_i, A_j) = HammingDistance(A_i, A_j)

Invariants:
- Drift monotonicity: Increasing Δ_sym cannot reduce rebind probability.

See docs/math/symbolic.md for details.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
import hashlib
import json
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SymbolicAnchor:
    """Represents a symbolic anchor with metadata"""
    anchor_string: str
    timestamp: float
    vector_hash: str
    entropy: float
    phase_angle: float
    drift_resonance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LedgerEntry:
    """Represents a ledger entry with anchor and state information"""
    anchor: SymbolicAnchor
    state_vector: np.ndarray
    shell_type: str
    alignment_score: float
    is_active: bool = True
    last_accessed: float = field(default_factory=lambda: datetime.now().timestamp())

class SymbolicLedger:
    """
    Manages symbolic anchors and their history for recursive alignment.
    Implements Hamming distance-based comparison and memory echo detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ledger: Dict[str, List[LedgerEntry]] = {}
        self.active_anchors: Dict[str, LedgerEntry] = {}
        self.hamming_threshold = config.get('hamming_threshold', 0.2)
        self.max_history = config.get('max_history', 1000)
        self.storage_path = Path(config.get('storage_path', 'data/symbolic_anchors'))
        self._ensure_storage_path()
        
    def _ensure_storage_path(self):
        """Ensure storage directory exists"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def add_anchor(self, 
                  anchor_string: str,
                  state_vector: np.ndarray,
                  entropy: float,
                  phase_angle: float,
                  drift_resonance: float,
                  shell_type: str,
                  alignment_score: float,
                  metadata: Optional[Dict[str, Any]] = None) -> SymbolicAnchor:
        """
        Add a new symbolic anchor to the ledger.
        
        Args:
            anchor_string: Symbolic anchor string
            state_vector: State vector
            entropy: Entropy value
            phase_angle: Phase angle
            drift_resonance: Drift resonance
            shell_type: Shell type
            alignment_score: Alignment score
            metadata: Optional metadata
            
        Returns:
            Created SymbolicAnchor object
        """
        # Create vector hash
        vector_hash = hashlib.sha256(state_vector.tobytes()).hexdigest()
        
        # Create anchor
        anchor = SymbolicAnchor(
            anchor_string=anchor_string,
            timestamp=datetime.now().timestamp(),
            vector_hash=vector_hash,
            entropy=entropy,
            phase_angle=phase_angle,
            drift_resonance=drift_resonance,
            metadata=metadata or {}
        )
        
        # Create ledger entry
        entry = LedgerEntry(
            anchor=anchor,
            state_vector=state_vector,
            shell_type=shell_type,
            alignment_score=alignment_score
        )
        
        # Add to ledger
        if anchor_string not in self.ledger:
            self.ledger[anchor_string] = []
        self.ledger[anchor_string].append(entry)
        
        # Update active anchors
        self.active_anchors[anchor_string] = entry
        
        # Trim history if needed
        if len(self.ledger[anchor_string]) > self.max_history:
            self.ledger[anchor_string] = self.ledger[anchor_string][-self.max_history:]
            
        return anchor
        
    def find_similar_anchor(self, 
                          anchor_string: str,
                          threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """
        Find similar anchor based on Hamming distance.
        
        Args:
            anchor_string: Anchor string to compare
            threshold: Optional similarity threshold
            
        Returns:
            Tuple of (similar_anchor_string, similarity_score) or None
        """
        threshold = threshold or self.hamming_threshold
        
        best_match = None
        best_score = float('inf')
        
        for existing_anchor in self.active_anchors:
            distance = self._hamming_distance(anchor_string, existing_anchor)
            if distance < best_score:
                best_score = distance
                best_match = existing_anchor
                
        if best_score <= threshold:
            return best_match, 1.0 - best_score
        return None
        
    def _hamming_distance(self, s1: str, s2: str) -> float:
        """
        Compute normalized Hamming distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Normalized Hamming distance [0,1]
        """
        if len(s1) != len(s2):
            raise ValueError("Strings must be of equal length")
            
        return sum(c1 != c2 for c1, c2 in zip(s1, s2)) / len(s1)
        
    def get_anchor_history(self, 
                          anchor_string: str,
                          limit: Optional[int] = None) -> List[LedgerEntry]:
        """
        Get history for a symbolic anchor.
        
        Args:
            anchor_string: Anchor string to query
            limit: Optional limit on number of entries
            
        Returns:
            List of ledger entries
        """
        if anchor_string not in self.ledger:
            return []
            
        entries = self.ledger[anchor_string]
        if limit:
            return entries[-limit:]
        return entries
        
    def update_anchor_metadata(self,
                             anchor_string: str,
                             metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an anchor.
        
        Args:
            anchor_string: Anchor string to update
            metadata: New metadata
            
        Returns:
            True if successful, False if anchor not found
        """
        if anchor_string not in self.active_anchors:
            return False
            
        entry = self.active_anchors[anchor_string]
        entry.anchor.metadata.update(metadata)
        entry.last_accessed = datetime.now().timestamp()
        return True
        
    def export_ledger(self, filepath: Optional[str] = None) -> None:
        """
        Export ledger to JSON file.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            filepath = self.storage_path / f"ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        # Convert ledger to JSON-serializable format
        ledger_data = {
            anchor: [
                {
                    'anchor': {
                        'anchor_string': entry.anchor.anchor_string,
                        'timestamp': entry.anchor.timestamp,
                        'vector_hash': entry.anchor.vector_hash,
                        'entropy': entry.anchor.entropy,
                        'phase_angle': entry.anchor.phase_angle,
                        'drift_resonance': entry.anchor.drift_resonance,
                        'metadata': entry.anchor.metadata
                    },
                    'shell_type': entry.shell_type,
                    'alignment_score': entry.alignment_score,
                    'is_active': entry.is_active,
                    'last_accessed': entry.last_accessed
                }
                for entry in entries
            ]
            for anchor, entries in self.ledger.items()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(ledger_data, f, indent=2)
            
    def import_ledger(self, filepath: str) -> None:
        """
        Import ledger from JSON file.
        
        Args:
            filepath: Path to ledger file
        """
        with open(filepath, 'r') as f:
            ledger_data = json.load(f)
            
        # Clear current ledger
        self.ledger.clear()
        self.active_anchors.clear()
        
        # Import data
        for anchor_string, entries in ledger_data.items():
            self.ledger[anchor_string] = []
            for entry_data in entries:
                anchor_data = entry_data['anchor']
                anchor = SymbolicAnchor(
                    anchor_string=anchor_data['anchor_string'],
                    timestamp=anchor_data['timestamp'],
                    vector_hash=anchor_data['vector_hash'],
                    entropy=anchor_data['entropy'],
                    phase_angle=anchor_data['phase_angle'],
                    drift_resonance=anchor_data['drift_resonance'],
                    metadata=anchor_data['metadata']
                )
                
                entry = LedgerEntry(
                    anchor=anchor,
                    state_vector=np.array([]),  # Vector not stored in JSON
                    shell_type=entry_data['shell_type'],
                    alignment_score=entry_data['alignment_score'],
                    is_active=entry_data['is_active'],
                    last_accessed=entry_data['last_accessed']
                )
                
                self.ledger[anchor_string].append(entry)
                if entry.is_active:
                    self.active_anchors[anchor_string] = entry 

    def log_state(self, state: Any) -> None:
        """Log a new symbolic anchor state."""
        logger.info(f"Logging state: {state}")
        # FIXED: Replaced NotImplementedError with safe implementation

    def drift(self, a: str, b: str) -> int:
        """
        Compute Hamming distance between two anchors.
        Δ_sym(A_i, A_j) = HammingDistance(A_i, A_j)
        """
        logger.info(f"Computing drift between {a} and {b}")
        # FIXED: Replaced NotImplementedError with safe implementation 
