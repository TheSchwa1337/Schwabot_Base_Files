"""
Cursor Engine
============

Implements the Quantum Triplet Walker cursor that synchronizes triplet data
through tick-deltas, aligns market structures, and maintains logs for various
patterns and triggers.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .braid_pattern_engine import BraidPatternEngine, BraidPattern
import logging
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class CursorState:
    """Represents the current state of the cursor"""
    triplet: Tuple[float, float, float]
    timestamp: float
    velocity: float = 0.0
    entropy: float = 0.0

def load_config():
    config_path = Path(__file__).resolve().parent / 'config/matrix_response_paths.yaml'
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return {"defaults": True}

class Cursor:
    """Quantum Triplet Walker cursor implementation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
        self.pattern_engine = BraidPatternEngine()
        self.state: Optional[CursorState] = None
        self.history: List[CursorState] = []
        
    def tick(self, triplet: Tuple[float, float, float], timestamp: float) -> Optional[BraidPattern]:
        """Process a new tick with triplet data"""
        if self.state is None:
            self.state = CursorState(triplet, timestamp)
            self.history.append(self.state)
            return None
            
        velocity = np.linalg.norm(np.array(triplet) - np.array(self.state.triplet))
        entropy = -np.sum(np.square(np.array(triplet) / np.sum(triplet)))
        
        self.state = CursorState(triplet=triplet, timestamp=timestamp, velocity=velocity, entropy=entropy)
        self.history.append(self.state)
        
        # Update pattern engine
        self.pattern_engine.add_strand(self.state.velocity, self.state.entropy)
        
        # Check for patterns
        return self.pattern_engine.classify()
        
    def tick_vector(self, triplet_series: List[Tuple[float, float, float]], timestamps: List[float]) -> List[BraidPattern]:
        patterns = []
        for t, ts in zip(triplet_series, timestamps):
            pattern = self.tick(t, ts)
            if pattern:
                patterns.append(pattern)
        return patterns
        
    def get_current_pattern(self) -> Optional[BraidPattern]:
        """Get the current braid pattern"""
        return self.pattern_engine.classify()
        
    def get_pattern_frequency(self, pattern_name: str) -> float:
        """Calculate frequency of a specific pattern in recent history"""
        count = 0
        for s in self.history:
            sim = self.pattern_engine.simulate_pattern_from_states([s.triplet])
            if sim and sim[0].name == pattern_name:
                count += 1
        return count / len(self.history) if self.history else 0.0
        
    def get_history(self, limit: Optional[int] = None) -> List[CursorState]:
        """Get cursor history, optionally limited to last N states"""
        if limit is None:
            return self.history
        return self.history[-limit:]
        
    def clear_history(self):
        """Clear cursor history"""
        self.history.clear()
        self.pattern_engine.clear_history()
        
    def add_custom_pattern(self, name: str, pattern: List[Tuple[int, float]]):
        """Add a custom pattern to the pattern library"""
        self.pattern_engine.add_custom_pattern(name, pattern)
        
    def reverse_tick(self, steps: int = 1) -> None:
        """Reverse the cursor by a specified number of steps"""
        for _ in range(min(steps, len(self.history))):
            self.history.pop()
        self.state = self.history[-1] if self.history else None
        
    def to_dict(self) -> Dict:
        """Convert cursor state to a dictionary"""
        return {
            "state": self.state.__dict__ if self.state else None,
            "history": [s.__dict__ for s in self.history]
        }
        
    def to_json(self) -> str:
        """Convert cursor state to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)
        
    def _calculate_angular_similarity(self, old_triplet: Tuple[float, float, float], new_triplet: Tuple[float, float, float]) -> float:
        """Calculate the angular similarity between two triplets"""
        vec1 = np.array(old_triplet)
        vec2 = np.array(new_triplet)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        return angle
        
    def calculate_angular_similarity(self, triplet1: Tuple[float, float, float], triplet2: Tuple[float, float, float]) -> float:
        """Calculate the angular similarity between two triplets"""
        vec1 = np.array(triplet1)
        vec2 = np.array(triplet2)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        return angle

# Example usage
if __name__ == "__main__":
    engine = Cursor()
    engine.add_custom_pattern("example", [(1, 0.5), (2, 0.3)])
    triplet1 = (1.0, 2.0, 3.0)
    triplet2 = (4.0, 5.0, 6.0)
    similarity = engine.calculate_angular_similarity(triplet1, triplet2)
    print(f"Angular Similarity: {similarity} degrees") 