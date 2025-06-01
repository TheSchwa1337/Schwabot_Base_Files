"""
Vector Matcher Module
===================

Converts tick vectors into triplet codes and matches them against the cyclic map.
Integrates with cyclic core for pattern validation and profit navigation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
import hashlib
from datetime import datetime

class MatchType(Enum):
    """Types of pattern matches"""
    EXACT = "exact"
    NEAR = "near"
    SYMMETRY = "symmetry"
    BREAK = "break"

@dataclass
class MatchResult:
    """Result of vector pattern matching"""
    triplet: str
    match_type: MatchType
    confidence: float
    distance: int
    timestamp: float
    metadata: Dict[str, Any]

class VectorMatcher:
    """Matches vectors against cyclic patterns"""
    
    def __init__(self, max_distance: int = 3):
        self.max_distance = max_distance
        self.cyclic_triplets = self._generate_cyclic_map()
        self.match_history = []
        self.max_history = 1000
    
    def _generate_cyclic_map(self) -> List[str]:
        """Generate the complete 1/998001 triplet sequence"""
        triplets = []
        for i in range(1, 1000):
            triplet = str(i).zfill(3)
            triplets.append(triplet)
        return triplets
    
    def vector_to_triplet(self, vector: np.ndarray) -> str:
        """
        Convert vector to 3-digit triplet string
        
        Args:
            vector: Input vector
            
        Returns:
            Triplet string
        """
        # Scale vector to 0-999 range and convert to triplet
        scaled = abs(vector[0] * 1000) % 1000
        return str(int(scaled)).zfill(3)
    
    def calculate_distance(self, from_triplet: str, to_triplet: str) -> int:
        """
        Calculate cyclic distance between triplets
        
        Args:
            from_triplet: Starting triplet
            to_triplet: Target triplet
            
        Returns:
            Cyclic distance
        """
        try:
            from_idx = self.cyclic_triplets.index(from_triplet)
            to_idx = self.cyclic_triplets.index(to_triplet)
            
            # Calculate both forward and backward distances
            forward_dist = (to_idx - from_idx) % len(self.cyclic_triplets)
            backward_dist = (from_idx - to_idx) % len(self.cyclic_triplets)
            
            # Return shortest distance with direction
            if forward_dist <= backward_dist:
                return forward_dist
            else:
                return -backward_dist
        except ValueError:
            return 0
    
    def match_vector(self, vector: np.ndarray, 
                    metadata: Optional[Dict] = None) -> MatchResult:
        """
        Match vector against cyclic patterns
        
        Args:
            vector: Input vector
            metadata: Additional metadata
            
        Returns:
            Match result
        """
        # Convert to triplet
        triplet = self.vector_to_triplet(vector)
        
        # Determine match type and confidence
        if triplet == "998":
            match_type = MatchType.SYMMETRY
            confidence = 1.0
            distance = 0
        elif triplet in self.cyclic_triplets:
            # Calculate distance from last match
            if self.match_history:
                last_triplet = self.match_history[-1].triplet
                distance = self.calculate_distance(last_triplet, triplet)
                
                if abs(distance) <= self.max_distance:
                    match_type = MatchType.NEAR
                    confidence = 1.0 - (abs(distance) / self.max_distance)
                else:
                    match_type = MatchType.EXACT
                    confidence = 0.8
            else:
                match_type = MatchType.EXACT
                confidence = 0.8
                distance = 0
        else:
            match_type = MatchType.BREAK
            confidence = 0.0
            distance = 0
        
        # Create match result
        result = MatchResult(
            triplet=triplet,
            match_type=match_type,
            confidence=confidence,
            distance=distance,
            timestamp=datetime.now().timestamp(),
            metadata=metadata or {}
        )
        
        # Add to history
        self.match_history.append(result)
        if len(self.match_history) > self.max_history:
            self.match_history = self.match_history[-self.max_history:]
        
        return result
    
    def find_similar_patterns(self, target_triplet: str, 
                            max_distance: int = 3) -> List[MatchResult]:
        """
        Find similar patterns in history
        
        Args:
            target_triplet: Target triplet
            max_distance: Maximum distance for similarity
            
        Returns:
            List of similar matches
        """
        similar = []
        
        for match in self.match_history:
            distance = self.calculate_distance(target_triplet, match.triplet)
            if abs(distance) <= max_distance:
                similar.append(match)
        
        return similar
    
    def detect_pattern_sequence(self, sequence_length: int = 5) -> Optional[List[str]]:
        """
        Detect recurring pattern sequences
        
        Args:
            sequence_length: Length of sequence to detect
            
        Returns:
            Detected sequence or None
        """
        if len(self.match_history) < sequence_length * 2:
            return None
        
        recent_triplets = [m.triplet for m in self.match_history[-sequence_length*2:]]
        
        # Look for repeating subsequences
        for i in range(len(recent_triplets) - sequence_length):
            sequence = recent_triplets[i:i+sequence_length]
            for j in range(i+sequence_length, len(recent_triplets)-sequence_length+1):
                if recent_triplets[j:j+sequence_length] == sequence:
                    return sequence
        
        return None
    
    def get_match_history(self, limit: int = 100) -> List[MatchResult]:
        """
        Get recent match history
        
        Args:
            limit: Maximum number of matches
            
        Returns:
            List of recent matches
        """
        return self.match_history[-limit:]
    
    def clear_history(self):
        """Clear match history"""
        self.match_history.clear()
    
    def export_state(self) -> Dict[str, Any]:
        """Export current matcher state"""
        return {
            "match_history": [
                {
                    "triplet": m.triplet,
                    "match_type": m.match_type.value,
                    "confidence": m.confidence,
                    "distance": m.distance,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata
                }
                for m in self.match_history
            ],
            "max_distance": self.max_distance
        }
    
    def import_state(self, state_data: Dict[str, Any]):
        """Import matcher state"""
        self.clear_history()
        
        # Import match history
        for match_data in state_data.get("match_history", []):
            match = MatchResult(
                triplet=match_data["triplet"],
                match_type=MatchType(match_data["match_type"]),
                confidence=match_data["confidence"],
                distance=match_data["distance"],
                timestamp=match_data["timestamp"],
                metadata=match_data["metadata"]
            )
            self.match_history.append(match)
        
        # Import settings
        self.max_distance = state_data.get("max_distance", self.max_distance)

# Example usage
if __name__ == "__main__":
    matcher = VectorMatcher()
    
    # Test vector sequence
    test_vectors = [
        np.array([0.001]),
        np.array([0.172]),
        np.array([0.345]),
        np.array([0.998]),
        np.array([0.002])
    ]
    
    print("Testing VectorMatcher:")
    print("=" * 40)
    
    for vector in test_vectors:
        result = matcher.match_vector(vector)
        print(f"Vector {vector[0]:.3f} -> {result.triplet} "
              f"({result.match_type.value}, conf: {result.confidence:.2f})")
        
        if result.distance != 0:
            print(f"  Distance: {result.distance}")
    
    # Test pattern detection
    print("\nPattern Detection:")
    pattern = matcher.detect_pattern_sequence(3)
    if pattern:
        print(f"Found pattern: {pattern}")
    
    # Test similar patterns
    print("\nSimilar Patterns:")
    similar = matcher.find_similar_patterns("001")
    for match in similar:
        print(f"  {match.triplet} (distance: {match.distance})") 