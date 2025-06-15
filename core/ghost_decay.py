"""
Ghost Decay System
==================

Manages decay of ghost signals (prior success echoes) to prevent overfit
to historical success patterns. Implements exponential decay with reinforcement
learning to maintain only relevant ghost patterns.

Mathematical Foundation:
W_g(t) = W₀ · e^(-α(t-t₀))

Where:
- W₀ = ghost's initial confidence from profitable pattern
- α = decay constant (0.01-0.05 typical)
- t-t₀ = time elapsed since ghost creation
- Auto-lockout ghosts below 0.1 weight
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class GhostStatus(Enum):
    """Ghost signal status"""
    ACTIVE = "active"
    DECAYING = "decaying"
    SUPPRESSED = "suppressed"
    LOCKED = "locked"
    REINFORCED = "reinforced"

@dataclass
class GhostSignal:
    """Individual ghost signal with decay tracking"""
    ghost_id: str
    creation_time: float
    initial_weight: float
    current_weight: float
    decay_constant: float
    profit_pattern: Dict[str, Any]
    success_count: int = 0
    failure_count: int = 0
    last_triggered: float = 0.0
    status: GhostStatus = GhostStatus.ACTIVE
    reinforcement_history: List[float] = field(default_factory=list)

@dataclass
class DecayMetrics:
    """Ghost decay system metrics"""
    total_ghosts_created: int = 0
    active_ghosts: int = 0
    suppressed_ghosts: int = 0
    locked_ghosts: int = 0
    avg_ghost_lifetime: float = 0.0
    total_reinforcements: int = 0
    decay_events: int = 0

class GhostDecaySystem:
    """
    Advanced ghost signal decay management system.
    
    Prevents overfit to historical success by implementing exponential decay
    of ghost signals unless they are reinforced by recent successful patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ghost decay system.
        
        Args:
            config: Configuration parameters for decay system
        """
        self.config = config or {}
        
        # Decay parameters
        self.default_decay_constant = self.config.get('decay_constant', 0.02)
        self.min_weight_threshold = self.config.get('min_weight', 0.1)
        self.reinforcement_threshold = self.config.get('reinforcement_threshold', 0.7)
        self.suppression_threshold = self.config.get('suppression_threshold', 0.05)
        
        # Reinforcement parameters
        self.reinforcement_boost = self.config.get('reinforcement_boost', 0.2)
        self.max_reinforcement_weight = self.config.get('max_weight', 2.0)
        self.pattern_similarity_threshold = self.config.get('similarity_threshold', 0.8)
        
        # Ghost storage and tracking
        self.active_ghosts: Dict[str, GhostSignal] = {}
        self.suppressed_ghosts: Dict[str, GhostSignal] = {}
        self.locked_ghosts: Dict[str, GhostSignal] = {}
        
        # Historical tracking
        self.ghost_history: deque = deque(maxlen=500)
        self.decay_events: deque = deque(maxlen=200)
        
        # Metrics
        self.metrics = DecayMetrics()
        
        logger.info("Ghost Decay System initialized with exponential decay")
    
    def create_ghost_signal(self, profit_pattern: Dict[str, Any], 
                          initial_confidence: float) -> str:
        """
        Create new ghost signal from profitable pattern.
        
        Args:
            profit_pattern: Pattern data that generated profit
            initial_confidence: Initial confidence/weight for the ghost
            
        Returns:
            Ghost ID for tracking
        """
        # Generate unique ghost ID
        ghost_id = self._generate_ghost_id()
        
        # Determine decay constant based on pattern strength
        decay_constant = self._calculate_decay_constant(profit_pattern, initial_confidence)
        
        # Create ghost signal
        ghost = GhostSignal(
            ghost_id=ghost_id,
            creation_time=time.time(),
            initial_weight=initial_confidence,
            current_weight=initial_confidence,
            decay_constant=decay_constant,
            profit_pattern=profit_pattern.copy(),
            status=GhostStatus.ACTIVE
        )
        
        # Store in active ghosts
        self.active_ghosts[ghost_id] = ghost
        self.ghost_history.append(ghost)
        
        # Update metrics
        self.metrics.total_ghosts_created += 1
        self.metrics.active_ghosts += 1
        
        logger.info(f"Ghost signal created: {ghost_id} with weight {initial_confidence:.3f}")
        
        return ghost_id
    
    def update_ghost_weights(self, current_time: Optional[float] = None) -> Dict[str, float]:
        """
        Update all ghost weights based on exponential decay.
        
        Args:
            current_time: Current timestamp (uses time.time() if None)
            
        Returns:
            Dictionary of ghost_id -> current_weight for active ghosts
        """
        if current_time is None:
            current_time = time.time()
        
        updated_weights = {}
        ghosts_to_suppress = []
        ghosts_to_lock = []
        
        # Update active ghosts
        for ghost_id, ghost in self.active_ghosts.items():
            # Calculate time elapsed
            time_elapsed = current_time - ghost.creation_time
            
            # Apply exponential decay
            decayed_weight = ghost.initial_weight * np.exp(-ghost.decay_constant * time_elapsed)
            ghost.current_weight = decayed_weight
            
            # Check thresholds
            if decayed_weight < self.suppression_threshold:
                ghosts_to_lock.append(ghost_id)
            elif decayed_weight < self.min_weight_threshold:
                ghosts_to_suppress.append(ghost_id)
            else:
                updated_weights[ghost_id] = decayed_weight
        
        # Process suppressions and locks
        for ghost_id in ghosts_to_suppress:
            self._suppress_ghost(ghost_id)
            
        for ghost_id in ghosts_to_lock:
            self._lock_ghost(ghost_id)
        
        logger.debug(f"Ghost weights updated: {len(updated_weights)} active, "
                    f"{len(ghosts_to_suppress)} suppressed, {len(ghosts_to_lock)} locked")
        
        return updated_weights
    
    def reinforce_ghost(self, ghost_id: str, reinforcement_strength: float, 
                       current_pattern: Dict[str, Any]) -> bool:
        """
        Reinforce ghost signal based on pattern match and success.
        
        Args:
            ghost_id: ID of ghost to reinforce
            reinforcement_strength: Strength of reinforcement [0, 1]
            current_pattern: Current pattern for similarity check
            
        Returns:
            True if reinforcement was applied
        """
        # Check if ghost exists and is active
        if ghost_id not in self.active_ghosts:
            return False
        
        ghost = self.active_ghosts[ghost_id]
        
        # Check pattern similarity
        similarity = self._calculate_pattern_similarity(ghost.profit_pattern, current_pattern)
        
        if similarity < self.pattern_similarity_threshold:
            logger.debug(f"Ghost {ghost_id} reinforcement rejected: low similarity {similarity:.3f}")
            return False
        
        # Apply reinforcement
        reinforcement_amount = reinforcement_strength * self.reinforcement_boost
        ghost.current_weight = min(
            ghost.current_weight + reinforcement_amount,
            self.max_reinforcement_weight
        )
        
        # Update ghost status and tracking
        ghost.status = GhostStatus.REINFORCED
        ghost.success_count += 1
        ghost.last_triggered = time.time()
        ghost.reinforcement_history.append(reinforcement_strength)
        
        # Update metrics
        self.metrics.total_reinforcements += 1
        
        logger.info(f"Ghost {ghost_id} reinforced: weight {ghost.current_weight:.3f} "
                   f"(similarity: {similarity:.3f})")
        
        return True
    
    def penalize_ghost(self, ghost_id: str, penalty_strength: float) -> bool:
        """
        Apply penalty to ghost signal for poor performance.
        
        Args:
            ghost_id: ID of ghost to penalize
            penalty_strength: Strength of penalty [0, 1]
            
        Returns:
            True if penalty was applied
        """
        if ghost_id not in self.active_ghosts:
            return False
        
        ghost = self.active_ghosts[ghost_id]
        
        # Apply penalty (accelerated decay)
        penalty_amount = penalty_strength * 0.3  # 30% weight reduction
        ghost.current_weight = max(ghost.current_weight - penalty_amount, 0.0)
        
        # Update failure tracking
        ghost.failure_count += 1
        
        # Check if ghost should be suppressed due to penalty
        if ghost.current_weight < self.min_weight_threshold:
            self._suppress_ghost(ghost_id)
            
        logger.info(f"Ghost {ghost_id} penalized: weight {ghost.current_weight:.3f}")
        
        return True
    
    def get_active_ghost_weights(self) -> Dict[str, float]:
        """Get current weights of all active ghosts."""
        return {ghost_id: ghost.current_weight 
                for ghost_id, ghost in self.active_ghosts.items()}
    
    def get_ghost_recommendations(self, current_pattern: Dict[str, Any], 
                                top_k: int = 5) -> List[Tuple[str, float, float]]:
        """
        Get top ghost recommendations based on pattern similarity and weight.
        
        Args:
            current_pattern: Current market/fractal pattern
            top_k: Number of top recommendations to return
            
        Returns:
            List of (ghost_id, weight, similarity) tuples
        """
        recommendations = []
        
        for ghost_id, ghost in self.active_ghosts.items():
            # Calculate pattern similarity
            similarity = self._calculate_pattern_similarity(ghost.profit_pattern, current_pattern)
            
            # Calculate recommendation score (weight * similarity)
            score = ghost.current_weight * similarity
            
            recommendations.append((ghost_id, ghost.current_weight, similarity, score))
        
        # Sort by score and return top k
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        return [(ghost_id, weight, similarity) 
                for ghost_id, weight, similarity, _ in recommendations[:top_k]]
    
    def _generate_ghost_id(self) -> str:
        """Generate unique ghost ID."""
        timestamp = int(time.time() * 1000)
        return f"ghost_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def _calculate_decay_constant(self, profit_pattern: Dict[str, Any], 
                                initial_confidence: float) -> float:
        """Calculate decay constant based on pattern characteristics."""
        base_decay = self.default_decay_constant
        
        # Adjust decay based on initial confidence
        confidence_factor = 1.0 - (initial_confidence * 0.3)  # Higher confidence = slower decay
        
        # Adjust decay based on pattern complexity
        pattern_complexity = len(profit_pattern.get('signals', []))
        complexity_factor = 1.0 + (pattern_complexity * 0.01)  # More complex = faster decay
        
        decay_constant = base_decay * confidence_factor * complexity_factor
        
        return np.clip(decay_constant, 0.005, 0.1)
    
    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], 
                                    pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns."""
        # Simple similarity based on common keys and value differences
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1 = pattern1[key]
            val2 = pattern2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(similarity)
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (exact match)
                similarities.append(1.0 if val1 == val2 else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _suppress_ghost(self, ghost_id: str):
        """Move ghost from active to suppressed state."""
        if ghost_id in self.active_ghosts:
            ghost = self.active_ghosts.pop(ghost_id)
            ghost.status = GhostStatus.SUPPRESSED
            self.suppressed_ghosts[ghost_id] = ghost
            
            self.metrics.active_ghosts -= 1
            self.metrics.suppressed_ghosts += 1
            
            logger.debug(f"Ghost {ghost_id} suppressed (weight: {ghost.current_weight:.3f})")
    
    def _lock_ghost(self, ghost_id: str):
        """Move ghost to locked state (permanent removal)."""
        ghost = None
        
        if ghost_id in self.active_ghosts:
            ghost = self.active_ghosts.pop(ghost_id)
            self.metrics.active_ghosts -= 1
        elif ghost_id in self.suppressed_ghosts:
            ghost = self.suppressed_ghosts.pop(ghost_id)
            self.metrics.suppressed_ghosts -= 1
        
        if ghost:
            ghost.status = GhostStatus.LOCKED
            self.locked_ghosts[ghost_id] = ghost
            self.metrics.locked_ghosts += 1
            
            # Record decay event
            decay_event = {
                "timestamp": time.time(),
                "ghost_id": ghost_id,
                "lifetime": time.time() - ghost.creation_time,
                "final_weight": ghost.current_weight,
                "success_count": ghost.success_count,
                "failure_count": ghost.failure_count
            }
            self.decay_events.append(decay_event)
            self.metrics.decay_events += 1
            
            logger.debug(f"Ghost {ghost_id} locked after {decay_event['lifetime']:.1f}s")
    
    def cleanup_old_ghosts(self, max_age_hours: float = 24.0):
        """Remove very old locked ghosts to free memory."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_ghosts = []
        for ghost_id, ghost in self.locked_ghosts.items():
            if current_time - ghost.creation_time > max_age_seconds:
                old_ghosts.append(ghost_id)
        
        for ghost_id in old_ghosts:
            del self.locked_ghosts[ghost_id]
            
        if old_ghosts:
            logger.info(f"Cleaned up {len(old_ghosts)} old locked ghosts")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive ghost decay system summary."""
        # Calculate average ghost lifetime
        if self.decay_events:
            avg_lifetime = np.mean([event["lifetime"] for event in self.decay_events])
        else:
            avg_lifetime = 0.0
        
        # Calculate success rates
        total_successes = sum(ghost.success_count for ghost in self.active_ghosts.values())
        total_failures = sum(ghost.failure_count for ghost in self.active_ghosts.values())
        success_rate = total_successes / max(total_successes + total_failures, 1)
        
        return {
            "total_ghosts_created": self.metrics.total_ghosts_created,
            "active_ghosts": len(self.active_ghosts),
            "suppressed_ghosts": len(self.suppressed_ghosts),
            "locked_ghosts": len(self.locked_ghosts),
            "avg_ghost_lifetime": avg_lifetime,
            "total_reinforcements": self.metrics.total_reinforcements,
            "ghost_success_rate": success_rate,
            "active_ghost_weights": self.get_active_ghost_weights(),
            "recent_decay_events": len([e for e in self.decay_events 
                                      if time.time() - e["timestamp"] < 3600])
        }
    
    def reset_system(self):
        """Reset entire ghost decay system."""
        self.active_ghosts.clear()
        self.suppressed_ghosts.clear()
        self.locked_ghosts.clear()
        self.ghost_history.clear()
        self.decay_events.clear()
        self.metrics = DecayMetrics()
        logger.info("Ghost decay system reset")

# Example usage and testing
if __name__ == "__main__":
    # Test ghost decay system
    decay_system = GhostDecaySystem()
    
    # Create test ghost signals
    pattern1 = {
        "braid_signal": 0.8,
        "paradox_signal": 0.6,
        "profit_delta": 75.0,
        "volatility": 0.3
    }
    
    pattern2 = {
        "braid_signal": 0.7,
        "paradox_signal": 0.5,
        "profit_delta": 60.0,
        "volatility": 0.4
    }
    
    # Create ghosts
    ghost1_id = decay_system.create_ghost_signal(pattern1, 0.9)
    ghost2_id = decay_system.create_ghost_signal(pattern2, 0.7)
    
    print(f"Created ghosts: {ghost1_id}, {ghost2_id}")
    
    # Simulate time passage and decay
    import time as time_module
    time_module.sleep(1)  # Wait 1 second
    
    # Update weights
    weights = decay_system.update_ghost_weights()
    print(f"Ghost weights after decay: {weights}")
    
    # Test reinforcement
    current_pattern = {
        "braid_signal": 0.82,
        "paradox_signal": 0.58,
        "profit_delta": 80.0,
        "volatility": 0.25
    }
    
    reinforced = decay_system.reinforce_ghost(ghost1_id, 0.8, current_pattern)
    print(f"Ghost reinforcement result: {reinforced}")
    
    # Get recommendations
    recommendations = decay_system.get_ghost_recommendations(current_pattern)
    print(f"Ghost recommendations: {recommendations}")
    
    # Get system summary
    summary = decay_system.get_system_summary()
    print(f"System summary: {summary}") 