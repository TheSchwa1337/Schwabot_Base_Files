#!/usr/bin/env python3
"""
🌌⚙️ ENTROPY DRIFT TRACKER
==========================

Tracks entropy variance between trade vectors over time.
Measures the instability of strategy vectors to predict optimal execution windows.

Core Concept: ΔE = ||Vₙ - Vₙ₋₁||
Where Vₙ is current vector and Vₙ₋₁ is previous vector.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class DriftSnapshot:
    """Snapshot of vector drift at a specific time"""
    timestamp: float
    vector: np.ndarray
    drift_value: float
    entropy_score: float

class EntropyDriftTracker:
    """
    Entropy Drift Tracker
    
    Tracks vector instability over time and computes drift-based warp windows
    for optimal trade execution timing.
    """
    
    def __init__(self, max_history: int = 100, warp_threshold: float = 0.15):
        """
        Initialize entropy drift tracker
        
        Args:
            max_history: Maximum number of vector snapshots to store
            warp_threshold: Threshold for warp window activation
        """
        self.history: Dict[str, deque] = {}
        self.max_history = max_history
        self.warp_threshold = warp_threshold
        self.drift_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Entropy Drift Tracker initialized (max_history: {max_history}, threshold: {warp_threshold})")
    
    def record_vector(self, strategy_id: str, vector: np.ndarray) -> float:
        """
        Record a new vector and compute its drift
        
        Args:
            strategy_id: Strategy identifier
            vector: Current profit vector
            
        Returns:
            Computed drift value
        """
        try:
            current_time = time.time()
            
            # Initialize history for new strategy
            if strategy_id not in self.history:
                self.history[strategy_id] = deque(maxlen=self.max_history)
                self.drift_stats[strategy_id] = {
                    'avg_drift': 0.0,
                    'max_drift': 0.0,
                    'min_drift': float('inf'),
                    'drift_count': 0
                }
            
            # Compute drift from previous vector
            drift_value = 0.0
            if len(self.history[strategy_id]) > 0:
                last_snapshot = self.history[strategy_id][-1]
                drift_value = np.linalg.norm(vector - last_snapshot.vector)
            
            # Compute entropy score (vector variance)
            entropy_score = np.std(vector) if len(vector) > 1 else 0.0
            
            # Create drift snapshot
            snapshot = DriftSnapshot(
                timestamp=current_time,
                vector=vector.copy(),
                drift_value=drift_value,
                entropy_score=entropy_score
            )
            
            # Add to history
            self.history[strategy_id].append(snapshot)
            
            # Update drift statistics
            stats = self.drift_stats[strategy_id]
            stats['drift_count'] += 1
            stats['avg_drift'] = (stats['avg_drift'] * (stats['drift_count'] - 1) + drift_value) / stats['drift_count']
            stats['max_drift'] = max(stats['max_drift'], drift_value)
            stats['min_drift'] = min(stats['min_drift'], drift_value)
            
            logger.debug(f"Recorded vector for {strategy_id}: drift={drift_value:.4f}, entropy={entropy_score:.4f}")
            return drift_value
            
        except Exception as e:
            logger.error(f"Error recording vector for {strategy_id}: {e}")
            return 0.0
    
    def compute_drift(self, strategy_id: str, window_size: Optional[int] = None) -> float:
        """
        Compute average drift over recent vectors
        
        Args:
            strategy_id: Strategy identifier
            window_size: Number of recent vectors to consider (None = all)
            
        Returns:
            Average drift value
        """
        try:
            if strategy_id not in self.history or len(self.history[strategy_id]) < 2:
                return 0.0
            
            vectors = list(self.history[strategy_id])
            
            # Use specified window size or all available
            if window_size is not None:
                vectors = vectors[-window_size:]
            
            if len(vectors) < 2:
                return 0.0
            
            # Compute drift between consecutive vectors
            drifts = []
            for i in range(1, len(vectors)):
                drift = np.linalg.norm(vectors[i].vector - vectors[i-1].vector)
                drifts.append(drift)
            
            avg_drift = np.mean(drifts) if drifts else 0.0
            return avg_drift
            
        except Exception as e:
            logger.error(f"Error computing drift for {strategy_id}: {e}")
            return 0.0
    
    def is_warp_window(self, strategy_id: str, threshold: Optional[float] = None) -> bool:
        """
        Check if current state is within a warp window
        
        Args:
            strategy_id: Strategy identifier
            threshold: Custom threshold (uses default if None)
            
        Returns:
            True if in warp window
        """
        try:
            drift = self.compute_drift(strategy_id)
            thresh = threshold or self.warp_threshold
            return drift > thresh
            
        except Exception as e:
            logger.error(f"Error checking warp window for {strategy_id}: {e}")
            return False
    
    def get_drift_trend(self, strategy_id: str, window_size: int = 10) -> str:
        """
        Get drift trend direction
        
        Args:
            strategy_id: Strategy identifier
            window_size: Number of recent vectors to analyze
            
        Returns:
            Trend direction: "increasing", "decreasing", "stable"
        """
        try:
            if strategy_id not in self.history or len(self.history[strategy_id]) < window_size:
                return "stable"
            
            vectors = list(self.history[strategy_id])[-window_size:]
            if len(vectors) < 3:
                return "stable"
            
            # Compute drift values
            drifts = []
            for i in range(1, len(vectors)):
                drift = np.linalg.norm(vectors[i].vector - vectors[i-1].vector)
                drifts.append(drift)
            
            if len(drifts) < 2:
                return "stable"
            
            # Compute trend
            first_half = np.mean(drifts[:len(drifts)//2])
            second_half = np.mean(drifts[len(drifts)//2:])
            
            if second_half > first_half * 1.2:
                return "increasing"
            elif second_half < first_half * 0.8:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error computing drift trend for {strategy_id}: {e}")
            return "stable"
    
    def get_entropy_score(self, strategy_id: str) -> float:
        """
        Get current entropy score for strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Current entropy score
        """
        try:
            if strategy_id not in self.history or len(self.history[strategy_id]) == 0:
                return 0.0
            
            latest = self.history[strategy_id][-1]
            return latest.entropy_score
            
        except Exception as e:
            logger.error(f"Error getting entropy score for {strategy_id}: {e}")
            return 0.0
    
    def predict_warp_delay(self, strategy_id: str, alpha: float = 100.0) -> float:
        """
        Predict optimal warp delay based on current drift
        
        Args:
            strategy_id: Strategy identifier
            alpha: Warp scaling constant (seconds per drift unit)
            
        Returns:
            Predicted delay in seconds
        """
        try:
            drift = self.compute_drift(strategy_id)
            delay = drift * alpha
            
            # Clamp delay to reasonable bounds
            delay = max(0.0, min(delay, 3600.0))  # 0 to 1 hour
            
            return delay
            
        except Exception as e:
            logger.error(f"Error predicting warp delay for {strategy_id}: {e}")
            return 0.0
    
    def get_drift_statistics(self, strategy_id: str) -> Dict[str, float]:
        """
        Get comprehensive drift statistics
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary of drift statistics
        """
        try:
            if strategy_id not in self.drift_stats:
                return {}
            
            stats = self.drift_stats[strategy_id].copy()
            
            # Add current drift and trend
            stats['current_drift'] = self.compute_drift(strategy_id)
            stats['drift_trend'] = self.get_drift_trend(strategy_id)
            stats['entropy_score'] = self.get_entropy_score(strategy_id)
            stats['warp_delay'] = self.predict_warp_delay(strategy_id)
            stats['in_warp_window'] = self.is_warp_window(strategy_id)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting drift statistics for {strategy_id}: {e}")
            return {}
    
    def cleanup_old_data(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up old drift data
        
        Args:
            max_age_hours: Maximum age of data to keep
            
        Returns:
            Number of snapshots removed
        """
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            total_removed = 0
            
            for strategy_id in list(self.history.keys()):
                initial_count = len(self.history[strategy_id])
                
                # Remove old snapshots
                self.history[strategy_id] = deque(
                    [s for s in self.history[strategy_id] if s.timestamp >= cutoff_time],
                    maxlen=self.max_history
                )
                
                removed = initial_count - len(self.history[strategy_id])
                total_removed += removed
                
                # Remove strategy if no data left
                if len(self.history[strategy_id]) == 0:
                    del self.history[strategy_id]
                    if strategy_id in self.drift_stats:
                        del self.drift_stats[strategy_id]
            
            logger.info(f"Cleaned up {total_removed} old drift snapshots")
            return total_removed
            
        except Exception as e:
            logger.error(f"Error cleaning up old drift data: {e}")
            return 0


def create_entropy_drift_tracker(max_history: int = 100, warp_threshold: float = 0.15) -> EntropyDriftTracker:
    """
    Factory function to create EntropyDriftTracker
    
    Args:
        max_history: Maximum number of vector snapshots to store
        warp_threshold: Threshold for warp window activation
        
    Returns:
        Initialized EntropyDriftTracker instance
    """
    return EntropyDriftTracker(max_history=max_history, warp_threshold=warp_threshold)


def test_entropy_drift_tracker():
    """Test function for entropy drift tracker"""
    print("🌌⚙️ Testing Entropy Drift Tracker")
    print("=" * 50)
    
    # Create tracker
    tracker = create_entropy_drift_tracker(max_history=50, warp_threshold=0.1)
    
    # Test data
    strategy_id = "test_strategy_warp"
    
    # Test 1: Record vectors and compute drift
    print("\n📝 Test 1: Recording Vectors and Computing Drift")
    vectors = [
        np.array([0.1, 0.2, 0.1]),
        np.array([0.2, 0.3, 0.2]),  # Higher drift
        np.array([0.1, 0.1, 0.1]),  # Lower drift
        np.array([0.4, 0.5, 0.4])   # High drift
    ]
    
    for i, vector in enumerate(vectors):
        drift = tracker.record_vector(strategy_id, vector)
        print(f"  Vector {i+1}: {vector} → Drift: {drift:.4f}")
    
    # Test 2: Compute average drift
    print("\n📊 Test 2: Computing Average Drift")
    avg_drift = tracker.compute_drift(strategy_id)
    print(f"  Average drift: {avg_drift:.4f}")
    
    # Test 3: Check warp window
    print("\n🕳️ Test 3: Checking Warp Window")
    in_warp = tracker.is_warp_window(strategy_id)
    print(f"  In warp window: {in_warp}")
    
    # Test 4: Get drift trend
    print("\n📈 Test 4: Getting Drift Trend")
    trend = tracker.get_drift_trend(strategy_id)
    print(f"  Drift trend: {trend}")
    
    # Test 5: Predict warp delay
    print("\n⏱️ Test 5: Predicting Warp Delay")
    delay = tracker.predict_warp_delay(strategy_id)
    print(f"  Predicted delay: {delay:.2f} seconds")
    
    # Test 6: Get statistics
    print("\n📊 Test 6: Getting Drift Statistics")
    stats = tracker.get_drift_statistics(strategy_id)
    print(f"  Drift statistics: {stats}")


if __name__ == "__main__":
    test_entropy_drift_tracker() 