"""
GPU-Accelerated Metrics
=====================

Implements GPU-accelerated versions of key metrics calculations
using CuPy for fast tensor operations.
"""

import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricTensors:
    """Container for GPU tensors used in metric calculations"""
    price_history: cp.ndarray
    volume_history: cp.ndarray
    entropy_history: cp.ndarray
    drift_vectors: cp.ndarray
    trust_scores: cp.ndarray
    bit_depths: cp.ndarray

class GPUMetrics:
    """GPU-accelerated metric calculations"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.tensors = None
        self._initialize_tensors()
        
    def _initialize_tensors(self):
        """Initialize GPU tensors"""
        self.tensors = MetricTensors(
            price_history=cp.zeros(self.window_size, dtype=cp.float32),
            volume_history=cp.zeros(self.window_size, dtype=cp.float32),
            entropy_history=cp.zeros(self.window_size, dtype=cp.float32),
            drift_vectors=cp.zeros((self.window_size, 3), dtype=cp.float32),
            trust_scores=cp.zeros(self.window_size, dtype=cp.float32),
            bit_depths=cp.zeros(self.window_size, dtype=cp.int32)
        )
        
    def update(self, price: float, volume: float, bit_depth: int):
        """Update metric tensors with new data"""
        # Shift all tensors left
        for tensor in [
            self.tensors.price_history,
            self.tensors.volume_history,
            self.tensors.entropy_history,
            self.tensors.drift_vectors,
            self.tensors.trust_scores,
            self.tensors.bit_depths
        ]:
            cp.roll(tensor, -1, axis=0)
            
        # Update with new values
        self.tensors.price_history[-1] = price
        self.tensors.volume_history[-1] = volume
        self.tensors.bit_depths[-1] = bit_depth
        
        # Recalculate derived metrics
        self._update_entropy()
        self._update_drift_vectors()
        self._update_trust_scores()
        
    def _update_entropy(self):
        """Calculate rolling entropy using GPU"""
        # Calculate price changes
        price_changes = cp.diff(self.tensors.price_history)
        
        # Calculate probability distribution
        hist, _ = cp.histogram(price_changes, bins=50, density=True)
        hist = cp.clip(hist, 1e-10, None)  # Avoid log(0)
        
        # Calculate entropy
        entropy = -cp.sum(hist * cp.log2(hist))
        self.tensors.entropy_history[-1] = entropy
        
    def _update_drift_vectors(self):
        """Calculate drift vectors using GPU"""
        # Calculate price momentum
        momentum = cp.diff(self.tensors.price_history, n=3)
        
        # Calculate volume trend
        volume_trend = cp.diff(self.tensors.volume_history, n=3)
        
        # Calculate entropy trend
        entropy_trend = cp.diff(self.tensors.entropy_history, n=3)
        
        # Combine into drift vector
        self.tensors.drift_vectors[-1] = cp.array([
            momentum[-1],
            volume_trend[-1],
            entropy_trend[-1]
        ])
        
    def _update_trust_scores(self):
        """Calculate trust scores using GPU"""
        # Calculate volatility
        returns = cp.diff(cp.log(self.tensors.price_history))
        volatility = cp.std(returns)
        
        # Calculate volume stability
        volume_stability = 1.0 / (1.0 + cp.std(self.tensors.volume_history))
        
        # Calculate entropy stability
        entropy_stability = 1.0 / (1.0 + cp.std(self.tensors.entropy_history))
        
        # Combine into trust score
        self.tensors.trust_scores[-1] = (
            0.4 * (1.0 / (1.0 + volatility)) +
            0.3 * volume_stability +
            0.3 * entropy_stability
        )
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        return {
            'entropy_rate': float(self.tensors.entropy_history[-1]),
            'drift_vector': self.tensors.drift_vectors[-1].get().tolist(),
            'trust_score': float(self.tensors.trust_scores[-1]),
            'volatility': float(cp.std(cp.diff(cp.log(self.tensors.price_history)))),
            'volume_stability': float(1.0 / (1.0 + cp.std(self.tensors.volume_history))),
            'entropy_stability': float(1.0 / (1.0 + cp.std(self.tensors.entropy_history)))
        }
        
    def compare_tensors(self, other: 'GPUMetrics') -> float:
        """Compare metric tensors with another instance"""
        # Calculate differences
        price_diff = cp.mean(cp.abs(self.tensors.price_history - other.tensors.price_history))
        volume_diff = cp.mean(cp.abs(self.tensors.volume_history - other.tensors.volume_history))
        entropy_diff = cp.mean(cp.abs(self.tensors.entropy_history - other.tensors.entropy_history))
        drift_diff = cp.mean(cp.abs(self.tensors.drift_vectors - other.tensors.drift_vectors))
        trust_diff = cp.mean(cp.abs(self.tensors.trust_scores - other.tensors.trust_scores))
        
        # Weighted average of differences
        return float(
            0.3 * price_diff +
            0.2 * volume_diff +
            0.2 * entropy_diff +
            0.2 * drift_diff +
            0.1 * trust_diff
        )
        
    def get_rolling_volatility(self, window: int = 20) -> float:
        """Calculate rolling volatility"""
        returns = cp.diff(cp.log(self.tensors.price_history[-window:]))
        return float(cp.std(returns))
        
    def get_drift_magnitude(self) -> float:
        """Calculate magnitude of current drift vector"""
        return float(cp.linalg.norm(self.tensors.drift_vectors[-1]))
        
    def get_trust_trend(self, window: int = 20) -> float:
        """Calculate trend in trust scores"""
        trust_scores = self.tensors.trust_scores[-window:]
        return float(cp.polyfit(cp.arange(window), trust_scores, 1)[0]) 