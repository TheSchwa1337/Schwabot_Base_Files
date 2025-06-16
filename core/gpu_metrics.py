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
import yaml
import json

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
    
    def __init__(self, window_size: int = 20, hist_bins: int = 100, trust_weights: Tuple[float, float, float] = (0.45, 0.35, 0.2)):
        self.window_size = window_size
        self.hist_bins = hist_bins
        self.trust_weights = trust_weights
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
        
    def _roll(self, tensor):
        tensor[:] = cp.roll(tensor, -1, axis=0)
        
    def _fetch(self, x):
        return x.get() if hasattr(x, 'get') else x

    def update(self, price: Optional[float] = None, volume: Optional[float] = None, bit_depth: Optional[int] = None):
        self._update(price, volume, bit_depth)
        
    def _update(self, price=None, volume=None, bit_depth=None):
        """Update metric tensors with new data"""
        # Apply roll first
        if price is not None:
            self._roll(self.tensors.price_history)
            self.tensors.price_history[-1] = price
        if volume is not None:
            self._roll(self.tensors.volume_history)
            self.tensors.volume_history[-1] = volume
        if bit_depth is not None:
            self._roll(self.tensors.bit_depths)
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
        hist_bins = self.hist_bins
        hist, _ = cp.histogram(price_changes, bins=hist_bins, density=True)
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
        vol_stab = 1.0 / (1.0 + cp.std(self.tensors.volume_history))
        
        # Calculate entropy stability
        ent_stab = 1.0 / (1.0 + cp.std(self.tensors.entropy_history))
        
        # Combine into trust score
        w1, w2, w3 = self.trust_weights
        trust = w1 * (1.0 / (1.0 + volatility)) + w2 * vol_stab + w3 * ent_stab
        self._roll(self.tensors.trust_scores)
        self.tensors.trust_scores[-1] = trust
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        return {
            'entropy': float(self.tensors.entropy_history[-1]),
            'drift': self._fetch(self.tensors.drift_vectors[-1]),
            'trust': float(self.tensors.trust_scores[-1]),
            'volatility': float(cp.std(cp.diff(cp.log(self.tensors.price_history))))
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
        if cp.count_nonzero(trust_scores) < 2:
            return 0.0
        trust_scores_cpu = cp.asnumpy(trust_scores)
        trend = np.polyfit(np.arange(len(trust_scores_cpu)), trust_scores_cpu, 1)[0]
        return float(trend)

    def _on_price_tick(self, data):
        price     = data['price']
        bit_depth = data.get('bit_depth',0)
        self._update(price=price, volume=None, bit_depth=bit_depth)

    def _on_volume_tick(self, data):
        volume = data['volume']
        self._update(price=None, volume=volume, bit_depth=None)

    def save_state(self, name):
        """Save the current state of the metrics."""
        state = {k: v.tolist() for k, v in self.tensors.__dict__.items()}
        with open(f'{name}.json', 'w') as file:
            json.dump(state, file)

    def load_state(self, name):
        """Load the state from a saved JSON file."""
        with open(f'{name}.json', 'r') as file:
            state = json.load(file)
        for key, value in state.items():
            arr = cp.asarray(value)
            self.tensors.__dict__[key][-len(arr):] = arr

    def __repr__(self):
        return f"GPUMetrics(price_history={self._fetch(self.tensors.price_history)}, volume_history={self._fetch(self.tensors.volume_history)}, entropy_history={self._fetch(self.tensors.entropy_history)}, drift_vectors={self._fetch(self.tensors.drift_vectors)}, trust_scores={self._fetch(self.tensors.trust_scores)}, bit_depths={self._fetch(self.tensors.bit_depths)})" 