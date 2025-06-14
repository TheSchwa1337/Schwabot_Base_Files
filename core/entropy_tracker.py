"""
Entropy Tracker Module
Handles Shannon entropy calculations across multiple time windows
and maintains rolling statistics for price, volume, and time entropy.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntropyState:
    """Represents a single entropy measurement at a point in time"""
    price_entropy: float
    volume_entropy: float
    time_entropy: float
    timestamp: float
    bit_pattern: Optional[int] = None
    tier: Optional[int] = None
    # Normalized values for Z-scoring
    price_normalized: float = 0.0
    volume_normalized: float = 0.0


class EntropyTracker:
    """
    Tracks entropy across multiple time windows and maintains
    rolling statistics for normalization and pattern detection.
    """
    
    def __init__(self, maxlen: int = 1000):
        """
        Initialize entropy tracker with rolling buffers.
        
        Args:
            maxlen: Maximum length of history buffer
        """
        self.history = deque(maxlen=maxlen)
        self.price_history = deque(maxlen=maxlen)
        self.volume_history = deque(maxlen=maxlen)
        self.timestamp_history = deque(maxlen=maxlen)
        
        # Rolling statistics for normalization
        self.rolling_mean = {'price': 0.0, 'volume': 0.0}
        self.rolling_std = {'price': 1.0, 'volume': 1.0}
        
        # Multi-window entropy calculations
        self.window_sizes = {
            'short': 5,    # Micro-tick behavior
            'mid': 16,     # Standard Ferris Wheel logic
            'long': 64     # Macro-pattern recognition
        }
        
        # Density buffers for variance tracking
        self.density_buffer = deque(maxlen=128)
        
    def update(self, price: float, volume: float, timestamp: float) -> EntropyState:
        """
        Process new tick data and calculate entropy components.
        
        Args:
            price: Current BTC price
            volume: Trading volume
            timestamp: Unix timestamp
            
        Returns:
            EntropyState with calculated entropies
        """
        # Store raw values
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.timestamp_history.append(timestamp)
        
        # Calculate log returns for price normalization (Rule #1)
        price_normalized = self._calculate_normalized_return(price)
        volume_normalized = self._normalize_volume(volume)
        
        # Calculate Shannon entropy for each component
        price_entropy = self._calculate_price_entropy()
        volume_entropy = self._calculate_volume_entropy()
        time_entropy = self._calculate_time_entropy()
        
        # Create entropy state
        state = EntropyState(
            price_entropy=price_entropy,
            volume_entropy=volume_entropy,
            time_entropy=time_entropy,
            timestamp=timestamp,
            price_normalized=price_normalized,
            volume_normalized=volume_normalized
        )
        
        # Add to history
        self.history.append(state)
        
        return state
    
    def _calculate_normalized_return(self, price: float) -> float:
        """
        Calculate normalized log return with Z-scoring (Rule #1).
        
        r_t = ln(P_t / P_{t-1})
        z_t = (r_t - μ_r) / σ_r
        """
        if len(self.price_history) < 2:
            return 0.0
        
        # Calculate log return
        prev_price = self.price_history[-2]
        log_return = np.log(price / prev_price) if prev_price > 0 else 0.0
        
        # Update rolling statistics
        if len(self.price_history) > 20:
            returns = [np.log(self.price_history[i] / self.price_history[i-1]) 
                      for i in range(1, len(self.price_history)) 
                      if self.price_history[i-1] > 0]
            self.rolling_mean['price'] = np.mean(returns)
            self.rolling_std['price'] = np.std(returns) or 1.0
        
        # Z-score normalization
        z_score = (log_return - self.rolling_mean['price']) / self.rolling_std['price']
        return z_score
    
    def _normalize_volume(self, volume: float) -> float:
        """Normalize volume using rolling Z-score."""
        if len(self.volume_history) > 20:
            self.rolling_mean['volume'] = np.mean(list(self.volume_history)[-100:])
            self.rolling_std['volume'] = np.std(list(self.volume_history)[-100:]) or 1.0
        
        z_score = (volume - self.rolling_mean['volume']) / self.rolling_std['volume']
        return z_score
    
    def _shannon_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """
        Calculate Shannon entropy for a data array (Rule #2).
        
        H = -Σ p_i * log2(p_i)
        """
        if len(data) < 2:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(data, bins=bins, density=True)
        
        # Normalize to probabilities
        hist = hist * (data.max() - data.min()) / bins
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        
        # Calculate Shannon entropy
        if len(hist) == 0:
            return 0.0
        
        # Normalize probabilities
        hist = hist / hist.sum()
        
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_price_entropy(self) -> float:
        """Calculate price entropy using normalized returns."""
        if len(self.history) < 2:
            return 0.0
        
        # Use normalized price changes
        price_changes = np.array([s.price_normalized for s in list(self.history)[-100:]])
        return self._shannon_entropy(price_changes)
    
    def _calculate_volume_entropy(self) -> float:
        """Calculate volume entropy using normalized volumes."""
        if len(self.history) < 2:
            return 0.0
        
        # Use normalized volume changes
        volume_changes = np.array([s.volume_normalized for s in list(self.history)[-100:]])
        return self._shannon_entropy(volume_changes)
    
    def _calculate_time_entropy(self) -> float:
        """Calculate time entropy from timestamp deltas."""
        if len(self.timestamp_history) < 2:
            return 0.0
        
        # Calculate time deltas
        time_deltas = np.diff(list(self.timestamp_history)[-100:])
        return self._shannon_entropy(time_deltas)
    
    def get_multi_window_entropies(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate entropy across multiple time windows (Rule #2).
        
        Returns:
            Dictionary with entropy values for short/mid/long windows
        """
        results = {}
        
        for window_name, window_size in self.window_sizes.items():
            if len(self.history) < window_size:
                results[window_name] = {
                    'price': 0.0,
                    'volume': 0.0,
                    'time': 0.0
                }
                continue
            
            # Get window data
            window_data = list(self.history)[-window_size:]
            
            # Calculate entropies for this window
            price_data = np.array([s.price_normalized for s in window_data])
            volume_data = np.array([s.volume_normalized for s in window_data])
            time_data = np.diff([s.timestamp for s in window_data])
            
            results[window_name] = {
                'price': self._shannon_entropy(price_data, bins=min(window_size, 50)),
                'volume': self._shannon_entropy(volume_data, bins=min(window_size, 50)),
                'time': self._shannon_entropy(time_data, bins=min(window_size-1, 50)) if len(time_data) > 0 else 0.0
            }
        
        return results
    
    def update_density_buffer(self, density: float):
        """Update density buffer for variance calculations."""
        self.density_buffer.append(density)
    
    def get_density_variance(self, window: str = 'mid') -> float:
        """
        Calculate density variance for specified window.
        
        Args:
            window: 'short', 'mid', or 'long'
            
        Returns:
            Variance of density values
        """
        window_size = self.window_sizes.get(window, 16)
        
        if len(self.density_buffer) < window_size:
            return 0.0
        
        densities = list(self.density_buffer)[-window_size:]
        return np.var(densities)
    
    def get_latest_state(self) -> Optional[EntropyState]:
        """Get the most recent entropy state."""
        return self.history[-1] if self.history else None
    
    def get_entropy_vector(self) -> Optional[np.ndarray]:
        """
        Get current entropy as a 3D vector for similarity calculations.
        
        Returns:
            numpy array [price_entropy, volume_entropy, time_entropy]
        """
        if not self.history:
            return None
        
        latest = self.history[-1]
        return np.array([
            latest.price_entropy,
            latest.volume_entropy,
            latest.time_entropy
        ]) 