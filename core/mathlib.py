"""
Core mathematical utilities for Schwabot strategy intelligence,
Schwafit validation, and recursive trade logic support.
"""

import numpy as np
from datetime import datetime
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

def delta_time(t1: datetime, t2: datetime) -> float:
    """Time delta in seconds."""
    return (t2 - t1).total_seconds()

def sum_profit_gradient(profit: np.ndarray, grad: np.ndarray) -> float:
    """Sum of profit weighted by node-gradient."""
    return np.sum(profit * grad)

def calculate_entropy(prices: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of price histogram."""
    hist, _ = np.histogram(prices, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0.0

def kelly_fraction(mu: float, rf: float, sigma2: float) -> float:
    """Kelly criterion for optimal fraction."""
    if sigma2 == 0:
        return 0.0
    return (mu - rf) / sigma2

def sha256_hash(*args) -> str:
    """SHA256 hash of concatenated stringified arguments."""
    s = ''.join(str(a) for a in args)
    return hashlib.sha256(s.encode()).hexdigest()

def combine_hash_logic(hash_block: str, entropy: float, kelly: float) -> str:
    """Combine hash, entropy, and kelly into a new hash block."""
    return sha256_hash(hash_block, entropy, kelly)

def exponential_decay(v0: float, t: float, lambda_: float) -> float:
    """Exponential decay function V(t) = V0 * exp(-lambda * t)."""
    return v0 * np.exp(-lambda_ * t)

@dataclass
class GradedProfitVector:
    """Standardized representation of a trade's behavioral fingerprint."""
    profit: float
    volume_allocated: float
    time_held: float
    signal_strength: float
    smart_money_score: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for vector operations."""
        return np.array([
            self.profit,
            self.volume_allocated,
            self.time_held,
            self.signal_strength,
            self.smart_money_score
        ], dtype=np.float32)

class CoreMathLib:
    """
    Core mathematical utilities for Schwabot strategy intelligence,
    Schwafit validation, and recursive trade logic support.
    """

    def __init__(self):
        """Initialize the math library."""
        pass

    # --- VECTOR OPS ---

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Cosine similarity in range [-1, 1]
        """
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes Euclidean distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Euclidean distance
        """
        return float(np.linalg.norm(a - b))

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Returns unit vector in same direction.
        
        Args:
            v: Input vector
            
        Returns:
            np.ndarray: Normalized unit vector
        """
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    # --- G(e) GRADING LOGIC ---

    def grading_vector(self, trade: Dict[str, Any]) -> GradedProfitVector:
        """
        Converts a trade dictionary into a standardized Graded Profit Vector.
        
        Args:
            trade: Dictionary containing trade metrics
            
        Returns:
            GradedProfitVector: Standardized trade vector
        """
        return GradedProfitVector(
            profit=trade.get('profit', 0.0),
            volume_allocated=trade.get('volume_allocated', 0.0),
            time_held=trade.get('time_held', 0.0),
            signal_strength=trade.get('signal_strength', 0.0),
            smart_money_score=trade.get('smart_money_score', 0.0)
        )

    def average_grade_vector(self, trades: List[GradedProfitVector]) -> GradedProfitVector:
        """
        Computes the average of a list of G(e) vectors.
        
        Args:
            trades: List of GradedProfitVector objects
            
        Returns:
            GradedProfitVector: Average vector
        """
        if not trades:
            return GradedProfitVector(0.0, 0.0, 0.0, 0.0, 0.0)
        
        arrays = [t.to_array() for t in trades]
        avg_array = np.mean(arrays, axis=0)
        return GradedProfitVector(
            profit=avg_array[0],
            volume_allocated=avg_array[1],
            time_held=avg_array[2],
            signal_strength=avg_array[3],
            smart_money_score=avg_array[4]
        )

    def profit_drift_vector(self, old: GradedProfitVector, new: GradedProfitVector) -> GradedProfitVector:
        """
        Computes the delta between two G(e) vectors.
        
        Args:
            old: Previous vector
            new: Current vector
            
        Returns:
            GradedProfitVector: Delta vector
        """
        old_array = old.to_array()
        new_array = new.to_array()
        delta = new_array - old_array
        return GradedProfitVector(
            profit=delta[0],
            volume_allocated=delta[1],
            time_held=delta[2],
            signal_strength=delta[3],
            smart_money_score=delta[4]
        )

    # --- SHELL STATE MATH ---

    def shell_entropy(self, distribution: List[float]) -> float:
        """
        Calculates entropy of a distribution (e.g. class probabilities).
        
        Args:
            distribution: List of probabilities
            
        Returns:
            float: Entropy value
        """
        distribution = np.array(distribution)
        distribution = distribution[distribution > 0]  # Remove zeros
        return float(-np.sum(distribution * np.log(distribution)))

    def phase_angle(self, vector: np.ndarray) -> float:
        """
        Calculates phase angle for a 2D signal vector.
        
        Args:
            vector: 2D vector
            
        Returns:
            float: Phase angle in radians
        """
        return float(np.arctan2(vector[1], vector[0]))

    def volatility(self, prices: List[float]) -> float:
        """
        Estimates volatility from price data using standard deviation.
        
        Args:
            prices: List of price values
            
        Returns:
            float: Volatility estimate
        """
        return float(np.std(prices))

    def drift(self, prices: List[float]) -> float:
        """
        Estimates linear drift (trend) over price data.
        
        Args:
            prices: List of price values
            
        Returns:
            float: Drift estimate
        """
        if len(prices) < 2:
            return 0.0
        return float(prices[-1] - prices[0]) / (len(prices) - 1)

    # --- FRACTAL LOGIC / SMART MONEY GAN ---

    def latent_similarity(self, z1: np.ndarray, z2: np.ndarray, threshold: float = 0.2) -> bool:
        """
        Returns True if two GAN latent vectors are similar enough.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            threshold: Similarity threshold
            
        Returns:
            bool: True if vectors are similar
        """
        dist = self.euclidean_distance(z1, z2)
        return dist <= threshold

    # --- SCHWAFIT VALIDATION HELPERS ---

    def dynamic_holdout_ratio(self, t: int, min_r: float = 0.1, max_r: float = 0.9, 
                            cycle: int = 1000, noise_scale: float = 0.03) -> float:
        """
        Computes time-dependent holdout ratio with sinusoidal modulation + noise.
        
        Args:
            t: Current time step
            min_r: Minimum ratio
            max_r: Maximum ratio
            cycle: Cycle period
            noise_scale: Noise amplitude
            
        Returns:
            float: Holdout ratio
        """
        alpha = (min_r + max_r) / 2
        beta = (max_r - min_r) / 2
        r_t = alpha + beta * np.sin(2 * np.pi * t / cycle) + np.random.normal(0, noise_scale)
        return float(np.clip(r_t, min_r, max_r))

    def compute_score(self, predicted: float, actual: float, scale: float = 1.0) -> float:
        """
        Computes a bounded accuracy or fitness score.
        
        Args:
            predicted: Predicted value
            actual: Actual value
            scale: Scaling factor
            
        Returns:
            float: Score in range [0, 1]
        """
        error = abs(predicted - actual)
        return float(max(0.0, 1.0 - (error / scale)))

    def score_strategy_performance(self, prediction: List[float], target: List[float]) -> float:
        """
        Scores a list of predicted outcomes vs target.
        
        Args:
            prediction: List of predicted values
            target: List of target values
            
        Returns:
            float: Performance score in range [0, 1]
        """
        if len(prediction) != len(target):
            return 0.0
        diffs = np.abs(np.array(prediction) - np.array(target))
        max_possible = np.max(target) if np.max(target) > 0 else 1
        return float(np.clip(1.0 - np.mean(diffs) / max_possible, 0.0, 1.0))

    # --- ADDITIONAL UTILITIES ---

    def spectral_entropy(self, signal: np.ndarray) -> float:
        """
        Calculates spectral entropy of a signal.
        
        Args:
            signal: Input signal
            
        Returns:
            float: Spectral entropy
        """
        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(signal)) ** 2
        # Normalize to get probability distribution
        spectrum = spectrum / np.sum(spectrum)
        # Calculate entropy
        return self.shell_entropy(spectrum)

    def entropy_slope(self, signal: np.ndarray, window_size: int = 10) -> float:
        """
        Calculates the rate of change of entropy over time.
        
        Args:
            signal: Input signal
            window_size: Size of sliding window
            
        Returns:
            float: Entropy slope (always positive)
        """
        if len(signal) < window_size:
            return 0.0
            
        entropies = []
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            entropies.append(self.shell_entropy(window))
            
        # Calculate slope using linear regression
        x = np.arange(len(entropies))
        slope, _ = np.polyfit(x, entropies, 1)
        return float(abs(slope))  # Return absolute value of slope

    def coherence_vector(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """
        Computes Pearson correlation coefficient between two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            np.ndarray: Correlation coefficient
        """
        return np.corrcoef(signal1, signal2)[0, 1] 