"""
Braid Fractal Implementation
===========================

Implements the Braid Fractal system for interference pattern analysis
between Forever, Paradox, and Eco fractals. This is the 4th fractal class
that manages inter-fractal resonance and conflict resolution.
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class BraidState:
    """State container for braid fractal calculations"""
    timestamp: float
    forever_value: float
    paradox_value: float
    eco_value: float
    braid_signal: float
    interference_pattern: str = "neutral"
    confidence: float = 0.5

class BraidFractal:
    """
    Braid Fractal implementation for inter-fractal interference analysis.
    
    The Braid Fractal computes interference patterns between the three core
    fractal systems and provides meta-signals for decision making.
    """
    
    def __init__(self, max_memory: int = 100, decay_factor: float = 0.95):
        """
        Initialize Braid Fractal system.
        
        Args:
            max_memory: Maximum number of historical states to retain
            decay_factor: Decay factor for historical influence
        """
        self.max_memory = max_memory
        self.decay_factor = decay_factor
        
        # Signal history for time-series analysis
        self.signal_history: deque = deque(maxlen=max_memory)
        self.state_history: deque = deque(maxlen=max_memory)
        
        # Interference pattern tracking
        self.pattern_weights = {
            "constructive": 0.0,
            "destructive": 0.0,
            "neutral": 1.0,
            "chaotic": 0.0
        }
        
        # Braid parameters
        self.convergence_threshold = 0.1
        self.divergence_threshold = 0.8
        self.chaos_threshold = 1.2
        
        logger.info("Braid Fractal initialized with interference pattern analysis")
    
    def update(self, f_vals: List[float], p_vals: List[float], 
               e_vals: List[float], t_range: List[float]) -> float:
        """
        Update braid fractal with new fractal values.
        
        Args:
            f_vals: Forever fractal values
            p_vals: Paradox fractal values  
            e_vals: Eco fractal values
            t_range: Time range for integration
            
        Returns:
            Current braid signal value
        """
        if len(f_vals) == 0 or len(p_vals) == 0 or len(e_vals) == 0:
            return 0.0
            
        # Calculate braid interlock function
        braid_signal = self._compute_braid_interlock(f_vals, p_vals, e_vals, t_range)
        
        # Store in history
        self.signal_history.append(braid_signal)
        
        # Create state record
        current_state = BraidState(
            timestamp=time.time(),
            forever_value=f_vals[-1] if f_vals else 0.0,
            paradox_value=p_vals[-1] if p_vals else 0.0,
            eco_value=e_vals[-1] if e_vals else 0.0,
            braid_signal=braid_signal,
            interference_pattern=self._classify_interference_pattern(f_vals[-1], p_vals[-1], e_vals[-1]),
            confidence=self._calculate_confidence()
        )
        
        self.state_history.append(current_state)
        
        # Update pattern weights
        self._update_pattern_weights(current_state)
        
        return braid_signal
    
    def _compute_braid_interlock(self, f_vals: List[float], p_vals: List[float], 
                                e_vals: List[float], t_range: List[float]) -> float:
        """
        Compute the braid interlock function:
        B(t) = ∫[F'(τ)·P(τ) - E(τ)²]dτ
        
        Args:
            f_vals: Forever fractal values
            p_vals: Paradox fractal values
            e_vals: Eco fractal values
            t_range: Time range
            
        Returns:
            Braid interlock value
        """
        if len(t_range) < 2:
            return 0.0
            
        braid = 0.0
        
        for i in range(len(t_range) - 1):
            dt = t_range[i + 1] - t_range[i] if i + 1 < len(t_range) else 0.01
            
            # Calculate F'(τ) - derivative of Forever fractal
            if i + 1 < len(f_vals):
                dF = f_vals[i + 1] - f_vals[i]
            else:
                dF = 0.0
                
            # Get current values
            P_val = p_vals[i] if i < len(p_vals) else 0.0
            E_val = e_vals[i] if i < len(e_vals) else 0.0
            
            # Compute integrand: F'(τ)·P(τ) - E(τ)²
            integrand = dF * P_val - E_val ** 2
            
            # Add to integral
            braid += integrand * dt
            
        return braid
    
    def _classify_interference_pattern(self, f_val: float, p_val: float, e_val: float) -> str:
        """
        Classify the current interference pattern between fractals.
        
        Args:
            f_val: Forever fractal value
            p_val: Paradox fractal value
            e_val: Eco fractal value
            
        Returns:
            Interference pattern classification
        """
        # Calculate alignment metrics
        fp_alignment = abs(f_val - p_val)
        pe_alignment = abs(p_val - e_val)
        ef_alignment = abs(e_val - f_val)
        
        avg_alignment = (fp_alignment + pe_alignment + ef_alignment) / 3.0
        
        # Classify based on alignment
        if avg_alignment < self.convergence_threshold:
            return "constructive"
        elif avg_alignment > self.divergence_threshold:
            return "destructive"
        elif avg_alignment > self.chaos_threshold:
            return "chaotic"
        else:
            return "neutral"
    
    def _calculate_confidence(self) -> float:
        """
        Calculate confidence in current braid signal based on historical stability.
        
        Returns:
            Confidence score [0, 1]
        """
        if len(self.signal_history) < 3:
            return 0.5
            
        # Calculate signal stability over recent history
        recent_signals = list(self.signal_history)[-10:]
        signal_variance = np.var(recent_signals)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = np.exp(-signal_variance)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _update_pattern_weights(self, state: BraidState):
        """
        Update pattern weights based on current state.
        
        Args:
            state: Current braid state
        """
        # Decay existing weights
        for pattern in self.pattern_weights:
            self.pattern_weights[pattern] *= self.decay_factor
            
        # Boost current pattern
        if state.interference_pattern in self.pattern_weights:
            self.pattern_weights[state.interference_pattern] += 0.1
            
        # Normalize weights
        total_weight = sum(self.pattern_weights.values())
        if total_weight > 0:
            for pattern in self.pattern_weights:
                self.pattern_weights[pattern] /= total_weight
    
    def averaged_braid(self, window: int = 10) -> float:
        """
        Get averaged braid signal over specified window.
        
        Args:
            window: Number of recent signals to average
            
        Returns:
            Averaged braid signal
        """
        if len(self.signal_history) < window:
            return 0.0
            
        recent_signals = list(self.signal_history)[-window:]
        return np.mean(recent_signals)
    
    def get_interference_summary(self) -> Dict[str, Any]:
        """
        Get summary of current interference patterns.
        
        Returns:
            Dictionary containing interference analysis
        """
        current_state = self.state_history[-1] if self.state_history else None
        
        return {
            'current_signal': current_state.braid_signal if current_state else 0.0,
            'current_pattern': current_state.interference_pattern if current_state else "neutral",
            'confidence': current_state.confidence if current_state else 0.0,
            'pattern_weights': self.pattern_weights.copy(),
            'signal_trend': self._calculate_signal_trend(),
            'stability_index': self._calculate_stability_index()
        }
    
    def _calculate_signal_trend(self) -> str:
        """Calculate trend direction of braid signal."""
        if len(self.signal_history) < 5:
            return "stable"
            
        recent = list(self.signal_history)[-5:]
        first_half = np.mean(recent[:2])
        second_half = np.mean(recent[-2:])
        
        diff = second_half - first_half
        
        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_stability_index(self) -> float:
        """Calculate stability index of braid system."""
        if len(self.signal_history) < 3:
            return 0.5
            
        # Calculate coefficient of variation
        signals = list(self.signal_history)
        mean_signal = np.mean(signals)
        std_signal = np.std(signals)
        
        if mean_signal == 0:
            return 0.0
            
        cv = std_signal / abs(mean_signal)
        stability = np.exp(-cv)  # Higher CV = lower stability
        
        return np.clip(stability, 0.0, 1.0)
    
    def predict_next_signal(self, horizon: int = 1) -> Tuple[float, float]:
        """
        Predict next braid signal value(s).
        
        Args:
            horizon: Number of steps to predict ahead
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        if len(self.signal_history) < 3:
            return 0.0, 0.0
            
        # Simple linear extrapolation
        recent_signals = list(self.signal_history)[-5:]
        
        # Calculate trend
        x = np.arange(len(recent_signals))
        y = np.array(recent_signals)
        
        # Linear regression
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            next_value = recent_signals[-1] + slope * horizon
        else:
            next_value = recent_signals[-1]
            
        # Confidence based on recent stability
        confidence = self._calculate_confidence()
        
        return next_value, confidence
    
    def detect_regime_change(self) -> bool:
        """
        Detect if there's been a significant regime change in braid patterns.
        
        Returns:
            True if regime change detected
        """
        if len(self.state_history) < 10:
            return False
            
        # Compare recent pattern distribution to historical
        recent_states = list(self.state_history)[-5:]
        historical_states = list(self.state_history)[:-5]
        
        recent_patterns = [s.interference_pattern for s in recent_states]
        historical_patterns = [s.interference_pattern for s in historical_states]
        
        # Calculate pattern distribution differences
        recent_dist = {p: recent_patterns.count(p) / len(recent_patterns) 
                      for p in set(recent_patterns)}
        historical_dist = {p: historical_patterns.count(p) / len(historical_patterns) 
                          for p in set(historical_patterns)}
        
        # Calculate KL divergence (simplified)
        all_patterns = set(recent_patterns + historical_patterns)
        kl_div = 0.0
        
        for pattern in all_patterns:
            p_recent = recent_dist.get(pattern, 1e-10)
            p_hist = historical_dist.get(pattern, 1e-10)
            kl_div += p_recent * np.log(p_recent / p_hist)
            
        # Threshold for regime change detection
        return kl_div > 0.5
    
    def reset(self):
        """Reset braid fractal state."""
        self.signal_history.clear()
        self.state_history.clear()
        self.pattern_weights = {
            "constructive": 0.0,
            "destructive": 0.0,
            "neutral": 1.0,
            "chaotic": 0.0
        }
        logger.info("Braid Fractal state reset")

# Example usage and testing
if __name__ == "__main__":
    # Test braid fractal
    braid = BraidFractal()
    
    # Simulate fractal values
    t_range = [0.0, 0.1, 0.2, 0.3, 0.4]
    f_vals = [0.5, 0.6, 0.7, 0.8, 0.9]  # Forever fractal
    p_vals = [0.3, 0.4, 0.2, 0.6, 0.5]  # Paradox fractal
    e_vals = [0.2, 0.3, 0.4, 0.3, 0.2]  # Eco fractal
    
    # Update braid
    signal = braid.update(f_vals, p_vals, e_vals, t_range)
    
    print(f"Braid signal: {signal}")
    print(f"Interference summary: {braid.get_interference_summary()}")
    
    # Test prediction
    next_val, confidence = braid.predict_next_signal()
    print(f"Predicted next signal: {next_val} (confidence: {confidence})") 