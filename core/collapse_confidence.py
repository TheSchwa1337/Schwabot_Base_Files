"""
Collapse Confidence Engine
==========================

Mathematically sound confidence scoring for collapse events and echo triggers.
Implements the core confidence equation that drives vault routing, ghost decay,
and fractal weight adjustments.

Mathematical Foundation:
Confidence = |Δprofit|/(σψ + ε) · coherence^λ

Where:
- Δprofit = projected gain/loss from echo
- σψ = braid phase volatility  
- ε = stability epsilon (prevents division by zero)
- coherence = moving mean of Braid + Paradox alignment
- λ ∈ [0.8, 1.5] = coherence amplification scalar
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class CollapseState:
    """Enhanced collapse state with mathematical confidence scoring"""
    timestamp: float
    collapse_id: str
    confidence: float  # Core confidence score [0.0, 1.0]
    profit_delta: float  # Expected profit change in basis points
    braid_volatility: float  # σψ - braid phase volatility
    coherence: float  # Fractal alignment measure
    coherence_amplification: float  # λ parameter
    stability_epsilon: float  # ε parameter
    raw_confidence: float  # Pre-normalization confidence
    confidence_components: Dict[str, float] = field(default_factory=dict)

@dataclass
class ConfidenceMetrics:
    """Tracking metrics for confidence calculation"""
    total_calculations: int = 0
    avg_confidence: float = 0.0
    confidence_variance: float = 0.0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    coherence_history: List[float] = field(default_factory=list)
    volatility_history: List[float] = field(default_factory=list)

class CollapseConfidenceEngine:
    """
    Mathematical confidence scoring engine for collapse events.
    
    Provides rigorous confidence calculation that drives:
    - Echo trigger weighting
    - Vault volume sizing  
    - Ghost reuse decisions
    - Fractal weight adjustments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize collapse confidence engine.
        
        Args:
            config: Configuration parameters for confidence calculation
        """
        self.config = config or {}
        
        # Core mathematical parameters
        self.stability_epsilon = self.config.get('stability_epsilon', 0.01)
        self.coherence_amplification_min = self.config.get('lambda_min', 0.8)
        self.coherence_amplification_max = self.config.get('lambda_max', 1.5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Historical tracking for coherence calculation
        self.braid_history: deque = deque(maxlen=50)
        self.paradox_history: deque = deque(maxlen=50)
        self.volatility_history: deque = deque(maxlen=30)
        self.confidence_history: deque = deque(maxlen=100)
        
        # Metrics tracking
        self.metrics = ConfidenceMetrics()
        
        # Adaptive parameters
        self.adaptive_lambda = True
        self.current_lambda = 1.0
        
        logger.info("Collapse Confidence Engine initialized with mathematical scoring")
    
    def calculate_collapse_confidence(self, profit_delta: float, braid_signal: float, 
                                    paradox_signal: float, recent_volatility: List[float]) -> CollapseState:
        """
        Calculate mathematically rigorous confidence score for collapse event.
        
        Args:
            profit_delta: Expected profit change in basis points
            braid_signal: Current braid fractal signal
            paradox_signal: Current paradox fractal signal  
            recent_volatility: Recent volatility measurements
            
        Returns:
            CollapseState with calculated confidence and components
        """
        # Store signals for coherence calculation
        self.braid_history.append(braid_signal)
        self.paradox_history.append(paradox_signal)
        
        # Calculate braid phase volatility (σψ)
        braid_volatility = self._calculate_braid_volatility(recent_volatility)
        self.volatility_history.append(braid_volatility)
        
        # Calculate fractal coherence
        coherence = self._calculate_fractal_coherence()
        
        # Determine coherence amplification (λ)
        lambda_param = self._calculate_coherence_amplification(coherence, profit_delta)
        
        # Core confidence calculation
        raw_confidence = self._calculate_raw_confidence(
            profit_delta, braid_volatility, coherence, lambda_param
        )
        
        # Normalize confidence to [0, 1] range
        normalized_confidence = self._normalize_confidence(raw_confidence)
        
        # Create confidence components breakdown
        components = {
            "profit_magnitude": abs(profit_delta),
            "volatility_stability": 1.0 / (braid_volatility + self.stability_epsilon),
            "coherence_factor": coherence ** lambda_param,
            "lambda_amplification": lambda_param,
            "raw_score": raw_confidence,
            "normalization_factor": normalized_confidence / max(raw_confidence, 1e-6)
        }
        
        # Create collapse state
        collapse_state = CollapseState(
            timestamp=time.time(),
            collapse_id=self._generate_collapse_id(),
            confidence=normalized_confidence,
            profit_delta=profit_delta,
            braid_volatility=braid_volatility,
            coherence=coherence,
            coherence_amplification=lambda_param,
            stability_epsilon=self.stability_epsilon,
            raw_confidence=raw_confidence,
            confidence_components=components
        )
        
        # Update metrics and history
        self._update_metrics(collapse_state)
        self.confidence_history.append(normalized_confidence)
        
        logger.debug(f"Collapse confidence calculated: {normalized_confidence:.3f} "
                    f"(profit: {profit_delta:.1f}bp, coherence: {coherence:.3f})")
        
        return collapse_state
    
    def _calculate_braid_volatility(self, recent_volatility: List[float]) -> float:
        """Calculate braid phase volatility (σψ)."""
        if not recent_volatility:
            return 0.1  # Default moderate volatility
            
        # Combine recent market volatility with braid signal volatility
        market_vol = np.std(recent_volatility) if len(recent_volatility) > 1 else recent_volatility[0]
        
        # Add braid signal volatility if we have history
        if len(self.braid_history) > 3:
            braid_vol = np.std(list(self.braid_history)[-10:])
            combined_vol = 0.7 * market_vol + 0.3 * braid_vol
        else:
            combined_vol = market_vol
            
        return np.clip(combined_vol, 0.01, 2.0)
    
    def _calculate_fractal_coherence(self) -> float:
        """Calculate coherence as moving mean of Braid + Paradox alignment."""
        if len(self.braid_history) < 3 or len(self.paradox_history) < 3:
            return 0.5  # Default moderate coherence
            
        # Get recent signals
        recent_braid = list(self.braid_history)[-10:]
        recent_paradox = list(self.paradox_history)[-10:]
        
        # Calculate alignment between braid and paradox signals
        min_length = min(len(recent_braid), len(recent_paradox))
        if min_length < 2:
            return 0.5
            
        braid_vals = recent_braid[-min_length:]
        paradox_vals = recent_paradox[-min_length:]
        
        # Coherence as inverse of signal divergence
        signal_divergence = np.mean([abs(b - p) for b, p in zip(braid_vals, paradox_vals)])
        
        # Convert divergence to coherence [0, 1]
        coherence = np.exp(-signal_divergence * 2.0)
        
        return np.clip(coherence, 0.1, 1.0)
    
    def _calculate_coherence_amplification(self, coherence: float, profit_delta: float) -> float:
        """Calculate λ parameter for coherence amplification."""
        if not self.adaptive_lambda:
            return self.current_lambda
            
        # Adaptive λ based on coherence and profit magnitude
        base_lambda = 1.0
        
        # Higher coherence → higher amplification
        coherence_factor = 0.3 * (coherence - 0.5)
        
        # Higher profit potential → more aggressive amplification
        profit_factor = 0.2 * np.tanh(abs(profit_delta) / 100.0)
        
        adaptive_lambda = base_lambda + coherence_factor + profit_factor
        
        # Clamp to valid range
        lambda_param = np.clip(adaptive_lambda, 
                              self.coherence_amplification_min, 
                              self.coherence_amplification_max)
        
        self.current_lambda = lambda_param
        return lambda_param
    
    def _calculate_raw_confidence(self, profit_delta: float, braid_volatility: float, 
                                 coherence: float, lambda_param: float) -> float:
        """
        Calculate raw confidence using core mathematical formula.
        
        Confidence = |Δprofit|/(σψ + ε) · coherence^λ
        """
        # Profit magnitude component
        profit_magnitude = abs(profit_delta)
        
        # Stability component (inverse volatility)
        stability_factor = profit_magnitude / (braid_volatility + self.stability_epsilon)
        
        # Coherence amplification component
        coherence_factor = coherence ** lambda_param
        
        # Combined raw confidence
        raw_confidence = stability_factor * coherence_factor
        
        return raw_confidence
    
    def _normalize_confidence(self, raw_confidence: float) -> float:
        """Normalize raw confidence to [0, 1] range."""
        # Use adaptive normalization based on historical range
        if len(self.confidence_history) > 10:
            # Use historical percentile normalization
            hist_values = list(self.confidence_history)
            p95 = np.percentile(hist_values, 95)
            p5 = np.percentile(hist_values, 5)
            
            if p95 > p5:
                normalized = (raw_confidence - p5) / (p95 - p5)
            else:
                normalized = 0.5
        else:
            # Use simple sigmoid normalization for cold start
            normalized = 2.0 / (1.0 + np.exp(-raw_confidence)) - 1.0
            
        return np.clip(normalized, 0.0, 1.0)
    
    def _generate_collapse_id(self) -> str:
        """Generate unique collapse event ID."""
        timestamp = int(time.time() * 1000)
        return f"collapse_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def _update_metrics(self, collapse_state: CollapseState):
        """Update tracking metrics."""
        self.metrics.total_calculations += 1
        
        # Update running averages
        alpha = 0.1  # Smoothing factor
        self.metrics.avg_confidence = (
            alpha * collapse_state.confidence + 
            (1 - alpha) * self.metrics.avg_confidence
        )
        
        # Update confidence categorization
        if collapse_state.confidence > 0.7:
            self.metrics.high_confidence_count += 1
        elif collapse_state.confidence < 0.3:
            self.metrics.low_confidence_count += 1
            
        # Update history tracking
        self.metrics.coherence_history.append(collapse_state.coherence)
        self.metrics.volatility_history.append(collapse_state.braid_volatility)
        
        # Keep history bounded
        if len(self.metrics.coherence_history) > 100:
            self.metrics.coherence_history.pop(0)
        if len(self.metrics.volatility_history) > 100:
            self.metrics.volatility_history.pop(0)
    
    def is_high_confidence(self, collapse_state: CollapseState) -> bool:
        """Check if collapse state meets high confidence threshold."""
        return collapse_state.confidence > self.confidence_threshold
    
    def get_confidence_grade(self, confidence: float) -> str:
        """Get human-readable confidence grade."""
        if confidence >= 0.8:
            return "EXCELLENT"
        elif confidence >= 0.6:
            return "GOOD"
        elif confidence >= 0.4:
            return "MODERATE"
        elif confidence >= 0.2:
            return "POOR"
        else:
            return "VERY_POOR"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if self.metrics.total_calculations == 0:
            return {"status": "no_calculations"}
            
        # Calculate variance if we have enough data
        if len(self.confidence_history) > 5:
            confidence_variance = np.var(list(self.confidence_history))
        else:
            confidence_variance = 0.0
            
        return {
            "total_calculations": self.metrics.total_calculations,
            "average_confidence": self.metrics.avg_confidence,
            "confidence_variance": confidence_variance,
            "high_confidence_rate": self.metrics.high_confidence_count / self.metrics.total_calculations,
            "low_confidence_rate": self.metrics.low_confidence_count / self.metrics.total_calculations,
            "current_lambda": self.current_lambda,
            "avg_coherence": np.mean(self.metrics.coherence_history) if self.metrics.coherence_history else 0.0,
            "avg_volatility": np.mean(self.metrics.volatility_history) if self.metrics.volatility_history else 0.0,
            "confidence_stability": 1.0 - confidence_variance if confidence_variance < 1.0 else 0.0
        }
    
    def adjust_threshold(self, new_threshold: float):
        """Dynamically adjust confidence threshold."""
        self.confidence_threshold = np.clip(new_threshold, 0.1, 0.9)
        logger.info(f"Confidence threshold adjusted to {self.confidence_threshold:.3f}")
    
    def reset_metrics(self):
        """Reset all metrics and history."""
        self.metrics = ConfidenceMetrics()
        self.confidence_history.clear()
        self.braid_history.clear()
        self.paradox_history.clear()
        self.volatility_history.clear()
        logger.info("Collapse confidence metrics reset")

# Example usage and testing
if __name__ == "__main__":
    # Test collapse confidence engine
    engine = CollapseConfidenceEngine()
    
    # Simulate collapse event
    profit_delta = 75.0  # 75 basis points expected profit
    braid_signal = 0.8
    paradox_signal = 0.6
    recent_volatility = [0.2, 0.3, 0.25, 0.4, 0.35]
    
    # Calculate confidence
    collapse_state = engine.calculate_collapse_confidence(
        profit_delta, braid_signal, paradox_signal, recent_volatility
    )
    
    print(f"Collapse Confidence: {collapse_state.confidence:.3f}")
    print(f"Confidence Grade: {engine.get_confidence_grade(collapse_state.confidence)}")
    print(f"High Confidence: {engine.is_high_confidence(collapse_state)}")
    print(f"Components: {collapse_state.confidence_components}")
    
    # Test multiple calculations
    for i in range(10):
        profit = np.random.uniform(-50, 100)
        braid = np.random.uniform(0, 1)
        paradox = np.random.uniform(0, 1)
        vol = [np.random.uniform(0.1, 0.5) for _ in range(5)]
        
        state = engine.calculate_collapse_confidence(profit, braid, paradox, vol)
        print(f"Test {i+1}: Confidence={state.confidence:.3f}, Profit={profit:.1f}bp")
    
    # Print metrics summary
    print(f"\nMetrics Summary: {engine.get_metrics_summary()}") 