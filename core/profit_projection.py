"""
Profit Projection Engine
========================

Forecasts profit trajectories P(t) over N future ticks based on:
- Braid fractal confidence and memory
- Volatility slope analysis  
- Recursive profit convergence patterns

Mathematical Foundation:
P(t) = Σ w_i(t) · f_i(t) where convergence → +∞ when aligned
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ProfitHorizon:
    """Container for profit projection data"""
    timestamp: float
    horizon_steps: int
    projected_profits: List[float]
    confidence_scores: List[float]
    volatility_impact: float
    braid_influence: float
    convergence_probability: float

class ProfitProjectionEngine:
    """
    Advanced profit trajectory forecasting using fractal synthesis.
    
    Implements recursive profit convergence mathematics:
    P(t+Δt) = P(t) + Σ[fractal_weights * fractal_signals * volatility_damping]
    """
    
    def __init__(self, max_horizon: int = 12, decay_rate: float = 0.95):
        """
        Initialize profit projection engine.
        
        Args:
            max_horizon: Maximum number of ticks to project ahead
            decay_rate: Decay rate for future projections
        """
        self.max_horizon = max_horizon
        self.decay_rate = decay_rate
        
        # Historical tracking
        self.projection_history: deque = deque(maxlen=100)
        self.accuracy_scores: deque = deque(maxlen=50)
        
        # Projection parameters
        self.volatility_damping = 0.8
        self.braid_amplification = 1.2
        self.convergence_threshold = 0.85
        
        logger.info(f"Profit Projection Engine initialized with {max_horizon} step horizon")
    
    def forecast_profit(self, braid_memory: List[float], tick_volatility: float, 
                       fractal_weights: Dict[str, float], current_profit: float = 0.0) -> ProfitHorizon:
        """
        Generate profit forecast over specified horizon.
        
        Args:
            braid_memory: Historical braid signal values
            tick_volatility: Current market volatility
            fractal_weights: Current fractal system weights
            current_profit: Current profit baseline
            
        Returns:
            ProfitHorizon containing projected trajectory
        """
        if not braid_memory:
            return self._empty_horizon()
            
        # Calculate base projection parameters
        braid_trend = self._calculate_braid_trend(braid_memory)
        volatility_factor = self._calculate_volatility_factor(tick_volatility)
        weight_momentum = self._calculate_weight_momentum(fractal_weights)
        
        # Generate horizon projections
        projected_profits = []
        confidence_scores = []
        
        for i in range(self.max_horizon):
            # Time decay factor
            time_decay = self.decay_rate ** i
            
            # Volatility impact (decreases with distance)
            vol_impact = volatility_factor * (1 - (tick_volatility * i / 10))
            
            # Braid influence (amplified by confidence)
            braid_influence = braid_trend * self.braid_amplification * time_decay
            
            # Weight momentum contribution
            momentum_contrib = weight_momentum * time_decay
            
            # Combined projection
            projected_profit = current_profit + (
                braid_influence * vol_impact + momentum_contrib
            ) * (i + 1)
            
            # Confidence calculation (decreases with horizon distance)
            confidence = self._calculate_projection_confidence(
                i, braid_memory, tick_volatility, fractal_weights
            )
            
            projected_profits.append(projected_profit)
            confidence_scores.append(confidence)
        
        # Calculate convergence probability
        convergence_prob = self._calculate_convergence_probability(
            projected_profits, confidence_scores
        )
        
        # Create horizon object
        horizon = ProfitHorizon(
            timestamp=time.time(),
            horizon_steps=self.max_horizon,
            projected_profits=projected_profits,
            confidence_scores=confidence_scores,
            volatility_impact=volatility_factor,
            braid_influence=braid_trend,
            convergence_probability=convergence_prob
        )
        
        # Store in history
        self.projection_history.append(horizon)
        
        return horizon
    
    def _calculate_braid_trend(self, braid_memory: List[float]) -> float:
        """Calculate trend direction and strength from braid memory."""
        if len(braid_memory) < 3:
            return 0.0
            
        # Linear regression on recent braid values
        recent_values = braid_memory[-5:] if len(braid_memory) >= 5 else braid_memory
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return np.clip(slope, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_volatility_factor(self, tick_volatility: float) -> float:
        """Convert volatility to projection factor."""
        # Higher volatility = higher potential but lower confidence
        base_factor = 1.0 - (tick_volatility * self.volatility_damping)
        return np.clip(base_factor, 0.1, 2.0)
    
    def _calculate_weight_momentum(self, fractal_weights: Dict[str, float]) -> float:
        """Calculate momentum from fractal weight distribution."""
        if not fractal_weights:
            return 0.0
            
        # Calculate weight variance (higher variance = more momentum)
        weights = list(fractal_weights.values())
        weight_var = np.var(weights)
        
        # Convert to momentum signal
        momentum = weight_var * 0.5
        return np.clip(momentum, -0.5, 0.5)
    
    def _calculate_projection_confidence(self, step: int, braid_memory: List[float], 
                                       volatility: float, weights: Dict[str, float]) -> float:
        """Calculate confidence for specific projection step."""
        # Base confidence decreases with distance
        base_confidence = 1.0 / (1.0 + step * 0.1)
        
        # Braid stability factor
        if len(braid_memory) >= 3:
            braid_stability = 1.0 - np.std(braid_memory[-3:])
            braid_stability = np.clip(braid_stability, 0.0, 1.0)
        else:
            braid_stability = 0.5
            
        # Volatility confidence (lower volatility = higher confidence)
        vol_confidence = 1.0 - np.clip(volatility, 0.0, 1.0)
        
        # Weight balance factor
        if weights:
            weight_balance = 1.0 - np.std(list(weights.values()))
            weight_balance = np.clip(weight_balance, 0.0, 1.0)
        else:
            weight_balance = 0.5
            
        # Combined confidence
        combined = base_confidence * braid_stability * vol_confidence * weight_balance
        return np.clip(combined, 0.0, 1.0)
    
    def _calculate_convergence_probability(self, profits: List[float], 
                                         confidences: List[float]) -> float:
        """Calculate probability of profit convergence."""
        if not profits or not confidences:
            return 0.0
            
        # Check for positive trend
        positive_trend = sum(1 for p in profits if p > 0) / len(profits)
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Stability check (lower variance = higher convergence probability)
        profit_stability = 1.0 - (np.std(profits) / (np.mean(np.abs(profits)) + 1e-6))
        profit_stability = np.clip(profit_stability, 0.0, 1.0)
        
        # Combined convergence probability
        convergence = positive_trend * avg_confidence * profit_stability
        return np.clip(convergence, 0.0, 1.0)
    
    def _empty_horizon(self) -> ProfitHorizon:
        """Return empty horizon for edge cases."""
        return ProfitHorizon(
            timestamp=time.time(),
            horizon_steps=0,
            projected_profits=[],
            confidence_scores=[],
            volatility_impact=0.0,
            braid_influence=0.0,
            convergence_probability=0.0
        )
    
    def get_optimal_hold_duration(self, horizon: ProfitHorizon) -> int:
        """
        Determine optimal hold duration based on profit projections.
        
        Args:
            horizon: Profit projection horizon
            
        Returns:
            Optimal number of ticks to hold position
        """
        if not horizon.projected_profits:
            return 0
            
        # Find peak profit point with sufficient confidence
        best_step = 0
        best_score = 0.0
        
        for i, (profit, confidence) in enumerate(zip(horizon.projected_profits, horizon.confidence_scores)):
            # Score combines profit potential with confidence
            score = profit * confidence
            
            if score > best_score and confidence > 0.3:
                best_score = score
                best_step = i + 1
                
        return best_step
    
    def update_accuracy(self, predicted_profit: float, actual_profit: float):
        """Update accuracy tracking for model improvement."""
        if predicted_profit == 0:
            accuracy = 0.0
        else:
            accuracy = 1.0 - abs(predicted_profit - actual_profit) / abs(predicted_profit)
            accuracy = np.clip(accuracy, 0.0, 1.0)
            
        self.accuracy_scores.append(accuracy)
        
        # Log accuracy metrics
        if len(self.accuracy_scores) >= 10:
            avg_accuracy = np.mean(list(self.accuracy_scores)[-10:])
            logger.info(f"Profit projection accuracy (last 10): {avg_accuracy:.3f}")
    
    def get_projection_summary(self) -> Dict[str, Any]:
        """Get summary of current projection capabilities."""
        if not self.projection_history:
            return {"status": "no_projections"}
            
        recent_horizon = self.projection_history[-1]
        avg_accuracy = np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0
        
        return {
            "last_projection_time": recent_horizon.timestamp,
            "convergence_probability": recent_horizon.convergence_probability,
            "max_projected_profit": max(recent_horizon.projected_profits) if recent_horizon.projected_profits else 0.0,
            "average_accuracy": avg_accuracy,
            "total_projections": len(self.projection_history),
            "braid_influence": recent_horizon.braid_influence,
            "volatility_impact": recent_horizon.volatility_impact
        }

# Example usage
if __name__ == "__main__":
    # Test profit projection
    engine = ProfitProjectionEngine()
    
    # Simulate data
    braid_memory = [0.1, 0.2, 0.3, 0.4, 0.5]
    tick_volatility = 0.3
    fractal_weights = {"forever": 0.8, "paradox": 0.6, "eco": 0.7, "braid": 0.9}
    
    # Generate projection
    horizon = engine.forecast_profit(braid_memory, tick_volatility, fractal_weights, 100.0)
    
    print(f"Projected profits: {horizon.projected_profits}")
    print(f"Convergence probability: {horizon.convergence_probability}")
    print(f"Optimal hold duration: {engine.get_optimal_hold_duration(horizon)} ticks") 