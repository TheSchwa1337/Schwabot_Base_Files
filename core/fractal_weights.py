"""
Fractal Weighting Bus
=====================

Real-time weighting system for fractal engines based on:
- Profit impact measurement
- Collapse prediction accuracy  
- Drift resilience performance
- Historical success rates

Mathematical Foundation:
w_i(t) = α·profit_impact_i + β·accuracy_i + γ·resilience_i + δ·momentum_i
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class FractalPerformance:
    """Performance metrics for a single fractal system"""
    fractal_name: str
    profit_impact: float = 0.0
    accuracy_score: float = 0.5
    resilience_score: float = 0.5
    momentum_score: float = 0.0
    success_rate: float = 0.5
    last_updated: float = 0.0

@dataclass
class WeightUpdate:
    """Record of weight adjustment"""
    timestamp: float
    fractal_name: str
    old_weight: float
    new_weight: float
    reason: str
    impact_magnitude: float

class FractalWeightBus:
    """
    Dynamic weighting system for fractal engines.
    
    Continuously adjusts fractal weights based on real-time performance
    metrics to optimize overall system profit convergence.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize fractal weighting bus.
        
        Args:
            initial_weights: Starting weights for each fractal system
        """
        # Default fractal systems
        self.fractal_names = ["forever", "paradox", "eco", "braid"]
        
        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            self.weights = {name: 1.0 for name in self.fractal_names}
            
        # Performance tracking
        self.performance_history: Dict[str, deque] = {
            name: deque(maxlen=100) for name in self.fractal_names
        }
        
        self.current_performance: Dict[str, FractalPerformance] = {
            name: FractalPerformance(fractal_name=name) for name in self.fractal_names
        }
        
        # Weight adjustment history
        self.weight_history: deque = deque(maxlen=200)
        
        # Adjustment parameters
        self.learning_rate = 0.1
        self.momentum_decay = 0.95
        self.min_weight = 0.1
        self.max_weight = 2.0
        
        # Performance coefficients
        self.profit_coefficient = 0.4
        self.accuracy_coefficient = 0.3
        self.resilience_coefficient = 0.2
        self.momentum_coefficient = 0.1
        
        logger.info(f"Fractal Weight Bus initialized with systems: {self.fractal_names}")
    
    def update_performance(self, fractal_name: str, feedback: Dict[str, Any]):
        """
        Update performance metrics for a specific fractal system.
        
        Args:
            fractal_name: Name of the fractal system
            feedback: Performance feedback dictionary
        """
        if fractal_name not in self.fractal_names:
            logger.warning(f"Unknown fractal system: {fractal_name}")
            return
            
        perf = self.current_performance[fractal_name]
        
        # Update profit impact
        if "profit_delta" in feedback:
            profit_impact = self._calculate_profit_impact(feedback["profit_delta"])
            perf.profit_impact = self._smooth_update(perf.profit_impact, profit_impact, 0.2)
            
        # Update accuracy score
        if "prediction_accuracy" in feedback:
            perf.accuracy_score = self._smooth_update(
                perf.accuracy_score, feedback["prediction_accuracy"], 0.15
            )
            
        # Update resilience score
        if "volatility_handling" in feedback:
            resilience = self._calculate_resilience_score(feedback)
            perf.resilience_score = self._smooth_update(perf.resilience_score, resilience, 0.1)
            
        # Update momentum score
        momentum = self._calculate_momentum_score(fractal_name, feedback)
        perf.momentum_score = self._smooth_update(perf.momentum_score, momentum, 0.1)
        
        # Update success rate
        if "success" in feedback:
            success_rate = self._update_success_rate(fractal_name, feedback["success"])
            perf.success_rate = success_rate
            
        perf.last_updated = time.time()
        
        # Store in history
        self.performance_history[fractal_name].append(perf)
        
        # Trigger weight recalculation
        self._recalculate_weights()
        
        logger.debug(f"Updated {fractal_name} performance: impact={perf.profit_impact:.3f}, "
                    f"accuracy={perf.accuracy_score:.3f}, resilience={perf.resilience_score:.3f}")
    
    def _calculate_profit_impact(self, profit_delta: float) -> float:
        """Calculate normalized profit impact score."""
        # Normalize profit delta to [0, 1] range
        impact = np.tanh(profit_delta / 100.0)  # Assuming profit in basis points
        return np.clip((impact + 1.0) / 2.0, 0.0, 1.0)
    
    def _calculate_resilience_score(self, feedback: Dict[str, Any]) -> float:
        """Calculate resilience score based on volatility handling."""
        volatility = feedback.get("volatility_handling", 0.5)
        stability = feedback.get("stability_maintained", True)
        
        base_score = volatility
        if not stability:
            base_score *= 0.5
            
        return np.clip(base_score, 0.0, 1.0)
    
    def _calculate_momentum_score(self, fractal_name: str, feedback: Dict[str, Any]) -> float:
        """Calculate momentum score based on recent performance trend."""
        history = self.performance_history[fractal_name]
        
        if len(history) < 3:
            return 0.5
            
        # Calculate trend in profit impact over recent history
        recent_impacts = [p.profit_impact for p in list(history)[-5:]]
        
        if len(recent_impacts) >= 2:
            trend = np.polyfit(range(len(recent_impacts)), recent_impacts, 1)[0]
            momentum = np.tanh(trend * 10)  # Scale and normalize
            return np.clip((momentum + 1.0) / 2.0, 0.0, 1.0)
            
        return 0.5
    
    def _update_success_rate(self, fractal_name: str, success: bool) -> float:
        """Update rolling success rate for fractal system."""
        history = self.performance_history[fractal_name]
        
        # Count recent successes
        recent_history = list(history)[-20:] if len(history) >= 20 else list(history)
        
        if not recent_history:
            return 0.5 if success else 0.4
            
        # Add current success to calculation
        success_count = sum(1 for p in recent_history if p.success_rate > 0.5)
        if success:
            success_count += 1
            
        total_count = len(recent_history) + 1
        return success_count / total_count
    
    def _smooth_update(self, current_value: float, new_value: float, alpha: float) -> float:
        """Apply exponential smoothing to metric updates."""
        return alpha * new_value + (1 - alpha) * current_value
    
    def _recalculate_weights(self):
        """Recalculate all fractal weights based on current performance."""
        new_weights = {}
        
        for fractal_name in self.fractal_names:
            perf = self.current_performance[fractal_name]
            
            # Calculate composite weight using performance coefficients
            composite_score = (
                self.profit_coefficient * perf.profit_impact +
                self.accuracy_coefficient * perf.accuracy_score +
                self.resilience_coefficient * perf.resilience_score +
                self.momentum_coefficient * perf.momentum_score
            )
            
            # Apply learning rate and momentum
            old_weight = self.weights[fractal_name]
            weight_delta = self.learning_rate * (composite_score - 0.5)  # Center around 0.5
            new_weight = old_weight + weight_delta
            
            # Apply bounds
            new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
            new_weights[fractal_name] = new_weight
            
            # Record weight change
            if abs(new_weight - old_weight) > 0.01:
                update_record = WeightUpdate(
                    timestamp=time.time(),
                    fractal_name=fractal_name,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    reason=f"Performance update: {composite_score:.3f}",
                    impact_magnitude=abs(new_weight - old_weight)
                )
                self.weight_history.append(update_record)
        
        # Normalize weights to prevent runaway growth
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            normalization_factor = len(self.fractal_names) / total_weight
            for name in new_weights:
                new_weights[name] *= normalization_factor
                
        self.weights = new_weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current fractal weights."""
        return self.weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for fractal_name in self.fractal_names:
            perf = self.current_performance[fractal_name]
            summary[fractal_name] = {
                "weight": self.weights[fractal_name],
                "profit_impact": perf.profit_impact,
                "accuracy_score": perf.accuracy_score,
                "resilience_score": perf.resilience_score,
                "momentum_score": perf.momentum_score,
                "success_rate": perf.success_rate,
                "last_updated": perf.last_updated
            }
            
        # Add system-wide metrics
        summary["system_metrics"] = {
            "total_weight": sum(self.weights.values()),
            "weight_variance": np.var(list(self.weights.values())),
            "dominant_fractal": max(self.weights.keys(), key=lambda k: self.weights[k]),
            "recent_adjustments": len([w for w in self.weight_history if time.time() - w.timestamp < 300])
        }
        
        return summary
    
    def handle_collapse_event(self, fractal_name: str, collapse_type: str):
        """Handle fractal collapse events with immediate weight adjustment."""
        if fractal_name not in self.fractal_names:
            return
            
        # Immediate weight reduction for collapsed fractal
        penalty_factor = {
            "paradox_collapse": 0.7,
            "eco_overstress": 0.8,
            "braid_inversion": 0.6,
            "forever_memory_overflow": 0.9
        }.get(collapse_type, 0.8)
        
        old_weight = self.weights[fractal_name]
        new_weight = old_weight * penalty_factor
        new_weight = max(new_weight, self.min_weight)
        
        self.weights[fractal_name] = new_weight
        
        # Record emergency adjustment
        emergency_update = WeightUpdate(
            timestamp=time.time(),
            fractal_name=fractal_name,
            old_weight=old_weight,
            new_weight=new_weight,
            reason=f"Emergency: {collapse_type}",
            impact_magnitude=old_weight - new_weight
        )
        self.weight_history.append(emergency_update)
        
        logger.warning(f"Emergency weight adjustment for {fractal_name}: {old_weight:.3f} → {new_weight:.3f} "
                      f"due to {collapse_type}")
    
    def boost_performing_fractal(self, fractal_name: str, boost_factor: float = 1.1):
        """Boost weight of well-performing fractal."""
        if fractal_name not in self.fractal_names:
            return
            
        old_weight = self.weights[fractal_name]
        new_weight = min(old_weight * boost_factor, self.max_weight)
        
        if new_weight != old_weight:
            self.weights[fractal_name] = new_weight
            
            boost_update = WeightUpdate(
                timestamp=time.time(),
                fractal_name=fractal_name,
                old_weight=old_weight,
                new_weight=new_weight,
                reason=f"Performance boost: {boost_factor}x",
                impact_magnitude=new_weight - old_weight
            )
            self.weight_history.append(boost_update)
            
            logger.info(f"Boosted {fractal_name} weight: {old_weight:.3f} → {new_weight:.3f}")
    
    def reset_weights(self):
        """Reset all weights to default values."""
        old_weights = self.weights.copy()
        self.weights = {name: 1.0 for name in self.fractal_names}
        
        # Record reset
        for name in self.fractal_names:
            reset_update = WeightUpdate(
                timestamp=time.time(),
                fractal_name=name,
                old_weight=old_weights[name],
                new_weight=1.0,
                reason="System reset",
                impact_magnitude=abs(old_weights[name] - 1.0)
            )
            self.weight_history.append(reset_update)
            
        logger.info("All fractal weights reset to default values")

# Example usage
if __name__ == "__main__":
    # Test fractal weight bus
    weight_bus = FractalWeightBus()
    
    # Simulate performance updates
    weight_bus.update_performance("forever", {
        "profit_delta": 50.0,
        "prediction_accuracy": 0.8,
        "volatility_handling": 0.7,
        "success": True
    })
    
    weight_bus.update_performance("paradox", {
        "profit_delta": -20.0,
        "prediction_accuracy": 0.4,
        "volatility_handling": 0.3,
        "success": False
    })
    
    # Check results
    weights = weight_bus.get_weights()
    summary = weight_bus.get_performance_summary()
    
    print(f"Updated weights: {weights}")
    print(f"Performance summary: {summary}")
    
    # Test collapse handling
    weight_bus.handle_collapse_event("paradox", "paradox_collapse")
    print(f"Weights after collapse: {weight_bus.get_weights()}") 