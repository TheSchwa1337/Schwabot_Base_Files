"""
Profit Trajectory Coprocessor
=============================

Implements the profit trajectory smoothing and vector calculation system for
thermal-aware processing decisions. This coprocessor analyzes profit momentum
over time to determine optimal GPU/CPU allocation and strategy weighting.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from collections import deque
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class ProfitZoneState(Enum):
    """Enumeration of profit zone states"""
    DRAWDOWN = "drawdown"
    STABLE = "stable"
    SURGING = "surging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

@dataclass
class TrajectoryVector:
    """Container for trajectory analysis results"""
    slope: float
    vector_strength: float  # 0-1 normalized
    zone_state: ProfitZoneState
    confidence: float
    momentum: float
    timestamp: datetime
    metadata: Dict

class ProfitTrajectoryCoprocessor:
    """
    Analyzes profit trajectories and generates vector signals for system optimization.
    
    This coprocessor implements the mathematical framework for profit-synchronized
    thermal-spike logic, providing drift coefficients and zone state detection.
    """
    
    def __init__(self, window_size: int = 10000, smoothing_beta: float = 0.95):
        """
        Initialize the profit trajectory coprocessor
        
        Args:
            window_size: Maximum number of profit data points to maintain
            smoothing_beta: Exponential smoothing coefficient (0.9-0.99)
        """
        self.window_size = window_size
        self.smoothing_beta = smoothing_beta
        self.profit_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.trajectory_history: List[TrajectoryVector] = []
        
        # Thresholds for zone state detection
        self.stable_threshold = 0.05  # ±5% slope considered stable
        self.surge_threshold = 0.15   # >15% slope considered surging
        self.volatility_threshold = 0.25  # Standard deviation threshold
        
        # Smoothed values
        self.smoothed_profit = 0.0
        self.smoothed_variance = 0.0
        
        # Last calculated vector
        self.last_vector: Optional[TrajectoryVector] = None
        
    def update(self, profit: float, tick_time: Optional[datetime] = None) -> TrajectoryVector:
        """
        Update profit trajectory with new data point and calculate vector
        
        Args:
            profit: Current profit value
            tick_time: Timestamp for this profit reading
            
        Returns:
            TrajectoryVector containing analysis results
        """
        if tick_time is None:
            tick_time = datetime.now(timezone.utc)
            
        # Add to history
        self.profit_history.append(profit)
        self.time_history.append(tick_time.timestamp())
        
        # Update smoothed values
        self._update_smoothed_values(profit)
        
        # Calculate trajectory vector
        vector = self._calculate_trajectory_vector(tick_time)
        
        # Update history
        self.trajectory_history.append(vector)
        if len(self.trajectory_history) > 1000:  # Keep last 1000 vectors
            self.trajectory_history = self.trajectory_history[-1000:]
            
        self.last_vector = vector
        
        logger.debug(f"Updated trajectory: zone={vector.zone_state.value}, "
                    f"vector_strength={vector.vector_strength:.3f}, "
                    f"slope={vector.slope:.3f}")
        
        return vector
        
    def _update_smoothed_values(self, profit: float) -> None:
        """Update exponentially smoothed profit and variance"""
        if len(self.profit_history) == 1:
            # First data point
            self.smoothed_profit = profit
            self.smoothed_variance = 0.0
        else:
            # Exponential smoothing
            self.smoothed_profit = (self.smoothing_beta * self.smoothed_profit + 
                                  (1 - self.smoothing_beta) * profit)
            
            # Update variance estimate
            deviation = profit - self.smoothed_profit
            self.smoothed_variance = (self.smoothing_beta * self.smoothed_variance + 
                                    (1 - self.smoothing_beta) * deviation**2)
                                    
    def _calculate_trajectory_vector(self, current_time: datetime) -> TrajectoryVector:
        """Calculate the current trajectory vector"""
        if len(self.profit_history) < 3:
            return TrajectoryVector(
                slope=0.0,
                vector_strength=0.0,
                zone_state=ProfitZoneState.UNKNOWN,
                confidence=0.0,
                momentum=0.0,
                timestamp=current_time,
                metadata={"reason": "insufficient_data"}
            )
            
        # Calculate slope using polynomial fit
        slope = self._calculate_profit_slope()
        
        # Calculate vector strength (sigmoid transformation)
        vector_strength = self._calculate_vector_strength(slope)
        
        # Determine zone state
        zone_state = self._determine_zone_state(slope)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence()
        
        # Calculate momentum
        momentum = self._calculate_momentum()
        
        # Prepare metadata
        metadata = {
            "data_points": len(self.profit_history),
            "smoothed_profit": self.smoothed_profit,
            "smoothed_variance": self.smoothed_variance,
            "time_span_minutes": (self.time_history[-1] - self.time_history[0]) / 60 if len(self.time_history) > 1 else 0
        }
        
        return TrajectoryVector(
            slope=slope,
            vector_strength=vector_strength,
            zone_state=zone_state,
            confidence=confidence,
            momentum=momentum,
            timestamp=current_time,
            metadata=metadata
        )
        
    def _calculate_profit_slope(self) -> float:
        """Calculate profit slope using cubic polynomial fit"""
        if len(self.profit_history) < 4:
            # Simple linear regression for small datasets
            x = np.arange(len(self.profit_history))
            y = np.array(list(self.profit_history))
            return np.polyfit(x, y, 1)[0] if len(y) > 1 else 0.0
            
        # Use cubic polynomial fit for better trajectory estimation
        try:
            x = np.array(list(self.time_history))
            y = np.array(list(self.profit_history))
            
            # Normalize time to prevent numerical issues
            x_normalized = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
            
            # Fit cubic polynomial
            poly_coeffs = np.polyfit(x_normalized, y, min(3, len(y) - 1))
            
            # Calculate derivative at current time (slope)
            derivative_coeffs = np.polyder(poly_coeffs)
            current_slope = np.polyval(derivative_coeffs, 1.0)  # At normalized time = 1
            
            return float(current_slope)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Polynomial fit failed: {e}. Using simple slope.")
            return self._calculate_simple_slope()
            
    def _calculate_simple_slope(self) -> float:
        """Fallback simple slope calculation"""
        if len(self.profit_history) < 2:
            return 0.0
            
        recent_profits = list(self.profit_history)[-10:]  # Last 10 points
        x = np.arange(len(recent_profits))
        return np.polyfit(x, recent_profits, 1)[0]
        
    def _calculate_vector_strength(self, slope: float) -> float:
        """Calculate vector strength using sigmoid transformation"""
        # Sigmoid transformation: V_profit = 1 / (1 + e^(-P'(t_now)))
        # Scale slope to reasonable range for sigmoid
        scaled_slope = slope * 10  # Adjust scaling factor as needed
        return 1.0 / (1.0 + np.exp(-scaled_slope))
        
    def _determine_zone_state(self, slope: float) -> ProfitZoneState:
        """Determine the current profit zone state"""
        abs_slope = abs(slope)
        volatility = np.sqrt(self.smoothed_variance)
        
        # Check for high volatility first
        if volatility > self.volatility_threshold:
            return ProfitZoneState.VOLATILE
            
        # Determine state based on slope
        if slope > self.surge_threshold:
            return ProfitZoneState.SURGING
        elif slope < -self.surge_threshold:
            return ProfitZoneState.DRAWDOWN
        elif abs_slope <= self.stable_threshold:
            return ProfitZoneState.STABLE
        else:
            # Moderate movement
            return ProfitZoneState.SURGING if slope > 0 else ProfitZoneState.DRAWDOWN
            
    def _calculate_confidence(self) -> float:
        """Calculate confidence in trajectory analysis"""
        if len(self.profit_history) < 3:
            return 0.0
            
        # Factors affecting confidence:
        # 1. Amount of data available
        data_factor = min(len(self.profit_history) / 100.0, 1.0)
        
        # 2. Consistency of trend (low variance relative to trend)
        trend_consistency = 1.0 / (1.0 + self.smoothed_variance) if self.smoothed_variance > 0 else 1.0
        
        # 3. Recency of data (all data is recent in our case)
        recency_factor = 1.0
        
        return data_factor * trend_consistency * recency_factor
        
    def _calculate_momentum(self) -> float:
        """Calculate momentum using exponential weighting of recent profits"""
        if len(self.profit_history) < 2:
            return 0.0
            
        recent_profits = list(self.profit_history)[-10:]  # Last 10 points
        weights = np.exp(np.linspace(-1, 0, len(recent_profits)))  # Exponential weights
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted momentum
        momentum = np.sum(weights * np.array(recent_profits))
        
        # Normalize to [-1, 1] range
        if self.smoothed_profit != 0:
            return np.tanh(momentum / abs(self.smoothed_profit))
        else:
            return 0.0
            
    def get_vector(self) -> Optional[TrajectoryVector]:
        """Get the most recent trajectory vector"""
        return self.last_vector
        
    def get_zone_state(self) -> ProfitZoneState:
        """Get current zone state"""
        return self.last_vector.zone_state if self.last_vector else ProfitZoneState.UNKNOWN
        
    def get_drift_coefficient(self) -> float:
        """Get drift coefficient for thermal/processing decisions"""
        if not self.last_vector:
            return 1.0  # Neutral coefficient
            
        # Base coefficient on vector strength and zone state
        base_coeff = self.last_vector.vector_strength
        
        # Adjust based on zone state
        zone_multiplier = {
            ProfitZoneState.SURGING: 1.2,
            ProfitZoneState.STABLE: 1.0,
            ProfitZoneState.DRAWDOWN: 0.8,
            ProfitZoneState.VOLATILE: 0.9,
            ProfitZoneState.UNKNOWN: 1.0
        }
        
        coefficient = base_coeff * zone_multiplier[self.last_vector.zone_state]
        
        # Apply confidence weighting
        confidence_weight = self.last_vector.confidence
        final_coefficient = (coefficient * confidence_weight + 
                           1.0 * (1 - confidence_weight))  # Blend with neutral
                           
        # Clamp to reasonable range
        return np.clip(final_coefficient, 0.5, 2.0)
        
    def should_expand_processing(self, threshold: float = 0.7) -> bool:
        """
        Determine if processing should be expanded based on trajectory
        
        Args:
            threshold: Vector strength threshold for expansion decision
            
        Returns:
            True if processing should be expanded
        """
        if not self.last_vector:
            return False
            
        # Expand if vector strength is high and zone state is favorable
        favorable_states = {ProfitZoneState.SURGING, ProfitZoneState.STABLE}
        
        return (self.last_vector.vector_strength > threshold and 
                self.last_vector.zone_state in favorable_states and
                self.last_vector.confidence > 0.5)
                
    def get_processing_allocation(self) -> Dict[str, float]:
        """
        Get recommended processing allocation percentages
        
        Returns:
            Dictionary with GPU/CPU allocation recommendations
        """
        if not self.last_vector:
            return {"gpu": 0.5, "cpu": 0.5}  # Default 50/50 split
            
        # Base allocation on vector strength and zone state
        vector_strength = self.last_vector.vector_strength
        
        if self.last_vector.zone_state == ProfitZoneState.SURGING:
            gpu_allocation = 0.3 + 0.4 * vector_strength  # 30-70% GPU
        elif self.last_vector.zone_state == ProfitZoneState.STABLE:
            gpu_allocation = 0.4 + 0.2 * vector_strength  # 40-60% GPU
        elif self.last_vector.zone_state == ProfitZoneState.DRAWDOWN:
            gpu_allocation = 0.2 + 0.2 * vector_strength  # 20-40% GPU
        else:  # VOLATILE or UNKNOWN
            gpu_allocation = 0.3  # Conservative 30% GPU
            
        # Apply confidence weighting
        confidence = self.last_vector.confidence
        gpu_allocation = gpu_allocation * confidence + 0.5 * (1 - confidence)
        
        # Ensure allocations sum to 1
        gpu_allocation = np.clip(gpu_allocation, 0.1, 0.9)
        cpu_allocation = 1.0 - gpu_allocation
        
        return {
            "gpu": gpu_allocation,
            "cpu": cpu_allocation,
            "recommended_burst": self.should_expand_processing()
        }
        
    def get_statistics(self) -> Dict[str, Union[float, int, str]]:
        """Get coprocessor statistics"""
        if not self.last_vector:
            return {"status": "no_data"}
            
        return {
            "data_points": len(self.profit_history),
            "current_slope": self.last_vector.slope,
            "vector_strength": self.last_vector.vector_strength,
            "zone_state": self.last_vector.zone_state.value,
            "confidence": self.last_vector.confidence,
            "momentum": self.last_vector.momentum,
            "drift_coefficient": self.get_drift_coefficient(),
            "smoothed_profit": self.smoothed_profit,
            "smoothed_variance": self.smoothed_variance,
            "processing_allocation": self.get_processing_allocation()
        }
        
    def reset(self) -> None:
        """Reset the coprocessor state"""
        self.profit_history.clear()
        self.time_history.clear()
        self.trajectory_history.clear()
        self.smoothed_profit = 0.0
        self.smoothed_variance = 0.0
        self.last_vector = None
        
        logger.info("Profit trajectory coprocessor reset")

# Example usage and testing
if __name__ == "__main__":
    # Initialize coprocessor
    coprocessor = ProfitTrajectoryCoprocessor(window_size=1000)
    
    # Simulate profit data
    np.random.seed(42)
    base_profit = 1000.0
    
    print("Simulating profit trajectory...")
    
    for i in range(50):
        # Simulate different profit patterns
        if i < 20:
            # Stable period
            profit = base_profit + np.random.normal(0, 10)
        elif i < 35:
            # Surging period
            profit = base_profit + (i - 20) * 5 + np.random.normal(0, 5)
        else:
            # Volatile period
            profit = base_profit + 75 + np.random.normal(0, 30)
            
        # Update coprocessor
        vector = coprocessor.update(profit)
        
        if i % 10 == 0:
            print(f"Step {i}: Profit={profit:.2f}, Zone={vector.zone_state.value}, "
                  f"Vector={vector.vector_strength:.3f}, Slope={vector.slope:.3f}")
    
    # Print final statistics
    stats = coprocessor.get_statistics()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Test processing recommendations
    allocation = coprocessor.get_processing_allocation()
    print(f"\nProcessing Allocation Recommendation:")
    print(f"  GPU: {allocation['gpu']:.1%}")
    print(f"  CPU: {allocation['cpu']:.1%}")
    print(f"  Burst Recommended: {allocation['recommended_burst']}") 