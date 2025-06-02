"""
Basket Swap Overlay Router
========================

Implements dynamic bit-map hierarchy for Schwabot's recursive trading intelligence.
Controls basket memory decisioning based on entropy state and profit confidence.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OverlayTier:
    """Defines a bit-depth tier with its characteristics"""
    name: str
    min_entropy: float
    max_entropy: float
    min_trust: float
    max_trust: float
    thermal_sensitivity: float
    memory_coherence: float
    phase_depth: float
    profit_threshold: float

class BasketSwapOverlayRouter:
    """Routes basket swaps through bit-depth tiers based on market conditions"""
    
    def __init__(self):
        # Define overlay tiers
        self.layers = {
            4: OverlayTier(
                name="quick_scalp",
                min_entropy=0.7,
                max_entropy=1.0,
                min_trust=0.3,
                max_trust=0.6,
                thermal_sensitivity=0.8,
                memory_coherence=0.3,
                phase_depth=0.2,
                profit_threshold=0.001
            ),
            8: OverlayTier(
                name="momentum_trade",
                min_entropy=0.5,
                max_entropy=0.8,
                min_trust=0.4,
                max_trust=0.7,
                thermal_sensitivity=0.6,
                memory_coherence=0.5,
                phase_depth=0.4,
                profit_threshold=0.002
            ),
            16: OverlayTier(
                name="daily_swing",
                min_entropy=0.3,
                max_entropy=0.6,
                min_trust=0.5,
                max_trust=0.8,
                thermal_sensitivity=0.4,
                memory_coherence=0.7,
                phase_depth=0.6,
                profit_threshold=0.003
            ),
            42: OverlayTier(
                name="trust_memory",
                min_entropy=0.2,
                max_entropy=0.5,
                min_trust=0.6,
                max_trust=0.9,
                thermal_sensitivity=0.3,
                memory_coherence=0.8,
                phase_depth=0.8,
                profit_threshold=0.004
            ),
            81: OverlayTier(
                name="entropy_coherence",
                min_entropy=0.0,
                max_entropy=0.3,
                min_trust=0.7,
                max_trust=1.0,
                thermal_sensitivity=0.2,
                memory_coherence=0.9,
                phase_depth=1.0,
                profit_threshold=0.005
            )
        }
        
        # Initialize routing history
        self.routing_history: List[Dict] = []
        
        # Initialize performance metrics
        self.performance_metrics = {
            'total_routes': 0,
            'successful_routes': 0,
            'avg_profit': 0.0,
            'avg_thermal_cost': 0.0,
            'tier_performance': {depth: {
                'count': 0,
                'success_rate': 0.0,
                'avg_profit': 0.0,
                'avg_thermal': 0.0
            } for depth in self.layers.keys()}
        }

    def route(
        self,
        entropy_level: float,
        trust_vector: List[float],
        thermal_state: float,
        memory_coherence: float,
        phase_depth: float,
        profit_gradient: float
    ) -> Tuple[int, float]:
        """Route swap decision through appropriate bit-depth tier"""
        # Calculate average trust
        avg_trust = np.mean(trust_vector)
        
        # Calculate route scores for each tier
        route_scores = {}
        for depth, tier in self.layers.items():
            # Calculate tier match score
            entropy_match = self._calculate_tier_match(
                entropy_level,
                tier.min_entropy,
                tier.max_entropy
            )
            
            trust_match = self._calculate_tier_match(
                avg_trust,
                tier.min_trust,
                tier.max_trust
            )
            
            thermal_match = 1.0 - abs(thermal_state - tier.thermal_sensitivity)
            memory_match = 1.0 - abs(memory_coherence - tier.memory_coherence)
            phase_match = 1.0 - abs(phase_depth - tier.phase_depth)
            profit_match = 1.0 if profit_gradient >= tier.profit_threshold else 0.0
            
            # Calculate weighted score
            route_scores[depth] = (
                entropy_match * 0.3 +
                trust_match * 0.2 +
                thermal_match * 0.15 +
                memory_match * 0.15 +
                phase_match * 0.1 +
                profit_match * 0.1
            )
        
        # Select best route
        best_depth = max(route_scores.items(), key=lambda x: x[1])[0]
        confidence = route_scores[best_depth]
        
        # Store routing decision
        self.routing_history.append({
            'timestamp': datetime.now(),
            'depth': best_depth,
            'confidence': confidence,
            'entropy': entropy_level,
            'trust': avg_trust,
            'thermal': thermal_state,
            'memory': memory_coherence,
            'phase': phase_depth,
            'profit': profit_gradient
        })
        
        # Keep history limited
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        return best_depth, confidence

    def _calculate_tier_match(
        self,
        value: float,
        min_val: float,
        max_val: float
    ) -> float:
        """Calculate how well a value matches a tier's range"""
        if min_val <= value <= max_val:
            # Calculate distance from range center
            center = (min_val + max_val) / 2
            distance = abs(value - center)
            max_distance = (max_val - min_val) / 2
            return 1.0 - (distance / max_distance)
        return 0.0

    def update_performance(
        self,
        depth: int,
        profit: float,
        thermal_cost: float
    ):
        """Update performance metrics for a tier"""
        self.performance_metrics['total_routes'] += 1
        if profit > 0:
            self.performance_metrics['successful_routes'] += 1
        
        # Update tier-specific metrics
        tier_metrics = self.performance_metrics['tier_performance'][depth]
        tier_metrics['count'] += 1
        
        # Update success rate
        if profit > 0:
            tier_metrics['success_rate'] = (
                (tier_metrics['success_rate'] * (tier_metrics['count'] - 1) + 1) /
                tier_metrics['count']
            )
        else:
            tier_metrics['success_rate'] = (
                tier_metrics['success_rate'] * (tier_metrics['count'] - 1) /
                tier_metrics['count']
            )
        
        # Update averages using exponential moving average
        alpha = 0.1  # Learning rate
        tier_metrics['avg_profit'] = (
            (1 - alpha) * tier_metrics['avg_profit'] +
            alpha * profit
        )
        tier_metrics['avg_thermal'] = (
            (1 - alpha) * tier_metrics['avg_thermal'] +
            alpha * thermal_cost
        )
        
        # Update global metrics
        self.performance_metrics['avg_profit'] = (
            (1 - alpha) * self.performance_metrics['avg_profit'] +
            alpha * profit
        )
        self.performance_metrics['avg_thermal_cost'] = (
            (1 - alpha) * self.performance_metrics['avg_thermal_cost'] +
            alpha * thermal_cost
        )

    def get_tier_stats(self) -> Dict:
        """Get statistics about tier performance"""
        return {
            'total_routes': self.performance_metrics['total_routes'],
            'success_rate': (
                self.performance_metrics['successful_routes'] /
                max(1, self.performance_metrics['total_routes'])
            ),
            'avg_profit': self.performance_metrics['avg_profit'],
            'avg_thermal_cost': self.performance_metrics['avg_thermal_cost'],
            'tier_performance': {
                depth: {
                    'count': metrics['count'],
                    'success_rate': metrics['success_rate'],
                    'avg_profit': metrics['avg_profit'],
                    'avg_thermal': metrics['avg_thermal']
                }
                for depth, metrics in self.performance_metrics['tier_performance'].items()
            }
        }

    def get_optimal_tier(
        self,
        entropy_level: float,
        trust_vector: List[float]
    ) -> int:
        """Get the optimal tier for current conditions"""
        # Calculate route scores
        route_scores = {}
        avg_trust = np.mean(trust_vector)
        
        for depth, tier in self.layers.items():
            # Calculate base score
            entropy_match = self._calculate_tier_match(
                entropy_level,
                tier.min_entropy,
                tier.max_entropy
            )
            trust_match = self._calculate_tier_match(
                avg_trust,
                tier.min_trust,
                tier.max_trust
            )
            
            # Get historical performance
            tier_metrics = self.performance_metrics['tier_performance'][depth]
            if tier_metrics['count'] > 0:
                performance_score = (
                    tier_metrics['success_rate'] * 0.6 +
                    np.tanh(tier_metrics['avg_profit']) * 0.4
                )
            else:
                performance_score = 0.5
            
            # Combine scores
            route_scores[depth] = (
                entropy_match * 0.4 +
                trust_match * 0.3 +
                performance_score * 0.3
            )
        
        return max(route_scores.items(), key=lambda x: x[1])[0]

# Example usage
if __name__ == "__main__":
    router = BasketSwapOverlayRouter()
    
    # Test routing
    depth, confidence = router.route(
        entropy_level=0.4,
        trust_vector=[0.7, 0.8, 0.9],
        thermal_state=0.5,
        memory_coherence=0.7,
        phase_depth=0.6,
        profit_gradient=0.003
    )
    
    print(f"Selected depth: {depth}, Confidence: {confidence:.4f}")
    
    # Update performance
    router.update_performance(depth, 0.002, 0.3)
    
    # Get tier stats
    stats = router.get_tier_stats()
    print("\nTier stats:")
    print(stats) 