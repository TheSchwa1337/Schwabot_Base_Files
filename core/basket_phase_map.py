"""
Basket Phase Map
===============

Implements the phase mapping system for Schwabot's recursive trading intelligence.
Tracks basket states, paradox pressure, and phase transitions.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class PhaseMetrics:
    """Container for phase-specific metrics"""
    profit_trend: float
    stability: float
    memory_coherence: float
    paradox_pressure: float
    entropy_rate: float
    thermal_state: float
    bit_depth: int
    trust_score: float

@dataclass
class PhaseRegion:
    """Defines a region in phase space"""
    name: str
    profit_trend_range: Tuple[float, float]
    stability_range: Tuple[float, float]
    memory_coherence_range: Tuple[float, float]
    paradox_pressure_range: Tuple[float, float]
    entropy_rate_range: Tuple[float, float]
    thermal_state_range: Tuple[float, float]
    bit_depth_range: Tuple[int, int]
    trust_score_range: Tuple[float, float]

class BasketPhaseMap:
    """Maps basket states to phase regions and tracks phase transitions"""
    
    def __init__(self):
        self.phase_memory: Dict[str, Dict] = {}
        self.phase_history: Dict[str, List[PhaseMetrics]] = {}
        
        # Define phase regions
        self.phase_regions = {
            'STABLE': PhaseRegion(
                name='STABLE',
                profit_trend_range=(0.001, float('inf')),
                stability_range=(0.7, 1.0),
                memory_coherence_range=(0.8, 1.0),
                paradox_pressure_range=(0.0, 2.0),
                entropy_rate_range=(0.0, 0.3),
                thermal_state_range=(0.0, 0.6),
                bit_depth_range=(16, 81),
                trust_score_range=(0.7, 1.0)
            ),
            'SMART_MONEY': PhaseRegion(
                name='SMART_MONEY',
                profit_trend_range=(0.001, float('inf')),
                stability_range=(0.4, 0.7),
                memory_coherence_range=(0.6, 0.8),
                paradox_pressure_range=(2.0, 5.0),
                entropy_rate_range=(0.3, 0.7),
                thermal_state_range=(0.4, 0.8),
                bit_depth_range=(42, 81),
                trust_score_range=(0.8, 1.0)
            ),
            'UNSTABLE': PhaseRegion(
                name='UNSTABLE',
                profit_trend_range=(-float('inf'), 0.0),
                stability_range=(0.0, 0.4),
                memory_coherence_range=(0.0, 0.6),
                paradox_pressure_range=(5.0, float('inf')),
                entropy_rate_range=(0.7, 1.0),
                thermal_state_range=(0.8, 1.0),
                bit_depth_range=(4, 16),
                trust_score_range=(0.0, 0.4)
            ),
            'OVERLOADED': PhaseRegion(
                name='OVERLOADED',
                profit_trend_range=(-float('inf'), 0.0),
                stability_range=(0.0, 0.3),
                memory_coherence_range=(0.0, 0.4),
                paradox_pressure_range=(10.0, float('inf')),
                entropy_rate_range=(0.8, 1.0),
                thermal_state_range=(0.9, 1.0),
                bit_depth_range=(4, 8),
                trust_score_range=(0.0, 0.2)
            )
        }

    def update_phase_entry(self, 
                          basket_id: str, 
                          sha_key: str, 
                          phase_depth: int, 
                          trust_score: float,
                          current_metrics: Dict[str, float]) -> None:
        """Update phase entry with current metrics"""
        metrics = PhaseMetrics(
            profit_trend=current_metrics.get('profit_gradient', 0.0),
            stability=current_metrics.get('variance_of_returns', 0.0),
            memory_coherence=current_metrics.get('memory_coherence_score', 1.0),
            paradox_pressure=self._calculate_paradox_pressure(
                current_metrics.get('entropy_rate', 0.0),
                current_metrics.get('profit_gradient', 0.0)
            ),
            entropy_rate=current_metrics.get('entropy_rate', 0.0),
            thermal_state=current_metrics.get('thermal_state', 0.0),
            bit_depth=phase_depth,
            trust_score=trust_score
        )
        
        # Update phase memory
        self.phase_memory[basket_id] = {
            'sha_key': sha_key,
            'phase_depth': phase_depth,
            'trust_score': trust_score,
            'last_swap': datetime.now(),
            'metrics': metrics
        }
        
        # Update phase history
        if basket_id not in self.phase_history:
            self.phase_history[basket_id] = []
        self.phase_history[basket_id].append(metrics)
        
        # Keep history limited
        if len(self.phase_history[basket_id]) > 1000:
            self.phase_history[basket_id] = self.phase_history[basket_id][-1000:]

    def _calculate_paradox_pressure(self, entropy_rate: float, profit_gradient: float) -> float:
        """Calculate paradox pressure based on entropy and profit"""
        epsilon = 1e-9  # Prevent division by zero
        
        if profit_gradient > 0:
            # Smart money zone: high entropy but profitable
            return entropy_rate / (profit_gradient + epsilon)
        else:
            # Unstable zone: high entropy and unprofitable
            return entropy_rate / (abs(profit_gradient) + epsilon) * 100

    def _is_in_region(self, metrics: PhaseMetrics, region: PhaseRegion) -> bool:
        """Check if metrics fall within a phase region"""
        return (
            region.profit_trend_range[0] <= metrics.profit_trend <= region.profit_trend_range[1] and
            region.stability_range[0] <= metrics.stability <= region.stability_range[1] and
            region.memory_coherence_range[0] <= metrics.memory_coherence <= region.memory_coherence_range[1] and
            region.paradox_pressure_range[0] <= metrics.paradox_pressure <= region.paradox_pressure_range[1] and
            region.entropy_rate_range[0] <= metrics.entropy_rate <= region.entropy_rate_range[1] and
            region.thermal_state_range[0] <= metrics.thermal_state <= region.thermal_state_range[1] and
            region.bit_depth_range[0] <= metrics.bit_depth <= region.bit_depth_range[1] and
            region.trust_score_range[0] <= metrics.trust_score <= region.trust_score_range[1]
        )

    def check_basket_swap_condition(self, basket_id: str) -> Tuple[str, float]:
        """Check if basket should be swapped based on current phase"""
        info = self.get_phase_info(basket_id)
        if not info or 'metrics' not in info:
            return 'NO_SWAP', 0.0
            
        metrics = info['metrics']
        
        # Check each phase region
        for region_name, region in self.phase_regions.items():
            if self._is_in_region(metrics, region):
                # Calculate swap urgency based on how far into the region we are
                urgency = self._calculate_swap_urgency(metrics, region)
                return region_name, urgency
                
        return 'NORMAL', 0.0

    def _calculate_swap_urgency(self, metrics: PhaseMetrics, region: PhaseRegion) -> float:
        """Calculate how urgently a swap is needed based on position in phase region"""
        # Normalize each metric to [0,1] within its region
        profit_urgency = (metrics.profit_trend - region.profit_trend_range[0]) / (region.profit_trend_range[1] - region.profit_trend_range[0])
        stability_urgency = (metrics.stability - region.stability_range[0]) / (region.stability_range[1] - region.stability_range[0])
        coherence_urgency = (metrics.memory_coherence - region.memory_coherence_range[0]) / (region.memory_coherence_range[1] - region.memory_coherence_range[0])
        paradox_urgency = (metrics.paradox_pressure - region.paradox_pressure_range[0]) / (region.paradox_pressure_range[1] - region.paradox_pressure_range[0])
        
        # Weight the urgencies based on region type
        if region.name == 'SMART_MONEY':
            weights = [0.3, 0.2, 0.2, 0.3]  # Emphasize profit and paradox
        elif region.name in ['UNSTABLE', 'OVERLOADED']:
            weights = [0.2, 0.3, 0.3, 0.2]  # Emphasize stability and coherence
        else:
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
            
        return np.average([profit_urgency, stability_urgency, coherence_urgency, paradox_urgency], weights=weights)

    def get_phase_info(self, basket_id: str) -> Dict:
        """Get current phase information for a basket"""
        return self.phase_memory.get(basket_id, {})

    def get_phase_history(self, basket_id: str, window: int = 100) -> List[PhaseMetrics]:
        """Get recent phase history for a basket"""
        if basket_id not in self.phase_history:
            return []
        return self.phase_history[basket_id][-window:]

    def get_phase_transitions(self, basket_id: str) -> List[Tuple[str, str, datetime]]:
        """Get history of phase transitions for a basket"""
        if basket_id not in self.phase_history:
            return []
            
        transitions = []
        history = self.phase_history[basket_id]
        
        for i in range(1, len(history)):
            prev_phase = self._get_phase_for_metrics(history[i-1])
            curr_phase = self._get_phase_for_metrics(history[i])
            
            if prev_phase != curr_phase:
                transitions.append((prev_phase, curr_phase, datetime.now()))
                
        return transitions

    def _get_phase_for_metrics(self, metrics: PhaseMetrics) -> str:
        """Determine which phase a set of metrics belongs to"""
        for region_name, region in self.phase_regions.items():
            if self._is_in_region(metrics, region):
                return region_name
        return 'NORMAL'

# Example usage
if __name__ == "__main__":
    phase_map = BasketPhaseMap()
    
    # Update phase entry
    metrics = {
        'profit_gradient': 0.002,
        'variance_of_returns': 0.5,
        'memory_coherence_score': 0.7,
        'entropy_rate': 0.4,
        'thermal_state': 0.5
    }
    
    phase_map.update_phase_entry(
        basket_id="test_basket",
        sha_key="test_sha",
        phase_depth=42,
        trust_score=0.8,
        current_metrics=metrics
    )
    
    # Check swap condition
    phase, urgency = phase_map.check_basket_swap_condition("test_basket")
    print(f"Current phase: {phase}, Swap urgency: {urgency:.2f}") 