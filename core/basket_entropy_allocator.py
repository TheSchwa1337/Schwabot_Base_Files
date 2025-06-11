"""
Basket Entropy Allocator
=======================

Implements entropy-guided allocation for Schwabot's recursive trading intelligence.
Manages entropy bands and allocation weights based on market conditions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from config.io_utils import load_config, ensure_config_exists

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EntropyBand:
    """Defines an entropy band with its characteristics"""
    lower_bound: float
    upper_bound: float
    base_weight: float
    thermal_sensitivity: float
    memory_coherence: float
    phase_depth: float

class BasketEntropyAllocator:
    """Manages entropy-guided allocation for baskets"""
    
    def __init__(self, config_path: Path = None):
        # Load configuration
        if config_path is None:
            config_path = ensure_config_exists('matrix_response_paths.yaml')
        
        try:
            self.config = load_config(config_path)
            logger.info(f"BasketEntropyAllocator initialized with config from {config_path}")
        except ValueError as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            self.config = {}
        
        # Define entropy bands
        self.entropy_bands = [
            EntropyBand(0.0, 0.2, 0.1, 0.8, 0.9, 0.1),  # Low entropy
            EntropyBand(0.2, 0.4, 0.2, 0.6, 0.7, 0.3),  # Medium-low entropy
            EntropyBand(0.4, 0.6, 0.4, 0.4, 0.5, 0.5),  # Medium entropy
            EntropyBand(0.6, 0.8, 0.2, 0.6, 0.3, 0.7),  # Medium-high entropy
            EntropyBand(0.8, 1.0, 0.1, 0.8, 0.1, 0.9)   # High entropy
        ]
        
        # Initialize allocation history
        self.allocation_history: List[Dict] = []
        
        # Initialize performance metrics
        self.performance_metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'avg_profit': 0.0,
            'avg_thermal_cost': 0.0,
            'band_performance': {i: {
                'count': 0,
                'success_rate': 0.0,
                'avg_profit': 0.0
            } for i in range(len(self.entropy_bands))}
        }
        
        # Initialize phase memory for basket tracking
        self.phase_memory: Dict[str, Dict] = {}

    def calculate_allocation_weights(
        self,
        market_entropy: List[float],
        thermal_state: float,
        memory_coherence: float,
        phase_depth: float
    ) -> Dict[int, float]:
        """Calculate allocation weights based on current conditions"""
        # Normalize inputs
        thermal_state = np.clip(thermal_state, 0.0, 1.0)
        memory_coherence = np.clip(memory_coherence, 0.0, 1.0)
        phase_depth = np.clip(phase_depth, 0.0, 1.0)
        
        # Calculate base weights for each band
        weights = {}
        for i, band in enumerate(self.entropy_bands):
            # Calculate entropy match score
            entropy_match = np.mean([
                self._calculate_band_match(entropy, band)
                for entropy in market_entropy
            ])
            
            # Calculate thermal adjustment
            thermal_adjustment = 1.0 - abs(thermal_state - band.thermal_sensitivity)
            
            # Calculate memory coherence adjustment
            memory_adjustment = 1.0 - abs(memory_coherence - band.memory_coherence)
            
            # Calculate phase depth adjustment
            phase_adjustment = 1.0 - abs(phase_depth - band.phase_depth)
            
            # Combine adjustments with base weight
            weight = band.base_weight * (
                entropy_match * 0.4 +
                thermal_adjustment * 0.2 +
                memory_adjustment * 0.2 +
                phase_adjustment * 0.2
            )
            
            weights[i] = float(weight)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights

    def _calculate_band_match(self, entropy: float, band: EntropyBand) -> float:
        """Calculate how well an entropy value matches a band"""
        if band.lower_bound <= entropy <= band.upper_bound:
            # Calculate distance from band center
            center = (band.lower_bound + band.upper_bound) / 2
            distance = abs(entropy - center)
            max_distance = (band.upper_bound - band.lower_bound) / 2
            return 1.0 - (distance / max_distance)
        return 0.0

    def update_performance(
        self,
        band_index: int,
        profit: float,
        thermal_cost: float
    ):
        """Update performance metrics for a band"""
        self.performance_metrics['total_allocations'] += 1
        if profit > 0:
            self.performance_metrics['successful_allocations'] += 1
        
        # Update band-specific metrics
        band_metrics = self.performance_metrics['band_performance'][band_index]
        band_metrics['count'] += 1
        
        # Update success rate
        if profit > 0:
            band_metrics['success_rate'] = (
                (band_metrics['success_rate'] * (band_metrics['count'] - 1) + 1) /
                band_metrics['count']
            )
        else:
            band_metrics['success_rate'] = (
                band_metrics['success_rate'] * (band_metrics['count'] - 1) /
                band_metrics['count']
            )
        
        # Update average profit using exponential moving average
        alpha = 0.1  # Learning rate
        band_metrics['avg_profit'] = (
            (1 - alpha) * band_metrics['avg_profit'] +
            alpha * profit
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

    def get_band_stats(self) -> Dict:
        """Get statistics about band performance"""
        return {
            'total_allocations': self.performance_metrics['total_allocations'],
            'success_rate': (
                self.performance_metrics['successful_allocations'] /
                max(1, self.performance_metrics['total_allocations'])
            ),
            'avg_profit': self.performance_metrics['avg_profit'],
            'avg_thermal_cost': self.performance_metrics['avg_thermal_cost'],
            'band_performance': {
                i: {
                    'count': metrics['count'],
                    'success_rate': metrics['success_rate'],
                    'avg_profit': metrics['avg_profit']
                }
                for i, metrics in self.performance_metrics['band_performance'].items()
            }
        }

    def get_optimal_band(self, market_entropy: List[float]) -> int:
        """Get the optimal band for current market entropy"""
        weights = self.calculate_allocation_weights(
            market_entropy,
            thermal_state=0.5,  # Default values
            memory_coherence=0.5,
            phase_depth=0.5
        )
        
        # Consider both weights and historical performance
        scores = {}
        for band_index, weight in weights.items():
            band_metrics = self.performance_metrics['band_performance'][band_index]
            if band_metrics['count'] > 0:
                scores[band_index] = (
                    weight * 0.4 +
                    band_metrics['success_rate'] * 0.3 +
                    np.tanh(band_metrics['avg_profit']) * 0.3
                )
            else:
                scores[band_index] = weight
        
        return max(scores.items(), key=lambda x: x[1])[0]

    def update_phase_entry(
        self,
        basket_id: str,
        sha_key: str,
        phase_depth: int,
        trust_score: float,
        current_metrics: Dict[str, float]
    ):
        """Update phase entry for a basket"""
        self.phase_memory[basket_id] = {
            'sha_key': sha_key,
            'phase_depth': phase_depth,
            'trust_score': trust_score,
            'metrics': current_metrics.copy(),
            'timestamp': datetime.now()
        }
        logger.info(f"Updated phase entry for basket {basket_id}")

    def check_basket_swap_condition(self, basket_id: str) -> Tuple[str, float]:
        """Check if basket should swap and return phase/urgency"""
        if basket_id not in self.phase_memory:
            return 'NORMAL', 0.0
        
        entry = self.phase_memory[basket_id]
        metrics = entry['metrics']
        
        # Calculate swap urgency based on metrics
        urgency = 0.0
        
        # High entropy rate increases urgency
        if metrics.get('entropy_rate', 0) > 0.8:
            urgency += 0.3
        
        # Low trust score increases urgency
        if entry['trust_score'] < 0.4:
            urgency += 0.4
        
        # High thermal state increases urgency
        if metrics.get('thermal_state', 0) > 0.7:
            urgency += 0.3
        
        # Determine phase
        if urgency > 0.7:
            return 'SMART_MONEY', urgency
        else:
            return 'NORMAL', urgency

# Data Provider Interface
class DataProvider:
    """Abstract interface for data providers"""
    
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        raise NotImplementedError("Subclasses must implement this method.")

# Example usage
if __name__ == "__main__":
    allocator = BasketEntropyAllocator()
    
    # Test market entropy
    market_entropy = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Calculate allocation weights
    weights = allocator.calculate_allocation_weights(
        market_entropy,
        thermal_state=0.6,
        memory_coherence=0.7,
        phase_depth=0.4
    )
    
    print("Allocation weights:")
    for band, weight in weights.items():
        print(f"Band {band}: {weight:.4f}")
    
    # Get optimal band
    optimal_band = allocator.get_optimal_band(market_entropy)
    print(f"\nOptimal band: {optimal_band}")
    
    # Test phase entry
    test_metrics = {
        'profit_gradient': 0.002,
        'variance_of_returns': 0.5,
        'memory_coherence_score': 0.7,
        'entropy_rate': 0.4,
        'thermal_state': 0.5
    }
    
    allocator.update_phase_entry(
        basket_id="test_basket",
        sha_key="test_sha",
        phase_depth=42,
        trust_score=0.8,
        current_metrics=test_metrics
    )
    
    phase, urgency = allocator.check_basket_swap_condition("test_basket")
    print(f"\nBasket phase: {phase}, urgency: {urgency:.2f}") 