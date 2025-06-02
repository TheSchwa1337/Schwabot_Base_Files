"""
Edge Vector Field
===============

Implements recursive scanning of historical tick and tensor data for Schwabot's trading intelligence.
Detects signature edge-case patterns and computes edge vector signatures.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class EdgeVector:
    """Container for edge vector metrics"""
    profit_gradient: float
    entropy_gradient: float
    tensor_variance: float
    volume_profile: float
    thermal_state: float
    memory_coherence: float
    paradox_pressure: float
    timestamp: datetime

class EdgeVectorField:
    """Scans for edge-case patterns and computes vector signatures"""
    
    def __init__(
        self,
        window_size: int = 100,
        min_confidence: float = 0.7
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence
        
        # Initialize vector field storage
        self.vector_field: Dict[str, List[EdgeVector]] = {}
        
        # Define edge patterns
        self.edge_patterns = {
            'inverse_profit_fork': {
                'profit_gradient_range': (-float('inf'), -0.002),
                'entropy_gradient_range': (0.1, float('inf')),
                'tensor_variance_range': (0.5, float('inf')),
                'volume_profile_range': (0.7, float('inf'))
            },
            'shadow_pump': {
                'profit_gradient_range': (0.002, float('inf')),
                'entropy_gradient_range': (-float('inf'), -0.1),
                'tensor_variance_range': (0.3, 0.7),
                'volume_profile_range': (0.5, 0.8)
            },
            'paradox_spike': {
                'profit_gradient_range': (-0.001, 0.001),
                'entropy_gradient_range': (0.2, float('inf')),
                'tensor_variance_range': (0.8, float('inf')),
                'volume_profile_range': (0.6, float('inf'))
            },
            'thermal_breakdown': {
                'profit_gradient_range': (-float('inf'), 0.0),
                'entropy_gradient_range': (0.0, float('inf')),
                'tensor_variance_range': (0.0, float('inf')),
                'volume_profile_range': (0.0, float('inf')),
                'thermal_threshold': 0.8
            }
        }
        
        # Initialize pattern detection history
        self.pattern_history: List[Dict] = []
        
        # Initialize performance metrics
        self.performance_metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'avg_profit': 0.0,
            'avg_thermal_cost': 0.0,
            'pattern_performance': {pattern: {
                'count': 0,
                'success_rate': 0.0,
                'avg_profit': 0.0,
                'avg_thermal': 0.0
            } for pattern in self.edge_patterns.keys()}
        }

    def compute_vector_signature(
        self,
        pair: str,
        market_data: Dict[str, any]
    ) -> Tuple[EdgeVector, float]:
        """Compute edge vector signature for current market state"""
        profit_history = market_data.get('profit_history', [])
        entropy_history = market_data.get('entropy_history', [])
        tensor_data = market_data.get('tensor_data', [])

        if not (profit_history and entropy_history and tensor_data):
            return None, 0.0

        # Calculate gradients
        profit_gradient = float(np.gradient(profit_history)[-1])
        entropy_gradient = float(np.gradient(entropy_history)[-1])
        tensor_variance = float(np.var(tensor_data))
        
        # Create edge vector
        vector = EdgeVector(
            profit_gradient=profit_gradient,
            entropy_gradient=entropy_gradient,
            tensor_variance=tensor_variance,
            volume_profile=market_data.get('volume_profile', 0.5),
            thermal_state=market_data.get('thermal_state', 0.5),
            memory_coherence=market_data.get('memory_coherence', 0.8),
            paradox_pressure=market_data.get('paradox_pressure', 2.0),
            timestamp=datetime.now()
        )
        
        # Store vector
        if pair not in self.vector_field:
            self.vector_field[pair] = []
        self.vector_field[pair].append(vector)
        
        # Keep history limited
        if len(self.vector_field[pair]) > self.window_size:
            self.vector_field[pair] = self.vector_field[pair][-self.window_size:]
        
        # Calculate confidence
        confidence = self._calculate_vector_confidence(vector)
        
        return vector, confidence

    def _calculate_vector_confidence(self, vector: EdgeVector) -> float:
        """Calculate confidence in vector signature"""
        pattern_scores = []
        
        for pattern, ranges in self.edge_patterns.items():
            # Calculate pattern match score
            profit_match = self._calculate_range_match(
                vector.profit_gradient,
                ranges['profit_gradient_range']
            )
            entropy_match = self._calculate_range_match(
                vector.entropy_gradient,
                ranges['entropy_gradient_range']
            )
            tensor_match = self._calculate_range_match(
                vector.tensor_variance,
                ranges['tensor_variance_range']
            )
            volume_match = self._calculate_range_match(
                vector.volume_profile,
                ranges['volume_profile_range']
            )
            
            # Special case for thermal breakdown
            if pattern == 'thermal_breakdown':
                thermal_match = 1.0 if vector.thermal_state >= ranges['thermal_threshold'] else 0.0
            else:
                thermal_match = 1.0
            
            # Calculate weighted score
            pattern_score = (
                profit_match * 0.3 +
                entropy_match * 0.3 +
                tensor_match * 0.2 +
                volume_match * 0.1 +
                thermal_match * 0.1
            )
            
            pattern_scores.append(pattern_score)
        
        # Return the highest confidence score or a low confidence if no match is found
        return max(pattern_scores, default=0.0)

    def _calculate_range_match(
        self,
        value: float,
        range_tuple: Tuple[float, float]
    ) -> float:
        """Calculate how well a value matches a range"""
        min_val, max_val = range_tuple
        
        if min_val <= value <= max_val:
            # Calculate distance from range center
            center = (min_val + max_val) / 2
            distance = abs(value - center)
            max_distance = (max_val - min_val) / 2
            return 1.0 - (distance / max_distance)
        return 0.0

    def detect_edge_patterns(
        self,
        vector: EdgeVector,
        confidence: float
    ) -> List[Tuple[str, float]]:
        """Detect edge patterns in vector signature"""
        if confidence < self.min_confidence:
            return []
            
        detected_patterns = []
        
        for pattern, ranges in self.edge_patterns.items():
            # Check if vector matches pattern
            profit_match = self._calculate_range_match(
                vector.profit_gradient,
                ranges['profit_gradient_range']
            )
            entropy_match = self._calculate_range_match(
                vector.entropy_gradient,
                ranges['entropy_gradient_range']
            )
            tensor_match = self._calculate_range_match(
                vector.tensor_variance,
                ranges['tensor_variance_range']
            )
            volume_match = self._calculate_range_match(
                vector.volume_profile,
                ranges['volume_profile_range']
            )
            
            # Special case for thermal breakdown
            if pattern == 'thermal_breakdown':
                thermal_match = 1.0 if vector.thermal_state >= ranges['thermal_threshold'] else 0.0
            else:
                thermal_match = 1.0
            
            # Calculate pattern confidence
            pattern_confidence = (
                profit_match * 0.3 +
                entropy_match * 0.3 +
                tensor_match * 0.2 +
                volume_match * 0.1 +
                thermal_match * 0.1
            )
            
            if pattern_confidence >= self.min_confidence:
                detected_patterns.append((pattern, pattern_confidence))
        
        return detected_patterns

    def update_pattern_performance(
        self,
        pattern: str,
        profit: float,
        thermal_cost: float
    ):
        """Update performance metrics for a pattern"""
        self.performance_metrics['total_detections'] += 1
        if profit > 0:
            self.performance_metrics['successful_detections'] += 1
        
        # Update pattern-specific metrics
        pattern_metrics = self.performance_metrics['pattern_performance'][pattern]
        pattern_metrics['count'] += 1
        
        # Update success rate
        if profit > 0:
            pattern_metrics['success_rate'] = (
                (pattern_metrics['success_rate'] * (pattern_metrics['count'] - 1) + 1) /
                pattern_metrics['count']
            )
        else:
            pattern_metrics['success_rate'] = (
                pattern_metrics['success_rate'] * (pattern_metrics['count'] - 1) /
                pattern_metrics['count']
            )
        
        # Update averages using exponential moving average
        alpha = 0.1  # Learning rate
        pattern_metrics['avg_profit'] = (
            (1 - alpha) * pattern_metrics['avg_profit'] +
            alpha * profit
        )
        pattern_metrics['avg_thermal'] = (
            (1 - alpha) * pattern_metrics['avg_thermal'] +
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

    def get_pattern_stats(self) -> Dict:
        """Get statistics about pattern detection"""
        return {
            'total_detections': self.performance_metrics['total_detections'],
            'success_rate': (
                self.performance_metrics['successful_detections'] /
                max(1, self.performance_metrics['total_detections'])
            ),
            'avg_profit': self.performance_metrics['avg_profit'],
            'avg_thermal_cost': self.performance_metrics['avg_thermal_cost'],
            'pattern_performance': {
                pattern: {
                    'count': metrics['count'],
                    'success_rate': metrics['success_rate'],
                    'avg_profit': metrics['avg_profit'],
                    'avg_thermal': metrics['avg_thermal']
                }
                for pattern, metrics in self.performance_metrics['pattern_performance'].items()
            }
        }

# Example usage
if __name__ == "__main__":
    field = EdgeVectorField()
    
    # Test market data
    market_data = {
        'profit_history': [0.001, 0.002, 0.003, 0.002, 0.001],
        'entropy_history': [0.2, 0.3, 0.4, 0.5, 0.6],
        'tensor_data': [0.1, 0.2, 0.3, 0.4, 0.5],
        'volume_profile': 0.7,
        'thermal_state': 0.5,
        'memory_coherence': 0.8,
        'paradox_pressure': 2.0
    }
    
    # Compute vector signature
    vector, confidence = field.compute_vector_signature('BTC/USD', market_data)
    print(f"Vector: {vector}, Confidence: {confidence}")
    
    # Detect patterns
    detected_patterns = field.detect_edge_patterns(vector, confidence)
    print("\nDetected patterns:")
    for pattern, conf in detected_patterns:
        print(f"{pattern}: {conf:.4f}")
    
    # Update performance
    field.update_pattern_performance('inverse_profit_fork', 0.002, 0.3)
    
    # Get pattern stats
    stats = field.get_pattern_stats()
    print("\nPattern stats:")
    print(stats) 