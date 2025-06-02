"""
Drift Exit Detector
=================

Implements edge-case drift entropy detection for Schwabot's recursive trading intelligence.
Detects subtle market exit zones and paradox fields before traditional swap thresholds.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class DriftMetrics:
    """Container for drift-specific metrics"""
    entropy_delta: float
    drift_confidence: float
    paradox_pressure: float
    trend_strength: float
    volume_profile: float
    thermal_state: float
    memory_coherence: float
    timestamp: datetime

class DriftExitDetector:
    """Detects drift exits and paradox fields in market signals"""
    
    def __init__(
        self,
        entropy_history: pd.DataFrame,
        trend_tracker: Dict[str, pd.DataFrame],
        window_size: int = 100
    ):
        self.entropy_history = entropy_history
        self.trend_tracker = trend_tracker
        self.window_size = window_size
        
        # Initialize drift metrics storage
        self.drift_metrics: Dict[str, List[DriftMetrics]] = {}
        
        # Define drift thresholds
        self.drift_thresholds = {
            'entropy_delta': -0.07,  # Significant entropy decrease
            'drift_confidence': 0.42,  # Minimum confidence threshold
            'paradox_pressure': 5.0,   # High paradox pressure
            'trend_strength': 0.3,     # Minimum trend strength
            'volume_profile': 0.6      # Volume confirmation threshold
        }
        
        # Initialize consciousness state tracking
        self.consciousness_states = {
            'BULL': {
                'entropy_range': (0.0, 0.3),
                'paradox_range': (0.0, 2.0),
                'drift_confidence_range': (0.65, 1.0)
            },
            'BEAR': {
                'entropy_range': (0.3, 0.7),
                'paradox_range': (2.0, 5.0),
                'drift_confidence_range': (0.42, 0.65)
            },
            'CRAB': {
                'entropy_range': (0.4, 0.6),
                'paradox_range': (1.0, 3.0),
                'drift_confidence_range': (0.5, 0.7)
            },
            'BLACK_SWAN': {
                'entropy_range': (0.7, 1.0),
                'paradox_range': (5.0, float('inf')),
                'drift_confidence_range': (0.0, 0.42)
            }
        }

    def detect_drift_exit(
        self,
        market_signal: Dict[str, any],
        pair: str
    ) -> Tuple[str, float]:
        """Detect drift exit conditions for a trading pair"""
        # Calculate entropy delta
        entropy_delta = self._calculate_entropy_delta(pair)
        
        # Get drift confidence
        drift_confidence = self._calculate_drift_confidence(pair)
        
        # Calculate paradox pressure
        paradox_pressure = self._calculate_paradox_pressure(
            market_signal.get('entropy_rate', 0.0),
            market_signal.get('profit_gradient', 0.0)
        )
        
        # Store metrics
        metrics = DriftMetrics(
            entropy_delta=entropy_delta,
            drift_confidence=drift_confidence,
            paradox_pressure=paradox_pressure,
            trend_strength=market_signal.get('trend_strength', 0.0),
            volume_profile=market_signal.get('volume_profile', 0.0),
            thermal_state=market_signal.get('thermal_state', 0.0),
            memory_coherence=market_signal.get('memory_coherence', 0.0),
            timestamp=datetime.now()
        )
        
        if pair not in self.drift_metrics:
            self.drift_metrics[pair] = []
        self.drift_metrics[pair].append(metrics)
        
        # Keep history limited
        if len(self.drift_metrics[pair]) > self.window_size:
            self.drift_metrics[pair] = self.drift_metrics[pair][-self.window_size:]
        
        # Determine consciousness state
        consciousness = self._determine_consciousness_state(metrics)
        
        # Check for drift exit conditions
        if self._check_drift_exit_conditions(metrics, consciousness):
            return 'drift_exit', self._calculate_exit_urgency(metrics, consciousness)
            
        # Check for ghost exit trigger
        if self._check_ghost_exit_trigger(metrics, consciousness):
            return 'ghost_exit_trigger', self._calculate_exit_urgency(metrics, consciousness)
            
        return 'hold', 0.0

    def _calculate_entropy_delta(self, pair: str) -> float:
        """Calculate entropy delta for a pair"""
        if pair not in self.entropy_history.columns:
            return 0.0
            
        recent_entropy = self.entropy_history[pair].tail(2)
        if len(recent_entropy) < 2:
            return 0.0
            
        return float(recent_entropy.iloc[-1] - recent_entropy.iloc[-2])

    def _calculate_drift_confidence(self, pair: str) -> float:
        """Calculate drift confidence for a pair"""
        if pair not in self.trend_tracker:
            return 0.5
            
        trend_data = self.trend_tracker[pair]
        if len(trend_data) < 2:
            return 0.5
            
        # Calculate confidence based on trend consistency
        recent_trend = trend_data.tail(self.window_size)
        trend_consistency = np.mean(np.abs(np.diff(recent_trend['trend'])))
        volume_consistency = np.mean(recent_trend['volume'] > recent_trend['volume'].mean())
        
        return float(0.7 * (1 - trend_consistency) + 0.3 * volume_consistency)

    def _calculate_paradox_pressure(
        self,
        entropy_rate: float,
        profit_gradient: float
    ) -> float:
        """Calculate paradox pressure based on entropy and profit"""
        epsilon = 1e-9  # Prevent division by zero
        
        if profit_gradient > 0:
            # Smart money zone: high entropy but profitable
            return entropy_rate / (profit_gradient + epsilon)
        else:
            # Unstable zone: high entropy and unprofitable
            return entropy_rate / (abs(profit_gradient) + epsilon) * 100

    def _determine_consciousness_state(self, metrics: DriftMetrics) -> str:
        """Determine current market consciousness state"""
        for state, ranges in self.consciousness_states.items():
            if (
                ranges['entropy_range'][0] <= metrics.entropy_delta <= ranges['entropy_range'][1] and
                ranges['paradox_range'][0] <= metrics.paradox_pressure <= ranges['paradox_range'][1] and
                ranges['drift_confidence_range'][0] <= metrics.drift_confidence <= ranges['drift_confidence_range'][1]
            ):
                return state
        return 'UNKNOWN'

    def _check_drift_exit_conditions(
        self,
        metrics: DriftMetrics,
        consciousness: str
    ) -> bool:
        """Check if drift exit conditions are met"""
        if consciousness == 'BULL':
            return (
                metrics.entropy_delta < self.drift_thresholds['entropy_delta'] and
                metrics.drift_confidence < self.drift_thresholds['drift_confidence'] and
                metrics.volume_profile > self.drift_thresholds['volume_profile']
            )
        elif consciousness == 'BEAR':
            return (
                metrics.entropy_delta > abs(self.drift_thresholds['entropy_delta']) and
                metrics.drift_confidence < self.drift_thresholds['drift_confidence'] and
                metrics.paradox_pressure > self.drift_thresholds['paradox_pressure']
            )
        elif consciousness == 'CRAB':
            return (
                abs(metrics.entropy_delta) > 0.1 and
                metrics.drift_confidence < 0.5 and
                metrics.trend_strength > self.drift_thresholds['trend_strength']
            )
        elif consciousness == 'BLACK_SWAN':
            return (
                metrics.paradox_pressure > self.drift_thresholds['paradox_pressure'] * 2 and
                metrics.entropy_delta < -0.2 and
                metrics.drift_confidence < 0.3
            )
        return False

    def _check_ghost_exit_trigger(
        self,
        metrics: DriftMetrics,
        consciousness: str
    ) -> bool:
        """Check for ghost exit trigger conditions"""
        if consciousness == 'BULL':
            return (
                metrics.drift_confidence < 0.65 and
                metrics.paradox_pressure > 3.0 and
                metrics.memory_coherence < 0.5
            )
        elif consciousness == 'BEAR':
            return (
                metrics.drift_confidence < 0.42 and
                metrics.entropy_delta > 0.1 and
                metrics.thermal_state > 0.8
            )
        elif consciousness == 'CRAB':
            return (
                metrics.drift_confidence < 0.5 and
                abs(metrics.entropy_delta) > 0.15 and
                metrics.volume_profile > 0.8
            )
        elif consciousness == 'BLACK_SWAN':
            return (
                metrics.paradox_pressure > 7.0 and
                metrics.drift_confidence < 0.3 and
                metrics.thermal_state > 0.9
            )
        return False

    def _calculate_exit_urgency(
        self,
        metrics: DriftMetrics,
        consciousness: str
    ) -> float:
        """Calculate exit urgency based on metrics and consciousness"""
        # Base urgency on how far metrics are from thresholds
        urgency = 0.0
        
        if consciousness == 'BULL':
            urgency = (
                (self.drift_thresholds['entropy_delta'] - metrics.entropy_delta) * 0.4 +
                (self.drift_thresholds['drift_confidence'] - metrics.drift_confidence) * 0.3 +
                (metrics.volume_profile - self.drift_thresholds['volume_profile']) * 0.3
            )
        elif consciousness == 'BEAR':
            urgency = (
                (metrics.entropy_delta - abs(self.drift_thresholds['entropy_delta'])) * 0.4 +
                (self.drift_thresholds['drift_confidence'] - metrics.drift_confidence) * 0.3 +
                (metrics.paradox_pressure - self.drift_thresholds['paradox_pressure']) * 0.3
            )
        elif consciousness == 'CRAB':
            urgency = (
                abs(metrics.entropy_delta) * 0.4 +
                (0.5 - metrics.drift_confidence) * 0.3 +
                (metrics.trend_strength - self.drift_thresholds['trend_strength']) * 0.3
            )
        elif consciousness == 'BLACK_SWAN':
            urgency = (
                (metrics.paradox_pressure - self.drift_thresholds['paradox_pressure'] * 2) * 0.4 +
                (0.3 - metrics.drift_confidence) * 0.3 +
                (metrics.thermal_state - 0.8) * 0.3
            )
            
        return float(np.clip(urgency, 0.0, 1.0))

    def get_drift_stats(self, pair: str) -> Dict:
        """Get statistics about drift detection for a pair"""
        if pair not in self.drift_metrics:
            return {}
            
        metrics = self.drift_metrics[pair]
        if not metrics:
            return {}
            
        return {
            'total_detections': len(metrics),
            'drift_exits': sum(1 for m in metrics if m.entropy_delta < self.drift_thresholds['entropy_delta']),
            'ghost_triggers': sum(1 for m in metrics if m.drift_confidence < self.drift_thresholds['drift_confidence']),
            'avg_paradox_pressure': float(np.mean([m.paradox_pressure for m in metrics])),
            'avg_drift_confidence': float(np.mean([m.drift_confidence for m in metrics])),
            'consciousness_distribution': {
                state: sum(1 for m in metrics if self._determine_consciousness_state(m) == state)
                for state in self.consciousness_states.keys()
            }
        }

# Example usage
if __name__ == "__main__":
    # Create sample data
    entropy_history = pd.DataFrame({
        'BTC/USD': [0.2, 0.3, 0.4, 0.5, 0.6]
    })
    
    trend_tracker = {
        'BTC/USD': pd.DataFrame({
            'trend': [0.1, 0.2, 0.3, 0.4, 0.5],
            'volume': [1000, 1200, 1100, 1300, 1400]
        })
    }
    
    detector = DriftExitDetector(entropy_history, trend_tracker)
    
    # Test market signal
    market_signal = {
        'entropy_rate': 0.4,
        'profit_gradient': 0.002,
        'trend_strength': 0.6,
        'volume_profile': 0.7,
        'thermal_state': 0.5,
        'memory_coherence': 0.8
    }
    
    # Detect drift exit
    action, urgency = detector.detect_drift_exit(market_signal, 'BTC/USD')
    print(f"Action: {action}, Urgency: {urgency:.4f}")
    
    # Get drift stats
    stats = detector.get_drift_stats('BTC/USD')
    print("\nDrift stats:")
    print(stats) 