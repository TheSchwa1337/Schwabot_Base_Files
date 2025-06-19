"""
Detonation Sequencer for Schwabot
Manages the execution of trading strategies with Smart Money integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import unittest
from aleph_core.entropy_analyzer import EntropyAnalyzer
from .smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneyMetrics
import json

@dataclass
class DetonationState:
    """Current state of the detonation sequencer"""
    pattern_id: str
    confidence: float
    smart_money_score: float
    spoof_score: float
    wall_score: float
    velocity: str
    liquidity_resonance: str
    timestamp: datetime
    phase: int = 0
    active: bool = True
    phase_velocity_classification: Optional[str] = None
    routing_tag: Optional[str] = None
    sequence_id: Optional[str] = None
    pattern_hash: Optional[str] = None

@dataclass
class PatternMetrics:
    """Metrics for pattern analysis."""
    entropy: float = 0.0
    stability: float = 0.0
    momentum: float = 0.0
    volatility: float = 0.0
    correlation: float = 0.0
    latency_ms: float = 0.0

@dataclass
class DetonationHistory:
    detonation_history: List[DetonationState] = field(default_factory=list)

class DetonationSequencer(DetonationHistory):
    """
    Manages the execution of trading strategies with Smart Money integration.
    Uses market microstructure analysis to enhance decision making.
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 smart_money_threshold: float = 0.6):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.smart_money_threshold = smart_money_threshold
        self.smart_money = SmartMoneyAnalyzer()
        self.current_state: Optional[DetonationState] = None
        self.sequence_id_counter = 0
        self.pattern_metrics = PatternMetrics()
        self.sequence_history = []
        self.last_update = datetime.now()
        
    def generate_timing_hash(self, pattern_data: Dict) -> str:
        """
        Generate timing hash for pattern detonation.
        """
        # Combine pattern data with current timestamp
        data_str = f"{pattern_data.get('pattern', '')}{datetime.now().timestamp()}"
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def analyze_pattern(self, pattern_data: Dict) -> PatternMetrics:
        """
        Analyze pattern for detonation potential.
        """
        # Calculate pattern metrics
        entropy = np.random.random()  # Replace with actual entropy calculation
        stability = np.random.random()  # Replace with actual stability calculation
        momentum = np.random.random()  # Replace with actual momentum calculation
        volatility = np.random.random()  # Replace with actual volatility calculation
        correlation = np.random.random()  # Replace with actual correlation calculation
        
        self.pattern_metrics = PatternMetrics(
            entropy=entropy,
            stability=stability,
            momentum=momentum,
            volatility=volatility,
            correlation=correlation
        )
        
        return self.pattern_metrics
    
    def initiate_detonation(self, 
                          payload: Dict,
                          price: float,
                          volume: float,
                          order_book: Dict,
                          trades: List[Dict]) -> Dict:
        """
        Initiate detonation protocol with Smart Money analysis.
        
        Args:
            payload: Strategy payload including pattern and confidence
            price: Current price
            volume: Current volume
            order_book: Current order book state
            trades: Recent trades
            
        Returns:
            Dict containing detonation decision and metrics
        """
        # Get Smart Money metrics
        smart_money_metrics = self.smart_money.analyze_tick(
            price=price,
            volume=volume,
            order_book=order_book,
            trades=trades
        )
        
        # Create detonation state
        state = DetonationState(
            pattern_id=payload.get('pattern', 'UNKNOWN'),
            confidence=payload.get('confidence', 0.0),
            smart_money_score=smart_money_metrics.smart_money_score,
            spoof_score=smart_money_metrics.spoof_score,
            wall_score=smart_money_metrics.wall_score,
            velocity=smart_money_metrics.velocity,
            liquidity_resonance=smart_money_metrics.liquidity_resonance,
            timestamp=datetime.now()
        )
        
        # Store state
        self.current_state = state
        self.detonation_history.append(state)
        
        # Make detonation decision
        should_detonate = (
            state.confidence > self.confidence_threshold and
            state.smart_money_score > self.smart_money_threshold and
            state.spoof_score < 0.5  # Avoid high spoofing periods
        )
        
        # Additional conditions based on Smart Money metrics
        if should_detonate:
            if state.velocity == "HIGH_UP" and state.liquidity_resonance == "SWEEP_RESONANCE":
                should_detonate = True
            elif state.velocity == "HIGH_DOWN" and state.wall_score > 0.7:
                should_detonate = True
            else:
                should_detonate = False
        
        # Update state attributes
        state.phase_velocity_classification = self.evaluate_velocity_matrix(state)
        state.routing_tag = (
            "HIGH_PRIORITY" if state.confidence > 0.9 and state.smart_money_score > 0.8 else
            "MID_PRIORITY" if state.confidence > 0.7 else
            "LOW_PRIORITY"
        )
        self.sequence_id_counter += 1
        state.sequence_id = f"DSEQ-{self.sequence_id_counter:05d}"
        state.pattern_hash = self.generate_batch_hash(state.sequence_id, state.timestamp)
        
        return {
            'detonation_activated': should_detonate,
            'confidence': state.confidence,
            'smart_money_score': state.smart_money_score,
            'spoof_score': state.spoof_score,
            'wall_score': state.wall_score,
            'velocity': state.velocity,
            'liquidity_resonance': state.liquidity_resonance,
            'reason': self._get_detonation_reason(state) if should_detonate else "Insufficient confidence or Smart Money conditions"
        }
        
    def _get_detonation_reason(self, state: DetonationState) -> str:
        """Generate human-readable reason for detonation decision"""
        reasons = []
        
        if state.confidence > self.confidence_threshold:
            reasons.append(f"High pattern confidence ({state.confidence:.2f})")
            
        if state.smart_money_score > self.smart_money_threshold:
            reasons.append(f"Strong Smart Money signal ({state.smart_money_score:.2f})")
            
        if state.velocity == "HIGH_UP" and state.liquidity_resonance == "SWEEP_RESONANCE":
            reasons.append("Bullish sweep with high velocity")
        elif state.velocity == "HIGH_DOWN" and state.wall_score > 0.7:
            reasons.append("Bearish wall with high velocity")
            
        return " + ".join(reasons)
        
    def get_detonation_metrics(self) -> Dict:
        """Get metrics about recent detonations"""
        if not self.detonation_history:
            return {}
            
        recent_states = self.detonation_history[-100:]  # Last 100 detonations
        
        return {
            'total_detonations': len(recent_states),
            'avg_confidence': np.mean([s.confidence for s in recent_states]),
            'avg_smart_money_score': np.mean([s.smart_money_score for s in recent_states]),
            'avg_spoof_score': np.mean([s.spoof_score for s in recent_states]),
            'velocity_distribution': {
                'HIGH_UP': sum(1 for s in recent_states if s.velocity == "HIGH_UP"),
                'HIGH_DOWN': sum(1 for s in recent_states if s.velocity == "HIGH_DOWN"),
                'NORMAL': sum(1 for s in recent_states if s.velocity == "NORMAL")
            },
            'liquidity_distribution': {
                'SWEEP': sum(1 for s in recent_states if "SWEEP" in s.liquidity_resonance),
                'CONSOLIDATION': sum(1 for s in recent_states if "CONSOLIDATION" in s.liquidity_resonance),
                'NORMAL': sum(1 for s in recent_states if "NORMAL" in s.liquidity_resonance)
            }
        }
    
    def update_sequence(self, new_phase: int) -> DetonationState:
        """
        Update detonation sequence state.
        """
        if not self.current_state.active:
            return self.current_state
        
        self.current_state.phase = new_phase
        
        # Update confidence based on phase
        if self.current_state.phase < 30:
            self.current_state.confidence *= 0.95  # Initial decay
        elif self.current_state.phase < 70:
            self.current_state.confidence *= 1.05  # Build-up phase
        else:
            self.current_state.confidence *= 0.98  # Final decay
        
        # Reset if sequence complete
        if self.current_state.phase >= 100:
            self.current_state.active = False
            self.current_state.phase = 0
        
        self.current_state.timestamp = datetime.now()
        return self.current_state
    
    def get_sequence_data(self) -> Dict:
        """
        Get complete sequence data for visualization.
        """
        return {
            'state': {
                'active': self.current_state.active,
                'phase': self.current_state.phase,
                'confidence': self.current_state.confidence,
                'pattern_hash': self.current_state.pattern_hash,
                'sequence_id': self.current_state.sequence_id
            },
            'metrics': {
                'entropy': self.pattern_metrics.entropy,
                'stability': self.pattern_metrics.stability,
                'momentum': self.pattern_metrics.momentum,
                'volatility': self.pattern_metrics.volatility,
                'correlation': self.pattern_metrics.correlation
            },
            'history': self.sequence_history[-5:],  # Last 5 sequences
            'timestamp': self.current_state.timestamp
        } 

    def evaluate_velocity_matrix(self, state: DetonationState) -> str:
        if state.velocity == "HIGH_UP" and state.spoof_score < 0.3:
            return "BULL_SURGE"
        elif state.velocity == "HIGH_DOWN" and state.wall_score > 0.7:
            return "BEAR_WALL"
        elif state.spoof_score > 0.8:
            return "SPOOF_ZONE"
        return "NEUTRAL"

    def export_detonation_log(self, filepath: str = "detonation_log.json") -> None:
        with open(filepath, "w") as f:
            json.dump([s.__dict__ for s in self.detonation_history], f, indent=2, default=str)

    def get_recent_summary(self) -> Dict[str, Any]:
        return {
            'last_pattern': self.current_state.pattern_id,
            'detonated': self.current_state.active,
            'routing_tag': self.current_state.routing_tag,
            'velocity_class': self.current_state.phase_velocity_classification,
            'phase': self.current_state.phase,
            'timestamp': self.current_state.timestamp.isoformat()
        }

    def generate_batch_hash(self, sequence_id: str, timestamp: datetime) -> str:
        key = f"{sequence_id}-{timestamp.timestamp()}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get_heatmap_matrix(self) -> np.ndarray:
        matrix = np.zeros((10, 10))
        for s in self.detonation_history:
            x = int(s.confidence * 10) % 10
            y = int(s.smart_money_score * 10) % 10
            matrix[x][y] += 1
        return matrix

class TestEntropyAnalyzer(unittest.TestCase):
    def test_empty_entropy_values(self):
        analyzer = EntropyAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.analyze_entropy_distribution([])

    def test_uniform_entropy_values(self):
        entropy_values = [42] * 144
        analyzer = EntropyAnalyzer()
        result = analyzer.analyze_entropy_distribution(entropy_values)
        self.assertEqual(result['unique_entropies'], 1)
        self.assertAlmostEqual(result['mean'], 42.0, places=5)

    def test_non_uniform_entropy_values(self):
        entropy_values = [i for i in range(144)]
        analyzer = EntropyAnalyzer()
        result = analyzer.analyze_entropy_distribution(entropy_values)
        self.assertEqual(result['unique_entropies'], 144)
        self.assertAlmostEqual(result['mean'], 71.999999, places=5)

    def test_large_dataset(self):
        entropy_values = np.random.randint(0, 144, size=100000)
        analyzer = EntropyAnalyzer()
        result = analyzer.analyze_entropy_distribution(entropy_values)
        self.assertEqual(result['unique_entropies'], len(np.unique(entropy_values)))
        self.assertAlmostEqual(result['mean'], np.mean(entropy_values), places=5)

class TestDetonationSequencer(unittest.TestCase):
    def test_initiate_detonation(self):
        sequencer = DetonationSequencer()
        pattern_data = {'pattern': 'A'}
        state = sequencer.initiate_detonation(pattern_data)
        self.assertTrue(state['detonation_activated'])
        self.assertEqual(state['confidence'], 0.0)
        self.assertEqual(state['smart_money_score'], 0.0)
        self.assertEqual(state['spoof_score'], 0.0)
        self.assertEqual(state['wall_score'], 0.0)
        self.assertEqual(state['velocity'], "NORMAL")
        self.assertEqual(state['liquidity_resonance'], "NORMAL")

    def test_update_sequence(self):
        sequencer = DetonationSequencer()
        pattern_data = {'pattern': 'A'}
        state = sequencer.initiate_detonation(pattern_data)
        for phase in range(1, 101):
            state = sequencer.update_sequence(phase)
            self.assertEqual(state.phase, phase)

    def test_sequence_completion(self):
        sequencer = DetonationSequencer()
        pattern_data = {'pattern': 'A'}
        state = sequencer.initiate_detonation(pattern_data)
        for _ in range(90):
            state = sequencer.update_sequence(state.phase + 1)
        self.assertFalse(state.active)

    def test_pattern_analysis(self):
        sequencer = DetonationSequencer()
        pattern_data = {'pattern': 'A'}
        metrics = sequencer.analyze_pattern(pattern_data)
        self.assertGreater(metrics.entropy, 0.0)
        self.assertGreater(metrics.stability, 0.0)
        self.assertGreater(metrics.momentum, 0.0)
        self.assertGreater(metrics.volatility, 0.0)
        self.assertGreater(metrics.correlation, 0.0)

if __name__ == '__main__':
    unittest.main() 