"""
Test suite for DLT Waveform Engine
"""

import unittest
from datetime import datetime, timedelta
from dlt_waveform_engine import DLTWaveformEngine, PhaseDomain
from .trailback import TrailbackLogic
from .flow_imbalance import FlowImbalance
from .smart_strategy_evaluator import select_strategy
from .quantum_visualizer import QuantumVisualizer
from .rde_visuals import RDEVisuals

class TestDLTWaveformEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DLTWaveformEngine()
        
    def test_phase_trust_initialization(self):
        """Test initial phase trust state"""
        for phase in PhaseDomain:
            trust = self.engine.phase_trust[phase]
            self.assertEqual(trust.successful_echoes, 0)
            self.assertEqual(trust.entropy_consistency, 0.0)
            
    def test_phase_trust_update(self):
        """Test updating phase trust metrics"""
        phase = PhaseDomain.MID
        
        # Update with success
        self.engine.update_phase_trust(phase, True, 0.9)
        trust = self.engine.phase_trust[phase]
        self.assertEqual(trust.successful_echoes, 1)
        self.assertGreater(trust.entropy_consistency, 0.0)
        
        # Update with failure
        self.engine.update_phase_trust(phase, False, 0.5)
        trust = self.engine.phase_trust[phase]
        self.assertEqual(trust.successful_echoes, 1)  # Should not increment
        
    def test_phase_trust_validation(self):
        """Test phase trust validation thresholds"""
        phase = PhaseDomain.MID
        threshold = self.engine.phase_thresholds[phase]
        
        # Should not be trusted initially
        self.assertFalse(self.engine.is_phase_trusted(phase))
        
        # Add successful echoes
        for _ in range(threshold):
            self.engine.update_phase_trust(phase, True, 0.9)
            
        # Should now be trusted
        self.assertTrue(self.engine.is_phase_trusted(phase))
        
    def test_trigger_addition(self):
        """Test adding trigger points to memory"""
        phase = PhaseDomain.SHORT
        window = timedelta(hours=2)
        
        # Add trigger
        self.engine.add_trigger(phase, window, 0.8, 0.6)
        
        # Verify trigger was added
        self.assertEqual(len(self.engine.triggers), 1)
        trigger = self.engine.triggers[0]
        self.assertEqual(trigger.phase, phase)
        self.assertEqual(trigger.time_window, window)
        self.assertEqual(trigger.diogenic_score, 0.8)
        self.assertEqual(trigger.frequency, 0.6)
        
        # Verify bitmap was updated
        bit_index = self.engine._get_bit_index(phase, window)
        self.assertTrue(self.engine.bitmap[bit_index])
        
    def test_trigger_score_computation(self):
        """Test trigger score computation"""
        phase = PhaseDomain.MID
        window = timedelta(days=5)
        
        # Add multiple triggers
        self.engine.add_trigger(phase, window, 0.8, 0.6)
        self.engine.add_trigger(phase, window, 0.9, 0.7)
        
        # Update phase trust to allow scoring
        for _ in range(self.engine.phase_thresholds[phase]):
            self.engine.update_phase_trust(phase, True, 0.9)
            
        # Compute score
        score = self.engine.compute_trigger_score(datetime.now(), phase)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_trade_trigger_evaluation(self):
        """Test trade trigger evaluation"""
        phase = PhaseDomain.MID
        current_time = datetime.now()
        
        # Setup trusted phase with triggers
        for _ in range(self.engine.phase_thresholds[phase]):
            self.engine.update_phase_trust(phase, True, 0.9)
            
        self.engine.add_trigger(phase, timedelta(days=5), 0.8, 0.6)
        
        # Evaluate trigger
        should_trigger, confidence = self.engine.evaluate_trade_trigger(
            phase, current_time, 0.9, 2000000
        )
        
        self.assertIsInstance(should_trigger, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_short_term_volume_validation(self):
        """Test volume validation for short-term trades"""
        phase = PhaseDomain.SHORT
        current_time = datetime.now()
        
        # Setup trusted phase
        for _ in range(self.engine.phase_thresholds[phase]):
            self.engine.update_phase_trust(phase, True, 0.9)
            
        # Test with low volume
        should_trigger, confidence = self.engine.evaluate_trade_trigger(
            phase, current_time, 0.9, 500000  # Below minimum
        )
        self.assertFalse(should_trigger)
        
        # Test with sufficient volume
        should_trigger, confidence = self.engine.evaluate_trade_trigger(
            phase, current_time, 0.9, 2000000  # Above minimum
        )
        self.assertIsInstance(should_trigger, bool)

class TrailbackLogic:
    def __init__(self, max_depth=12):
        self.history = []  # [(price_delta, volume, timestamp)]

    def record(self, price_delta, volume, ts):
        self.history.append((price_delta, volume, ts))
        if len(self.history) > 1000:
            self.history.pop(0)

    def detect_absorption(self, depth=5):
        """Detect if price delta narrowed despite rising volume"""
        recent = self.history[-depth:]
        prices = [x[0] for x in recent]
        volumes = [x[1] for x in recent]
        if len(prices) < depth:
            return False
        return (
            max(prices) - min(prices) < 0.25 * sum(volumes) / len(volumes)
        )

class FlowImbalance:
    def __init__(self):
        self.window = []

    def update(self, bid_volume, ask_volume):
        imbalance = bid_volume - ask_volume
        self.window.append(imbalance)
        if len(self.window) > 50:
            self.window.pop(0)

    def dominant_side(self):
        avg = sum(self.window[-10:]) / 10
        if avg > 0:
            return 'BID_DOMINANT'
        elif avg < 0:
            return 'ASK_DOMINANT'
        return 'NEUTRAL'

def select_strategy(trailback_result, flow_dominance, vacuum_flag, pattern_state):
    if trailback_result and vacuum_flag and flow_dominance == 'BID_DOMINANT':
        return 'hidden_buy'
    elif flow_dominance == 'ASK_DOMINANT' and pattern_state == 'chaotic':
        return 'panic_short'
    elif pattern_state == 'quantum':
        return 'bounce'
    return 'wait'

class QuantumVisualizer:
    def __init__(self):
        self.strategy_pie_chart = {}
        self.pattern_frequency_plot = {}

    def update_strategy_pie(self, strategy):
        if strategy in self.strategy_pie_chart:
            self.strategy_pie_chart[strategy] += 1
        else:
            self.strategy_pie_chart[strategy] = 1

    def update_pattern_frequency(self, pattern_state):
        if pattern_state in self.pattern_frequency_plot:
            self.pattern_frequency_plot[pattern_state] += 1
        else:
            self.pattern_frequency_plot[pattern_state] = 1

    def display_dashboard(self):
        # Display the dashboard with pie chart and frequency plot
        pass

class RDEVisuals:
    def __init__(self, quantum_visualizer):
        self.quantum_visualizer = quantum_visualizer

    def log_spin(self, spin_data):
        strategy = spin_data.get("smart_strategy")
        if strategy:
            self.quantum_visualizer.update_strategy_pie(strategy)
        pattern_state = spin_data.get("pattern_label")
        if pattern_state:
            self.quantum_visualizer.update_pattern_frequency(pattern_state)

    def display_dashboard(self):
        # Display the dashboard
        pass

class RDECore:
    def __init__(self):
        self.trailback = TrailbackLogic()
        self.flow = FlowImbalance()
        self.quantum_visualizer = QuantumVisualizer()
        self.rde_visuals = RDEVisuals(self.quantum_visualizer)

    def update_signals(self, delta, volume, ts, orderbook, pattern_label):
        self.trailback.record(delta, volume, ts)
        self.flow.update(orderbook['bids'][0][1], orderbook['asks'][0][1])
        flow_side = self.flow.dominant_side()
        strategy = select_strategy(
            trailback_result=self.trailback.detect_absorption(),
            flow_dominance=flow_side,
            vacuum_flag=detect_liquidity_vacuum(orderbook),
            pattern_state=pattern_label
        )
        meta["smart_strategy"] = strategy

    def log_spin(self, spin_data):
        # Log the spin data with smart strategy
        self.rde_visuals.log_spin(spin_data)

class FaultZoneMatrix:
    def __init__(self):
        self.fault_vectors = {}

    def record_fault_vector(self, vector, type="panic", strength=0.87):
        # Generate a unique identifier for the fault vector
        panic_id = f"{type}_{strength}"
        
        # Store the fault vector in the dictionary
        self.fault_vectors[panic_id] = {
            "vector": vector,
            "type": type,
            "strength": strength
        }
        print(f"Fault vector recorded: {panic_id}")

    def get_fault_vector(self, panic_id):
        # Retrieve a fault vector by its ID
        return self.fault_vectors.get(panic_id)

if __name__ == '__main__':
    unittest.main() 