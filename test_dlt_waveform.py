"""
Test suite for DLT Waveform Engine
"""

import unittest
from datetime import datetime, timedelta
from dlt_waveform_engine import DLTWaveformEngine, PhaseDomain

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

if __name__ == '__main__':
    unittest.main() 