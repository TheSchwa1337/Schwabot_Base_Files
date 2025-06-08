"""
Test Suite for Zygote Propagation Engine
======================================

Comprehensive test coverage for ZygotePropagationEngine including:
- Archetype matching
- Phase awareness
- Anomaly detection
- Memory integration
- Safety thresholds
"""

import unittest
import numpy as np
from datetime import datetime
from typing import Dict, Any
import yaml
from pathlib import Path
import json
import os

from ..zygote_propagation_engine import ZygotePropagationEngine, PropagationTrace
from ..phase_handler import PhaseHandler
from ..gan_filter import GANFilter

class TestZygotePropagationEngine(unittest.TestCase):
    """Test suite for ZygotePropagationEngine"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test configuration and fixtures"""
        # Create test config
        cls.test_config = {
            'core': {
                'memory_window': 10,
                'temporal_coherence_threshold': 0.6,
                'phase_stability_threshold': 0.5
            },
            'gan_config': {
                'input_dim': 3,
                'hidden_dim': 32,
                'latent_dim': 16
            },
            'phase': {
                'velocity_window': 5,
                'stability_window': 10
            }
        }
        
        # Save test config
        cls.config_path = Path("test_propagation_config.yaml")
        with open(cls.config_path, 'w') as f:
            yaml.dump(cls.test_config, f)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if cls.config_path.exists():
            cls.config_path.unlink()
            
    def setUp(self):
        """Initialize test fixtures"""
        self.engine = ZygotePropagationEngine(self.test_config)
        
        # Create test shell data
        self.test_shell = {
            'uuid': 'test_shell_123',
            'vector': np.array([0.8, 0.7, 0.3]),
            'profit_band': 'EXPANSION',
            'phase_state': np.array([0.2, 0.8, 0.4, 0.6]),
            'timestamp': datetime.now().timestamp()
        }
        
    def test_archetype_matching(self):
        """Test archetype matching with phase awareness"""
        matched_archetype, confidence = self.engine.match_archetype_with_phase(
            self.test_shell['vector'],
            self.test_shell['phase_state']
        )
        
        self.assertIsNotNone(matched_archetype)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertIn(matched_archetype, self.engine.archetypes)
        
    def test_temporal_coherence(self):
        """Test temporal coherence calculation"""
        # Add some shell history
        for i in range(5):
            self.engine.shell_history.append({
                'vector': np.array([0.8 + i*0.1, 0.7, 0.3]),
                'timestamp': datetime.now().timestamp()
            })
            
        coherence = self.engine._compute_temporal_coherence(
            self.test_shell['vector'],
            self.engine.shell_history
        )
        
        self.assertTrue(0.0 <= coherence <= 1.0)
        
    def test_quantum_anomaly_detection(self):
        """Test quantum anomaly detection"""
        # Create an anomalous vector
        anomalous_vector = np.array([0.1, 0.9, 0.1])  # Far from any archetype
        
        anomaly = self.engine._detect_quantum_anomaly(
            anomalous_vector,
            np.pi/3  # 60 degrees
        )
        
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly['type'], 'paradox_fractal')
        self.assertTrue(anomaly['spawn_new_archetype'])
        
    def test_plasma_detection(self):
        """Test plasma state detection"""
        plasma_metrics = self.engine._detect_plasma(self.test_shell['vector'])
        
        self.assertIn('energy', plasma_metrics)
        self.assertIn('turbulence', plasma_metrics)
        self.assertTrue(plasma_metrics['energy'] >= 0.0)
        self.assertTrue(plasma_metrics['turbulence'] >= 0.0)
        
    def test_propagation_cycle(self):
        """Test full propagation cycle"""
        result = self.engine.propagate(self.test_shell)
        
        self.assertIn('uuid', result)
        self.assertIn('matched_archetype', result)
        self.assertIn('confidence', result)
        self.assertIn('profit_band', result)
        self.assertIn('phase_metrics', result)
        self.assertIn('plasma_metrics', result)
        
        # Check trace recording
        self.assertTrue(len(self.engine.trace_history) > 0)
        latest_trace = self.engine.trace_history[-1]
        self.assertEqual(latest_trace.uuid, self.test_shell['uuid'])
        
    def test_klein_bottle_state(self):
        """Test Klein bottle state computation"""
        # Create a recursive fold propagation result
        propagation_result = {
            'matched_archetype': 'recursive_fold',
            'confidence': 0.8,
            'phase_alignment': 0.9
        }
        
        klein_state = self.engine._compute_klein_state(propagation_result)
        
        self.assertIsNotNone(klein_state)
        self.assertEqual(klein_state['state'], 'klein_bottle')
        self.assertEqual(klein_state['confidence'], 0.8)
        
    def test_trace_export(self):
        """Test trace export functionality"""
        # Generate some traces
        for i in range(3):
            self.engine.propagate({
                'uuid': f'test_shell_{i}',
                'vector': np.array([0.8, 0.7, 0.3]),
                'profit_band': 'EXPANSION',
                'phase_state': np.array([0.2, 0.8, 0.4, 0.6]),
                'timestamp': datetime.now().timestamp()
            })
            
        # Export traces
        export_path = "test_traces.json"
        self.engine.export_traces(export_path)
        
        # Verify export
        self.assertTrue(os.path.exists(export_path))
        with open(export_path, 'r') as f:
            traces = json.load(f)
            self.assertEqual(len(traces), 3)
            
        # Clean up
        os.remove(export_path)
        
    def test_propagation_summary(self):
        """Test propagation summary generation"""
        # Generate some traces
        for i in range(5):
            self.engine.propagate({
                'uuid': f'test_shell_{i}',
                'vector': np.array([0.8, 0.7, 0.3]),
                'profit_band': 'EXPANSION',
                'phase_state': np.array([0.2, 0.8, 0.4, 0.6]),
                'timestamp': datetime.now().timestamp()
            })
            
        summary = self.engine.get_propagation_summary()
        
        self.assertIn('total_traces', summary)
        self.assertIn('recent_archetypes', summary)
        self.assertIn('average_confidence', summary)
        self.assertIn('anomaly_count', summary)
        
        self.assertEqual(summary['total_traces'], 5)
        self.assertEqual(len(summary['recent_archetypes']), 5)
        self.assertTrue(0.0 <= summary['average_confidence'] <= 1.0)
        
    def test_safety_thresholds(self):
        """Test safety threshold enforcement"""
        # Create a high-risk shell
        risky_shell = {
            'uuid': 'risky_shell',
            'vector': np.array([0.1, 0.9, 0.1]),  # Far from archetypes
            'profit_band': 'EXPANSION',
            'phase_state': np.array([0.9, 0.1, 0.9, 0.1]),  # Unstable phase
            'timestamp': datetime.now().timestamp()
        }
        
        result = self.engine.propagate(risky_shell)
        
        # Should detect anomaly and use fallback strategy
        self.assertIsNotNone(result.get('anomaly'))
        self.assertLess(result['confidence'], 0.5)
        
if __name__ == '__main__':
    unittest.main() 