"""
Test Suite for Zygot Shell Core
==============================

Comprehensive test coverage for ZygotShell implementation including:
- Unit tests for core functionality
- Integration tests for control hooks
- Symbolic encoding validation
- Edge case handling
- Async operation support
- Precision regression tests
"""

import unittest
import numpy as np
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from core.zygot_shell import (
    ZygotShell, 
    ZygotShellState, 
    ZalgoField,
    DRIFT_SHELL_BOUNDARY,
    ZYGOT_SUBHARMONIC_A,
    ZYGOT_SUBHARMONIC_B
)

class DummyCursor:
    """Mock cursor for testing shell generation"""
    def __init__(self, timestamp=None, triplet=None):
        self.triplet = triplet or (1.0, 1.0, 1.0)
        self.delta_idx = 0
        self.braid_angle = DRIFT_SHELL_BOUNDARY
        self.timestamp = timestamp if timestamp else datetime.utcnow().timestamp()

class TestZygotShell(unittest.TestCase):
    """Core test suite for ZygotShell functionality"""

    def setUp(self):
        """Initialize test fixtures"""
        self.zygot = ZygotShell()
        self.cursor = DummyCursor()
        self.test_vector = np.array([1.0, 0.0, 0.0])
        self.test_entropy = 0.5
        self.test_phase = np.pi

    def test_shell_generation(self):
        """Test basic shell generation and state validation"""
        shell = self.zygot.process_shell_state(
            vector=self.test_vector,
            phase_angle=self.test_phase,
            entropy=self.test_entropy
        )
        
        self.assertIsInstance(shell, ZygotShellState)
        self.assertEqual(shell.vector.shape[0], 3)
        self.assertGreaterEqual(shell.shell_radius, 0.0)
        self.assertIsNotNone(shell.zalgo_field)
        self.assertIn(shell.shell_type, ["stable", "resonant", "collapse"])

    def test_alignment_score_bounds(self):
        """Test alignment score calculation and bounds"""
        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        zalgo = ZalgoField(
            field_vector=vec2,
            entropy_level=1.0,
            collapse_risk=0.0,
            ghost_signals=[0.0, 0.0, 0.0],
            timestamp=time.time()
        )
        score = self.zygot._compute_alignment_score(vec1, zalgo)
        self.assertAlmostEqual(score, 0.0)

        # Test parallel vectors
        zalgo.field_vector = vec1
        score = self.zygot._compute_alignment_score(vec1, zalgo)
        self.assertAlmostEqual(score, 1.0)

        # Test zero vector
        zero_vec = np.zeros(3)
        score = self.zygot._compute_alignment_score(zero_vec, zalgo)
        self.assertEqual(score, 0.0)

    def test_drift_resonance_range(self):
        """Test drift resonance calculation and range validation"""
        # Test at π
        resonance = self.zygot.compute_drift_resonance(np.pi, 1.0)
        self.assertAlmostEqual(resonance, 1.0)

        # Test at 0
        resonance = self.zygot.compute_drift_resonance(0.0, 1.0)
        self.assertLess(resonance, 1.0)

        # Test with high entropy
        resonance = self.zygot.compute_drift_resonance(np.pi, 2.0)
        self.assertLess(resonance, 1.0)

    def test_stability_index(self):
        """Test stability index calculation"""
        # Test perfect stability
        stability = self.zygot.compute_stability_index(1.0, 1.0)
        self.assertAlmostEqual(stability, 1.0)

        # Test with custom config
        config = {'alpha_v': 10.0, 'alpha_s': 10.0, 'alpha_c': 0.5}
        stability = self.zygot.compute_stability_index(1.0, 1.0, config)
        self.assertAlmostEqual(stability, 1.0)

        # Test with unstable signals
        stability = self.zygot.compute_stability_index(2.0, 2.0)
        self.assertLess(stability, 1.0)

    def test_zalgo_field_generation(self):
        """Test Zalgo field generation and properties"""
        field = self.zygot.compute_zalgo_field(self.test_vector, self.test_entropy)
        
        self.assertIsInstance(field, ZalgoField)
        self.assertEqual(len(field.ghost_signals), 3)
        self.assertTrue(0.0 <= field.collapse_risk <= 1.0)
        self.assertIsNotNone(field.field_vector)

    def test_ghost_signals(self):
        """Test ghost signal generation properties"""
        signals = self.zygot._generate_ghost_signals(self.test_vector, self.test_entropy)
        
        self.assertEqual(len(signals), 3)
        self.assertTrue(all(-1.0 <= s <= 1.0 for s in signals))
        self.assertTrue(all(isinstance(s, float) for s in signals))

    def test_shell_state_classification(self):
        """Test shell state classification logic"""
        # Test stable state
        shell_type = self.zygot.classify_shell_state(0.8, 0.8)
        self.assertEqual(shell_type, "stable")

        # Test resonant state
        shell_type = self.zygot.classify_shell_state(0.4, 0.4)
        self.assertEqual(shell_type, "resonant")

        # Test collapse state
        shell_type = self.zygot.classify_shell_state(0.2, 0.2)
        self.assertEqual(shell_type, "collapse")

    def test_state_history(self):
        """Test state history tracking"""
        # Generate multiple states
        states = []
        for i in range(5):
            state = self.zygot.process_shell_state(
                vector=self.test_vector,
                phase_angle=self.test_phase + i,
                entropy=self.test_entropy
            )
            states.append(state)

        # Test history retrieval
        recent = self.zygot.get_recent_states(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1], states[-1])

    def test_export_shell_map(self):
        """Test shell map export functionality"""
        # Generate test state
        state = self.zygot.process_shell_state(
            vector=self.test_vector,
            phase_angle=self.test_phase,
            entropy=self.test_entropy
        )

        # Export to temporary file
        temp_path = Path("temp_shell_map.json")
        self.zygot.export_shell_map(str(temp_path))

        # Verify file contents
        self.assertTrue(temp_path.exists())
        with open(temp_path) as f:
            data = json.load(f)
            self.assertIn("timestamp", data)
            self.assertIn("states", data)
            self.assertTrue(len(data["states"]) > 0)

        # Cleanup
        temp_path.unlink()

    def test_plot_shell_evolution(self):
        """Test shell evolution plotting"""
        # Generate test data
        timestamps = [time.time() + i for i in range(5)]
        alignment_scores = [0.8] * 5
        drift_resonances = [0.7] * 5

        # Test plotting (should not raise)
        self.zygot.plot_shell_evolution(
            timestamps,
            alignment_scores,
            drift_resonances,
            "Test Evolution"
        )

class TestZygotControlHooks(unittest.TestCase):
    """Test suite for Zygot control hooks and integration points"""

    def setUp(self):
        self.zygot = ZygotShell()
        self.cursor = DummyCursor()
        self.shell = self.zygot.process_shell_state(
            vector=np.array([1.0, 0.0, 0.0]),
            phase_angle=np.pi,
            entropy=0.5
        )

    def test_ghost_hash_injection(self):
        """Test ghost hash injection"""
        ghost_id = "ghost_420"
        modified = ZygotControlHooks.inject_ghost_hash(self.shell, ghost_id)
        self.assertEqual(modified.ghost_hash, ghost_id)

    def test_matrix_origin_registration(self):
        """Test matrix origin registration"""
        strategy_id = "test_strategy"
        result = ZygotControlHooks.register_matrix_origin(self.shell, strategy_id)
        
        self.assertEqual(result["strategy"], strategy_id)
        self.assertEqual(result["timestamp"], self.shell.timestamp)
        self.assertEqual(result["alignment"], self.shell.alignment_score)
        self.assertEqual(result["radius"], self.shell.shell_radius)

    def test_ncco_forward_payload(self):
        """Test NCCO data forwarding"""
        payload = ZygotControlHooks.forward_to_ncco(self.shell)
        
        self.assertIn("vector", payload)
        self.assertIn("drift", payload)
        self.assertIn("score", payload)
        self.assertEqual(payload["drift"], self.shell.drift_resonance)
        self.assertEqual(payload["score"], self.shell.alignment_score)

    def test_corridor_inversion(self):
        """Test vector inversion for ghost corridors"""
        inverted = ZygotControlHooks.invert_for_corridor(self.shell)
        
        self.assertIsNotNone(inverted)
        self.assertEqual(inverted.shape, self.shell.vector.shape)
        dot_product = np.dot(self.shell.vector, inverted)
        self.assertLess(dot_product, 0)

    def test_symbolic_encoding(self):
        """Test symbolic encoding generation"""
        # Test encoding generation
        encoded = ZygotControlHooks.symbolic_encode(self.shell)
        self.assertEqual(len(encoded), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in encoded))

        # Test encoding changes with state
        modified_shell = self.shell
        modified_shell.shell_radius += 0.1
        new_encoded = ZygotControlHooks.symbolic_encode(modified_shell)
        self.assertNotEqual(encoded, new_encoded)

class TestAsyncZygotShell(unittest.TestCase):
    """Test suite for async ZygotShell operations"""

    async def async_drift_loop(self, shell_gen, cursor, ticks):
        """Async generator for shell state evolution"""
        for t in range(ticks):
            await asyncio.sleep(0.01)
            yield shell_gen.process_shell_state(
                vector=np.array([1.0, 0.0, 0.0]),
                phase_angle=np.pi + t * 0.1,
                entropy=0.5
            )

    def test_async_shell_generation(self):
        """Test async shell generation and evolution"""
        async def run_test():
            states = []
            async for state in self.async_drift_loop(self.zygot, self.cursor, 5):
                states.append(state)
            return states

        # Run async test
        states = asyncio.run(run_test())
        
        self.assertEqual(len(states), 5)
        self.assertTrue(all(isinstance(s, ZygotShellState) for s in states))
        self.assertTrue(all(s.phase_angle > states[i-1].phase_angle 
                          for i, s in enumerate(states[1:], 1)))

if __name__ == "__main__":
    unittest.main() 