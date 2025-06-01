"""
Test suite for the enhanced fractal core implementation.
"""

import unittest
import numpy as np
import time
from core.enhanced_fractal_core import EnhancedFractalCore, QuantizationProfile, FractalState

class TestEnhancedFractalCore(unittest.TestCase):
    """Test cases for the enhanced fractal core implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profile = QuantizationProfile(
            decay_power=1.5,
            terms=12,
            dimension=8,
            epsilon_q=0.003,
            precision=1e-3
        )
        self.core = EnhancedFractalCore(self.profile)
        
    def test_forever_fractal(self):
        """Test the forever fractal signal generation."""
        t = time.time()
        signal = self.core.forever_fractal(t)
        
        # Check signal properties
        self.assertIsInstance(signal, float)
        self.assertTrue(-1 <= signal <= 1)  # Bounded by sine waves
        
        # Test phase shift
        shifted = self.core.forever_fractal(t, phase_shift=np.pi/2)
        self.assertNotEqual(signal, shifted)  # Different phase should give different signal
        
    def test_generate_fractal_vector(self):
        """Test fractal vector generation."""
        t = time.time()
        vector = self.core.generate_fractal_vector(t)
        
        # Check vector properties
        self.assertEqual(len(vector), self.profile.dimension)
        self.assertTrue(all(-1 <= v <= 1 for v in vector))  # All components bounded
        
    def test_quantize_vector(self):
        """Test vector quantization."""
        vector = [0.123456, -0.789012, 0.345678]
        quantized = self.core.quantize_vector(vector)
        
        # Check quantization properties
        self.assertEqual(len(vector), len(quantized))
        for v, q in zip(vector, quantized):
            self.assertTrue(abs(v - q) <= self.profile.precision)
            
    def test_validate_spacing(self):
        """Test vector spacing validation."""
        # Valid spacing
        vectors = [
            [0.0, 0.0],
            [0.005, 0.005]  # Distance > epsilon_q
        ]
        self.assertTrue(self.core.validate_spacing(vectors))
        
        # Invalid spacing
        vectors = [
            [0.0, 0.0],
            [0.001, 0.001]  # Distance < epsilon_q
        ]
        self.assertFalse(self.core.validate_spacing(vectors))
        
    def test_fft_non_aliasing_check(self):
        """Test FFT non-aliasing check."""
        # Non-aliasing case
        vector = [math.sin(2 * math.pi * i / 8) for i in range(8)]
        self.assertTrue(self.core.fft_non_aliasing_check([vector]))
        
        # Aliasing case (high frequency components)
        vector = [math.sin(2 * math.pi * i * 4 / 8) for i in range(8)]
        self.assertFalse(self.core.fft_non_aliasing_check([vector]))
        
    def test_spectral_entropy(self):
        """Test spectral entropy calculation."""
        # Pure sine wave (low entropy)
        vector = [math.sin(2 * math.pi * i / 8) for i in range(8)]
        entropy = self.core.spectral_entropy(vector)
        self.assertTrue(0 <= entropy <= 1)
        
        # Random noise (high entropy)
        vector = np.random.randn(8)
        entropy = self.core.spectral_entropy(vector)
        self.assertTrue(0 <= entropy <= 1)
        
    def test_validate_entropy_bandwidth(self):
        """Test entropy bandwidth validation."""
        # Valid entropy
        vectors = [
            [math.sin(2 * math.pi * i / 8) for i in range(8)],  # Low entropy
            np.random.randn(8)  # High entropy
        ]
        self.assertTrue(self.core.validate_entropy_bandwidth(vectors))
        
        # Invalid entropy (all low)
        vectors = [
            [math.sin(2 * math.pi * i / 8) for i in range(8)],
            [math.sin(2 * math.pi * i / 8 + 0.1) for i in range(8)]
        ]
        self.assertFalse(self.core.validate_entropy_bandwidth(vectors))
        
    def test_compute_entropy_slope(self):
        """Test entropy slope computation."""
        # Create test states
        states = [
            FractalState(
                vector=[0.0] * 8,
                timestamp=time.time(),
                phase=0.0,
                entropy=0.5,
                recursive_depth=0
            ),
            FractalState(
                vector=[0.0] * 8,
                timestamp=time.time() + 1.0,
                phase=0.0,
                entropy=0.6,
                recursive_depth=0
            )
        ]
        
        slope = self.core.compute_entropy_slope(states)
        self.assertIsInstance(slope, float)
        self.assertTrue(slope > 0)  # Entropy increased
        
    def test_profit_tree(self):
        """Test profit allocation tree."""
        pattern_hash = "test_pattern"
        
        # Update profit
        self.core.update_profit_tree(pattern_hash, 1.0)
        self.assertEqual(self.core.get_cumulative_profit(pattern_hash), 1.0)
        
        # Update again
        self.core.update_profit_tree(pattern_hash, 2.0)
        self.assertEqual(self.core.get_cumulative_profit(pattern_hash), 3.0)
        
    def test_compute_dormant_score(self):
        """Test dormant score computation."""
        # Add test history
        self.core.entropy_slope_history = [-0.1] * 10
        self.core.harmonic_power_history = [0.05] * 10
        self.core.cpu_render_history = [150.0] * 10
        
        score = self.core.compute_dormant_score()
        self.assertTrue(0 <= score <= 1)
        self.assertTrue(score > 0.7)  # Should trigger dormant state
        
    def test_process_recursive_state(self):
        """Test recursive state processing."""
        vector = [0.0] * 8
        result = self.core.process_recursive_state(vector)
        
        # Check result structure
        self.assertIn("depth", result)
        self.assertIn("dormant_state", result)
        self.assertIn("entropy", result)
        self.assertIn("entropy_slope", result)
        self.assertIn("cyclic_detected", result)
        self.assertIn("recursive_result", result)
        self.assertIn("profit", result)
        
        # Check recursive depth
        self.assertEqual(result["depth"], 0)
        self.assertIsNotNone(result["recursive_result"])
        
    def test_detect_cyclic_pattern(self):
        """Test cyclic pattern detection."""
        # Test pattern reversal
        vector = [998998 / 998001] + [0.0] * 7
        self.assertTrue(self.core.detect_cyclic_pattern(vector))
        self.assertIsNotNone(self.core.pattern_reversal_key)
        
        # Test normal pattern
        vector = [0.5] * 8
        self.assertFalse(self.core.detect_cyclic_pattern(vector))
        
    def test_state_persistence(self):
        """Test state saving and loading."""
        # Add some state
        vector = [0.0] * 8
        self.core.process_recursive_state(vector)
        
        # Save state
        self.core.save_state("test_state.json")
        
        # Load state
        loaded_core = EnhancedFractalCore.load_state("test_state.json")
        
        # Compare states
        self.assertEqual(len(self.core.state_history), len(loaded_core.state_history))
        self.assertEqual(self.core.dormant_state, loaded_core.dormant_state)
        self.assertEqual(self.core.last_profit_snapshot, loaded_core.last_profit_snapshot)
        
if __name__ == '__main__':
    unittest.main() 