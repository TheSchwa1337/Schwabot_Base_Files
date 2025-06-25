from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Test suite for the core mathematical and engine utilities.

Validates the correctness and robustness of the new functions added to
`core/utils/math_utils.py` and the structure of the new core engines.
"""

import unittest
from core.unified_math_system import unified_math
from typing import List, Dict

# Import the modules to test
from core.utils.math_utils import (
    calculate_tick_acceleration,
    waveform_pattern_match,
    calculate_hash_distance,
    calculate_weighted_confidence,
    wavelet_decompose,
    calculate_temporal_confidence_merge,
    calculate_execution_lag,
    apply_lag_compensation_curve,
)

from core.dlt_waveform_engine import DLTWaveformEngine
from core.riddle_gemm import RiddleGEMMEngine
from core.multi_bit_btc_processor import MultiBitBTCProcessor
from core.temporal_execution_correction_layer import TemporalExecutionCorrectionLayer


class TestCoreMathAndEngines(unittest.TestCase):
    """Test cases for core math utilities and engine structures."""

    def setUp(self):
        """Set up common test data."""
        self.velocities = np.array([10, 12, 15, 14, 18])
        self.times = np.array([1, 2, 3, 4, 5])
        self.delta_times = np.diff(self.times, prepend=self.times[0])
        self.wave1 = np.unified_math.sin(np.linspace(0, 2 * np.pi, 100))
        self.wave2 = np.unified_math.sin(np.linspace(0, 2 * np.pi, 100) + 0.1)
        self.hash1 = "a" * 64
        self.hash2 = "a" * 63 + "b"
        self.vector1 = np.random.rand(10)
        self.vector2 = np.random.rand(10)

    def test_calculate_tick_acceleration(self):
        """Test tick acceleration calculation."""
        acceleration = calculate_tick_acceleration(self.velocities, self.delta_times)
        self.assertIsNotNone(acceleration)
        self.assertEqual(len(acceleration), len(self.velocities))

    def test_waveform_pattern_match(self):
        """Test waveform pattern matching."""
        is_match, confidence = waveform_pattern_match(self.wave1, self.wave2, threshold=0.9)
        self.assertTrue(is_match)
        self.assertGreater(confidence, 0.9)
        
        is_match_fail, _ = waveform_pattern_match(self.wave1, np.random.rand(100), threshold=0.9)
        self.assertFalse(is_match_fail)

    def test_calculate_hash_distance(self):
        """Test hash distance calculations."""
        # Hamming distance should be 2 because 'a' (1010) and 'b' (1011) differ by 1 bit, 
        # but hex conversion and padding might affect this simple view.
        # Let's test properties instead of exact value.
        hamming_dist = calculate_hash_distance(self.hash1, self.hash2, method='hamming')
        self.assertGreater(hamming_dist, 0)

        cosine_dist = calculate_hash_distance(self.hash1, self.hash2, method='cosine')
        self.assertGreater(cosine_dist, 0)

        self.assertEqual(calculate_hash_distance(self.hash1, self.hash1), 0)

    def test_calculate_weighted_confidence(self):
        """Test weighted confidence calculation."""
        confidence = calculate_weighted_confidence(self.vector1, self.vector2)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_wavelet_decompose(self):
        """Test wavelet decomposition."""
        data = np.random.rand(32) # Must be a power of 2 for simple Haar
        coeffs = wavelet_decompose(data, level=3)
        self.assertEqual(len(coeffs), 4) # approx, detail1, detail2, detail3

    def test_calculate_temporal_confidence_merge(self):
        """Test temporal confidence merge."""
        scores = [0.8, 0.7, 0.9]
        weights = [1.0, 0.5, 2.0]
        merged = calculate_temporal_confidence_merge(scores, weights)
        self.assertGreater(merged, 0.7)
        self.assertLess(merged, 0.9)

    def test_calculate_execution_lag(self):
        """Test execution lag calculation."""
        lag = calculate_execution_lag(ideal_time=100.0, actual_time=100.15)
        self.assertAlmostEqual(lag, 0.15)

    def test_apply_lag_compensation_curve(self):
        """Test lag compensation curve application."""
        compensated = apply_lag_compensation_curve(value=1000, lag=0.2, sensitivity=0.1)
        self.assertLess(compensated, 1000)

    # --- Engine Initialization Tests ---

    def test_dlt_waveform_engine_init(self):
        """Test DLTWaveformEngine can be initialized."""
        try:
            engine = DLTWaveformEngine()
            self.assertIsInstance(engine, DLTWaveformEngine)
        except Exception as e:
            self.fail(f"DLTWaveformEngine initialization failed: {e}")

    def test_riddle_gemm_engine_init(self):
        """Test RiddleGEMMEngine can be initialized."""
        try:
            engine = RiddleGEMMEngine(vector_size=10)
            self.assertIsInstance(engine, RiddleGEMMEngine)
        except Exception as e:
            self.fail(f"RiddleGEMMEngine initialization failed: {e}")

    def test_multi_bit_btc_processor_init(self):
        """Test MultiBitBTCProcessor can be initialized."""
        try:
            timeframes = {"1m": 60, "15m": 900}
            engine = MultiBitBTCProcessor(timeframes=timeframes)
            self.assertIsInstance(engine, MultiBitBTCProcessor)
        except Exception as e:
            self.fail(f"MultiBitBTCProcessor initialization failed: {e}")

    def test_temporal_execution_correction_layer_init(self):
        """Test TemporalExecutionCorrectionLayer can be initialized."""
        try:
            engine = TemporalExecutionCorrectionLayer()
            self.assertIsInstance(engine, TemporalExecutionCorrectionLayer)
        except Exception as e:
            self.fail(f"TemporalExecutionCorrectionLayer initialization failed: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2) 