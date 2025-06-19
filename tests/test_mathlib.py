"""
Test suite for CoreMathLib functionality.
Tests all mathematical utilities and operations.
"""

import unittest
import numpy as np
from core.mathlib import CoreMathLib, GradedProfitVector


class TestCoreMathLib(unittest.TestCase):
    """Test cases for CoreMathLib functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.math = CoreMathLib()

    def test_cosine_similarity(self):
        """Test cosine similarity calculations."""
        # Test identical vectors
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        self.assertAlmostEqual(self.math.cosine_similarity(a, b), 1.0)

        # Test orthogonal vectors
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        self.assertAlmostEqual(self.math.cosine_similarity(a, b), 0.0)

        # Test opposite vectors
        a = np.array([1, 0, 0])
        b = np.array([-1, 0, 0])
        self.assertAlmostEqual(self.math.cosine_similarity(a, b), -1.0)

        # Test zero vectors
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        self.assertEqual(self.math.cosine_similarity(a, b), 0.0)

    def test_euclidean_distance(self):
        """Test Euclidean distance calculations."""
        # Test identical points
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertEqual(self.math.euclidean_distance(a, b), 0.0)

        # Test simple distance
        a = np.array([0, 0])
        b = np.array([3, 4])
        self.assertEqual(self.math.euclidean_distance(a, b), 5.0)

        # Test higher dimensions
        a = np.array([1, 1, 1, 1])
        b = np.array([2, 2, 2, 2])
        self.assertAlmostEqual(self.math.euclidean_distance(a, b), 2.0)

    def test_normalize_vector(self):
        """Test vector normalization."""
        # Test unit vector
        v = np.array([1, 0, 0])
        normalized = self.math.normalize_vector(v)
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0)

        # Test arbitrary vector
        v = np.array([3, 4])
        normalized = self.math.normalize_vector(v)
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0)

        # Test zero vector
        v = np.array([0, 0, 0])
        normalized = self.math.normalize_vector(v)
        np.testing.assert_array_equal(normalized, v)

    def test_grading_vector(self):
        """Test GradedProfitVector creation and operations."""
        # Test creation from dictionary
        trade = {
            'profit': 1.0,
            'volume_allocated': 100.0,
            'time_held': 60.0,
            'signal_strength': 0.8,
            'smart_money_score': 0.9
        }
        vector = self.math.grading_vector(trade)
        self.assertIsInstance(vector, GradedProfitVector)
        self.assertEqual(vector.profit, 1.0)
        self.assertEqual(vector.volume_allocated, 100.0)

        # Test missing values
        trade = {'profit': 1.0}
        vector = self.math.grading_vector(trade)
        self.assertEqual(vector.signal_strength, 0.0)
        self.assertEqual(vector.smart_money_score, 0.0)

    def test_average_grade_vector(self):
        """Test average grade vector calculations."""
        vectors = [
            GradedProfitVector(1.0, 100.0, 60.0, 0.8, 0.9),
            GradedProfitVector(2.0, 200.0, 120.0, 0.7, 0.8),
            GradedProfitVector(3.0, 300.0, 180.0, 0.6, 0.7)
        ]

        avg = self.math.average_grade_vector(vectors)
        self.assertIsInstance(avg, GradedProfitVector)
        self.assertAlmostEqual(avg.profit, 2.0)
        self.assertAlmostEqual(avg.volume_allocated, 200.0)

        # Test empty list
        avg = self.math.average_grade_vector([])
        self.assertEqual(avg.profit, 0.0)
        self.assertEqual(avg.volume_allocated, 0.0)

    def test_shell_entropy(self):
        """Test shell entropy calculations."""
        # Test uniform distribution
        dist = [0.25, 0.25, 0.25, 0.25]
        entropy = self.math.shell_entropy(dist)
        self.assertAlmostEqual(entropy, np.log(4))

        # Test deterministic distribution
        dist = [1.0, 0.0, 0.0]
        entropy = self.math.shell_entropy(dist)
        self.assertEqual(entropy, 0.0)

        # Test with zeros
        dist = [0.5, 0.0, 0.5]
        entropy = self.math.shell_entropy(dist)
        self.assertAlmostEqual(entropy, np.log(2))

    def test_phase_angle(self):
        """Test phase angle calculations."""
        # Test 0 degrees
        vector = np.array([1, 0])
        angle = self.math.phase_angle(vector)
        self.assertAlmostEqual(angle, 0.0)

        # Test 90 degrees
        vector = np.array([0, 1])
        angle = self.math.phase_angle(vector)
        self.assertAlmostEqual(angle, np.pi / 2)

        # Test 180 degrees
        vector = np.array([-1, 0])
        angle = self.math.phase_angle(vector)
        self.assertAlmostEqual(angle, np.pi)

    def test_volatility(self):
        """Test volatility calculations."""
        # Test constant prices
        prices = [100.0, 100.0, 100.0]
        vol = self.math.volatility(prices)
        self.assertEqual(vol, 0.0)

        # Test varying prices
        prices = [100.0, 110.0, 90.0, 105.0]
        vol = self.math.volatility(prices)
        self.assertGreater(vol, 0.0)

    def test_drift(self):
        """Test drift calculations."""
        # Test upward drift
        prices = [100.0, 110.0, 120.0, 130.0]
        drift = self.math.drift(prices)
        self.assertEqual(drift, 10.0)

        # Test downward drift
        prices = [130.0, 120.0, 110.0, 100.0]
        drift = self.math.drift(prices)
        self.assertEqual(drift, -10.0)

        # Test insufficient data
        prices = [100.0]
        drift = self.math.drift(prices)
        self.assertEqual(drift, 0.0)

    def test_latent_similarity(self):
        """Test latent vector similarity checks."""
        # Test similar vectors
        z1 = np.array([1.0, 0.0])
        z2 = np.array([1.1, 0.1])
        self.assertTrue(self.math.latent_similarity(z1, z2, threshold=0.2))

        # Test dissimilar vectors
        z1 = np.array([1.0, 0.0])
        z2 = np.array([0.0, 1.0])
        self.assertFalse(self.math.latent_similarity(z1, z2, threshold=0.2))

    def test_dynamic_holdout_ratio(self):
        """Test dynamic holdout ratio calculations."""
        ratios = []
        for t in range(100):
            ratio = self.math.dynamic_holdout_ratio(t, min_r=0.1, max_r=0.9)
            ratios.append(ratio)
            self.assertGreaterEqual(ratio, 0.1)
            self.assertLessEqual(ratio, 0.9)

        # Test that ratios vary
        self.assertGreater(np.std(ratios), 0)

    def test_compute_score(self):
        """Test score computation."""
        # Test perfect prediction
        score = self.math.compute_score(1.0, 1.0)
        self.assertEqual(score, 1.0)

        # Test poor prediction
        score = self.math.compute_score(0.0, 1.0)
        self.assertEqual(score, 0.0)

        # Test scaled prediction
        score = self.math.compute_score(0.5, 1.0, scale=2.0)
        self.assertEqual(score, 0.75)

    def test_score_strategy_performance(self):
        """Test strategy performance scoring."""
        # Test perfect predictions
        pred = [1.0, 2.0, 3.0]
        target = [1.0, 2.0, 3.0]
        score = self.math.score_strategy_performance(pred, target)
        self.assertEqual(score, 1.0)

        # Test poor predictions
        pred = [0.0, 0.0, 0.0]
        target = [1.0, 2.0, 3.0]
        score = self.math.score_strategy_performance(pred, target)
        self.assertLess(score, 1.0)

        # Test mismatched lengths
        pred = [1.0, 2.0]
        target = [1.0, 2.0, 3.0]
        score = self.math.score_strategy_performance(pred, target)
        self.assertEqual(score, 0.0)

    def test_spectral_entropy(self):
        """Test spectral entropy calculations."""
        # Test constant signal
        signal = np.ones(100)
        entropy = self.math.spectral_entropy(signal)
        self.assertAlmostEqual(entropy, 0.0)

        # Test random signal
        signal = np.random.randn(100)
        entropy = self.math.spectral_entropy(signal)
        self.assertGreater(entropy, 0.0)

    def test_entropy_slope(self):
        """Test entropy slope calculations."""
        # Test constant signal
        signal = np.ones(100)
        slope = self.math.entropy_slope(signal)
        self.assertEqual(slope, 0.0)

        # Test increasing entropy
        signal = np.linspace(0, 1, 100)
        slope = self.math.entropy_slope(signal)
        self.assertGreater(slope, 0.0)

        # Test insufficient data
        signal = np.ones(5)
        slope = self.math.entropy_slope(signal, window_size=10)
        self.assertEqual(slope, 0.0)

    def test_coherence_vector(self):
        """Test coherence vector calculations."""
        # Test identical signals
        signal1 = np.sin(np.linspace(0, 2 * np.pi, 100))
        signal2 = signal1.copy()
        coherence = self.math.coherence_vector(signal1, signal2)
        self.assertTrue(np.allclose(coherence, 1.0))

        # Test uncorrelated signals
        signal1 = np.sin(np.linspace(0, 2 * np.pi, 100))
        signal2 = np.cos(np.linspace(0, 2 * np.pi, 100))
        coherence = self.math.coherence_vector(signal1, signal2)
        self.assertFalse(np.allclose(coherence, 1.0))


if __name__ == '__main__':
    unittest.main()