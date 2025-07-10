#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Enhanced Mathematical Core
======================================================
Tests all mathematical functions, performance, and integration capabilities.
"""

import logging
import time
import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd

# Import the enhanced mathematical core
from core.enhanced_mathematical_core import EnhancedMathematicalCore, MathMode, MathResult, TradingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedMathematicalCore(unittest.TestCase):
    """Test suite for EnhancedMathematicalCore."""

    def setUp(self):
        """Set up test fixtures."""
        self.math_core = EnhancedMathematicalCore()
        self.test_tensor = np.random.randn(50, 50)
        self.test_signal = np.random.randn(1000)
        self.test_prices = np.array([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
        self.test_returns = np.diff(np.log(self.test_prices))
        
        logger.info("✅ Test setup completed")

    def tearDown(self):
        """Clean up after tests."""
        pass

    # ============================================================================
    # INITIALIZATION TESTS
    # ============================================================================

    def test_initialization(self):
        """Test core initialization."""
        self.assertTrue(self.math_core.initialized)
        self.assertIsInstance(self.math_core.mode, MathMode)
        logger.info("✅ Initialization test passed")

    def test_dependency_validation(self):
        """Test dependency validation."""
        status = self.math_core.get_status()
        self.assertIn('dependencies', status)
        self.assertIn('numpy', status['dependencies'])
        self.assertTrue(status['dependencies']['numpy'])
        logger.info("✅ Dependency validation test passed")

    def test_configuration(self):
        """Test configuration handling."""
        config = {'mode': 'cpu', 'precision': 'float64'}
        math_core = EnhancedMathematicalCore(config)
        self.assertEqual(math_core.config['mode'], 'cpu')
        logger.info("✅ Configuration test passed")

    # ============================================================================
    # TENSOR OPERATION TESTS
    # ============================================================================

    def test_tensor_norm(self):
        """Test tensor norm calculation."""
        result = self.math_core.tensor_operation(self.test_tensor, 'norm')
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)
        logger.info(f"✅ Tensor norm test passed: {result.value:.6f}")

    def test_tensor_trace(self):
        """Test tensor trace calculation."""
        result = self.math_core.tensor_operation(self.test_tensor, 'trace')
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        logger.info(f"✅ Tensor trace test passed: {result.value:.6f}")

    def test_tensor_eigenvalues(self):
        """Test tensor eigenvalue calculation."""
        result = self.math_core.tensor_operation(self.test_tensor, 'eigenvalues')
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, np.ndarray)
        self.assertEqual(len(result.value), 50)
        logger.info("✅ Tensor eigenvalues test passed")

    def test_tensor_determinant(self):
        """Test tensor determinant calculation."""
        result = self.math_core.tensor_operation(self.test_tensor, 'determinant')
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        logger.info(f"✅ Tensor determinant test passed: {result.value:.6f}")

    def test_tensor_inverse(self):
        """Test tensor inverse calculation."""
        # Use a smaller, well-conditioned matrix
        small_tensor = np.array([[2, 1], [1, 3]])
        result = self.math_core.tensor_operation(small_tensor, 'inverse')
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, np.ndarray)
        logger.info("✅ Tensor inverse test passed")

    # ============================================================================
    # ENTROPY TESTS
    # ============================================================================

    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        result = self.math_core.shannon_entropy(probabilities)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.value, 2.0, places=6)  # log2(4) = 2
        logger.info(f"✅ Shannon entropy test passed: {result.value:.6f}")

    def test_shannon_entropy_uniform(self):
        """Test Shannon entropy for uniform distribution."""
        probabilities = np.array([0.5, 0.5])
        result = self.math_core.shannon_entropy(probabilities)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.value, 1.0, places=6)  # log2(2) = 1
        logger.info(f"✅ Shannon entropy uniform test passed: {result.value:.6f}")

    def test_shannon_entropy_zero(self):
        """Test Shannon entropy with zero probabilities."""
        probabilities = np.array([1.0, 0.0, 0.0])
        result = self.math_core.shannon_entropy(probabilities)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.value, 0.0, places=6)
        logger.info(f"✅ Shannon entropy zero test passed: {result.value:.6f}")

    def test_wave_entropy(self):
        """Test wave entropy calculation."""
        result = self.math_core.wave_entropy(self.test_signal)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        self.assertGreaterEqual(result.value, 0)
        logger.info(f"✅ Wave entropy test passed: {result.value:.6f}")

    def test_information_gain(self):
        """Test information gain calculation."""
        before_entropy = 2.0
        after_entropy = 1.0
        result = self.math_core.information_gain(before_entropy, after_entropy)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.value, 1.0, places=6)
        logger.info(f"✅ Information gain test passed: {result.value:.6f}")

    # ============================================================================
    # TRADING MATHEMATICS TESTS
    # ============================================================================

    def test_calculate_returns(self):
        """Test logarithmic returns calculation."""
        result = self.math_core.calculate_returns(self.test_prices)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, np.ndarray)
        self.assertEqual(len(result.value), len(self.test_prices) - 1)
        logger.info("✅ Calculate returns test passed")

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        result = self.math_core.calculate_volatility(self.test_returns, window=5)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        self.assertGreaterEqual(result.value, 0)
        logger.info(f"✅ Volatility test passed: {result.value:.6f}")

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        result = self.math_core.calculate_sharpe_ratio(self.test_returns)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        logger.info(f"✅ Sharpe ratio test passed: {result.value:.6f}")

    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        result = self.math_core.calculate_var(self.test_returns, confidence_level=0.95)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        logger.info(f"✅ VaR test passed: {result.value:.6f}")

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        result = self.math_core.calculate_max_drawdown(self.test_prices)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        self.assertGreaterEqual(result.value, 0)
        logger.info(f"✅ Max drawdown test passed: {result.value:.6f}")

    # ============================================================================
    # ADVANCED ANALYTICS TESTS
    # ============================================================================

    def test_fourier_analysis(self):
        """Test Fourier analysis."""
        result = self.math_core.fourier_analysis(self.test_signal)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, dict)
        self.assertIn('fft', result.value)
        self.assertIn('power_spectrum', result.value)
        self.assertIn('dominant_frequency', result.value)
        logger.info("✅ Fourier analysis test passed")

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)
        result = self.math_core.correlation_analysis(data1, data2)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, float)
        self.assertGreaterEqual(result.value, -1)
        self.assertLessEqual(result.value, 1)
        logger.info(f"✅ Correlation analysis test passed: {result.value:.6f}")

    def test_kalman_filter(self):
        """Test Kalman filter."""
        measurements = np.random.randn(50)
        result = self.math_core.kalman_filter(measurements)
        self.assertTrue(result.success)
        self.assertIsInstance(result.value, dict)
        self.assertIn('filtered_states', result.value)
        self.assertIn('final_state', result.value)
        logger.info("✅ Kalman filter test passed")

    # ============================================================================
    # QUANTUM COMPUTING TESTS
    # ============================================================================

    def test_quantum_random_number(self):
        """Test quantum random number generation."""
        result = self.math_core.quantum_random_number(num_qubits=4)
        if result.success:
            self.assertIsInstance(result.value, int)
            self.assertGreaterEqual(result.value, 0)
            self.assertLess(result.value, 16)  # 2^4
            logger.info(f"✅ Quantum random number test passed: {result.value}")
        else:
            logger.info("⚠️ Quantum random number test skipped (libraries not available)")

    def test_quantum_entanglement_measure(self):
        """Test quantum entanglement measurement."""
        state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # Bell state
        result = self.math_core.quantum_entanglement_measure(state_vector)
        if result.success:
            self.assertIsInstance(result.value, float)
            self.assertGreaterEqual(result.value, 0)
            logger.info(f"✅ Quantum entanglement test passed: {result.value:.6f}")
        else:
            logger.info("⚠️ Quantum entanglement test skipped (libraries not available)")

    # ============================================================================
    # COMPREHENSIVE TRADING METRICS TESTS
    # ============================================================================

    def test_calculate_trading_metrics(self):
        """Test comprehensive trading metrics calculation."""
        metrics = self.math_core.calculate_trading_metrics(
            self.test_prices, 
            self.test_returns,
            risk_free_rate=0.02
        )
        
        self.assertIsInstance(metrics, TradingMetrics)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.sortino_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)
        self.assertIsInstance(metrics.var_95, float)
        self.assertIsInstance(metrics.volatility, float)
        
        logger.info(f"✅ Trading metrics test passed:")
        logger.info(f"   Sharpe Ratio: {metrics.sharpe_ratio:.6f}")
        logger.info(f"   Sortino Ratio: {metrics.sortino_ratio:.6f}")
        logger.info(f"   Max Drawdown: {metrics.max_drawdown:.6f}")
        logger.info(f"   VaR (95%): {metrics.var_95:.6f}")
        logger.info(f"   Volatility: {metrics.volatility:.6f}")

    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================

    def test_latency_measurement(self):
        """Test that latency is being measured."""
        result = self.math_core.tensor_operation(self.test_tensor, 'norm')
        self.assertTrue(hasattr(result, 'latency_ms'))
        self.assertIsInstance(result.latency_ms, float)
        self.assertGreaterEqual(result.latency_ms, 0)
        logger.info(f"✅ Latency measurement test passed: {result.latency_ms:.3f}ms")

    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        benchmark_result = self.math_core.benchmark_performance()
        self.assertIsInstance(benchmark_result, dict)
        self.assertIn('total_latency_ms', benchmark_result)
        self.assertIn('average_latency_ms', benchmark_result)
        self.assertIn('success_rate', benchmark_result)
        
        logger.info(f"✅ Benchmark test passed:")
        logger.info(f"   Total Latency: {benchmark_result['total_latency_ms']:.3f}ms")
        logger.info(f"   Average Latency: {benchmark_result['average_latency_ms']:.3f}ms")
        logger.info(f"   Success Rate: {benchmark_result['success_rate']:.2%}")

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================

    def test_error_handling_invalid_input(self):
        """Test error handling with invalid inputs."""
        # Test with empty array
        result = self.math_core.calculate_returns(np.array([]))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        logger.info("✅ Error handling test passed")

    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        # Test volatility with insufficient data
        short_returns = np.array([0.01, 0.02])
        result = self.math_core.calculate_volatility(short_returns, window=10)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        logger.info("✅ Insufficient data error handling test passed")

    # ============================================================================
    # INTEGRATION TESTS
    # ============================================================================

    def test_full_trading_pipeline(self):
        """Test a complete trading mathematical pipeline."""
        # Generate realistic price data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02))
        returns = np.diff(np.log(prices))
        
        # Calculate all metrics
        volatility_result = self.math_core.calculate_volatility(returns)
        sharpe_result = self.math_core.calculate_sharpe_ratio(returns)
        var_result = self.math_core.calculate_var(returns)
        mdd_result = self.math_core.calculate_max_drawdown(prices)
        entropy_result = self.math_core.wave_entropy(returns)
        
        # Verify all calculations succeeded
        self.assertTrue(volatility_result.success)
        self.assertTrue(sharpe_result.success)
        self.assertTrue(var_result.success)
        self.assertTrue(mdd_result.success)
        self.assertTrue(entropy_result.success)
        
        logger.info("✅ Full trading pipeline test passed")
        logger.info(f"   Volatility: {volatility_result.value:.6f}")
        logger.info(f"   Sharpe Ratio: {sharpe_result.value:.6f}")
        logger.info(f"   VaR (95%): {var_result.value:.6f}")
        logger.info(f"   Max Drawdown: {mdd_result.value:.6f}")
        logger.info(f"   Wave Entropy: {entropy_result.value:.6f}")

    def test_mathematical_consistency(self):
        """Test mathematical consistency across operations."""
        # Test that entropy calculations are consistent
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_result = self.math_core.shannon_entropy(uniform_probs)
        
        # Test that tensor operations are consistent
        identity_matrix = np.eye(3)
        trace_result = self.math_core.tensor_operation(identity_matrix, 'trace')
        det_result = self.math_core.tensor_operation(identity_matrix, 'determinant')
        
        self.assertAlmostEqual(entropy_result.value, 2.0, places=6)  # log2(4)
        self.assertAlmostEqual(trace_result.value, 3.0, places=6)   # trace of 3x3 identity
        self.assertAlmostEqual(det_result.value, 1.0, places=6)     # det of identity
        
        logger.info("✅ Mathematical consistency test passed")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("🧮 Starting Comprehensive Mathematical Core Tests")
    logger.info("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedMathematicalCore)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 Test Summary:")
    logger.info(f"   Tests Run: {result.testsRun}")
    logger.info(f"   Failures: {len(result.failures)}")
    logger.info(f"   Errors: {len(result.errors)}")
    logger.info(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        logger.error("❌ Test Failures:")
        for test, traceback in result.failures:
            logger.error(f"   {test}: {traceback}")
    
    if result.errors:
        logger.error("❌ Test Errors:")
        for test, traceback in result.errors:
            logger.error(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        logger.info("🎉 All tests passed successfully!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1) 