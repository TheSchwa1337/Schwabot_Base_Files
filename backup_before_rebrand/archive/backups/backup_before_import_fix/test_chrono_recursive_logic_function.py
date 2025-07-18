#!/usr/bin/env python3
"""
Test Suite for Chrono-Recursive Logic Function (CRLF)

Comprehensive tests for the CRLF implementation including:
- Core mathematical functions
- Recursive state propagation
- Trigger state determination
- Performance tracking
- Integration scenarios
"""

import logging
import time
import unittest
from typing import Any, Dict, List

import numpy as np

    ChronoRecursiveLogicFunction,
    CRLFState,
    CRLFResponse,
    CRLFTriggerState,
    create_crlf
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestChronoRecursiveLogicFunction(unittest.TestCase):
    """Test suite for Chrono-Recursive Logic Function."""

    def setUp(self):
        """Set up test fixtures."""
        self.crlf = create_crlf()
        self.test_strategy_vector = np.array([0.6, 0.4, 0.3, 0.7])
        self.test_profit_curve = np.array([100, 105, 103, 108, 110, 107, 112])
        self.test_market_entropy = 0.3

    def test_crlf_initialization(self):
        """Test CRLF initialization."""
        self.assertIsNotNone(self.crlf)
        self.assertIsNotNone(self.crlf.state)
        self.assertEqual(len(self.crlf.state.psi), 4)
        self.assertEqual(self.crlf.state.recursion_depth, 0)
        self.assertEqual(self.crlf.state.total_executions, 0)

    def test_compute_crlf_basic(self):
        """Test basic CRLF computation."""
        response = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        self.assertIsInstance(response, CRLFResponse)
        self.assertIsInstance(response.crlf_output, float)
        self.assertIsInstance(response.trigger_state, CRLFTriggerState)
        self.assertIsInstance(response.psi_n, np.ndarray)
        self.assertIsInstance(response.confidence, float)
        self.assertIsInstance(response.recommendations, dict)

        # Check value ranges
        self.assertGreaterEqual(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
        self.assertGreaterEqual(response.entropy_updated, 0.0)
        self.assertLessEqual(response.entropy_updated, 1.0)

    def test_recursive_state_function(self):
        """Test recursive state function computation."""
        # First computation
        response1 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        # Second computation with different strategy vector
        new_strategy_vector = np.array([0.8, 0.2, 0.5, 0.6])
        response2 = self.crlf.compute_crlf()
            new_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        # Check that recursion depth increased
        self.assertEqual(response1.recursion_depth, 0)
        self.assertEqual(response2.recursion_depth, 1)

        # Check that psi_n is different between computations
        self.assertFalse(np.array_equal(response1.psi_n, response2.psi_n))

    def test_strategy_gradient_computation(self):
        """Test strategy gradient computation."""
        # Test with positive profit trend
        positive_profit_curve = np.array([100, 102, 105, 108, 110, 112, 115])
        response = self.crlf.compute_crlf()
            self.test_strategy_vector,
            positive_profit_curve,
            self.test_market_entropy
        )

        # Test with negative profit trend
        negative_profit_curve = np.array([100, 98, 95, 92, 90, 88, 85])
        response2 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            negative_profit_curve,
            self.test_market_entropy
        )

        # The responses should be different due to different gradients
        self.assertNotEqual(response.crlf_output, response2.crlf_output)

    def test_entropy_update(self):
        """Test entropy update mechanism."""
        initial_entropy = self.crlf.state.entropy

        # First computation
        response1 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        # Second computation with different entropy
        response2 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            0.8  # Higher entropy
        )

        # Entropy should be updated
        self.assertNotEqual(initial_entropy, response1.entropy_updated)
        self.assertNotEqual(response1.entropy_updated, response2.entropy_updated)

        # Higher input entropy should result in higher updated entropy
        self.assertGreater(response2.entropy_updated, response1.entropy_updated)

    def test_trigger_state_determination(self):
        """Test trigger state determination based on CRLF output."""
        # Test different scenarios by manipulating the state
        test_cases = []
            (-0.5, CRLFTriggerState.RECURSIVE_RESET),
            (0.1, CRLFTriggerState.HOLD),
            (0.5, CRLFTriggerState.ESCALATE),
            (1.2, CRLFTriggerState.ESCALATE),
            (2.0, CRLFTriggerState.OVERRIDE)
        ]

        for expected_output, expected_state in test_cases:
            # Manually set the CRLF output by manipulating the computation
            # This is a simplified test - in practice, the output is computed
            with self.subTest(expected_output=expected_output):
                # Create a custom state with manipulated parameters
                custom_state = CRLFState()
                    tau=0.0,
                    psi=self.test_strategy_vector,
                    delta_t=0.0,
                    entropy=0.1
                )
                custom_crlf = ChronoRecursiveLogicFunction(custom_state)

                # Force the output by manipulating the computation
                # This is a test-specific approach
                response = custom_crlf._determine_trigger_state(expected_output)
                self.assertEqual(response, expected_state)

    def test_confidence_computation(self):
        """Test confidence computation."""
        # Test with low entropy and high CRLF output
        response = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            0.1  # Low entropy
        )

        # Low entropy should result in higher confidence
        self.assertGreater(response.confidence, 0.5)

        # Test with high entropy
        response2 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            0.9  # High entropy
        )

        # High entropy should result in lower confidence
        self.assertLess(response2.confidence, response.confidence)

    def test_recommendations_generation(self):
        """Test recommendations generation."""
        response = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        recommendations = response.recommendations

        # Check required fields
        required_fields = ['action', 'confidence', 'risk_adjustment', 'strategy_weights', 'temporal_urgency']
        for field in required_fields:
            self.assertIn(field, recommendations)

        # Check strategy weights
        strategy_weights = recommendations['strategy_weights']
        expected_strategies = ['momentum', 'scalping', 'mean_reversion', 'swing']
        for strategy in expected_strategies:
            self.assertIn(strategy, strategy_weights)
            self.assertGreaterEqual(strategy_weights[strategy], 0.0)
            self.assertLessEqual(strategy_weights[strategy], 1.0)

        # Check that weights sum to approximately 1.0
        total_weight = sum(strategy_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=1)

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Run multiple computations
        for i in range(5):
            self.crlf.compute_crlf()
                self.test_strategy_vector,
                self.test_profit_curve,
                self.test_market_entropy
            )

        # Check performance summary
        summary = self.crlf.get_performance_summary()

        self.assertIn('total_executions', summary)
        self.assertEqual(summary['total_executions'], 5)

        self.assertIn('current_recursion_depth', summary)
        self.assertGreaterEqual(summary['current_recursion_depth'], 0)

        self.assertIn('average_confidence', summary)
        self.assertGreaterEqual(summary['average_confidence'], 0.0)
        self.assertLessEqual(summary['average_confidence'], 1.0)

        self.assertIn('trigger_state_distribution', summary)
        self.assertIsInstance(summary['trigger_state_distribution'], dict)

    def test_state_history_management(self):
        """Test state history management."""
        # Run computations to build history
        for i in range(15):
            self.crlf.compute_crlf()
                self.test_strategy_vector,
                self.test_profit_curve,
                self.test_market_entropy
            )

        # Check that history is maintained
        self.assertGreater(len(self.crlf.state.psi_history), 0)
        self.assertGreater(len(self.crlf.state.entropy_history), 0)
        self.assertGreater(len(self.crlf.state.crlf_output_history), 0)

        # Check that history doesn't exceed maximum'
        max_history = 100
        self.assertLessEqual(len(self.crlf.state.psi_history), max_history)
        self.assertLessEqual(len(self.crlf.state.entropy_history), max_history)
        self.assertLessEqual(len(self.crlf.state.crlf_output_history), max_history)

    def test_recursion_depth_limits(self):
        """Test recursion depth limits."""
        # Run many computations to test recursion depth limits
        for i in range(20):
            self.crlf.compute_crlf()
                self.test_strategy_vector,
                self.test_profit_curve,
                self.test_market_entropy
            )

        # Check that recursion depth doesn't exceed maximum'
        self.assertLessEqual(self.crlf.state.recursion_depth, self.crlf.state.max_recursion_depth)

    def test_fallback_response(self):
        """Test fallback response when computation fails."""
        # Create a CRLF with invalid state to trigger fallback
        invalid_state = CRLFState()
            tau=0.0,
            psi=np.array([]),  # Invalid empty array
            delta_t=0.0,
            entropy=0.1
        )

        # This should trigger the fallback response
        # Note: The actual implementation handles this gracefully
        # This test verifies the fallback mechanism exists
        fallback_response = self.crlf._create_fallback_response()

        self.assertEqual(fallback_response.trigger_state, CRLFTriggerState.RECURSIVE_RESET)
        self.assertEqual(fallback_response.crlf_output, -1.0)
        self.assertEqual(fallback_response.confidence, 0.0)
        self.assertIn('fallback_strategy', fallback_response.recommendations)

    def test_reset_functionality(self):
        """Test CRLF reset functionality."""
        # Run some computations
        for i in range(3):
            self.crlf.compute_crlf()
                self.test_strategy_vector,
                self.test_profit_curve,
                self.test_market_entropy
            )

        # Check that state has been modified
        self.assertGreater(self.crlf.state.total_executions, 0)
        self.assertGreater(len(self.crlf.execution_history), 0)

        # Reset the CRLF
        self.crlf.reset_state()

        # Check that state has been reset
        self.assertEqual(self.crlf.state.total_executions, 0)
        self.assertEqual(len(self.crlf.execution_history), 0)
        self.assertEqual(self.crlf.state.recursion_depth, 0)

    def test_temporal_decay(self):
        """Test temporal decay mechanism."""
        # First computation
        response1 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        # Wait a bit
        time.sleep(0.1)

        # Second computation
        response2 = self.crlf.compute_crlf()
            self.test_strategy_vector,
            self.test_profit_curve,
            self.test_market_entropy
        )

        # The outputs should be different due to temporal decay
        self.assertNotEqual(response1.crlf_output, response2.crlf_output)

    def test_strategy_alignment_tracking(self):
        """Test strategy alignment tracking."""
        # Run computations
        for i in range(5):
            self.crlf.compute_crlf()
                self.test_strategy_vector,
                self.test_profit_curve,
                self.test_market_entropy
            )

        # Check alignment scores
        self.assertGreater(len(self.crlf.strategy_alignment_scores), 0)

        # All alignment scores should be between 0 and 1
        for score in self.crlf.strategy_alignment_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_integration_with_zpe_zbe(self):
        """Test CRLF integration with ZPE-ZBE concepts."""
        # Create a scenario that simulates ZPE-ZBE integration
        from core.zpe_zbe_core import QuantumSyncStatus, ZBEBalance, ZPEVector

        # Simulate ZPE-ZBE enhanced strategy vector
        zpe_enhanced_strategy = np.array([0.7, 0.3, 0.4, 0.8])  # Enhanced with quantum factors

        response = self.crlf.compute_crlf()
            zpe_enhanced_strategy,
            self.test_profit_curve,
            self.test_market_entropy
        )

        # The response should be valid even with ZPE-ZBE enhanced inputs
        self.assertIsInstance(response, CRLFResponse)
        self.assertIsInstance(response.crlf_output, float)
        self.assertIsInstance(response.trigger_state, CRLFTriggerState)

        # Check that recommendations include quantum-aware adjustments
        recommendations = response.recommendations
        self.assertIn('risk_adjustment', recommendations)
        self.assertIn('strategy_weights', recommendations)
        self.assertIn('temporal_urgency', recommendations)


class TestCRLFEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for CRLF."""

    def setUp(self):
        """Set up test fixtures."""
        self.crlf = create_crlf()

    def test_empty_profit_curve(self):
        """Test CRLF with empty profit curve."""
        strategy_vector = np.array([0.5, 0.5, 0.5, 0.5])
        empty_profit_curve = np.array([])

        response = self.crlf.compute_crlf()
            strategy_vector,
            empty_profit_curve,
            0.3
        )

        # Should handle gracefully
        self.assertIsInstance(response, CRLFResponse)

    def test_single_point_profit_curve(self):
        """Test CRLF with single point profit curve."""
        strategy_vector = np.array([0.5, 0.5, 0.5, 0.5])
        single_point_curve = np.array([100])

        response = self.crlf.compute_crlf()
            strategy_vector,
            single_point_curve,
            0.3
        )

        # Should handle gracefully
        self.assertIsInstance(response, CRLFResponse)

    def test_extreme_entropy_values(self):
        """Test CRLF with extreme entropy values."""
        strategy_vector = np.array([0.5, 0.5, 0.5, 0.5])
        profit_curve = np.array([100, 105, 103, 108, 110])

        # Test with very low entropy
        response1 = self.crlf.compute_crlf()
            strategy_vector,
            profit_curve,
            0.0
        )

        # Test with very high entropy
        response2 = self.crlf.compute_crlf()
            strategy_vector,
            profit_curve,
            1.0
        )

        # Both should be valid responses
        self.assertIsInstance(response1, CRLFResponse)
        self.assertIsInstance(response2, CRLFResponse)

        # High entropy should result in lower confidence
        self.assertGreater(response1.confidence, response2.confidence)

    def test_extreme_strategy_vectors(self):
        """Test CRLF with extreme strategy vector values."""
        profit_curve = np.array([100, 105, 103, 108, 110])

        # Test with all zeros
        zero_vector = np.array([0.0, 0.0, 0.0, 0.0])
        response1 = self.crlf.compute_crlf(zero_vector, profit_curve, 0.3)

        # Test with all ones
        ones_vector = np.array([1.0, 1.0, 1.0, 1.0])
        response2 = self.crlf.compute_crlf(ones_vector, profit_curve, 0.3)

        # Both should be valid responses
        self.assertIsInstance(response1, CRLFResponse)
        self.assertIsInstance(response2, CRLFResponse)


def run_crlf_tests():
    """Run all CRLF tests."""
    logger.info("🧪 Running Chrono-Recursive Logic Function Tests")
    logger.info("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestChronoRecursiveLogicFunction))
    test_suite.addTest(unittest.makeSuite(TestCRLFEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Tests run: {result.testsRun}")
    logger.info(f"   Failures: {len(result.failures)}")
    logger.info(f"   Errors: {len(result.errors)}")

    if result.failures:
        logger.error("❌ Test Failures:")
        for test, traceback in result.failures:
            logger.error(f"   {test}: {traceback}")

    if result.errors:
        logger.error("❌ Test Errors:")
        for test, traceback in result.errors:
            logger.error(f"   {test}: {traceback}")

    if result.wasSuccessful():
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Some tests failed!")

    return result.wasSuccessful()


if __name__ == '__main__':
    run_crlf_tests() 