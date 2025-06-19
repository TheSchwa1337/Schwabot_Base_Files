"""
Test suite for the Timing Manager
Tests the implementation of the mathematical timing framework.
"""

import unittest
import time
from core.timing_manager import TimingManager


class TestTimingManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.timing_manager = TimingManager(
            recursion_coefficient=0.5,
            memory_decay_rate=0.1,
            phase_sync_rate=0.2
        )
        self.current_time = time.time()

    def test_forever_fractal_calculation(self):
        """Test Forever Fractal stabilization calculation"""
        result = self.timing_manager.calculate_forever_fractal(
            self.current_time)
        self.assertIsInstance(result, float)
        self.assertTrue(-1 <= result <= 1)  # Result should be normalized

    def test_paradox_fractal_calculation(self):
        """Test Paradox Fractal resolution calculation"""
        chaos_integral = 0.5
        result = self.timing_manager.calculate_paradox_fractal(
            self.current_time,
            chaos_integral
        )
        self.assertIsInstance(result, float)
        self.assertTrue(result >= 0)  # Should be non-negative

    def test_echo_fractal_calculation(self):
        """Test Echo Fractal memory preservation calculation"""
        result = self.timing_manager.calculate_echo_fractal(self.current_time)
        self.assertIsInstance(result, float)
        self.assertTrue(-1 <= result <= 1)  # Result should be normalized

    def test_phase_transition_calculation(self):
        """Test smooth phase transition calculation"""
        result = self.timing_manager.calculate_phase_transition(
            self.current_time)
        self.assertIsInstance(result, float)
        self.assertTrue(-1 <= result <= 1)  # Result should be normalized

    def test_timing_state_update(self):
        """Test timing state update with market data"""
        market_data = {
            'volatility': 0.5,
            'volume': 1000,
            'price_changes': [0.1, -0.2, 0.3, -0.1]
        }

        self.timing_manager.update_timing_state(self.current_time, market_data)
        metrics = self.timing_manager.get_timing_metrics()

        # Verify all metrics are present and valid
        self.assertIn('forever_fractal', metrics)
        self.assertIn('paradox_resolution', metrics)
        self.assertIn('echo_memory', metrics)
        self.assertIn('phase_alignment', metrics)
        self.assertIn('recursion_depth', metrics)
        self.assertIn('memory_weight', metrics)

        # Verify metric ranges
        for metric, value in metrics.items():
            self.assertIsInstance(value, float)
            if metric != 'recursion_depth':
                self.assertTrue(-1 <= value <= 1)
            else:
                self.assertTrue(value >= 0)

    def test_recursion_depth_calculation(self):
        """Test recursion depth calculation based on market data"""
        market_data = {
            'volatility': 0.5,
            'volume': 1000
        }

        depth = self.timing_manager._calculate_recursion_depth(market_data)
        self.assertIsInstance(depth, int)
        self.assertTrue(depth >= 0)

    def test_chaos_integral_calculation(self):
        """Test chaos integral calculation from price changes"""
        market_data = {
            'price_changes': [0.1, -0.2, 0.3, -0.1]
        }

        integral = self.timing_manager._calculate_chaos_integral(market_data)
        self.assertIsInstance(integral, float)
        self.assertTrue(integral >= 0)

    def test_state_functions(self):
        """Test various state calculation functions"""
        t = self.current_time

        # Test recursion state
        rec_state = self.timing_manager._recursion_state(t)
        self.assertIsInstance(rec_state, float)
        self.assertTrue(-1 <= rec_state <= 1)

        # Test unstable state
        unstable = self.timing_manager._unstable_state(t)
        self.assertIsInstance(unstable, float)
        self.assertTrue(-1 <= unstable <= 1)

        # Test stable state
        stable = self.timing_manager._stable_state(t)
        self.assertIsInstance(stable, float)
        self.assertTrue(-1 <= stable <= 1)

        # Test observer state
        observer = self.timing_manager._observer_state(t)
        self.assertIsInstance(observer, float)
        self.assertTrue(-1 <= observer <= 1)

    def test_transition_integral(self):
        """Test transition integral calculation"""
        t = self.current_time
        integral = self.timing_manager._calculate_transition_integral(t)
        self.assertIsInstance(integral, float)
        self.assertTrue(-1 <= integral <= 1)

    def test_memory_management(self):
        """Test echo memory management"""
        # Update state multiple times
        for _ in range(1100):  # More than the 1000 limit
            self.timing_manager.update_timing_state(
                self.current_time,
                {'volatility': 0.5, 'volume': 1000}
            )

        # Verify memory size is maintained
        self.assertEqual(len(self.timing_manager.state.echo_memory), 1000)


if __name__ == '__main__':
    unittest.main()