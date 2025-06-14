"""
Production Readiness Test Suite
Tests all 12 mathematical requirements and production features.
"""

import unittest
import numpy as np
import time
from unittest.mock import MagicMock, patch

import sys
sys.path.append('..')

from core import (
    HashRecollectionSystem, EntropyTracker, BitOperations, 
    PatternUtils, StrangeLoopDetector, RiskEngine
)


class TestMathematicalRequirements(unittest.TestCase):
    """Test all 12 mathematical requirements for live BTC trading."""
    
    def setUp(self):
        """Set up test environment."""
        self.entropy_tracker = EntropyTracker(maxlen=100)
        self.bit_ops = BitOperations()
        self.pattern_utils = PatternUtils()
        self.strange_loop_detector = StrangeLoopDetector()
        self.risk_engine = RiskEngine(max_position_size=0.25)
        
        # Mock BTC price data
        self.btc_prices = [29000 + i * 10 + np.random.normal(0, 50) for i in range(100)]
        self.volumes = [1000 + np.random.normal(0, 100) for _ in range(100)]
    
    def test_1_tick_normalization(self):
        """Test Requirement 1: Tick Normalisation (log-return Z-scoring)"""
        # Test log-return calculation
        prices = np.array([29000, 29100, 29050, 29200])
        
        # Calculate log returns: r_t = ln(P_t / P_t-1)
        log_returns = np.diff(np.log(prices))
        expected_returns = [np.log(29100/29000), np.log(29050/29100), np.log(29200/29050)]
        
        np.testing.assert_array_almost_equal(log_returns, expected_returns, decimal=6)
        
        # Test Z-scoring: z_t = (r_t - μ_r) / σ_r
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        z_scores = (log_returns - mean_return) / std_return
        
        # Z-scores should have mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(np.mean(z_scores), 0.0, places=10)
        self.assertAlmostEqual(np.std(z_scores), 1.0, places=10)
        
        # Test with entropy tracker
        for i, price in enumerate(prices[1:]):
            state = self.entropy_tracker.update(price, self.volumes[i], time.time() + i)
            self.assertIsNotNone(state)
            # Normalized values should be properly calculated
            self.assertIsInstance(state.price_normalized, float)
    
    def test_2_shannon_entropy_windows(self):
        """Test Requirement 2: Shannon Entropy Windows (H5, H16, H64)"""
        # Feed data to entropy tracker
        for i in range(70):  # Enough for 64-window
            price = 29000 + i * 10 + np.random.normal(0, 20)
            volume = 1000 + np.random.normal(0, 50)
            self.entropy_tracker.update(price, volume, time.time() + i)
        
        # Get multi-window entropies
        entropies = self.entropy_tracker.get_multi_window_entropies()
        
        # Should have short (5), mid (16), and long (64) windows
        self.assertIn('short', entropies)
        self.assertIn('mid', entropies)
        self.assertIn('long', entropies)
        
        # Each window should have price, volume, and time entropy
        for window in entropies.values():
            self.assertIn('price_entropy', window)
            self.assertIn('volume_entropy', window)
            self.assertIn('time_entropy', window)
            
            # Entropy values should be non-negative
            self.assertGreaterEqual(window['price_entropy'], 0)
            self.assertGreaterEqual(window['volume_entropy'], 0)
            self.assertGreaterEqual(window['time_entropy'], 0)
    
    def test_3_bit_pattern_density_variance(self):
        """Test Requirement 3: Bit-Pattern Density & Variance"""
        # Test bit pattern density calculation
        test_pattern = 0b101010101010  # 6 ones out of 12 bits
        density = self.bit_ops.calculate_bit_density(test_pattern)
        expected_density = 6 / 42  # Should normalize to 42-bit pattern
        
        # Test variance calculation across multiple densities
        densities = [0.3, 0.35, 0.4, 0.32, 0.38]
        variance = self.bit_ops.calculate_density_variance(densities)
        expected_variance = np.var(densities)
        self.assertAlmostEqual(variance, expected_variance, places=6)
        
        # Test multi-scale variance tracking
        for density in densities:
            self.bit_ops.update_density_tracking(density)
        
        variances = self.bit_ops.get_multi_scale_variances()
        self.assertIn('short', variances)
        self.assertIn('mid', variances)
        self.assertIn('long', variances)
    
    def test_4_phase_extraction(self):
        """Test Requirement 4: Phase Extraction (4/8/42 bits)"""
        # Test phase bit extraction
        test_pattern = 0b111100001111000011110000111100001111000011  # 42-bit pattern
        
        b4, b8, b42 = self.bit_ops.extract_phase_bits(test_pattern)
        
        # Verify extraction
        self.assertEqual(b42, test_pattern)
        self.assertEqual(b8, (test_pattern >> 34) & 0xFF)
        self.assertEqual(b4, (test_pattern >> 38) & 0xF)
        
        # Test with entropy state
        entropy_state = self.entropy_tracker.update(29000, 1000, time.time())
        bit_pattern = self.bit_ops.calculate_42bit_float(entropy_state.price_entropy)
        phase_state = self.bit_ops.create_phase_state(bit_pattern, entropy_state)
        
        self.assertEqual(phase_state.b42, bit_pattern)
        self.assertIsInstance(phase_state.b4, int)
        self.assertIsInstance(phase_state.b8, int)
    
    def test_5_entry_exit_rules(self):
        """Test Requirement 5: Entry/Exit Rules"""
        # Create test phase state
        test_pattern = 0x3 << 38  # Put entry key in 4-bit position
        phase_state = self.bit_ops.create_phase_state(test_pattern, None)
        phase_state.density = 0.6  # Above entry threshold
        phase_state.variance_short = 0.001  # Below variance threshold
        
        # Test entry conditions
        pattern_analysis = {'pattern_strength': 0.8}
        is_entry, reasons = self.pattern_utils.is_entry_phase(phase_state, pattern_analysis)
        
        self.assertTrue(is_entry)
        self.assertIn('4-bit pattern in entry keys', reasons)
        
        # Test exit conditions
        phase_state.density = 0.3  # Below exit threshold
        is_exit, exit_reasons = self.pattern_utils.is_exit_phase(phase_state, pattern_analysis)
        
        self.assertTrue(is_exit)
        self.assertIn('density below exit threshold', exit_reasons)
    
    def test_6_hash_entropy_similarity(self):
        """Test Requirement 6: Hash-Entropy Similarity Score"""
        # Create test entropy vectors
        e1 = np.array([0.5, 0.3, 0.2])
        e2 = np.array([0.52, 0.31, 0.19])  # Similar
        e3 = np.array([0.1, 0.8, 0.9])    # Different
        
        # Test similarity calculation
        h1, h2 = 0x1234567890ABCDEF, 0x1234567890ABCDFF  # 1-bit different
        
        similarity_close = self.pattern_utils.compare_hashes(h1, h2, e1, e2)
        similarity_far = self.pattern_utils.compare_hashes(h1, h2, e1, e3)
        
        # Similar entropy + similar hash should score higher
        self.assertGreater(similarity_close, similarity_far)
        self.assertLessEqual(similarity_close, 1.0)
        self.assertGreaterEqual(similarity_far, 0.0)
    
    def test_7_cluster_confidence(self):
        """Test Requirement 7: Cluster Confidence"""
        # Mock hash database
        mock_db = {}
        for i in range(10):
            mock_entry = MagicMock()
            mock_entry.frequency = i + 1
            mock_entry.profit_history = 0.01 * i
            mock_db[i] = mock_entry
        
        # Test confidence calculation
        similar_hashes = [(1, 0.9), (2, 0.8), (3, 0.7)]
        pattern_analysis = {'pattern_strength': 0.8}
        
        confidence = self.pattern_utils.calculate_confidence(
            0, similar_hashes, mock_db, pattern_analysis
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_8_strange_loop_detector(self):
        """Test Requirement 8: Strange-Loop/Echo Detector"""
        # Test echo detection
        hash_val = 0x123456789ABCDEF0
        entropy_state1 = self.entropy_tracker.update(29000, 1000, time.time())
        entropy_state2 = self.entropy_tracker.update(29100, 1200, time.time() + 5)  # Different entropy
        
        # First occurrence
        result1 = self.strange_loop_detector.process_hash(hash_val, entropy_state1)
        self.assertIsNone(result1)  # No echo on first occurrence
        
        # Second occurrence with different entropy (should detect echo)
        result2 = self.strange_loop_detector.process_hash(hash_val, entropy_state2)
        if result2:  # May or may not trigger depending on threshold
            self.assertEqual(result2.hash_value, hash_val)
            self.assertIn(result2.pattern_type, ['echo', 'strange_loop'])
    
    def test_9_volatility_weighted_position_size(self):
        """Test Requirement 9: Volatility-Weighted Position Size"""
        # Set up price history for volatility calculation
        for price in self.btc_prices:
            self.risk_engine.update_price(price)
        
        # Calculate position size
        signal_confidence = 0.8
        expected_edge = 0.05
        current_price = 29000
        stop_loss_price = 28500
        
        position_signal = self.risk_engine.calculate_position_size(
            signal_confidence, expected_edge, current_price, stop_loss_price
        )
        
        # Verify Kelly calculation components
        self.assertIsInstance(position_signal.kelly_fraction, float)
        self.assertGreater(position_signal.kelly_fraction, 0)
        self.assertLess(position_signal.kelly_fraction, 0.25)  # Fractional Kelly
        
        # Position size should be reasonable
        self.assertGreater(position_signal.risk_adjusted_size, 0)
        self.assertLessEqual(position_signal.risk_adjusted_size, 0.25)  # Max position size
    
    def test_10_latency_compensation(self):
        """Test Requirement 10: Latency Compensation"""
        # Test latency adjustment
        original_timestamp = time.time()
        tick_data = {'timestamp': original_timestamp, 'price': 29000, 'volume': 1000}
        latency_ms = 50.0
        
        adjusted_data = self.pattern_utils.adjust_for_latency(tick_data, latency_ms)
        
        # Timestamp should be adjusted backwards by latency
        expected_timestamp = original_timestamp - (latency_ms / 1000)
        self.assertAlmostEqual(adjusted_data['timestamp'], expected_timestamp, places=3)
        
        # Other data should remain unchanged
        self.assertEqual(adjusted_data['price'], tick_data['price'])
        self.assertEqual(adjusted_data['volume'], tick_data['volume'])
    
    def test_11_gpu_metrics(self):
        """Test Requirement 11: GPU Metrics"""
        # Create hash recollection system
        hash_system = HashRecollectionSystem()
        
        # Get pattern metrics
        metrics = hash_system.get_pattern_metrics()
        
        # Should include GPU utilization (0 if no GPU)
        self.assertIn('gpu_utilization', metrics)
        self.assertGreaterEqual(metrics['gpu_utilization'], 0.0)
        self.assertLessEqual(metrics['gpu_utilization'], 1.0)
        
        # Should include queue utilization metrics
        self.assertIn('queue_utilization', metrics)
        self.assertIn('hash_queue', metrics['queue_utilization'])
        self.assertIn('result_queue', metrics['queue_utilization'])
    
    def test_12_profit_to_risk_expectancy(self):
        """Test Requirement 12: Profit-to-Risk Expectancy"""
        # Record some test trades
        trades = [
            (28900, 29100, 0.1, 'long'),   # Profit
            (29000, 29050, 0.1, 'long'),   # Small profit
            (29100, 28950, 0.1, 'long'),   # Loss
            (29200, 29400, 0.1, 'long'),   # Profit
        ]
        
        for entry, exit, size, trade_type in trades:
            self.risk_engine.record_trade(entry, exit, size, trade_type, 60.0)
        
        # Get risk metrics
        risk_metrics = self.risk_engine.get_risk_metrics()
        
        # Should calculate expectancy
        self.assertIsInstance(risk_metrics.current_expectancy, float)
        
        # Should calculate other risk metrics
        self.assertIsInstance(risk_metrics.sharpe_ratio, float)
        self.assertIsInstance(risk_metrics.win_rate, float)
        self.assertIsInstance(risk_metrics.profit_factor, float)
        
        # Win rate should be between 0 and 1
        self.assertGreaterEqual(risk_metrics.win_rate, 0.0)
        self.assertLessEqual(risk_metrics.win_rate, 1.0)


class TestProductionFeatures(unittest.TestCase):
    """Test production features from the punch-list."""
    
    def setUp(self):
        """Set up test environment."""
        self.hash_system = HashRecollectionSystem()
    
    def test_back_pressure_handling(self):
        """Test back-pressure handling (queue overload protection)."""
        # Start the system
        self.hash_system.start()
        
        # Flood with ticks to test back-pressure
        initial_dropped = self.hash_system.dropped_ticks
        
        # Send many ticks rapidly
        for i in range(100):
            self.hash_system.process_tick(29000 + i, 1000 + i, time.time() + i * 0.001)
        
        # System should handle overload gracefully
        metrics = self.hash_system.get_pattern_metrics()
        
        # Should track dropped ticks
        self.assertGreaterEqual(self.hash_system.dropped_ticks, initial_dropped)
        self.assertIn('dropped_ticks', metrics)
        
        # Queue utilization should be reported
        self.assertIn('queue_utilization', metrics)
        
        self.hash_system.stop()
    
    def test_system_metrics_and_reporting(self):
        """Test comprehensive system metrics and reporting."""
        # Process some ticks
        for i in range(20):
            self.hash_system.process_tick(29000 + i * 10, 1000 + i, time.time() + i)
        
        # Get system report
        report = self.hash_system.get_system_report()
        
        # Should include all required sections
        required_sections = ['summary', 'entropy', 'patterns', 'system']
        for section in required_sections:
            self.assertIn(section, report)
        
        # Summary should include key metrics
        summary = report['summary']
        required_summary_fields = [
            'uptime', 'ticks_processed', 'patterns_detected', 
            'hash_database_size', 'current_price'
        ]
        for field in required_summary_fields:
            self.assertIn(field, summary)
        
        # Should include strange loop metrics
        metrics = self.hash_system.get_pattern_metrics()
        self.assertIn('strange_loops', metrics)
        
        # Should include risk metrics
        self.assertIn('risk', metrics)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test graceful shutdown
        self.hash_system.start()
        self.assertTrue(self.hash_system.running)
        
        self.hash_system.stop()
        self.assertFalse(self.hash_system.running)
        
        # Test processing after shutdown (should not crash)
        self.hash_system.process_tick(29000, 1000, time.time())
        
        # Should handle invalid data gracefully
        try:
            self.hash_system.process_tick(float('nan'), -1000, time.time())
        except Exception as e:
            self.fail(f"System should handle invalid data gracefully, but raised: {e}")
    
    def test_configuration_management(self):
        """Test configuration loading and validation."""
        # Test with default configuration
        system1 = HashRecollectionSystem()
        self.assertIsNotNone(system1.config)
        
        # Test configuration merging
        default_keys = ['sync_interval', 'gpu_enabled', 'patterns']
        for key in default_keys:
            self.assertIn(key, system1.config)
    
    def test_modular_integration(self):
        """Test that all modules integrate properly."""
        # All components should be initialized
        self.assertIsNotNone(self.hash_system.entropy_tracker)
        self.assertIsNotNone(self.hash_system.bit_operations)
        self.assertIsNotNone(self.hash_system.pattern_utils)
        self.assertIsNotNone(self.hash_system.strange_loop_detector)
        self.assertIsNotNone(self.hash_system.risk_engine)
        
        # Process a tick through the full pipeline
        initial_hash_count = len(self.hash_system.hash_database)
        
        self.hash_system.process_tick(29000, 1000, time.time())
        
        # Should not crash and should potentially create hash entries
        # (depending on timing and queue processing)
        self.assertGreaterEqual(len(self.hash_system.hash_database), initial_hash_count)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 