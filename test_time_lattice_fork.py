"""
Test suite for Time Lattice Fork implementation
"""

import unittest
import numpy as np
from time_lattice_fork import TimeLatticeFork, NodeType
from datetime import datetime

class TestTimeLatticeFork(unittest.TestCase):
    def setUp(self):
        self.lattice = TimeLatticeFork(
            rsi_period=14,
            entropy_threshold=0.5,
            ghost_window=16
        )
        
    def test_rsi_calculation(self):
        # Test RSI calculation with known values
        prices = np.array([100, 102, 101, 103, 102, 104, 103, 105])
        rsi = self.lattice.calculate_rsi(prices)
        self.assertEqual(len(rsi), len(prices))
        self.assertTrue(all(0 <= x <= 100 for x in rsi))
        
    def test_ghost_pattern_detection(self):
        # Test ghost pattern detection
        current_price = 100.0
        expected_swing = 101.0  # 1% expected swing
        timestamp = datetime.now().timestamp()
        
        # Should detect ghost pattern
        ghost_hash = self.lattice.detect_ghost_pattern(
            current_price,
            expected_swing,
            timestamp
        )
        self.assertIsNotNone(ghost_hash)
        self.assertEqual(len(self.lattice.ghost_patterns), 1)
        
        # Should not detect ghost pattern
        current_price = 100.0
        expected_swing = 102.0  # 2% expected swing
        ghost_hash = self.lattice.detect_ghost_pattern(
            current_price,
            expected_swing,
            timestamp
        )
        self.assertIsNone(ghost_hash)
        
    def test_node_resonance(self):
        # Test node resonance calculation
        for node_type in NodeType:
            # Add some test nodes
            for i in range(5):
                self.lattice.update_node(
                    node_type=node_type,
                    value=100.0 + i,
                    rsi=50.0 + i,
                    hash_delta=0.1,
                    entropy=0.5
                )
            
            resonance = self.lattice.calculate_node_resonance(node_type)
            self.assertTrue(0 <= resonance <= 1)
            
    def test_lattice_signal(self):
        # Test signal generation
        # Add test data to nodes
        timestamp = datetime.now().timestamp()
        for node_type in NodeType:
            self.lattice.update_node(
                node_type=node_type,
                value=100.0,
                rsi=50.0,
                hash_delta=0.1,
                entropy=0.5
            )
        
        signal = self.lattice.get_lattice_signal()
        self.assertIn('action', signal)
        self.assertIn('confidence', signal)
        self.assertIn('reason', signal)
        self.assertIn(signal['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertTrue(0 <= signal['confidence'] <= 1)
        
    def test_process_tick(self):
        # Test full tick processing
        result = self.lattice.process_tick(
            price=100.0,
            volume=1.0,
            timestamp=datetime.now().timestamp()
        )
        
        self.assertIn('signal', result)
        self.assertIn('rsi', result)
        self.assertIn('hash_delta', result)
        self.assertIn('entropy', result)
        self.assertIn('ghost_hash', result)
        
        # Verify RSI range
        self.assertTrue(0 <= result['rsi'] <= 100)
        
        # Verify hash delta
        self.assertTrue(0 <= result['hash_delta'] <= 1)
        
        # Verify entropy
        self.assertTrue(result['entropy'] >= 0)

if __name__ == '__main__':
    unittest.main() 