"""
Test suite for the Recursive Profit Allocation System
Tests the implementation of non-linear profit calculation through recursive expansion.
"""

import unittest
import numpy as np
from datetime import datetime
import time
from core.recursive_profit import RecursiveProfitAllocationSystem, RecursiveMarketState

class TestRecursiveProfitAllocationSystem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.profit_system = RecursiveProfitAllocationSystem(max_memory_depth=1000)
        
        # Install test profit methods
        self.profit_system.install_profit_method("scalping_tff", 25.0)
        self.profit_system.install_profit_method("swing_tpf", 35.0)
        self.profit_system.install_profit_method("echo_memory", 20.0)
        self.profit_system.install_profit_method("recursive_arbitrage", 15.0)
        self.profit_system.install_profit_method("reserve_buffer", 5.0)
        
        # Create test market states
        self.entry_state = RecursiveMarketState(
            timestamp=datetime.now(),
            price=100.0,
            volume=1000.0,
            tff_stability_index=0.95,
            infinite_recursion_depth=15,
            paradox_resolution_count=2,
            paradox_stability_score=0.88,
            memory_coherence_level=0.92,
            historical_echo_strength=0.85,
            quantum_coherence=0.91,
            recursive_momentum=0.05
        )
        
        self.current_state = RecursiveMarketState(
            timestamp=datetime.now(),
            price=105.0,
            volume=1200.0,
            tff_stability_index=0.97,
            infinite_recursion_depth=18,
            paradox_resolution_count=1,
            paradox_stability_score=0.92,
            memory_coherence_level=0.94,
            historical_echo_strength=0.88,
            quantum_coherence=0.93,
            recursive_momentum=0.06
        )
        
    def test_tff_profit_expansion(self):
        """Test Forever Fractal profit expansion calculation"""
        profit = self.profit_system.calculate_tff_profit_expansion(
            self.entry_state,
            self.current_state
        )
        self.assertIsInstance(profit, float)
        self.assertGreater(profit, 0)  # Should be positive for price increase
        
    def test_fractal_layer_profit(self):
        """Test fractal layer profit calculation"""
        layer_profit = self.profit_system.calculate_fractal_layer_profit(
            self.entry_state,
            self.current_state,
            layer=5
        )
        self.assertIsInstance(layer_profit, float)
        
    def test_tpf_paradox_profit(self):
        """Test Paradox Fractal profit resolution"""
        base_profit = self.current_state.price - self.entry_state.price
        tff_profit = self.profit_system.calculate_tff_profit_expansion(
            self.entry_state,
            self.current_state
        )
        
        resolved_profit = self.profit_system.calculate_tpf_paradox_profit(
            base_profit,
            tff_profit,
            self.current_state
        )
        
        self.assertIsInstance(resolved_profit, float)
        self.assertGreater(resolved_profit, 0)
        
    def test_tef_memory_profit(self):
        """Test Echo Fractal memory profit calculation"""
        # Add some history
        for i in range(60):
            state = RecursiveMarketState(
                timestamp=datetime.now(),
                price=100.0 + i * 0.1,
                volume=1000.0 + i * 10,
                tff_stability_index=0.95,
                infinite_recursion_depth=15,
                paradox_resolution_count=2,
                paradox_stability_score=0.88,
                memory_coherence_level=0.92,
                historical_echo_strength=0.85,
                quantum_coherence=0.91,
                recursive_momentum=0.05
            )
            self.profit_system.state_history.append(state)
            
        memory_profit = self.profit_system.calculate_tef_memory_profit(
            self.current_state
        )
        self.assertIsInstance(memory_profit, float)
        
    def test_pattern_similarity(self):
        """Test pattern similarity calculation between states"""
        similarity = self.profit_system.calculate_pattern_similarity(
            self.entry_state,
            self.current_state
        )
        self.assertIsInstance(similarity, float)
        self.assertTrue(0 <= similarity <= 1)
        
    def test_predictive_movement_profit(self):
        """Test predictive movement profit calculation"""
        prediction = self.profit_system.calculate_predictive_movement_profit(
            self.current_state
        )
        
        self.assertIsInstance(prediction, dict)
        self.assertIn('tff_movement', prediction)
        self.assertIn('tpf_movement', prediction)
        self.assertIn('tef_movement', prediction)
        self.assertIn('unified_movement', prediction)
        self.assertIn('profit_potential', prediction)
        
        for value in prediction.values():
            self.assertIsInstance(value, float)
            
    def test_movement_prediction_methods(self):
        """Test individual movement prediction methods"""
        # Test TFF prediction
        tff_pred = self.profit_system.predict_tff_movement(
            self.current_state,
            horizon=10
        )
        self.assertIsInstance(tff_pred, float)
        
        # Test TPF prediction
        tpf_pred = self.profit_system.predict_tpf_movement(
            self.current_state,
            horizon=10
        )
        self.assertIsInstance(tpf_pred, float)
        
        # Test TEF prediction
        tef_pred = self.profit_system.predict_tef_movement(
            self.current_state,
            horizon=10
        )
        self.assertIsInstance(tef_pred, float)
        
    def test_unified_predictions(self):
        """Test unified recursive predictions"""
        unified = self.profit_system.unify_recursive_predictions(
            tff_pred=0.1,
            tpf_pred=0.08,
            tef_pred=0.12
        )
        self.assertIsInstance(unified, float)
        
    def test_movement_profit_potential(self):
        """Test movement profit potential calculation"""
        # Test small movement
        small_potential = self.profit_system.calculate_movement_profit_potential(0.05)
        self.assertIsInstance(small_potential, float)
        
        # Test large movement
        large_potential = self.profit_system.calculate_movement_profit_potential(0.2)
        self.assertIsInstance(large_potential, float)
        
        # Test negligible movement
        zero_potential = self.profit_system.calculate_movement_profit_potential(0.0005)
        self.assertEqual(zero_potential, 0.0)
        
    def test_profit_allocation(self):
        """Test profit allocation across methods"""
        total_profit = 1000.0
        allocation = self.profit_system.allocate_profit_across_methods(total_profit)
        
        self.assertIsInstance(allocation, dict)
        self.assertEqual(len(allocation), 5)  # Should have 5 methods
        
        # Verify allocations sum to total profit
        total_allocated = sum(allocation.values())
        self.assertAlmostEqual(total_allocated, total_profit)
        
    def test_market_tick_processing(self):
        """Test complete market tick processing"""
        result = self.profit_system.process_market_tick(self.current_state)
        
        self.assertIsInstance(result, dict)
        self.assertIn('timestamp', result)
        self.assertIn('total_recursive_profit', result)
        self.assertIn('tff_profit', result)
        self.assertIn('tpf_profit', result)
        self.assertIn('tef_profit', result)
        self.assertIn('movement_prediction', result)
        self.assertIn('profit_allocation', result)
        self.assertIn('quantum_coherence', result)
        self.assertIn('system_status', result)
        
    def test_insufficient_history(self):
        """Test behavior with insufficient history"""
        # Create new system with no history
        new_system = RecursiveProfitAllocationSystem()
        result = new_system.process_market_tick(self.current_state)
        
        self.assertEqual(result['status'], 'insufficient_history')
        
if __name__ == '__main__':
    unittest.main() 