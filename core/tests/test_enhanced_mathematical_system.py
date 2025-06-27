# -*- coding: utf-8 -*-
"""
Test Suite for Enhanced Unified Mathematical System
==================================================

Comprehensive test suite for the enhanced unified mathematical system that validates
all bit-phase logic, portfolio vectorization, fabricated logic gates, volumetric
structures, BTC price mapping, and backlog integration.

Test Coverage:
- Bit-phase tensor operations (2-bit, 4-bit, 8-bit, 16-bit, 42-bit, 256-bit)
- Portfolio vector creation and pathway mapping
- Fabricated logic gates with hash contrast
- Volumetric structure calculations
- BTC price mapping to 16-bit for Ferris RDE
- Tensor contraction and mathematical operations
- Hash memory encoding and entropy compensation
- Backlog system integration
- Visualization hooks
- Error handling and edge cases

Flake8 compliant with comprehensive type hints and error handling.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

# Import the enhanced mathematical system
try:
    from core.enhanced_unified_mathematical_system import (
        EnhancedUnifiedMathematicalSystem, BitPhase, MathOperation, PortfolioAsset,
        BitPhaseResult, PortfolioVector, FabricatedLogicGate, VolumetricStructure,
        BacklogHashEntry, get_enhanced_math_system
    )
    ENHANCED_MATH_AVAILABLE = True
except ImportError as e:
    ENHANCED_MATH_AVAILABLE = False
    print(f"Enhanced mathematical system not available: {e}")


class TestEnhancedUnifiedMathematicalSystem(unittest.TestCase):
    """Test suite for the enhanced unified mathematical system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ENHANCED_MATH_AVAILABLE:
            self.skipTest("Enhanced mathematical system not available")
        
        self.math_system = EnhancedUnifiedMathematicalSystem()
        self.test_strategy_id = 12345
        self.test_assets = [PortfolioAsset.BTC, PortfolioAsset.ETH, PortfolioAsset.XRP]
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_backlog_file = os.path.join(self.temp_dir, "test_backlog.json")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.test_backlog_file):
            os.remove(self.test_backlog_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_bit_phase_tensor_basic(self):
        """Test basic bit phase tensor operations."""
        result = self.math_system.bit_phase_tensor(self.test_strategy_id)
        
        # Verify result structure
        self.assertIsInstance(result, BitPhaseResult)
        self.assertEqual(result.strategy_id, self.test_strategy_id)
        self.assertEqual(result.mode, 'auto')
        
        # Verify bit phase calculations
        self.assertEqual(result.phi_2, self.test_strategy_id & 0b11)
        self.assertEqual(result.phi_4, self.test_strategy_id & 0b1111)
        self.assertEqual(result.phi_8, (self.test_strategy_id >> 4) & 0b11111111)
        self.assertEqual(result.phi_16, (self.test_strategy_id >> 12) & 0b1111111111111111)
        self.assertEqual(result.phi_42, (self.test_strategy_id >> 28) & 0x3FFFFFFFFFF)
        
        # Verify SHA-256 hash
        self.assertIsInstance(result.phi_256, str)
        self.assertEqual(len(result.phi_256), 64)  # SHA-256 hex digest length
        
        # Verify entropy and compression scores
        self.assertGreaterEqual(result.entropy_score, 0.0)
        self.assertLessEqual(result.entropy_score, 1.0)
        self.assertGreaterEqual(result.compression_ratio, 0.0)
        self.assertLessEqual(result.compression_ratio, 1.0)
    
    def test_bit_phase_tensor_caching(self):
        """Test bit phase tensor caching functionality."""
        # First call
        result1 = self.math_system.bit_phase_tensor(self.test_strategy_id)
        
        # Second call should use cache
        result2 = self.math_system.bit_phase_tensor(self.test_strategy_id)
        
        # Results should be identical
        self.assertEqual(result1.phi_4, result2.phi_4)
        self.assertEqual(result1.phi_8, result2.phi_8)
        self.assertEqual(result1.phi_42, result2.phi_42)
        self.assertEqual(result1.phi_256, result2.phi_256)
        
        # Cache should contain the result
        self.assertIn(self.test_strategy_id, self.math_system.bit_phase_cache)
    
    def test_bit_phase_tensor_edge_cases(self):
        """Test bit phase tensor with edge cases."""
        # Test with zero
        result_zero = self.math_system.bit_phase_tensor(0)
        self.assertEqual(result_zero.phi_4, 0)
        self.assertEqual(result_zero.phi_8, 0)
        self.assertEqual(result_zero.phi_42, 0)
        
        # Test with maximum values
        max_32bit = 0xFFFFFFFF
        result_max = self.math_system.bit_phase_tensor(max_32bit)
        self.assertEqual(result_max.phi_4, 15)  # 0b1111
        self.assertEqual(result_max.phi_8, 255)  # 0b11111111
        self.assertEqual(result_max.phi_16, 65535)  # 0b1111111111111111
        
        # Test with negative values (should handle gracefully)
        result_negative = self.math_system.bit_phase_tensor(-12345)
        self.assertIsInstance(result_negative, BitPhaseResult)
    
    def test_portfolio_vector_creation(self):
        """Test portfolio vector creation and pathway mapping."""
        portfolio = self.math_system.create_portfolio_vector(self.test_assets)
        
        # Verify result structure
        self.assertIsInstance(portfolio, PortfolioVector)
        self.assertEqual(len(portfolio.assets), len(self.test_assets))
        self.assertEqual(len(portfolio.weights), len(self.test_assets))
        self.assertEqual(len(portfolio.pathway_mapping), len(self.test_assets))
        self.assertEqual(len(portfolio.strategy_hashes), len(self.test_assets))
        
        # Verify weights sum to 1.0
        total_weight = sum(portfolio.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=10)
        
        # Verify pathway mappings are 16-bit integers
        for asset in self.test_assets:
            pathway = portfolio.pathway_mapping[asset]
            self.assertIsInstance(pathway, int)
            self.assertGreaterEqual(pathway, 0)
            self.assertLessEqual(pathway, 65535)  # 16-bit max
        
        # Verify strategy hashes are strings
        for asset in self.test_assets:
            strategy_hash = portfolio.strategy_hashes[asset]
            self.assertIsInstance(strategy_hash, str)
            self.assertGreater(len(strategy_hash), 0)
        
        # Verify timestamp
        self.assertIsInstance(portfolio.timestamp, datetime)
    
    def test_portfolio_vector_with_weights(self):
        """Test portfolio vector creation with custom weights."""
        custom_weights = {
            PortfolioAsset.BTC: 0.5,
            PortfolioAsset.ETH: 0.3,
            PortfolioAsset.XRP: 0.2
        }
        
        portfolio = self.math_system.create_portfolio_vector(self.test_assets, custom_weights)
        
        # Verify custom weights are used
        for asset, weight in custom_weights.items():
            self.assertEqual(portfolio.weights[asset], weight)
        
        # Verify weights sum to 1.0
        total_weight = sum(portfolio.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=10)
    
    def test_fabricated_logic_gate_creation(self):
        """Test fabricated logic gate creation."""
        normalized_bit_state = 42
        hash_segment = "a1b2c3d4"
        
        gate = self.math_system.create_fabricated_logic_gate(normalized_bit_state, hash_segment)
        
        # Verify result structure
        self.assertIsInstance(gate, FabricatedLogicGate)
        self.assertEqual(gate.normalized_bit_state, normalized_bit_state)
        self.assertEqual(gate.hash_segment, hash_segment)
        
        # Verify XOR calculation
        hash_int = int(hash_segment, 16)
        expected_xor = normalized_bit_state ^ hash_int
        self.assertEqual(gate.xor_result, expected_xor)
        
        # Verify route selector is generated
        self.assertIsInstance(gate.route_selector, str)
        self.assertGreater(len(gate.route_selector), 0)
        
        # Verify success probability and energy cost
        self.assertGreaterEqual(gate.success_probability, 0.0)
        self.assertLessEqual(gate.success_probability, 1.0)
        self.assertGreaterEqual(gate.energy_cost, 0.0)
        
        # Verify gate is stored
        self.assertIn(gate.gate_id, self.math_system.fabricated_gates)
    
    def test_volumetric_structure_calculation(self):
        """Test volumetric structure calculation."""
        asset = PortfolioAsset.BTC
        price = 50000.0
        volume = 1000.0
        historical_data = [48000.0, 49000.0, 50000.0, 51000.0, 52000.0]
        
        structure = self.math_system.calculate_volumetric_structure(asset, price, volume, historical_data)
        
        # Verify result structure
        self.assertIsInstance(structure, VolumetricStructure)
        self.assertEqual(structure.asset, asset)
        self.assertEqual(structure.price, price)
        
        # Verify volatility calculation
        self.assertGreaterEqual(structure.volatility, 0.0)
        
        # Verify historical bounce calculation
        self.assertGreaterEqual(structure.historical_bounce, 0.0)
        
        # Verify volume gradient
        self.assertGreaterEqual(structure.volume_gradient, 0.0)
        
        # Verify confidence score
        self.assertGreaterEqual(structure.confidence_score, 0.0)
        self.assertLessEqual(structure.confidence_score, 1.0)
        
        # Verify structure is stored
        self.assertIn(asset, self.math_system.volumetric_structures)
    
    def test_btc_price_mapping_16bit(self):
        """Test BTC price mapping to 16-bit for Ferris RDE integration."""
        btc_price = 50000.0
        ferris_phase = "mid"
        
        entry = self.math_system.map_btc_price_16bit(btc_price, ferris_phase)
        
        # Verify result structure
        self.assertIsInstance(entry, BacklogHashEntry)
        self.assertEqual(entry.btc_price, btc_price)
        self.assertEqual(entry.ferris_phase, ferris_phase)
        
        # Verify 16-bit mapping
        self.assertIsInstance(entry.mapped_16bit, int)
        self.assertGreaterEqual(entry.mapped_16bit, 0)
        self.assertLessEqual(entry.mapped_16bit, 65535)
        
        # Verify hash sequence
        self.assertIsInstance(entry.hash_sequence, str)
        self.assertGreater(len(entry.hash_sequence), 0)
        
        # Verify profit factor
        self.assertGreaterEqual(entry.profit_factor, 0.0)
        self.assertLessEqual(entry.profit_factor, 1.0)
        
        # Verify memory persistence
        self.assertGreaterEqual(entry.memory_persistence, 0.0)
        self.assertLessEqual(entry.memory_persistence, 1.0)
        
        # Verify timestamp
        self.assertIsInstance(entry.timestamp, datetime)
        
        # Verify entry is stored in backlog
        self.assertIn(entry, self.math_system.backlog_entries)
    
    def test_btc_price_mapping_edge_cases(self):
        """Test BTC price mapping with edge cases."""
        # Test minimum price
        min_entry = self.math_system.map_btc_price_16bit(10000.0, "low")
        self.assertEqual(min_entry.mapped_16bit, 0)
        
        # Test maximum price
        max_entry = self.math_system.map_btc_price_16bit(100000.0, "high")
        self.assertEqual(max_entry.mapped_16bit, 65535)
        
        # Test price below minimum (should be clamped)
        below_min_entry = self.math_system.map_btc_price_16bit(5000.0, "low")
        self.assertEqual(below_min_entry.mapped_16bit, 0)
        
        # Test price above maximum (should be clamped)
        above_max_entry = self.math_system.map_btc_price_16bit(150000.0, "high")
        self.assertEqual(above_max_entry.mapped_16bit, 65535)
    
    def test_tensor_contraction(self):
        """Test tensor contraction operations."""
        A = np.random.random((3, 4))
        B = np.random.random((4, 2))
        
        result = self.math_system.tensor_contraction(A, B)
        
        # Verify result shape
        expected_shape = (3, 2)
        self.assertEqual(result.shape, expected_shape)
        
        # Verify result type
        self.assertEqual(result.dtype, self.math_system.precision)
        
        # Verify result is not all zeros (basic sanity check)
        self.assertFalse(np.allclose(result, 0))
    
    def test_tensor_contraction_edge_cases(self):
        """Test tensor contraction with edge cases."""
        # Test with identity matrix
        I = np.eye(3)
        result = self.math_system.tensor_contraction(I, I)
        self.assertTrue(np.allclose(result, I))
        
        # Test with zero matrix
        Z = np.zeros((3, 3))
        result = self.math_system.tensor_contraction(Z, I)
        self.assertTrue(np.allclose(result, 0))
        
        # Test with incompatible shapes (should handle gracefully)
        A = np.random.random((3, 4))
        B = np.random.random((5, 2))  # Incompatible
        result = self.math_system.tensor_contraction(A, B)
        self.assertIsInstance(result, np.ndarray)
    
    def test_hash_memory_encoding(self):
        """Test hash memory encoding."""
        test_data = "test_string_data"
        
        hash_result = self.math_system.hash_memory_encoding(test_data)
        
        # Verify result is SHA-256 hash
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)  # SHA-256 hex digest length
        
        # Verify result is hexadecimal
        try:
            int(hash_result, 16)
        except ValueError:
            self.fail("Hash result is not hexadecimal")
        
        # Test with different data types
        test_array = np.array([1, 2, 3, 4, 5])
        hash_array = self.math_system.hash_memory_encoding(test_array)
        self.assertEqual(len(hash_array), 64)
        
        test_bytes = b"test_bytes_data"
        hash_bytes = self.math_system.hash_memory_encoding(test_bytes)
        self.assertEqual(len(hash_bytes), 64)
    
    def test_entropy_compensation(self):
        """Test entropy compensation calculations."""
        # Create test data
        data = np.random.random(100)
        
        compensated_data = self.math_system.entropy_compensation(data)
        
        # Verify result shape
        self.assertEqual(compensated_data.shape, data.shape)
        
        # Verify result type
        self.assertEqual(compensated_data.dtype, self.math_system.precision)
        
        # Verify result is not identical to input (compensation applied)
        self.assertFalse(np.allclose(compensated_data, data))
        
        # Test with different compensation factors
        compensated_data_high = self.math_system.entropy_compensation(data, 2.0)
        compensated_data_low = self.math_system.entropy_compensation(data, 0.5)
        
        # Results should be different for different factors
        self.assertFalse(np.allclose(compensated_data_high, compensated_data_low))
    
    def test_entropy_compensation_edge_cases(self):
        """Test entropy compensation with edge cases."""
        # Test with empty array
        empty_data = np.array([])
        result = self.math_system.entropy_compensation(empty_data)
        self.assertEqual(result.size, 0)
        
        # Test with single element
        single_data = np.array([1.0])
        result = self.math_system.entropy_compensation(single_data)
        self.assertEqual(result.size, 1)
        
        # Test with all zeros
        zero_data = np.zeros(10)
        result = self.math_system.entropy_compensation(zero_data)
        self.assertEqual(result.shape, zero_data.shape)
    
    def test_visualization_hooks(self):
        """Test visualization hooks functionality."""
        hook_called = False
        hook_data = None
        
        def test_hook(data):
            nonlocal hook_called, hook_data
            hook_called = True
            hook_data = data
        
        # Add visualization hook
        self.math_system.add_visualization_hook('test_hook', test_hook)
        
        # Trigger hook by performing an operation
        self.math_system.bit_phase_tensor(12345)
        
        # Hook should be called
        self.assertTrue(hook_called)
        self.assertIsNotNone(hook_data)
    
    def test_backlog_data_export(self):
        """Test backlog data export functionality."""
        # Create some backlog entries
        self.math_system.map_btc_price_16bit(50000.0, "mid")
        self.math_system.map_btc_price_16bit(51000.0, "high")
        self.math_system.map_btc_price_16bit(49000.0, "low")
        
        # Export backlog data
        self.math_system.export_backlog_data(self.test_backlog_file)
        
        # Verify file exists
        self.assertTrue(os.path.exists(self.test_backlog_file))
        
        # Verify file content
        with open(self.test_backlog_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify structure
        self.assertIn('backlog_entries', data)
        self.assertIn('btc_price_history', data)
        
        # Verify entries
        self.assertGreater(len(data['backlog_entries']), 0)
        self.assertGreater(len(data['btc_price_history']), 0)
        
        # Verify entry structure
        entry = data['backlog_entries'][0]
        required_fields = ['timestamp', 'btc_price', 'mapped_16bit', 'hash_sequence', 
                          'ferris_phase', 'profit_factor', 'memory_persistence', 'api_synced']
        for field in required_fields:
            self.assertIn(field, entry)
    
    def test_system_statistics(self):
        """Test system statistics functionality."""
        # Perform some operations
        self.math_system.bit_phase_tensor(12345)
        self.math_system.create_portfolio_vector(self.test_assets)
        self.math_system.map_btc_price_16bit(50000.0, "mid")
        
        # Get statistics
        stats = self.math_system.get_statistics()
        
        # Verify statistics structure
        required_fields = [
            'operation_count', 'error_count', 'success_rate',
            'bit_phase_cache_size', 'portfolio_vectors_count',
            'fabricated_gates_count', 'volumetric_structures_count',
            'backlog_entries_count', 'btc_price_history_count',
            'visualization_hooks_count', 'precision', 'epsilon'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Verify operation count
        self.assertGreater(stats['operation_count'], 0)
        
        # Verify success rate
        self.assertGreaterEqual(stats['success_rate'], 0.0)
        self.assertLessEqual(stats['success_rate'], 1.0)
        
        # Verify cache sizes
        self.assertGreaterEqual(stats['bit_phase_cache_size'], 0)
        self.assertGreaterEqual(stats['portfolio_vectors_count'], 0)
        self.assertGreaterEqual(stats['backlog_entries_count'], 0)
    
    def test_error_handling(self):
        """Test error handling in mathematical operations."""
        # Test with invalid inputs
        try:
            # This should handle the error gracefully
            result = self.math_system.bit_phase_tensor("invalid_id")
            self.assertIsInstance(result, BitPhaseResult)
        except Exception as e:
            self.fail(f"Bit phase tensor should handle invalid input gracefully: {e}")
        
        # Test tensor contraction with incompatible shapes
        try:
            A = np.random.random((3, 4))
            B = np.random.random((5, 2))  # Incompatible
            result = self.math_system.tensor_contraction(A, B)
            self.assertIsInstance(result, np.ndarray)
        except Exception as e:
            self.fail(f"Tensor contraction should handle incompatible shapes gracefully: {e}")
    
    def test_history_management(self):
        """Test history management functionality."""
        # Perform operations to create history
        self.math_system.bit_phase_tensor(12345)
        self.math_system.create_portfolio_vector(self.test_assets)
        self.math_system.map_btc_price_16bit(50000.0, "mid")
        
        # Verify history exists
        self.assertGreater(len(self.math_system.operation_history), 0)
        self.assertGreater(len(self.math_system.bit_phase_cache), 0)
        self.assertGreater(len(self.math_system.portfolio_vectors), 0)
        self.assertGreater(len(self.math_system.backlog_entries), 0)
        
        # Clear history
        self.math_system.clear_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.math_system.operation_history), 0)
        self.assertEqual(len(self.math_system.bit_phase_cache), 0)
        self.assertEqual(len(self.math_system.portfolio_vectors), 0)
        self.assertEqual(len(self.math_system.backlog_entries), 0)
    
    def test_global_instance(self):
        """Test global enhanced mathematical system instance."""
        # Get global instance
        global_system = get_enhanced_math_system()
        
        # Verify it's the same instance
        self.assertIs(global_system, get_enhanced_math_system())
        
        # Verify it's a valid instance
        self.assertIsInstance(global_system, EnhancedUnifiedMathematicalSystem)
        
        # Test basic functionality
        result = global_system.bit_phase_tensor(12345)
        self.assertIsInstance(result, BitPhaseResult)


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    if not ENHANCED_MATH_AVAILABLE:
        print("❌ Enhanced mathematical system not available - skipping tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedUnifiedMathematicalSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1) 