"""
Tests for GPU Offload Manager
===========================

Verifies GPU offloading, thermal management, and profit tracking functionality.
"""

import unittest
import numpy as np
import time
from pathlib import Path
import os
import tempfile
import yaml

from ..gpu_offload_manager import GPUOffloadManager, OffloadState
from ..zbe_temperature_tensor import ZBETemperatureTensor
from ..profit_tensor import ProfitTensorStore

class TestGPUOffloadManager(unittest.TestCase):
    """Test cases for GPU Offload Manager"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary config directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name) / "config"
        self.config_dir.mkdir()
        
        # Create test config
        self.config_path = self.config_dir / "gpu_config.yaml"
        self.config = {
            'gpu_safety': {
                'max_utilization': 0.8,
                'max_temperature': 75.0,
                'min_data_size': 1000,
                'memory_pool_size': 1024
            },
            'thermal': {
                'optimal_temp': 60.0,
                'max_temp': 85.0,
                'thermal_decay': 0.95,
                'efficiency_threshold': 0.5
            },
            'bit_depths': [
                {
                    'depth': 4,
                    'profit_threshold': 0.1,
                    'thermal_threshold': 70.0,
                    'memory_limit': 0.5
                },
                {
                    'depth': 8,
                    'profit_threshold': 0.2,
                    'thermal_threshold': 65.0,
                    'memory_limit': 0.6
                }
            ],
            'profit': {
                'history_window': 1000,
                'min_profit_threshold': 0.2,
                'profit_decay': 0.95,
                'thermal_weight': 0.3
            },
            'logging': {
                'level': 'INFO',
                'file': str(self.config_dir / "gpu_offload.log")
            },
            'environment': {
                'force_cpu': False
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # Initialize manager
        self.manager = GPUOffloadManager(config_path=str(self.config_path))
        
    def tearDown(self):
        """Clean up test environment"""
        self.manager.stop()
        self.temp_dir.cleanup()
        
    def test_gpu_availability(self):
        """Test GPU availability detection"""
        # Force CPU mode
        os.environ["SCHWABOT_FORCE_CPU"] = "1"
        manager = GPUOffloadManager(config_path=str(self.config_path))
        self.assertFalse(manager.gpu_available)
        
        # Reset environment
        os.environ["SCHWABOT_FORCE_CPU"] = "0"
        
    def test_thermal_management(self):
        """Test thermal-aware execution"""
        # Create test data
        data = np.random.rand(2000)
        
        def gpu_func(arr):
            return arr * 2
            
        def cpu_func(arr):
            return arr * 2
            
        # Simulate high temperature
        self.manager.zbe_tensor.optimal_temp = 40.0
        self.manager.zbe_tensor.max_temp = 45.0
        
        # Process operation
        state = self.manager._process_operation({
            'operation_id': 'test_op',
            'data': data,
            'gpu_func': gpu_func,
            'cpu_func': cpu_func
        })
        
        # Should fall back to CPU due to thermal conditions
        self.assertTrue(state.fallback_used)
        
    def test_profit_tracking(self):
        """Test profit-based execution decisions"""
        # Create test data
        data = np.random.rand(2000)
        
        def gpu_func(arr):
            return arr * 2
            
        def cpu_func(arr):
            return arr * 2
            
        # Set low profit potential
        self.manager.profit_store.update_profit('test_op', 0.1, 0.8)
        
        # Process operation
        state = self.manager._process_operation({
            'operation_id': 'test_op',
            'data': data,
            'gpu_func': gpu_func,
            'cpu_func': cpu_func
        })
        
        # Should fall back to CPU due to low profit
        self.assertTrue(state.fallback_used)
        
    def test_bit_depth_selection(self):
        """Test bit depth selection based on conditions"""
        # Create test data
        data = np.random.rand(2000)
        
        def gpu_func(arr):
            return arr * 2
            
        def cpu_func(arr):
            return arr * 2
            
        # Set good conditions for 8-bit
        self.manager.zbe_tensor.optimal_temp = 60.0
        self.manager.profit_store.update_profit('test_op', 0.3, 0.8)
        
        # Process operation
        state = self.manager._process_operation({
            'operation_id': 'test_op',
            'data': data,
            'gpu_func': gpu_func,
            'cpu_func': cpu_func
        })
        
        # Should use 8-bit depth
        self.assertEqual(state.bit_depth, 8)
        
    def test_memory_pool(self):
        """Test memory pool allocation"""
        if not self.manager.gpu_available:
            self.skipTest("GPU not available")
            
        # Create test data
        data = np.random.rand(2000)
        
        def gpu_func(arr):
            return arr * 2
            
        def cpu_func(arr):
            return arr * 2
            
        # Process operation
        state = self.manager._process_operation({
            'operation_id': 'test_op',
            'data': data,
            'gpu_func': gpu_func,
            'cpu_func': cpu_func
        })
        
        # Check memory pool was used
        self.assertIsNotNone(self.manager.mem_pool)
        
    def test_fault_handling(self):
        """Test fault handling and recovery"""
        # Create test data
        data = np.random.rand(2000)
        
        def gpu_func(arr):
            raise RuntimeError("Test error")
            
        def cpu_func(arr):
            return arr * 2
            
        # Process operation
        state = self.manager._process_operation({
            'operation_id': 'test_op',
            'data': data,
            'gpu_func': gpu_func,
            'cpu_func': cpu_func
        })
        
        # Should fall back to CPU and record error
        self.assertTrue(state.fallback_used)
        self.assertFalse(state.success)
        self.assertIsNotNone(state.error_message)
        
    def test_history_management(self):
        """Test operation history management"""
        # Create test data
        data = np.random.rand(2000)
        
        def gpu_func(arr):
            return arr * 2
            
        def cpu_func(arr):
            return arr * 2
            
        # Process multiple operations
        for i in range(1500):  # More than history window
            state = self.manager._process_operation({
                'operation_id': f'test_op_{i}',
                'data': data,
                'gpu_func': gpu_func,
                'cpu_func': cpu_func
            })
            
        # Check history size is limited
        self.assertLessEqual(len(self.manager.offload_history), 
                           self.config['profit']['history_window'])
        
if __name__ == '__main__':
    unittest.main() 