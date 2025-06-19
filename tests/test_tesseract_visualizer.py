"""
Tests for the TesseractVisualizer class
"""

import unittest
import numpy as np
import time
from core.tesseract_visualizer import TesseractVisualizer, TesseractMetrics


class TestTesseractVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = TesseractVisualizer(
            tensor_dims=(5, 5, 5, 3),
            update_interval=0.1,
            history_size=100,
            debug_mode=True
        )

    def tearDown(self):
        self.visualizer.stop_visualization()

    def test_initialization(self):
        """Test visualizer initialization"""
        self.assertEqual(self.visualizer.tensor_dims, (5, 5, 5, 3))
        self.assertEqual(self.visualizer.update_interval, 0.1)
        self.assertEqual(self.visualizer.history_size, 100)
        self.assertTrue(self.visualizer.debug_mode)
        self.assertFalse(self.visualizer.running)

    def test_4d_to_3d_projection(self):
        """Test 4D to 3D projection methods"""
        # Create test tensor
        test_tensor = np.random.rand(5, 5, 5, 3)

        # Test slice projection
        slice_proj = self.visualizer._project_4d_to_3d(
            test_tensor,
            method='slice'
        )
        self.assertEqual(slice_proj.shape, (5, 5, 5))

        # Test sum projection
        sum_proj = self.visualizer._project_4d_to_3d(test_tensor, method='sum')
        self.assertEqual(sum_proj.shape, (5, 5, 5))

        # Test max projection
        max_proj = self.visualizer._project_4d_to_3d(test_tensor, method='max')
        self.assertEqual(max_proj.shape, (5, 5, 5))

        # Test average projection
        avg_proj = self.visualizer._project_4d_to_3d(
            test_tensor,
            method='average'
        )
        self.assertEqual(avg_proj.shape, (5, 5, 5))

        # Test invalid method
        invalid_proj = self.visualizer._project_4d_to_3d(
            test_tensor,
            method='invalid'
        )
        self.assertEqual(invalid_proj.shape, (5, 5, 5))

    def test_metrics_calculation(self):
        """Test metrics calculation"""
        test_tensor = np.random.rand(5, 5, 5, 3)
        metrics = self.visualizer._calculate_metrics(test_tensor)

        self.assertIsInstance(metrics, TesseractMetrics)
        self.assertGreaterEqual(metrics.magnitude, 0)
        self.assertGreaterEqual(metrics.centroid_distance, 0)
        self.assertGreaterEqual(metrics.stability, 0)
        self.assertLessEqual(metrics.stability, 1)
        self.assertGreaterEqual(metrics.entropy, 0)

    def test_visualization_update(self):
        """Test visualization update with tensor data"""
        test_tensor = np.random.rand(5, 5, 5, 3)
        current_time = time.time()

        self.visualizer.update_visualization(
            thermal_tensor=test_tensor,
            memory_tensor=test_tensor,
            profit_tensor=test_tensor,
            current_time=current_time
        )

        self.assertEqual(len(self.visualizer.metrics_history), 1)
        self.assertEqual(len(self.visualizer.pattern_history), 1)
        self.assertEqual(len(self.visualizer.projection_history), 1)

    def test_history_management(self):
        """Test history size management"""
        test_tensor = np.random.rand(5, 5, 5, 3)

        # Add more items than history_size
        for _ in range(150):
            self.visualizer.update_visualization(
                thermal_tensor=test_tensor,
                memory_tensor=test_tensor,
                profit_tensor=test_tensor,
                current_time=time.time()
            )

        # Check that history is trimmed
        self.assertLessEqual(
            len(self.visualizer.metrics_history),
            self.visualizer.history_size
        )
        self.assertLessEqual(
            len(self.visualizer.pattern_history),
            self.visualizer.history_size
        )
        self.assertLessEqual(
            len(self.visualizer.projection_history),
            self.visualizer.history_size
        )

    def test_debug_info(self):
        """Test debug information retrieval"""
        debug_info = self.visualizer.get_debug_info()

        self.assertTrue(debug_info['debug_mode'])
        self.assertIn('last_update', debug_info)
        self.assertIn('projection_errors', debug_info)
        self.assertIn('dimension_mismatches', debug_info)
        self.assertIn('performance_metrics', debug_info)

    def test_visualization_stats(self):
        """Test visualization statistics"""
        stats = self.visualizer.get_visualization_stats()

        self.assertIn('total_updates', stats)
        self.assertIn('current_metrics', stats)
        self.assertIn('pattern_history_size', stats)
        self.assertIn('projection_history_size', stats)
        self.assertIn('tensor_dimensions', stats)
        self.assertIn('update_interval', stats)
        self.assertIn('running', stats)

    def test_start_stop_visualization(self):
        """Test visualization start/stop functionality"""
        self.visualizer.start_visualization()
        self.assertTrue(self.visualizer.running)

        self.visualizer.stop_visualization()
        self.assertFalse(self.visualizer.running)

        # Test multiple start/stop cycles
        for _ in range(3):
            self.visualizer.start_visualization()
            self.assertTrue(self.visualizer.running)
            self.visualizer.stop_visualization()
            self.assertFalse(self.visualizer.running)


if __name__ == '__main__':
    unittest.main()