from datetime import datetime
from unittest.mock import Mock

"""
Tests for the ResourceSequencer class
"""

import unittest  # noqa: F401
from unittest.mock import patch, MagicMock  # noqa: F401
from core.resource_sequencer import (  # noqa: F401
    ResourceSequencer,
    ResourceState,
    ResourceMetrics,
    SequenceMetrics
)


class TestResourceSequencer(unittest.TestCase):
    def setUp(self):
        self.sequencer = ResourceSequencer(
            max_cpu_percent=80.0,
            max_gpu_percent=85.0,
            max_memory_percent=75.0,
            thermal_threshold=0.8,
            retry_max_attempts=3,
            retry_base_delay=1.0
        )

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('GPUtil.getGPUs')
    def test_get_resource_metrics(self, mock_gpus, mock_memory, mock_cpu):
        # Mock CPU usage
        mock_cpu.return_value = 50.0

        # Mock memory usage
        mock_memory.return_value = MagicMock(percent=60.0)

        # Mock GPU usage
        mock_gpu = MagicMock()
        mock_gpu.load = 0.4
        mock_gpus.return_value = [mock_gpu]

        metrics = self.sequencer.get_resource_metrics()

        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.gpu_percent, 40.0)
        self.assertIsInstance(metrics.timestamp, datetime)  # noqa: F821
        self.assertIsInstance(metrics.resource_state, ResourceState)

    def test_calculate_position_size(self):
        # Test optimal state
        with patch.object(
            self.sequencer,
                'get_resource_metrics'
        ) as mock_metrics:
            mock_metrics.return_value = ResourceMetrics(
                cpu_percent=30.0,
                gpu_percent=20.0,
                memory_percent=40.0,
                thermal_state=0.3,
                timestamp=datetime.now(),  # noqa: F821
                resource_state=ResourceState.OPTIMAL
            )
            size = self.sequencer.calculate_position_size(1.0)
            self.assertEqual(size, 1.0)

        # Test critical state
        with patch.object(
            self.sequencer,
                'get_resource_metrics'
        ) as mock_metrics:
            mock_metrics.return_value = ResourceMetrics(
                cpu_percent=90.0,
                gpu_percent=85.0,
                memory_percent=80.0,
                thermal_state=0.9,
                timestamp=datetime.now(),  # noqa: F821
                resource_state=ResourceState.CRITICAL
            )
            size = self.sequencer.calculate_position_size(1.0)
            self.assertEqual(size, 0.1)  # Minimum position size

    def test_sequence_management(self):
        # Test starting a sequence
        sequence_id = "test_001"
        success = self.sequencer.start_sequence(sequence_id, 0.02, 0.01)
        self.assertTrue(success)
        self.assertIn(sequence_id, self.sequencer.active_sequences)

        # Test updating sequence
        self.sequencer.update_sequence(sequence_id, True, 0.015)
        self.assertNotIn(sequence_id, self.sequencer.active_sequences)
        self.assertEqual(len(self.sequencer.sequence_history), 1)

        # Test sequence metrics
        sequence = self.sequencer.sequence_history[0]
        self.assertEqual(sequence.success_rate, 1.0)
        self.assertEqual(sequence.profit_target, 0.02)
        self.assertEqual(sequence.max_drawdown, 0.01)

    def test_retry_logic(self):
        sequence_id = "test_002"
        self.sequencer.start_sequence(sequence_id, 0.02, 0.01)

        # Test retry decision
        should_retry, delay = self.sequencer.should_retry_sequence(sequence_id)
        self.assertTrue(should_retry)
        self.assertGreater(delay, 0)

        # Test max retries
        sequence = self.sequencer.active_sequences[sequence_id]
        sequence.retry_count = self.sequencer.retry_max_attempts
        should_retry, _ = self.sequencer.should_retry_sequence(sequence_id)
        self.assertFalse(should_retry)

    def test_optimal_sequence_params(self):
        # Test with no history
        params = self.sequencer.get_optimal_sequence_params()
        self.assertEqual(
            params['position_size'],
            self.sequencer.base_position_size
        )
        self.assertEqual(params['profit_target'], 0.02)
        self.assertEqual(params['max_drawdown'], 0.01)

        # Add some successful sequences
        for i in range(3):
            sequence_id = f"test_{i}"
            self.sequencer.start_sequence(sequence_id, 0.03, 0.015)
            self.sequencer.update_sequence(sequence_id, True, 0.025)

        # Test with history
        params = self.sequencer.get_optimal_sequence_params()
        self.assertGreater(params['position_size'], 0)
        self.assertGreater(params['profit_target'], 0)
        self.assertGreater(params['max_drawdown'], 0)

    def test_resource_report(self):
        report = self.sequencer.get_resource_report()
        self.assertIn('current_state', report)
        self.assertIn('cpu_usage', report)
        self.assertIn('gpu_usage', report)
        self.assertIn('memory_usage', report)
        self.assertIn('thermal_state', report)
        self.assertIn('active_sequences', report)
        self.assertIn('sequence_success_rate', report)
        self.assertIn('average_thermal_cost', report)


if __name__ == '__main__':
    unittest.main()