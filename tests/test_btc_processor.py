from unittest.mock import patch

"""
BTC Data Processor Tests
======================

Test suite for BTC data processing and hash generation functionality.
"""

import unittest  # noqa: F401
import asyncio  # noqa: F401
import json  # noqa: F401
import torch  # noqa: F401
from datetime import datetime  # noqa: F821
from unittest.mock import Mock, patch, AsyncMock  # noqa: F401

from core.btc_data_processor import BTCDataProcessor  # noqa: F401


class TestBTCDataProcessor(unittest.TestCase):
    """Test BTC data processor functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = {
            'websocket': {'uri': 'ws://test'},
            'buffer_size': 10,
            'hash_buffer_size': 5,
            'entropy_threshold': 0.7
        }
        self.processor = BTCDataProcessor()
        self.processor.config = self.config

    def tearDown(self):
        """Clean up test environment"""
        asyncio.run(self.processor.shutdown())

    @patch('websockets.connect')
    async def test_websocket_connection(self, mock_connect):
        """Test WebSocket connection handling"""
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws

        # Simulate incoming data
        test_data = {
            'price': '50000.0',
            'volume': '100.0',
            'timestamp': datetime.now().isoformat()  # noqa: F821
        }
        mock_ws.recv.return_value = json.dumps(test_data)

        # Start processing pipeline
        pipeline_task = asyncio.create_task(
            self.processor.start_processing_pipeline())

        # Wait for data processing
        await asyncio.sleep(0.1)

        # Verify data was processed
        latest_data = self.processor.get_latest_data()
        self.assertIsNotNone(latest_data['price_data'])
        self.assertEqual(float(latest_data['price_data']['price']), 50000.0)

        # Clean up
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass

    @patch('torch.cuda.is_available')
    def test_gpu_setup(self, mock_cuda_available):
        """Test GPU setup and fallback"""
        # Test GPU available
        mock_cuda_available.return_value = True
        processor = BTCDataProcessor()
        self.assertTrue(processor.use_gpu)
        self.assertEqual(processor.device.type, 'cuda')

        # Test GPU unavailable
        mock_cuda_available.return_value = False
        processor = BTCDataProcessor()
        self.assertFalse(processor.use_gpu)
        self.assertEqual(processor.device.type, 'cpu')

    async def test_hash_generation(self):
        """Test hash generation functionality"""
        # Create test data
        test_data = {
            'price': 50000.0,
            'volume': 100.0,
            'timestamp': datetime.now().isoformat()  # noqa: F821
        }

        # Generate hash
        if self.processor.use_gpu:
            hash_value = await self.processor._generate_hash_gpu(test_data)
        else:
            hash_value = await self.processor._generate_hash_cpu(test_data)

        # Verify hash
        self.assertIsNotNone(hash_value)
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA-256 hash length

    async def test_entropy_validation(self):
        """Test entropy validation"""
        # Create test hash
        test_hash = '0' * 64  # Low entropy hash

        # Validate entropy
        entropy_value = await self.processor.entropy_engine.calculate_entropy(test_hash)

        # Verify entropy
        self.assertLess(entropy_value, self.config['entropy_threshold'])

    async def test_price_correlations(self):
        """Test price correlation calculations"""
        # Create test data
        test_data = [
            {'price': 50000.0, 'volume': 100.0},
            {'price': 51000.0, 'volume': 110.0},
            {'price': 52000.0, 'volume': 120.0}
        ]

        for data in test_data:
            self.processor.data_buffer.append(data)

        # Calculate correlations
        correlations = await self.processor._calculate_price_correlations()

        # Verify correlations
        self.assertIn('price_volume', correlations)
        self.assertIn('price_entropy', correlations)
        self.assertGreater(correlations['price_volume'], 0.0)

    def test_buffer_management(self):
        """Test buffer size management"""
        # Fill buffer beyond limit
        for i in range(self.config['buffer_size'] + 5):
            self.processor.data_buffer.append({
                'price': float(i),
                'volume': float(i),
                'timestamp': datetime.now().isoformat()  # noqa: F821
            })

        # Verify buffer size
        self.assertEqual(
            len(self.processor.data_buffer),
            self.config['buffer_size']
        )

    @patch('torch.cuda.Stream')
    async def test_gpu_stream_management(self, mock_stream):
        """Test GPU stream management"""
        if not self.processor.use_gpu:
            self.skipTest("GPU not available")

        # Verify streams were created
        self.assertEqual(len(self.processor.streams), 3)

        # Test stream usage
        with torch.cuda.stream(self.processor.streams[0]):
            tensor = torch.tensor(
                [1.0,
                    2.0,
                    3.0],
                device=self.processor.device
            )
            _ = tensor * 2  # noqa: F841

        self.assertTrue(torch.all(result == torch.tensor(
            [2.0, 4.0, 6.0], device=self.processor.device)))

    async def test_error_handling(self):
        """Test error handling in processing pipeline"""
        # Simulate processing error
        with patch.object(
            self.processor,
                '_process_price_data',
                side_effect=Exception("Test error")
        ):
            # Add data to queue
            await self.processor.processing_queue.put({'price': '50000.0'})

            # Verify error was handled
            await asyncio.sleep(0.1)
            self.assertEqual(len(self.processor.data_buffer), 0)

    def test_performance_metrics(self):
        """Test performance monitoring"""
        # Verify performance monitoring is enabled
        self.assertTrue(self.processor.config['performance']['enabled'])

        # Check metrics collection
        metrics = self.processor._collect_performance_metrics()
        self.assertIn('processing_time', metrics)
        self.assertIn('hash_generation_time', metrics)
        self.assertIn('correlation_calculation_time', metrics)


if __name__ == '__main__':
    unittest.main()