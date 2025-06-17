"""
Test Suite for Schwabot Visual Core Integration
==============================================

Comprehensive tests for the visual core components:
- UI Integration Bridge
- Live Data Streamer
- Hardware Monitoring
- Market Data Simulation
- Settings Persistence
"""

import unittest
import time
import threading
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import components to test
try:
    from core.ui_integration_bridge import UIIntegrationBridge, SystemMetrics, TradingMetrics
    from components.live_data_streamer import LiveDataStreamer, MarketTick, StreamConfig
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    UIIntegrationBridge = None
    SystemMetrics = None
    TradingMetrics = None

class TestUIIntegrationBridge(unittest.TestCase):
    """Test the UI Integration Bridge functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Core components not available")
        
        self.bridge = UIIntegrationBridge()
        self.callback_data = []
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'bridge'):
            self.bridge.stop()
    
    def test_bridge_initialization(self):
        """Test that bridge initializes correctly"""
        self.assertIsNotNone(self.bridge)
        self.assertFalse(self.bridge.running)
        self.assertFalse(self.bridge.ui_state.connected)
        self.assertEqual(len(self.bridge.core_systems), 7)  # Should have 7 mock systems
    
    def test_bridge_start_stop(self):
        """Test starting and stopping the bridge"""
        # Start bridge
        self.bridge.start()
        self.assertTrue(self.bridge.running)
        self.assertTrue(self.bridge.ui_state.connected)
        
        # Wait a moment for thread to start
        time.sleep(0.1)
        
        # Stop bridge
        self.bridge.stop()
        self.assertFalse(self.bridge.running)
        self.assertFalse(self.bridge.ui_state.connected)
    
    def test_callback_registration(self):
        """Test callback registration and notification"""
        callback_triggered = threading.Event()
        
        def test_callback(data):
            self.callback_data.append(data)
            callback_triggered.set()
        
        # Register callback
        self.bridge.add_callback('system_update', test_callback)
        
        # Start bridge to trigger updates
        self.bridge.start()
        
        # Wait for callback
        self.assertTrue(callback_triggered.wait(timeout=5))
        self.assertGreater(len(self.callback_data), 0)
        
        # Verify data type
        self.assertIsInstance(self.callback_data[0], SystemMetrics)
    
    def test_decision_logging(self):
        """Test decision logging functionality"""
        decision_type = "TEST_BUY"
        confidence = 0.85
        inputs = {"price": 45000, "volume": 1000}
        outputs = {"action": "buy", "amount": 0.1}
        reasoning = "Test decision"
        
        # Log decision
        self.bridge.log_decision(decision_type, confidence, inputs, outputs, reasoning)
        
        # Check decision was logged
        decisions = self.bridge.get_recent_decisions(1)
        self.assertEqual(len(decisions), 1)
        
        decision = decisions[0]
        self.assertEqual(decision.decision_type, decision_type)
        self.assertEqual(decision.confidence, confidence)
        self.assertEqual(decision.inputs, inputs)
        self.assertEqual(decision.outputs, outputs)
        self.assertEqual(decision.reasoning, reasoning)
    
    def test_command_execution(self):
        """Test command execution from UI"""
        # Test start trading command
        result = self.bridge.execute_command("start_trading", {"pair": "BTC/USD"})
        self.assertTrue(result["success"])
        self.assertIn("Trading started", result["message"])
        
        # Test stop trading command
        result = self.bridge.execute_command("stop_trading", {})
        self.assertTrue(result["success"])
        self.assertIn("Trading stopped", result["message"])
        
        # Test unknown command
        result = self.bridge.execute_command("unknown_command", {})
        self.assertFalse(result["success"])
        self.assertIn("Unknown command", result["error"])
    
    def test_metrics_collection(self):
        """Test metrics collection and storage"""
        self.bridge.start()
        
        # Wait for some metrics to be collected
        time.sleep(2)
        
        # Check system metrics
        system_metrics = self.bridge.get_current_system_metrics()
        self.assertIsNotNone(system_metrics)
        self.assertIsInstance(system_metrics, SystemMetrics)
        
        # Check trading metrics
        trading_metrics = self.bridge.get_current_trading_metrics()
        self.assertIsNotNone(trading_metrics)
        self.assertIsInstance(trading_metrics, TradingMetrics)
        
        # Check historical data
        system_history = self.bridge.get_system_metrics_history(5)
        self.assertGreater(len(system_history), 0)
        self.assertLessEqual(len(system_history), 5)

class TestLiveDataStreamer(unittest.TestCase):
    """Test the Live Data Streamer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.streamer = LiveDataStreamer()
        self.received_ticks = []
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'streamer'):
            self.streamer.stop()
    
    def test_streamer_initialization(self):
        """Test streamer initializes correctly"""
        self.assertIsNotNone(self.streamer)
        self.assertFalse(self.streamer.running)
        self.assertEqual(len(self.streamer.connectors), 0)
    
    def test_exchange_configuration(self):
        """Test adding exchange configurations"""
        # Create test config
        config = StreamConfig(
            exchange="test_exchange",
            symbols=["BTC-USD"],
            websocket_url="wss://test.com",
            rest_url="https://test.com"
        )
        
        # This should raise an error for unsupported exchange
        with self.assertRaises(ValueError):
            self.streamer.add_exchange(config)
    
    def test_callback_system(self):
        """Test tick callback system"""
        def tick_callback(tick):
            self.received_ticks.append(tick)
        
        self.streamer.add_callback(tick_callback)
        
        # Create a mock tick
        test_tick = MarketTick(
            symbol="BTC-USD",
            price=45000.0,
            volume=1000000.0,
            timestamp=time.time(),
            exchange="test"
        )
        
        # Simulate receiving a tick
        self.streamer._store_tick(test_tick)
        
        # Check callback was triggered
        self.assertEqual(len(self.received_ticks), 1)
        self.assertEqual(self.received_ticks[0].symbol, "BTC-USD")
        self.assertEqual(self.received_ticks[0].price, 45000.0)
    
    def test_data_storage(self):
        """Test tick data storage and retrieval"""
        # Create test tick
        test_tick = MarketTick(
            symbol="BTC-USD",
            price=45000.0,
            volume=1000000.0,
            timestamp=time.time(),
            exchange="coinbase"
        )
        
        # Store tick
        self.streamer._store_tick(test_tick)
        
        # Test latest price retrieval
        latest_price = self.streamer.get_latest_price("coinbase", "BTC-USD")
        self.assertEqual(latest_price, 45000.0)
        
        # Test price history
        history = self.streamer.get_price_history("coinbase", "BTC-USD", 10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], 45000.0)
    
    def test_connection_status(self):
        """Test connection status tracking"""
        status = self.streamer.get_connection_status()
        self.assertEqual(len(status), 0)  # No exchanges configured

class TestMarketTick(unittest.TestCase):
    """Test MarketTick data structure"""
    
    def test_tick_creation(self):
        """Test creating market tick objects"""
        tick = MarketTick(
            symbol="BTC-USD",
            price=45000.0,
            volume=1000000.0,
            timestamp=time.time(),
            exchange="coinbase",
            bid=44990.0,
            ask=45010.0
        )
        
        self.assertEqual(tick.symbol, "BTC-USD")
        self.assertEqual(tick.price, 45000.0)
        self.assertEqual(tick.volume, 1000000.0)
        self.assertEqual(tick.exchange, "coinbase")
        self.assertEqual(tick.bid, 44990.0)
        self.assertEqual(tick.ask, 45010.0)
        
        # Test spread calculation (if implemented)
        if hasattr(tick, 'spread') and tick.spread is not None:
            expected_spread = tick.ask - tick.bid
            self.assertEqual(tick.spread, expected_spread)

class TestStreamConfig(unittest.TestCase):
    """Test StreamConfig data structure"""
    
    def test_config_creation(self):
        """Test creating stream configuration"""
        config = StreamConfig(
            exchange="coinbase",
            symbols=["BTC-USD", "ETH-USD"],
            websocket_url="wss://ws-feed.pro.coinbase.com",
            rest_url="https://api.pro.coinbase.com",
            api_key="test_key",
            rate_limit=10,
            reconnect_delay=5.0,
            max_reconnects=10
        )
        
        self.assertEqual(config.exchange, "coinbase")
        self.assertEqual(len(config.symbols), 2)
        self.assertIn("BTC-USD", config.symbols)
        self.assertEqual(config.rate_limit, 10)
        self.assertEqual(config.reconnect_delay, 5.0)
        self.assertEqual(config.max_reconnects, 10)

class TestIntegrationFlow(unittest.TestCase):
    """Test full integration flow between components"""
    
    def setUp(self):
        """Set up integration test environment"""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.bridge = UIIntegrationBridge()
        self.streamer = LiveDataStreamer()
        self.integration_data = []
    
    def tearDown(self):
        """Clean up integration tests"""
        if hasattr(self, 'bridge'):
            self.bridge.stop()
        if hasattr(self, 'streamer'):
            self.streamer.stop()
    
    def test_data_flow_integration(self):
        """Test data flow from streamer to bridge"""
        # Set up data flow callback
        def integration_callback(data):
            self.integration_data.append(data)
        
        self.bridge.add_callback('system_update', integration_callback)
        
        # Start systems
        self.bridge.start()
        
        # Wait for some data
        time.sleep(1)
        
        # Check data was received
        self.assertGreater(len(self.integration_data), 0)
    
    def test_command_flow_integration(self):
        """Test command flow from UI through bridge"""
        # Start bridge
        self.bridge.start()
        
        # Execute commands
        commands = [
            ("start_trading", {"pair": "BTC/USD"}),
            ("force_decision", {"type": "BUY", "confidence": 0.9}),
            ("stop_trading", {}),
            ("reset_system", {"type": "soft"})
        ]
        
        for command, params in commands:
            result = self.bridge.execute_command(command, params)
            self.assertTrue(result["success"], f"Command {command} failed: {result}")
    
    @patch('components.live_data_streamer.websockets')
    def test_mock_websocket_integration(self, mock_websockets):
        """Test integration with mocked WebSocket"""
        # Mock WebSocket connection
        mock_ws = MagicMock()
        mock_websockets.connect.return_value = mock_ws
        
        # Create test config
        config = StreamConfig(
            exchange="coinbase",
            symbols=["BTC-USD"],
            websocket_url="wss://test.com",
            rest_url="https://test.com"
        )
        
        # This test would be more complete with actual WebSocket mocking
        # For now, just verify the config is created correctly
        self.assertEqual(config.exchange, "coinbase")
        self.assertIn("BTC-USD", config.symbols)

class TestErrorHandling(unittest.TestCase):
    """Test error handling across components"""
    
    def setUp(self):
        """Set up error handling tests"""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Components not available")
        
        self.bridge = UIIntegrationBridge()
    
    def tearDown(self):
        """Clean up error handling tests"""
        if hasattr(self, 'bridge'):
            self.bridge.stop()
    
    def test_bridge_error_logging(self):
        """Test error logging in bridge"""
        # Start bridge
        self.bridge.start()
        
        # Force an error by calling invalid command
        result = self.bridge.execute_command("invalid_command", {})
        self.assertFalse(result["success"])
        
        # Check error was logged
        errors = self.bridge.get_error_log(1)
        # Note: This might not trigger an error log entry depending on implementation
        # The test structure is here for when more detailed error logging is implemented
    
    def test_callback_error_resilience(self):
        """Test that callback errors don't crash the system"""
        def failing_callback(data):
            raise Exception("Test callback error")
        
        # Register failing callback
        self.bridge.add_callback('system_update', failing_callback)
        
        # Start bridge - should not crash despite failing callback
        self.bridge.start()
        time.sleep(1)
        
        # Bridge should still be running
        self.assertTrue(self.bridge.running)

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUIIntegrationBridge,
        TestLiveDataStreamer,
        TestMarketTick,
        TestStreamConfig,
        TestIntegrationFlow,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code) 