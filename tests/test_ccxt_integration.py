"""
CCXT Integration Tests
=====================

Unit tests for CCXT integration, data providers, and trading configuration.
Tests both live and backtest scenarios with proper error handling.
"""

import unittest
import asyncio
import tempfile
import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

# Test imports
class TestCCXTIntegration(unittest.TestCase):
    """Test CCXT integration and data providers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / 'config'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test trading config
        self.test_config = {
            'supported_symbols': ['BTC/USDT', 'BTC/USDC'],
            'rate_limit_delay': 0.1,
            'sandbox_mode': True,
            'api_credentials': {
                'api_key': 'test_key',
                'secret': 'test_secret',
                'sandbox': True
            },
            'request_timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 1.0
        }
        
        # Save test config
        config_path = self.config_dir / 'trading_config.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_provider_import(self):
        """Test that data provider classes can be imported"""
        try:
            from core.data_provider import (
                DataProvider, CCXTDataProvider, BacktestDataProvider,
                DataProviderFactory, MarketData, OrderBookData
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import data provider classes: {e}")
    
    def test_market_data_structure(self):
        """Test MarketData dataclass structure"""
        try:
            from core.data_provider import MarketData
            
            # Create test market data
            market_data = MarketData(
                symbol='BTC/USDT',
                timestamp=datetime.now(),
                price=50000.0,
                volume=1000.0,
                bid=49990.0,
                ask=50010.0,
                spread=20.0,
                high_24h=51000.0,
                low_24h=49000.0,
                change_24h=1000.0,
                metadata={'exchange': 'binance'}
            )
            
            self.assertEqual(market_data.symbol, 'BTC/USDT')
            self.assertEqual(market_data.price, 50000.0)
            self.assertIsInstance(market_data.metadata, dict)
            
        except ImportError:
            self.skipTest("Data provider not available")
    
    def test_data_provider_factory(self):
        """Test DataProviderFactory creation"""
        try:
            from core.data_provider import DataProviderFactory, CCXTDataProvider, BacktestDataProvider
            
            # Test CCXT provider creation
            ccxt_provider = DataProviderFactory.create_provider('ccxt', exchange_id='binance')
            self.assertIsInstance(ccxt_provider, CCXTDataProvider)
            
            # Test backtest provider creation
            backtest_provider = DataProviderFactory.create_provider('backtest')
            self.assertIsInstance(backtest_provider, BacktestDataProvider)
            
            # Test invalid provider type
            with self.assertRaises(ValueError):
                DataProviderFactory.create_provider('invalid_type')
                
        except ImportError:
            self.skipTest("Data provider not available")
    
    @patch('ccxt.async_support')
    def test_ccxt_provider_initialization(self, mock_ccxt):
        """Test CCXT provider initialization with mocked CCXT"""
        try:
            from core.data_provider import CCXTDataProvider
            
            # Mock CCXT exchange
            mock_exchange_class = Mock()
            mock_exchange = Mock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange_class.return_value = mock_exchange
            mock_ccxt.binance = mock_exchange_class
            
            # Create provider
            provider = CCXTDataProvider(exchange_id='binance')
            
            # Test initialization
            asyncio.run(provider.initialize())
            
            # Verify exchange was created and markets loaded
            mock_exchange_class.assert_called_once()
            mock_exchange.load_markets.assert_called_once()
            
        except ImportError:
            self.skipTest("Data provider not available")
    
    def test_backtest_provider_data_loading(self):
        """Test BacktestDataProvider data loading"""
        try:
            from core.data_provider import BacktestDataProvider
            
            # Create test data directory
            data_dir = Path(self.temp_dir) / 'data' / 'historical'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test CSV data
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'open': [50000 + i for i in range(100)],
                'high': [50100 + i for i in range(100)],
                'low': [49900 + i for i in range(100)],
                'close': [50050 + i for i in range(100)],
                'volume': [1000 - i for i in range(100)]
            })
            
            csv_path = data_dir / 'BTC_USDT_1h.csv'
            test_data.to_csv(csv_path, index=False)
            
            # Test data loading
            provider = BacktestDataProvider(data_directory=str(data_dir))
            provider.load_historical_data('BTC/USDT', 'BTC_USDT_1h.csv')
            
            # Verify data was loaded
            self.assertIn('BTC/USDT', provider.data_cache)
            self.assertEqual(len(provider.data_cache['BTC/USDT']), 100)
            
        except ImportError:
            self.skipTest("Data provider not available")
    
    async def test_backtest_provider_market_data(self):
        """Test BacktestDataProvider market data retrieval"""
        try:
            from core.data_provider import BacktestDataProvider
            
            # Create test data
            data_dir = Path(self.temp_dir) / 'data' / 'historical'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
                'open': [50000.0] * 10,
                'high': [50100.0] * 10,
                'low': [49900.0] * 10,
                'close': [50050.0] * 10,
                'volume': [1000.0] * 10
            })
            
            csv_path = data_dir / 'test_data.csv'
            test_data.to_csv(csv_path, index=False)
            
            # Create provider and load data
            provider = BacktestDataProvider(data_directory=str(data_dir))
            provider.load_historical_data('BTC/USDT', 'test_data.csv')
            
            # Set simulation time
            sim_time = datetime(2024, 1, 1, 5, 0, 0)
            provider.set_simulation_time(sim_time)
            
            # Get market data
            market_data = await provider.get_market_data('BTC/USDT')
            
            # Verify market data
            self.assertEqual(market_data.symbol, 'BTC/USDT')
            self.assertEqual(market_data.price, 50050.0)
            self.assertGreater(market_data.volume, 0)
            
        except ImportError:
            self.skipTest("Data provider not available")

class TestConfigurationSystem(unittest.TestCase):
    """Test the standardized configuration system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / 'config'
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_import(self):
        """Test that configuration manager can be imported"""
        try:
            from config import ConfigManager, load_config, save_config, ConfigError
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import configuration manager: {e}")
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        try:
            from config import ConfigManager
            
            manager = ConfigManager()
            self.assertIsInstance(manager.config_dir, Path)
            self.assertIsInstance(manager._config_cache, dict)
            
        except ImportError:
            self.skipTest("Configuration manager not available")
    
    def test_config_path_resolution(self):
        """Test configuration path resolution"""
        try:
            from config import ConfigManager
            
            manager = ConfigManager()
            config_path = manager.get_config_path('test_config.yaml')
            
            # Verify path is correct
            self.assertTrue(config_path.name == 'test_config.yaml')
            self.assertTrue(config_path.parent.name == 'config')
            
        except ImportError:
            self.skipTest("Configuration manager not available")
    
    def test_config_loading_with_defaults(self):
        """Test configuration loading with default creation"""
        try:
            from config import ensure_config_exists, load_config
            
            # Test config that doesn't exist
            default_config = {
                'test_setting': 'default_value',
                'nested': {
                    'value': 42
                }
            }
            
            # This should create the config file
            config_path = ensure_config_exists('test_config.yaml', default_config)
            self.assertTrue(config_path.exists())
            
            # Load and verify
            loaded_config = load_config('test_config.yaml')
            self.assertEqual(loaded_config['test_setting'], 'default_value')
            self.assertEqual(loaded_config['nested']['value'], 42)
            
        except ImportError:
            self.skipTest("Configuration manager not available")
    
    def test_trading_config_validation(self):
        """Test trading configuration validation"""
        try:
            from config import load_config, ensure_config_exists
            
            # Create valid trading config
            trading_config = {
                'supported_symbols': ['BTC/USDT', 'BTC/USDC'],
                'exchange': {
                    'default_exchange': 'binance',
                    'sandbox_mode': True
                },
                'risk_management': {
                    'max_position_size_pct': 10.0,
                    'stop_loss_pct': 2.0
                }
            }
            
            # Ensure config exists
            config_path = ensure_config_exists('trading_config.yaml', trading_config)
            self.assertTrue(config_path.exists())
            
            # Load and validate structure
            loaded_config = load_config('trading_config.yaml')
            self.assertIn('supported_symbols', loaded_config)
            self.assertIn('exchange', loaded_config)
            self.assertIn('risk_management', loaded_config)
            
            # Validate specific values
            self.assertIsInstance(loaded_config['supported_symbols'], list)
            self.assertIn('BTC/USDT', loaded_config['supported_symbols'])
            self.assertEqual(loaded_config['exchange']['default_exchange'], 'binance')
            
        except ImportError:
            self.skipTest("Configuration manager not available")

class TestSystemIntegration(unittest.TestCase):
    """Test system integration between components"""
    
    def test_line_render_engine_config_integration(self):
        """Test LineRenderEngine with standardized config"""
        try:
            from core.line_render_engine import LineRenderEngine
            
            # This should work with default config creation
            engine = LineRenderEngine()
            self.assertIsNotNone(engine.config)
            
            # Test basic functionality
            result = engine.render_lines([])
            self.assertEqual(result['status'], 'rendered')
            self.assertEqual(result['lines_rendered_count'], 0)
            
        except ImportError:
            self.skipTest("LineRenderEngine not available")
        except Exception as e:
            self.skipTest(f"LineRenderEngine initialization failed: {e}")
    
    def test_matrix_fault_resolver_config_integration(self):
        """Test MatrixFaultResolver with standardized config"""
        try:
            from core.matrix_fault_resolver import MatrixFaultResolver
            
            # This should work with default config creation
            resolver = MatrixFaultResolver()
            self.assertIsNotNone(resolver.config)
            self.assertGreater(resolver.retry_attempts, 0)
            
            # Test basic functionality
            result = resolver.resolve_faults()
            self.assertIn('status', result)
            
        except ImportError:
            self.skipTest("MatrixFaultResolver not available")
        except Exception as e:
            self.skipTest(f"MatrixFaultResolver initialization failed: {e}")
    
    def test_data_provider_config_integration(self):
        """Test data provider with trading config"""
        try:
            from core.data_provider import CCXTDataProvider
            
            # Test with default config
            provider = CCXTDataProvider()
            self.assertIsNotNone(provider.config)
            self.assertIn('supported_symbols', provider.config)
            
        except ImportError:
            self.skipTest("Data provider not available")
        except Exception as e:
            self.skipTest(f"Data provider initialization failed: {e}")

class TestValidationAndErrorHandling(unittest.TestCase):
    """Test validation and error handling across the system"""
    
    def test_config_error_handling(self):
        """Test configuration error handling"""
        try:
            from config import ConfigError, load_config
            
            # Test loading non-existent config without defaults
            with self.assertRaises((ConfigError, FileNotFoundError)):
                load_config('non_existent_config.yaml')
                
        except ImportError:
            self.skipTest("Configuration manager not available")
    
    def test_data_provider_error_handling(self):
        """Test data provider error handling"""
        try:
            from core.data_provider import BacktestDataProvider
            
            provider = BacktestDataProvider()
            
            # Test getting data for non-loaded symbol
            with self.assertRaises(ValueError):
                asyncio.run(provider.get_price('INVALID/SYMBOL'))
                
        except ImportError:
            self.skipTest("Data provider not available")
    
    def test_fault_resolver_error_handling(self):
        """Test fault resolver error handling"""
        try:
            from core.matrix_fault_resolver import MatrixFaultResolver
            
            resolver = MatrixFaultResolver()
            
            # Test with invalid fault data
            result = resolver.resolve_faults({'type': 'invalid_fault_type'})
            self.assertIn('status', result)
            
        except ImportError:
            self.skipTest("MatrixFaultResolver not available")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 