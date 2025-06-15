"""
CCXT Trading Integration Demo
============================

Comprehensive demonstration of CCXT integration for BTC/USDC trading
including live data fetching, backtesting, and system validation.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demonstrate_config_system():
    """Demonstrate the standardized configuration system"""
    print("\n🔧 Configuration System Demonstration")
    print("=" * 50)
    
    try:
        from config import load_config, ensure_config_exists, ConfigError
        
        print("✅ Configuration system imported successfully")
        
        # Test trading config creation
        trading_config = {
            'supported_symbols': ['BTC/USDT', 'BTC/USDC', 'ETH/USDT'],
            'exchange': {
                'default_exchange': 'binance',
                'sandbox_mode': True,
                'rate_limit_delay': 0.1
            },
            'risk_management': {
                'max_position_size_pct': 10.0,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 5.0
            },
            'primary_pairs': {
                'btc_usdc': 'BTC/USDC',
                'btc_usdt': 'BTC/USDT'
            }
        }
        
        # Ensure config exists
        config_path = ensure_config_exists('trading_config.yaml', trading_config)
        print(f"📁 Trading config ensured at: {config_path}")
        
        # Load and validate
        loaded_config = load_config('trading_config.yaml')
        print(f"✅ Config loaded successfully")
        print(f"  Supported symbols: {loaded_config['supported_symbols']}")
        print(f"  Primary BTC/USDC pair: {loaded_config['primary_pairs']['btc_usdc']}")
        print(f"  Exchange: {loaded_config['exchange']['default_exchange']}")
        print(f"  Sandbox mode: {loaded_config['exchange']['sandbox_mode']}")
        
    except ImportError as e:
        print(f"❌ Configuration system not available: {e}")
    except Exception as e:
        print(f"❌ Configuration demo error: {e}")

async def demonstrate_data_providers():
    """Demonstrate data provider functionality"""
    print("\n📊 Data Provider Demonstration")
    print("=" * 50)
    
    try:
        from core.data_provider import (
            DataProviderFactory, CCXTDataProvider, BacktestDataProvider,
            MarketData, OrderBookData
        )
        
        print("✅ Data providers imported successfully")
        
        # Test data provider factory
        print(f"\n🏭 Testing Data Provider Factory:")
        
        # Create CCXT provider (without actual connection)
        ccxt_provider = DataProviderFactory.create_provider('ccxt', exchange_id='binance')
        print(f"  ✅ CCXT provider created: {type(ccxt_provider).__name__}")
        print(f"  Exchange ID: {ccxt_provider.exchange_id}")
        print(f"  Supported symbols: {ccxt_provider.supported_symbols}")
        
        # Create backtest provider
        backtest_provider = DataProviderFactory.create_provider('backtest')
        print(f"  ✅ Backtest provider created: {type(backtest_provider).__name__}")
        
        # Test MarketData structure
        print(f"\n📈 Testing MarketData Structure:")
        sample_market_data = MarketData(
            symbol='BTC/USDC',
            timestamp=datetime.now(),
            price=45000.0,
            volume=1500.0,
            bid=44990.0,
            ask=45010.0,
            spread=20.0,
            high_24h=46000.0,
            low_24h=44000.0,
            change_24h=500.0,
            metadata={'exchange': 'binance', 'source': 'demo'}
        )
        
        print(f"  Symbol: {sample_market_data.symbol}")
        print(f"  Price: ${sample_market_data.price:,.2f}")
        print(f"  24h Change: ${sample_market_data.change_24h:,.2f}")
        print(f"  Spread: ${sample_market_data.spread:.2f}")
        print(f"  Volume: {sample_market_data.volume:,.2f}")
        
    except ImportError as e:
        print(f"❌ Data providers not available: {e}")
    except Exception as e:
        print(f"❌ Data provider demo error: {e}")

async def demonstrate_backtest_provider():
    """Demonstrate backtesting data provider"""
    print("\n🔄 Backtest Provider Demonstration")
    print("=" * 50)
    
    try:
        from core.data_provider import BacktestDataProvider
        
        # Create test data directory
        data_dir = Path('data/historical')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample BTC/USDC data
        print("📊 Generating sample BTC/USDC historical data...")
        
        # Create realistic price data with some volatility
        dates = pd.date_range('2024-01-01', periods=168, freq='1H')  # 1 week of hourly data
        base_price = 45000.0
        
        # Generate price movements with some trend and volatility
        price_changes = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Add some trend (slight upward bias)
            trend = 0.001 * (i / len(dates))
            # Add volatility
            volatility = 0.02 * (2 * pd.np.random.random() - 1)
            # Add some mean reversion
            mean_reversion = -0.001 * (current_price - base_price) / base_price
            
            change = trend + volatility + mean_reversion
            current_price *= (1 + change)
            price_changes.append(current_price)
        
        # Create OHLCV data
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': price_changes,
            'high': [p * (1 + abs(pd.np.random.normal(0, 0.005))) for p in price_changes],
            'low': [p * (1 - abs(pd.np.random.normal(0, 0.005))) for p in price_changes],
            'close': price_changes,
            'volume': [1000 + pd.np.random.normal(0, 200) for _ in price_changes]
        })
        
        # Ensure high >= close >= low
        test_data['high'] = test_data[['high', 'close']].max(axis=1)
        test_data['low'] = test_data[['low', 'close']].min(axis=1)
        
        # Save test data
        csv_path = data_dir / 'BTC_USDC_1h.csv'
        test_data.to_csv(csv_path, index=False)
        print(f"  ✅ Sample data saved to: {csv_path}")
        print(f"  Data points: {len(test_data)}")
        print(f"  Date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
        print(f"  Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
        
        # Test backtest provider
        print(f"\n🧪 Testing Backtest Provider:")
        provider = BacktestDataProvider(data_directory=str(data_dir))
        
        # Load the data
        provider.load_historical_data('BTC/USDC', 'BTC_USDC_1h.csv')
        print(f"  ✅ Historical data loaded for BTC/USDC")
        
        # Set simulation time to middle of dataset
        sim_time = dates[len(dates)//2]
        provider.set_simulation_time(sim_time)
        print(f"  🕐 Simulation time set to: {sim_time}")
        
        # Get market data at simulation time
        market_data = await provider.get_market_data('BTC/USDC')
        print(f"  📊 Market data retrieved:")
        print(f"    Price: ${market_data.price:,.2f}")
        print(f"    Volume: {market_data.volume:,.2f}")
        print(f"    Spread: ${market_data.spread:.2f}")
        print(f"    24h Change: ${market_data.change_24h:,.2f}")
        
        # Get order book simulation
        order_book = await provider.get_order_book('BTC/USDC', limit=20)
        print(f"  📋 Order book simulated:")
        print(f"    Bid depth: {order_book.depth['bid_depth']:.2f}")
        print(f"    Ask depth: {order_book.depth['ask_depth']:.2f}")
        print(f"    Top bid: ${order_book.bids[0][0]:.2f}")
        print(f"    Top ask: ${order_book.asks[0][0]:.2f}")
        
        # Get historical data range
        historical_data = await provider.get_historical_data(
            'BTC/USDC', '1h', 
            sim_time - timedelta(hours=24), 
            limit=24
        )
        print(f"  📈 Historical data (24h): {len(historical_data)} records")
        print(f"    Price change: ${historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]:.2f}")
        
    except ImportError as e:
        print(f"❌ Backtest provider not available: {e}")
    except Exception as e:
        print(f"❌ Backtest demo error: {e}")

async def demonstrate_ccxt_provider():
    """Demonstrate CCXT provider (without actual connection)"""
    print("\n🌐 CCXT Provider Demonstration")
    print("=" * 50)
    
    try:
        from core.data_provider import CCXTDataProvider
        
        print("📡 Testing CCXT Provider Configuration:")
        
        # Create provider with configuration
        provider = CCXTDataProvider(exchange_id='binance')
        print(f"  ✅ CCXT provider created")
        print(f"  Exchange: {provider.exchange_id}")
        print(f"  Supported symbols: {provider.supported_symbols}")
        print(f"  Rate limit delay: {provider.rate_limit_delay}s")
        
        # Show configuration
        config = provider.config
        print(f"\n⚙️  Provider Configuration:")
        print(f"  Sandbox mode: {config.get('sandbox_mode', 'Not set')}")
        print(f"  Request timeout: {config.get('request_timeout', 'Not set')}s")
        print(f"  Retry attempts: {config.get('retry_attempts', 'Not set')}")
        
        # Test connection status (without actual connection)
        is_connected = await provider.is_connected()
        print(f"  Connection status: {'Connected' if is_connected else 'Not connected'}")
        
        print(f"\n💡 Note: Actual CCXT connection requires:")
        print(f"  1. Valid API credentials in trading_config.yaml")
        print(f"  2. CCXT library installed: pip install ccxt")
        print(f"  3. Network connectivity to exchange")
        print(f"  4. Call to provider.initialize() method")
        
    except ImportError as e:
        print(f"❌ CCXT provider not available: {e}")
        print(f"💡 Install CCXT: pip install ccxt")
    except Exception as e:
        print(f"❌ CCXT demo error: {e}")

async def demonstrate_system_integration():
    """Demonstrate system integration with trading components"""
    print("\n🔗 System Integration Demonstration")
    print("=" * 50)
    
    try:
        from core.line_render_engine import LineRenderEngine
        from core.matrix_fault_resolver import MatrixFaultResolver
        from core.data_provider import BacktestDataProvider
        
        print("🚀 Testing Integrated Trading System:")
        
        # Initialize components
        print(f"\n1️⃣  Initializing Components:")
        render_engine = LineRenderEngine()
        fault_resolver = MatrixFaultResolver()
        data_provider = BacktestDataProvider()
        
        print(f"  ✅ LineRenderEngine initialized")
        print(f"  ✅ MatrixFaultResolver initialized")
        print(f"  ✅ BacktestDataProvider initialized")
        
        # Simulate trading workflow
        print(f"\n2️⃣  Simulating Trading Workflow:")
        
        # Step 1: Get market data (simulated)
        print(f"  📊 Fetching market data...")
        market_data = {
            'symbol': 'BTC/USDC',
            'price': 45000.0,
            'volume': 1500.0,
            'timestamp': datetime.now().isoformat()
        }
        print(f"    Current BTC/USDC price: ${market_data['price']:,.2f}")
        
        # Step 2: Process with render engine
        print(f"  🎨 Processing with render engine...")
        line_data = [
            {
                'path': [44000, 44500, 45000, 45200, 45000],
                'profit': 0.5,
                'entropy': 0.3,
                'type': 'btc_usdc_trend',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        render_result = render_engine.render_lines(line_data)
        print(f"    ✅ Rendered {render_result['lines_rendered_count']} lines")
        print(f"    System load: Memory {render_result['system_metrics']['memory_percent']:.1f}%")
        
        # Step 3: Test fault resolution
        print(f"  🔧 Testing fault resolution...")
        fault_data = {
            'type': 'network_timeout',
            'severity': 'medium',
            'context': {'symbol': 'BTC/USDC', 'operation': 'price_fetch'}
        }
        
        fault_result = fault_resolver.resolve_faults(fault_data)
        print(f"    ✅ Fault resolved: {fault_result['status']}")
        print(f"    Method: {fault_result['method']}")
        
        # Step 4: Get system statistics
        print(f"\n3️⃣  System Statistics:")
        
        render_stats = render_engine.get_line_statistics()
        fault_stats = fault_resolver.get_fault_statistics()
        
        print(f"  Render Engine:")
        print(f"    Total lines: {render_stats['total_lines']}")
        print(f"    Active lines: {render_stats['active_lines']}")
        print(f"    Average score: {render_stats['average_score']:.3f}")
        
        print(f"  Fault Resolver:")
        print(f"    Total resolutions: {fault_stats['total_resolutions']}")
        print(f"    Error rate: {fault_stats['error_rate']:.1%}")
        print(f"    Average resolution time: {fault_stats['average_resolution_time']:.3f}s")
        
        print(f"\n✅ System integration test completed successfully!")
        
    except ImportError as e:
        print(f"❌ System components not available: {e}")
    except Exception as e:
        print(f"❌ System integration error: {e}")

async def demonstrate_validation_pipeline():
    """Demonstrate validation and error handling pipeline"""
    print("\n🛡️  Validation Pipeline Demonstration")
    print("=" * 50)
    
    try:
        from core.matrix_fault_resolver import MatrixFaultResolver
        from config import ConfigError
        
        print("🔍 Testing Validation and Error Handling:")
        
        resolver = MatrixFaultResolver()
        
        # Test various fault scenarios
        fault_scenarios = [
            {
                'type': 'config_error',
                'severity': 'high',
                'context': {'config_file': 'trading_config.yaml'}
            },
            {
                'type': 'validation_error', 
                'severity': 'medium',
                'context': {'field': 'symbol', 'value': 'INVALID/PAIR'}
            },
            {
                'type': 'data_corruption',
                'severity': 'high',
                'context': {'data_source': 'price_feed'}
            }
        ]
        
        print(f"\n📋 Testing Fault Resolution Scenarios:")
        for i, scenario in enumerate(fault_scenarios, 1):
            print(f"\n  Scenario {i}: {scenario['type']} ({scenario['severity']})")
            result = resolver.resolve_faults(scenario)
            print(f"    Status: {result['status']}")
            print(f"    Method: {result['method']}")
            print(f"    Action: {result.get('action_taken', 'N/A')}")
        
        # Test system health validation
        print(f"\n🏥 System Health Validation:")
        health_status = resolver.validate_system_health()
        print(f"  Overall status: {health_status['status']}")
        print(f"  Issues found: {len(health_status['issues'])}")
        if health_status['issues']:
            for issue in health_status['issues']:
                print(f"    ⚠️  {issue}")
        if health_status['recommendations']:
            print(f"  Recommendations:")
            for rec in health_status['recommendations']:
                print(f"    💡 {rec}")
        
        print(f"\n✅ Validation pipeline test completed!")
        
    except ImportError as e:
        print(f"❌ Validation components not available: {e}")
    except Exception as e:
        print(f"❌ Validation demo error: {e}")

async def main():
    """Run all demonstrations"""
    print("CCXT Trading Integration Demo")
    print("============================")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all demonstrations
    await demonstrate_config_system()
    await demonstrate_data_providers()
    await demonstrate_backtest_provider()
    await demonstrate_ccxt_provider()
    await demonstrate_system_integration()
    await demonstrate_validation_pipeline()
    
    print(f"\n🎉 All Demonstrations Complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📋 Summary of CCXT Integration Features:")
    print(f"  ✅ Standardized configuration management with YAML validation")
    print(f"  ✅ Data provider abstraction for live and backtest scenarios")
    print(f"  ✅ BTC/USDC trading pair support with fallback to BTC/USDT")
    print(f"  ✅ Comprehensive error handling and fault resolution")
    print(f"  ✅ System health monitoring and validation")
    print(f"  ✅ Integration with existing render engine and fault resolver")
    print(f"  ✅ Async/await support for high-performance trading")
    print(f"  ✅ Backtesting capabilities with historical data")
    
    print(f"\n🚀 Next Steps for Live Trading:")
    print(f"  1. Install CCXT: pip install ccxt")
    print(f"  2. Configure API credentials in config/trading_config.yaml")
    print(f"  3. Test with sandbox/testnet first")
    print(f"  4. Implement risk management rules")
    print(f"  5. Set up monitoring and alerting")
    print(f"  6. Start with small position sizes")

if __name__ == "__main__":
    asyncio.run(main()) 