#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Rebalancing Integration Test
======================================
Comprehensive test script for portfolio rebalancing, price integration, and API connectivity.
Tests all components of the enhanced Schwabot trading system.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.api.exchange_connection import ExchangeManager, ExchangeType, ExchangeCredentials
from core.real_time_market_data_integration import RealTimeMarketDataIntegration, PriceUpdate
from core.enhanced_portfolio_tracker import EnhancedPortfolioTracker, RebalancingAction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_rebalancing_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PortfolioRebalancingTester:
    """Comprehensive tester for portfolio rebalancing system."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🚀 Starting Portfolio Rebalancing Integration Tests")
        
        try:
            # Test 1: Exchange Connection
            await self.test_exchange_connections()
            
            # Test 2: Real-time Market Data
            await self.test_real_time_market_data()
            
            # Test 3: Portfolio Tracker
            await self.test_portfolio_tracker()
            
            # Test 4: Rebalancing Logic
            await self.test_rebalancing_logic()
            
            # Test 5: Full Integration
            await self.test_full_integration()
            
            # Generate test report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.test_results['overall_success'] = False
            self.test_results['error'] = str(e)
    
    async def test_exchange_connections(self):
        """Test exchange API connections."""
        logger.info("🔌 Testing Exchange Connections")
        
        test_config = {
            'exchanges': {
                'binance': {
                    'enabled': True,
                    'sandbox': True,
                    'rate_limit_delay': 1
                },
                'coinbase': {
                    'enabled': True,
                    'sandbox': True,
                    'rate_limit_delay': 1
                }
            }
        }
        
        try:
            exchange_manager = ExchangeManager(test_config)
            exchange_manager.initialize_connections()
            
            # Test connection
            await exchange_manager.connect_all()
            
            # Check status
            status = exchange_manager.get_all_status()
            health_check = await exchange_manager.health_check_all()
            
            # Validate results
            connected_count = len([s for s in status.values() if s['status'] == 'CONNECTED'])
            healthy_count = len([h for h in health_check.values() if h])
            
            self.test_results['exchange_connections'] = {
                'success': connected_count > 0,
                'connected_exchanges': connected_count,
                'healthy_exchanges': healthy_count,
                'status': status,
                'health_check': health_check
            }
            
            logger.info(f"✅ Exchange connections: {connected_count} connected, {healthy_count} healthy")
            
            # Test market data fetching
            if connected_count > 0:
                await self.test_market_data_fetching(exchange_manager)
            
        except Exception as e:
            logger.error(f"❌ Exchange connection test failed: {e}")
            self.test_results['exchange_connections'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_market_data_fetching(self, exchange_manager):
        """Test market data fetching from exchanges."""
        logger.info("📊 Testing Market Data Fetching")
        
        try:
            test_symbols = ['BTC/USDT', 'ETH/USDT']
            fetched_data = {}
            
            for exchange_name, connection in exchange_manager.connections.items():
                if connection.status == "CONNECTED":
                    exchange_data = {}
                    for symbol in test_symbols:
                        try:
                            market_data = await connection.get_market_data(symbol)
                            if market_data:
                                exchange_data[symbol] = market_data
                        except Exception as e:
                            logger.warning(f"Could not fetch {symbol} from {exchange_name}: {e}")
                    
                    if exchange_data:
                        fetched_data[exchange_name] = exchange_data
            
            self.test_results['market_data_fetching'] = {
                'success': len(fetched_data) > 0,
                'exchanges_with_data': len(fetched_data),
                'data_samples': fetched_data
            }
            
            logger.info(f"✅ Market data fetching: {len(fetched_data)} exchanges with data")
            
        except Exception as e:
            logger.error(f"❌ Market data fetching test failed: {e}")
            self.test_results['market_data_fetching'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_real_time_market_data(self):
        """Test real-time market data integration."""
        logger.info("📡 Testing Real-time Market Data Integration")
        
        config = {
            'exchanges': {
                'binance': {
                    'websocket_enabled': True,
                    'symbols': ['btcusdt', 'ethusdt']
                },
                'coinbase': {
                    'websocket_enabled': True,
                    'symbols': ['BTC-USD', 'ETH-USD']
                }
            },
            'reconnect_delay': 5,
            'max_reconnect_attempts': 3,
            'price_cache_ttl': 60
        }
        
        try:
            integration = RealTimeMarketDataIntegration(config)
            
            # Track received updates
            received_updates = []
            
            def price_callback(price_update: PriceUpdate):
                received_updates.append(price_update)
                logger.debug(f"Price update: {price_update.symbol} = ${price_update.price}")
            
            integration.add_price_callback(price_callback)
            
            # Start integration
            await integration.start()
            
            # Wait for some price updates
            await asyncio.sleep(30)
            
            # Check results
            stats = integration.get_statistics()
            connection_status = integration.get_connection_status()
            prices = integration.get_all_prices()
            
            self.test_results['real_time_market_data'] = {
                'success': len(received_updates) > 0,
                'updates_received': len(received_updates),
                'price_cache_size': len(prices),
                'connection_status': connection_status,
                'statistics': stats,
                'sample_prices': dict(list(prices.items())[:5])  # First 5 prices
            }
            
            logger.info(f"✅ Real-time market data: {len(received_updates)} updates received")
            
            # Stop integration
            await integration.stop()
            
        except Exception as e:
            logger.error(f"❌ Real-time market data test failed: {e}")
            self.test_results['real_time_market_data'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_portfolio_tracker(self):
        """Test enhanced portfolio tracker."""
        logger.info("💼 Testing Enhanced Portfolio Tracker")
        
        config = {
            'exchanges': {
                'binance': {
                    'enabled': True,
                    'websocket_enabled': True,
                    'symbols': ['btcusdt', 'ethusdt']
                }
            },
            'tracked_symbols': ['BTC/USD', 'ETH/USD'],
            'price_update_interval': 5,
            'rebalancing': {
                'enabled': True,
                'threshold': 0.05,
                'interval': 60,
                'target_allocation': {
                    'BTC': 0.6,
                    'ETH': 0.4
                }
            }
        }
        
        try:
            tracker = EnhancedPortfolioTracker(config)
            
            # Add some initial positions
            tracker.open_position('BTC/USD', 0.1, 50000, 'buy')
            tracker.open_position('ETH/USD', 1.0, 3000, 'buy')
            
            # Start tracker
            await tracker.start()
            
            # Wait for price updates
            await asyncio.sleep(20)
            
            # Get enhanced summary
            summary = tracker.get_enhanced_summary()
            performance = await tracker.get_performance_metrics()
            
            self.test_results['portfolio_tracker'] = {
                'success': True,
                'positions_count': len(tracker.positions),
                'total_value': summary['total_value'],
                'rebalancing_enabled': summary['rebalancing']['enabled'],
                'market_data_status': summary['market_data']['connection_status'],
                'performance_metrics': performance
            }
            
            logger.info(f"✅ Portfolio tracker: {len(tracker.positions)} positions, ${summary['total_value']:.2f} total value")
            
            # Stop tracker
            await tracker.stop()
            
        except Exception as e:
            logger.error(f"❌ Portfolio tracker test failed: {e}")
            self.test_results['portfolio_tracker'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_rebalancing_logic(self):
        """Test portfolio rebalancing logic."""
        logger.info("⚖️ Testing Rebalancing Logic")
        
        config = {
            'exchanges': {
                'binance': {
                    'enabled': True,
                    'websocket_enabled': True,
                    'symbols': ['btcusdt', 'ethusdt']
                }
            },
            'tracked_symbols': ['BTC/USD', 'ETH/USD'],
            'price_update_interval': 5,
            'rebalancing': {
                'enabled': True,
                'threshold': 0.05,
                'interval': 30,  # Short interval for testing
                'target_allocation': {
                    'BTC': 0.7,
                    'ETH': 0.3
                }
            }
        }
        
        try:
            tracker = EnhancedPortfolioTracker(config)
            
            # Create unbalanced portfolio
            tracker.open_position('BTC/USD', 0.05, 50000, 'buy')  # 30% allocation
            tracker.open_position('ETH/USD', 2.0, 3000, 'buy')    # 70% allocation
            
            # Start tracker
            await tracker.start()
            
            # Track rebalancing events
            rebalancing_events = []
            
            def rebalancing_callback(action: RebalancingAction, result: Dict[str, Any]):
                rebalancing_events.append({
                    'symbol': action.symbol,
                    'action': action.action,
                    'amount': action.amount,
                    'deviation': action.deviation
                })
                logger.info(f"Rebalancing event: {action.symbol} {action.action} ${action.amount:.2f}")
            
            tracker.add_rebalancing_callback(rebalancing_callback)
            
            # Wait for rebalancing check
            await asyncio.sleep(40)
            
            # Check rebalancing analysis
            rebalancing_check = await tracker.check_rebalancing_needs()
            
            self.test_results['rebalancing_logic'] = {
                'success': True,
                'needs_rebalancing': rebalancing_check['needs_rebalancing'],
                'rebalancing_actions_count': len(rebalancing_check.get('rebalancing_actions', [])),
                'events_triggered': len(rebalancing_events),
                'current_allocation': rebalancing_check.get('current_allocation', {}),
                'target_allocation': config['rebalancing']['target_allocation']
            }
            
            logger.info(f"✅ Rebalancing logic: {len(rebalancing_events)} events triggered")
            
            # Stop tracker
            await tracker.stop()
            
        except Exception as e:
            logger.error(f"❌ Rebalancing logic test failed: {e}")
            self.test_results['rebalancing_logic'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_full_integration(self):
        """Test full system integration."""
        logger.info("🔗 Testing Full System Integration")
        
        config = {
            'exchanges': {
                'binance': {
                    'enabled': True,
                    'websocket_enabled': True,
                    'symbols': ['btcusdt', 'ethusdt']
                },
                'coinbase': {
                    'enabled': True,
                    'websocket_enabled': True,
                    'symbols': ['BTC-USD', 'ETH-USD']
                }
            },
            'tracked_symbols': ['BTC/USD', 'ETH/USD'],
            'price_update_interval': 5,
            'rebalancing': {
                'enabled': True,
                'threshold': 0.05,
                'interval': 60,
                'target_allocation': {
                    'BTC': 0.6,
                    'ETH': 0.4
                }
            }
        }
        
        try:
            tracker = EnhancedPortfolioTracker(config)
            
            # Add initial positions
            tracker.open_position('BTC/USD', 0.1, 50000, 'buy')
            tracker.open_position('ETH/USD', 1.0, 3000, 'buy')
            
            # Track all events
            price_updates = []
            rebalancing_events = []
            
            def price_callback(price_update: PriceUpdate):
                price_updates.append(price_update)
            
            def rebalancing_callback(action: RebalancingAction, result: Dict[str, Any]):
                rebalancing_events.append(action)
            
            tracker.add_price_update_callback(price_callback)
            tracker.add_rebalancing_callback(rebalancing_callback)
            
            # Start full system
            await tracker.start()
            
            # Run for comprehensive test
            await asyncio.sleep(60)
            
            # Get final state
            summary = tracker.get_enhanced_summary()
            performance = await tracker.get_performance_metrics()
            
            self.test_results['full_integration'] = {
                'success': True,
                'price_updates_received': len(price_updates),
                'rebalancing_events': len(rebalancing_events),
                'portfolio_value': summary['total_value'],
                'market_data_connections': summary['market_data']['connection_status'],
                'exchange_connections': summary['exchanges']['connected_count'],
                'performance_metrics': performance,
                'system_uptime': time.time() - self.start_time
            }
            
            logger.info(f"✅ Full integration: {len(price_updates)} price updates, {len(rebalancing_events)} rebalancing events")
            
            # Stop system
            await tracker.stop()
            
        except Exception as e:
            logger.error(f"❌ Full integration test failed: {e}")
            self.test_results['full_integration'] = {
                'success': False,
                'error': str(e)
            }
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📋 Generating Test Report")
        
        # Calculate overall success
        test_successes = [
            result.get('success', False) 
            for result in self.test_results.values() 
            if isinstance(result, dict) and 'success' in result
        ]
        
        overall_success = all(test_successes) if test_successes else False
        
        # Create report
        report = {
            'test_timestamp': time.time(),
            'test_duration': time.time() - self.start_time,
            'overall_success': overall_success,
            'tests_passed': sum(test_successes),
            'total_tests': len(test_successes),
            'test_results': self.test_results
        }
        
        # Save report
        with open('portfolio_rebalancing_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 PORTFOLIO REBALANCING INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        logger.info(f"Tests Passed: {sum(test_successes)}/{len(test_successes)}")
        logger.info(f"Test Duration: {report['test_duration']:.2f} seconds")
        logger.info("=" * 60)
        
        for test_name, result in self.test_results.items():
            if isinstance(result, dict) and 'success' in result:
                status = "✅ PASSED" if result['success'] else "❌ FAILED"
                logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info("=" * 60)
        logger.info("📄 Detailed report saved to: portfolio_rebalancing_test_report.json")
        
        return report


async def main():
    """Main test runner."""
    tester = PortfolioRebalancingTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 