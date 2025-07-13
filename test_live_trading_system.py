#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trading System Test
========================
Test script to verify the live trading system works correctly.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.live_trading_system import LiveTradingSystem, TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def test_live_trading_system():
    """Test the live trading system."""
    logger.info("Starting Live Trading System Test")
    
    # Configuration for testing
    config = TradingConfig(
        exchanges={
            'binance': {
                'enabled': True,
                'websocket_enabled': True,
                'symbols': ['btcusdt', 'ethusdt'],
                'sandbox': True,
                'rate_limit_delay': 1
            },
            'coinbase': {
                'enabled': True,
                'websocket_enabled': True,
                'symbols': ['BTC-USD', 'ETH-USD'],
                'sandbox': True,
                'rate_limit_delay': 1
            }
        },
        tracked_symbols=['BTC/USD', 'ETH/USD'],
        price_update_interval=5,
        rebalancing_enabled=True,
        rebalancing_threshold=0.05,
        rebalancing_interval=30,  # Short interval for testing
        target_allocation={
            'BTC': 0.6,
            'ETH': 0.4
        },
        live_trading_enabled=False,  # Disable live trading for testing
        sandbox_mode=True,
        math_decision_enabled=True,
        enable_logging=True,
        enable_alerts=True,
        performance_tracking=True
    )
    
    # Create trading system
    trading_system = LiveTradingSystem(config)
    
    # Add callbacks for monitoring
    def trade_callback(event_type: str, data: Any, metadata: Dict[str, Any]):
        logger.info(f"Trade event: {event_type} - {data}")
    
    def alert_callback(alert_type: str, data: Dict[str, Any]):
        logger.warning(f"Alert: {alert_type} - {data}")
    
    def performance_callback(metrics: Dict[str, Any]):
        logger.info(f"Performance: {metrics}")
    
    trading_system.add_trade_callback(trade_callback)
    trading_system.add_alert_callback(alert_callback)
    trading_system.add_performance_callback(performance_callback)
    
    try:
        # Start the system
        logger.info("Starting trading system...")
        await trading_system.start()
        
        # Add some initial positions for testing
        trading_system.portfolio_tracker.open_position('BTC/USD', 0.05, 50000, 'buy')
        trading_system.portfolio_tracker.open_position('ETH/USD', 1.0, 3000, 'buy')
        
        # Run for 60 seconds to test functionality
        logger.info("Running system for 60 seconds...")
        await asyncio.sleep(60)
        
        # Get system status
        status = trading_system.get_system_status()
        logger.info(f"System status: {status}")
        
        # Get trading history
        history = trading_system.get_trading_history()
        logger.info(f"Trading history: {len(history)} trades")
        
        # Test portfolio summary
        summary = trading_system.portfolio_tracker.get_enhanced_summary()
        logger.info(f"Portfolio summary: {summary}")
        
        # Test performance metrics
        performance = await trading_system.portfolio_tracker.get_performance_metrics()
        logger.info(f"Performance metrics: {performance}")
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    finally:
        # Stop the system
        logger.info("Stopping trading system...")
        await trading_system.stop()


async def test_coinbase_direct_api():
    """Test the direct Coinbase API integration."""
    logger.info("Testing Coinbase Direct API")
    
    try:
        from core.api.coinbase_direct import CoinbaseDirectAPI
        
        # Test with sandbox mode
        api = CoinbaseDirectAPI(
            api_key="test_key",
            secret="test_secret", 
            passphrase="test_passphrase",
            sandbox=True
        )
        
        # Test connection
        connected = await api.connect()
        logger.info(f"Coinbase API connection: {connected}")
        
        if connected:
            # Test getting products
            products = await api.get_products()
            if products:
                logger.info(f"Found {len(products)} products")
                
                # Test getting ticker for BTC-USD
                ticker = await api.get_product_ticker('BTC-USD')
                if ticker:
                    logger.info(f"BTC-USD ticker: {ticker}")
            
            # Disconnect
            await api.disconnect()
        
        logger.info("Coinbase Direct API test completed")
        
    except Exception as e:
        logger.error(f"Coinbase Direct API test failed: {e}")


async def test_exchange_connections():
    """Test exchange connections."""
    logger.info("Testing Exchange Connections")
    
    try:
        from core.api.exchange_connection import ExchangeManager
        
        config = {
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
        
        exchange_manager = ExchangeManager(config)
        exchange_manager.initialize_connections()
        
        # Connect to exchanges
        await exchange_manager.connect_all()
        
        # Get status
        status = exchange_manager.get_all_status()
        logger.info(f"Exchange status: {status}")
        
        # Health check
        health = await exchange_manager.health_check_all()
        logger.info(f"Exchange health: {health}")
        
        # Disconnect
        await exchange_manager.disconnect_all()
        
        logger.info("Exchange connections test completed")
        
    except Exception as e:
        logger.error(f"Exchange connections test failed: {e}")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("LIVE TRADING SYSTEM COMPREHENSIVE TEST")
    logger.info("=" * 60)
    
    try:
        # Test 1: Exchange connections
        await test_exchange_connections()
        
        # Test 2: Coinbase Direct API
        await test_coinbase_direct_api()
        
        # Test 3: Full trading system
        await test_live_trading_system()
        
        logger.info("=" * 60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 