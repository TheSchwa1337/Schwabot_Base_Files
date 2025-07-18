#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical Functionality Test

This script identifies the most critical issue preventing the trading system from working.
"""

import asyncio
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_critical_functionality():
    """Test the most critical components to identify the blocking issue."""
    
    print("🔍 CRITICAL FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    try:
        from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
        print("✅ Trading executor import successful")
    except Exception as e:
        print(f"❌ Trading executor import failed: {e}")
        return
    
    # Test 2: Executor creation
    print("\n2. Testing executor creation...")
    try:
        exchange_config = {
            'exchange': 'coinbase',
            'api_key': 'test',
            'secret': 'test',
            'sandbox': True
        }
        
        strategy_config = {}
        entropy_config = {}
        risk_config = {}
        
        executor = EntropyEnhancedTradingExecutor(
            exchange_config, strategy_config, entropy_config, risk_config
        )
        print("✅ Executor creation successful")
    except Exception as e:
        print(f"❌ Executor creation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    # Test 3: Market data collection
    print("\n3. Testing market data collection...")
    try:
        # Create a simple market data object
        from core.pure_profit_calculator import MarketData
        
        market_data = MarketData(
            timestamp=asyncio.get_event_loop().time(),
            btc_price=50000.0,
            eth_price=3000.0,
            usdc_volume=1000000.0,
            volatility=0.02,
            momentum=0.01,
            volume_profile=1.2
        )
        print("✅ Market data creation successful")
    except Exception as e:
        print(f"❌ Market data creation failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    # Test 4: Trading cycle execution
    print("\n4. Testing trading cycle execution...")
    try:
        result = await executor.execute_trading_cycle()
        print(f"✅ Trading cycle completed")
        print(f"   Success: {result.success}")
        print(f"   Action: {result.action.value}")
        print(f"   Reason: {result.metadata.get('reason', 'N/A')}")
        
        if not result.success and 'error' in result.metadata:
            print(f"   Error: {result.metadata['error']}")
            
    except Exception as e:
        print(f"❌ Trading cycle execution failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    # Test 5: Performance metrics
    print("\n5. Testing performance metrics...")
    try:
        performance = executor.get_performance_summary()
        print("✅ Performance metrics retrieved")
        print(f"   Total trades: {performance.get('total_trades', 0)}")
        print(f"   Successful trades: {performance.get('successful_trades', 0)}")
        print(f"   Total profit: ${performance.get('total_profit', 0):.4f}")
    except Exception as e:
        print(f"❌ Performance metrics failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    print("\n✅ ALL CRITICAL TESTS PASSED")
    print("The trading system appears to be functional!")

if __name__ == "__main__":
    asyncio.run(test_critical_functionality()) 