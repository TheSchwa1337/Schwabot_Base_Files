#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test to Identify Critical Issue
"""

import asyncio

async def simple_test():
    print("🔍 SIMPLE CRITICAL TEST")
    print("=" * 30)
    
    try:
        # Test 1: Import
        print("1. Importing...")
        from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
        print("✅ Import successful")
        
        # Test 2: Create executor
        print("2. Creating executor...")
        executor = EntropyEnhancedTradingExecutor(
            exchange_config={'exchange': 'coinbase', 'sandbox': True},
            strategy_config={},
            entropy_config={},
            risk_config={}
        )
        print("✅ Executor created")
        
        # Test 3: Execute one cycle
        print("3. Executing trading cycle...")
        result = await executor.execute_trading_cycle()
        print(f"✅ Cycle completed: {result.success}")
        print(f"   Action: {result.action.value}")
        
        if not result.success:
            print(f"   Reason: {result.metadata.get('reason', 'unknown')}")
        
        # Test 4: Get performance
        print("4. Getting performance...")
        perf = executor.get_performance_summary()
        print(f"✅ Performance: {perf.get('total_trades', 0)} trades")
        
        print("\n🎯 CRITICAL ISSUE IDENTIFIED:")
        if not result.success:
            error = result.metadata.get('error', '')
            if 'exchange' in error.lower():
                print("❌ EXCHANGE CONNECTION ISSUE")
                print("   The system cannot connect to the exchange API.")
                print("   This is the most critical issue preventing live trading.")
                print("\n💡 SOLUTION:")
                print("   1. Add real API keys to exchange_config")
                print("   2. Test with sandbox=True first")
                print("   3. Ensure network connectivity")
            elif 'market' in error.lower():
                print("❌ MARKET DATA ISSUE")
                print("   Cannot fetch market data from exchange.")
            else:
                print(f"❌ UNKNOWN ISSUE: {error}")
        else:
            print("✅ NO CRITICAL ISSUES FOUND")
            print("   The trading system is functional!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_test()) 