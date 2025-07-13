#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to check module availability
"""

import sys
import os

def test_simple_import(module_name):
    """Test simple module import."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - basic import OK")
        return True
    except Exception as e:
        print(f"❌ {module_name} - Error: {e}")
        return False

def main():
    print("🔍 Testing basic module imports...")
    print("=" * 50)
    
    # Test core modules
    modules = [
        "core.enhanced_entropy_randomization_system",
        "core.self_generating_strategy_system", 
        "core.unified_memory_registry_system",
        "core.unified_mathematical_bridge",
        "core.complete_internalized_scalping_system"
    ]
    
    working = 0
    for module in modules:
        if test_simple_import(module):
            working += 1
    
    print("=" * 50)
    print(f"📊 Results: {working}/{len(modules)} modules working")
    
    # Test the complete scalping system
    print("\n🎯 Testing complete scalping system...")
    try:
        from core.complete_internalized_scalping_system import complete_scalping_system
        print("✅ Complete scalping system imported successfully")
        
        # Test basic functionality
        import asyncio
        
        async def test_scalping():
            test_data = {
                'symbol': 'BTC',
                'price': 50000.0,
                'volume': 1000.0,
                'volatility': 0.02,
                'moving_average': 50100.0,
                'liquidity': 0.8
            }
            
            test_portfolio = {
                'total_value': 10000.0,
                'available_balance': 5000.0,
                'positions': {}
            }
            
            execution = await complete_scalping_system.execute_scalping_cycle(
                test_data, test_portfolio
            )
            
            print(f"✅ Scalping cycle executed: {execution.decision.signal.value}")
            print(f"💰 Profit/Loss: {execution.profit_loss:.4f}")
            print(f"🎯 Confidence: {execution.decision.confidence:.3f}")
            
            return True
            
        # Run test
        result = asyncio.run(test_scalping())
        if result:
            print("✅ Complete scalping system test PASSED")
        else:
            print("❌ Complete scalping system test FAILED")
            
    except Exception as e:
        print(f"❌ Complete scalping system test failed: {e}")

if __name__ == "__main__":
    main() 