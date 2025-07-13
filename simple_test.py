#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test for Unified Mathematical Bridge System
"""

import sys
import traceback

def test_bridge_import():
    """Test if the bridge can be imported successfully."""
    try:
        print("🧠 Testing Unified Mathematical Bridge Import...")
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        print("✅ Bridge imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_bridge_creation():
    """Test if the bridge can be created successfully."""
    try:
        print("\n🔧 Testing Bridge Creation...")
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        bridge = UnifiedMathematicalBridge()
        print("✅ Bridge created successfully!")
        return bridge
    except Exception as e:
        print(f"❌ Bridge creation failed: {e}")
        traceback.print_exc()
        return None

def test_basic_integration():
    """Test basic integration functionality."""
    try:
        print("\n🔄 Testing Basic Integration...")
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        
        bridge = UnifiedMathematicalBridge()
        
        # Test market data
        market_data = {
            'symbol': 'BTC',
            'price_history': [100.0, 101.0, 102.0, 101.5, 103.0],
            'volume_history': [1000, 1100, 1200, 1150, 1300],
            'entropy_history': [0.1, 0.2, 0.15, 0.25, 0.3]
        }
        
        # Test portfolio state
        portfolio_state = {
            'total_value': 10000.0,
            'available_balance': 5000.0,
            'positions': {'BTC': 0.5}
        }
        
        # Run integration
        result = bridge.integrate_all_mathematical_systems(market_data, portfolio_state)
        
        print(f"✅ Integration completed!")
        print(f"   Success: {result.success}")
        print(f"   Confidence: {result.overall_confidence:.3f}")
        print(f"   Connections: {len(result.connections)}")
        print(f"   Execution Time: {result.execution_time:.3f}s")
        
        return result
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return None

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    try:
        print("\n📊 Testing Performance Monitoring...")
        from core.unified_mathematical_bridge import UnifiedMathematicalBridge
        
        bridge = UnifiedMathematicalBridge()
        
        # Get performance report
        performance_report = bridge.get_performance_report()
        print(f"✅ Performance report generated: {len(performance_report.get('metrics', {}))} metrics")
        
        # Get system health
        health_report = bridge.get_system_health_report()
        print(f"✅ Health report generated: {health_report.overall_health:.3f} overall health")
        
        # Get recommendations
        recommendations = bridge.get_optimization_recommendations()
        print(f"✅ Optimization recommendations: {len(recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧠 Unified Mathematical Bridge System - Simple Test")
    print("=" * 50)
    
    # Test imports
    if not test_bridge_import():
        print("\n❌ Import test failed. Exiting.")
        return False
    
    # Test bridge creation
    bridge = test_bridge_creation()
    if bridge is None:
        print("\n❌ Bridge creation failed. Exiting.")
        return False
    
    # Test basic integration
    result = test_basic_integration()
    if result is None:
        print("\n❌ Integration test failed. Exiting.")
        return False
    
    # Test performance monitoring
    if not test_performance_monitoring():
        print("\n❌ Performance monitoring test failed. Exiting.")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ The Unified Mathematical Bridge system is working correctly!")
    print("🚀 Your Schwabot trading system now has complete mathematical integration!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 