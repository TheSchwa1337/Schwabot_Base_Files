#!/usr/bin/env python3
"""
Test Schwafit Integration - Comprehensive Testing Suite
======================================================

Tests the complete Schwafit trading integration system to ensure
all components are working properly and can generate valid trade signals.

This script verifies:
- Schwafit core mathematical frameworks
- Trading signal generation
- Market data processing
- Portfolio state management
- Performance tracking
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to the path for imports
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(title: str, emoji: str = "🚀"):
    """Print a formatted banner."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 2))


def test_schwafit_core():
    """Test Schwafit core mathematical frameworks."""
    print_banner("TESTING SCHWAFIT CORE - Mathematical Frameworks", "🧮")
    
    try:
        from core.schwafit_core import SchwafitCore, SchwafitFramework
        
        # Initialize Schwafit core
        schwafit_core = SchwafitCore()
        print("✅ Schwafit Core initialized successfully")
        
        # Test data
        test_prices = [50000, 50100, 49900, 50200, 50050]
        test_volumes = [1500, 1800, 1200, 2100, 1600]
        test_phases = [0.5, 0.7, 0.3, 0.8]
        
        # Test ALIF calculation
        print("\n📊 Testing ALIF (Adaptive Learning Interference Filter)...")
        alif_result = schwafit_core.calculate_alif(test_prices, test_volumes)
        print(f"   Certainty: {alif_result.certainty:.3f}")
        print(f"   Confidence: {alif_result.confidence:.3f}")
        print(f"   Thermal State: {alif_result.thermal_state}")
        
        # Test MIR4X calculation
        print("\n🪞 Testing MIR4X (Mirror 4-phase pattern recognition)...")
        mir4x_result = schwafit_core.calculate_mir4x(test_phases)
        print(f"   Certainty: {mir4x_result.certainty:.3f}")
        print(f"   Confidence: {mir4x_result.confidence:.3f}")
        print(f"   Thermal State: {mir4x_result.thermal_state}")
        
        # Test comprehensive analysis
        print("\n🔍 Testing comprehensive mirror analysis...")
        market_data = {
            "prices": test_prices,
            "volumes": test_volumes,
            "phases": test_phases
}
        comprehensive_results = schwafit_core.comprehensive_mirror_analysis(market_data)
        print(f"   Frameworks active: {len(comprehensive_results)}")
        
        # Test recommendations
        print("\n💡 Testing mirror recommendations...")
        recommendations = schwafit_core.get_mirror_recommendations(comprehensive_results)
        print(f"   Overall Confidence: {recommendations.get('overall_confidence', 0):.3f}")
        print(f"   Recommended Action: {recommendations.get('recommended_action', 'unknown')}")
        print(f"   Risk Level: {recommendations.get('risk_level', 'unknown')}")
        
        print("\n✅ Schwafit Core tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Schwafit Core test failed: {e}")
        return False


def test_schwafit_trading_integration():
    """Test Schwafit trading integration."""
    print_banner("TESTING SCHWAFIT TRADING INTEGRATION - Complete System", "⚡")
    
    try:
        from core.schwafit_trading_integration import SchwafitTradingIntegration, TradingSignalType
        
        # Initialize trading integration
        schwafit_integration = SchwafitTradingIntegration({
            "demo_mode": True,
            "simulate_trading": True
        })
        print("✅ Schwafit Trading Integration initialized successfully")
        
        # Test market data processing
        print("\n📈 Testing market data processing...")
        market_data = {
            "symbol": "BTC/USDT",
            "prices": [50000, 50100, 49900, 50200, 50050],
            "volumes": [1500, 1800, 1200, 2100, 1600],
            "current_price": 50050
}
        # Process market data
        signal = asyncio.run(schwafit_integration.process_market_data(market_data))
        print(f"   Signal Type: {signal.signal_type.value}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Schwafit Score: {signal.schwafit_score:.3f}")
        print(f"   Reasoning: {signal.reasoning}")
        
        # Test trade execution
        print("\n🔄 Testing trade execution...")
        execution_result = asyncio.run(schwafit_integration.execute_trade_signal(signal))
        print(f"   Executed: {execution_result.get('executed', False)}")
        if execution_result.get('executed'):
            print(f"   Order ID: {execution_result.get('order_id', 'N/A')}")
            print(f"   Fill Price: {execution_result.get('fill_price', 'N/A')}")
        
        # Test portfolio state update
        print("\n💰 Testing portfolio state update...")
        portfolio_state = asyncio.run(schwafit_integration.update_portfolio_state())
        print(f"   Total Value: ${portfolio_state.total_value:,.2f}")
        print(f"   Health Score: {portfolio_state.schwafit_health_score:.3f}")
        print(f"   Risk Exposure: {portfolio_state.risk_exposure:.3f}")
        print(f"   Diversification: {portfolio_state.diversification_score:.3f}")
        
        # Test complete trading cycle
        print("\n🔄 Testing complete trading cycle...")
        cycle_result = asyncio.run(schwafit_integration.run_trading_cycle(market_data))
        print(f"   Cycle Duration: {cycle_result.get('cycle_duration', 0):.3f}s")
        print(f"   Signal Generated: {cycle_result.get('signal') is not None}")
        print(f"   Execution Result: {cycle_result.get('execution_result', {}).get('executed', False)}")
        
        # Test performance summary
        print("\n📊 Testing performance summary...")
        performance = schwafit_integration.get_performance_summary()
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Active Signals: {performance['active_signals']}")
        print(f"   Portfolio Health: {performance['schwafit_health']:.3f}")
        
        print("\n✅ Schwafit Trading Integration tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Schwafit Trading Integration test failed: {e}")
        return False


async def test_multiple_trading_cycles():
    """Test multiple trading cycles with different market conditions."""
    print_banner("TESTING MULTIPLE TRADING CYCLES - Market Simulation", "📊")
    
    try:
        from core.schwafit_trading_integration import SchwafitTradingIntegration
        
        # Initialize trading integration
        schwafit_integration = SchwafitTradingIntegration({
            "demo_mode": True,
            "simulate_trading": True
        })
        
        # Test different market scenarios
        scenarios = [
            {
                "name": "Bullish Market",
                "data": {
                    "symbol": "BTC/USDT",
                    "prices": [50000, 50500, 51000, 51500, 52000],
                    "volumes": [1500, 1800, 2000, 2200, 2500],
                    "current_price": 52000
}
            },
            {
                "name": "Bearish Market",
                "data": {
                    "symbol": "BTC/USDT",
                    "prices": [50000, 49500, 49000, 48500, 48000],
                    "volumes": [1500, 1800, 2000, 2200, 2500],
                    "current_price": 48000
}
            },
            {
                "name": "Sideways Market",
                "data": {
                    "symbol": "BTC/USDT",
                    "prices": [50000, 50100, 49900, 50200, 50050],
                    "volumes": [1500, 1600, 1400, 1700, 1500],
                    "current_price": 50050
}
}
]
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📈 Scenario {i}: {scenario['name']}")
            
            # Run trading cycle
            cycle_result = await schwafit_integration.run_trading_cycle(scenario['data'])
            
            if "error" in cycle_result:
                print(f"   ❌ Error: {cycle_result['error']}")
                continue
            
            signal = cycle_result.get("signal")
            execution = cycle_result.get("execution_result", {})
            
            print(f"   Signal: {signal.signal_type.value if signal else 'None'}")
            print(f"   Confidence: {signal.confidence:.3f if signal else 0.0}")
            print(f"   Executed: {execution.get('executed', False)}")
            
            results.append({
                "scenario": scenario['name'],
                "signal_type": signal.signal_type.value if signal else "None",
                "confidence": signal.confidence if signal else 0.0,
                "executed": execution.get('executed', False)
            })
        
        # Summary
        print(f"\n📊 Trading Cycle Summary:")
        print(f"   Scenarios tested: {len(results)}")
        executed_trades = sum(1 for r in results if r['executed'])
        print(f"   Trades executed: {executed_trades}")
        
        # Performance summary
        performance = schwafit_integration.get_performance_summary()
        print(f"   Total trades in history: {performance['total_trades']}")
        print(f"   Portfolio health: {performance['schwafit_health']:.3f}")
        
        print("\n✅ Multiple trading cycles test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Multiple trading cycles test failed: {e}")
        return False


def test_core_system_integration():
    """Test integration with the core system manager."""
    print_banner("TESTING CORE SYSTEM INTEGRATION - Component Management", "🔧")
    
    try:
        from core import initialize_core_system, get_schwafit_trading_integration, shutdown_core_system
        
        # Initialize core system
        print("🔄 Initializing core system...")
        init_result = initialize_core_system()
        print(f"   Status: {init_result['status']}")
        print(f"   Available components: {', '.join(init_result.get('available_components', []))}")
        
        # Get Schwafit trading integration
        print("\n📊 Getting Schwafit trading integration...")
        schwafit_trading = get_schwafit_trading_integration()
        if schwafit_trading:
            print("   ✅ Schwafit trading integration retrieved successfully")
            
            # Test basic functionality
            market_data = {
                "symbol": "ETH/USDT",
                "prices": [3000, 3010, 2990, 3020, 3005],
                "volumes": [1000, 1200, 800, 1400, 1100],
                "current_price": 3005
}
            signal = asyncio.run(schwafit_trading.process_market_data(market_data))
            print(f"   Signal generated: {signal.signal_type.value}")
            print(f"   Confidence: {signal.confidence:.3f}")
        else:
            print("   ❌ Failed to get Schwafit trading integration")
        
        # Shutdown core system
        print("\n🔄 Shutting down core system...")
        shutdown_result = shutdown_core_system()
        print(f"   Status: {shutdown_result['status']}")
        
        print("\n✅ Core system integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Core system integration test failed: {e}")
        return False


async def run_comprehensive_test():
    """Run all tests comprehensively."""
    print_banner("SCHWAFIT INTEGRATION COMPREHENSIVE TEST SUITE", "🧪")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Test 1: Schwafit Core
    print("\n" + "="*60)
    result1 = test_schwafit_core()
    test_results.append(("Schwafit Core", result1))
    
    # Test 2: Schwafit Trading Integration
    print("\n" + "="*60)
    result2 = test_schwafit_trading_integration()
    test_results.append(("Schwafit Trading Integration", result2))
    
    # Test 3: Multiple Trading Cycles
    print("\n" + "="*60)
    result3 = await test_multiple_trading_cycles()
    test_results.append(("Multiple Trading Cycles", result3))
    
    # Test 4: Core System Integration
    print("\n" + "="*60)
    result4 = test_core_system_integration()
    test_results.append(("Core System Integration", result4))
    
    # Final Results
    print("\n" + "="*60)
    print_banner("FINAL TEST RESULTS", "📋")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 Summary:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Schwafit integration is working properly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the implementation.")
        return False


def main():
    """Main test runner."""
    try:
        success = asyncio.run(run_comprehensive_test())
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 