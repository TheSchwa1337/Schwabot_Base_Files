#!/usr/bin/env python3
"""
Integration Test - Test the complete Schwabot system integration.
"""

import asyncio
import logging
from core.clean_strategy_integration_bridge import StrategyIntegrationBridge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_integration():
    """Test the complete integration system."""
    print("🚀 Testing Schwabot Integration System")
    print("=" * 50)
    
    try:
        # Initialize the strategy integration bridge
        print("1. Initializing Strategy Integration Bridge...")
        bridge = StrategyIntegrationBridge()
        
        # Get integration summary
        summary = bridge.get_integration_summary()
        print(f"✅ Bridge initialized successfully (v{summary['version']})")
        
        # Show component availability
        components = summary["components_available"]
        print("\n2. Component Availability:")
        for component, available in components.items():
            status = "✅" if available else "❌"
            print(f"   {status} {component}: {'Available' if available else 'Not Available'}")
        
        # Show orchestration state
        orchestration = summary["orchestration_state"]
        print(f"\n3. Orchestration State:")
        print(f"   Total Strategies Active: {orchestration['total_strategies_active']}")
        print(f"   Wall Street Strategies: {orchestration['wall_street_strategies_active']}")
        print(f"   Schwabot Strategies: {orchestration['schwabot_strategies_active']}")
        
        # Test signal processing
        print("\n4. Testing Signal Processing...")
        signals = await bridge.process_integrated_trading_signal(
            asset="BTC/USD",
            price=52000.0,
            volume=1500.0,
            timeframe="1h"
        )
        
        print(f"   Generated {len(signals)} integrated signals")
        
        if signals:
            print("\n5. Signal Details:")
            for i, signal in enumerate(signals, 1):
                print(f"   Signal {i}:")
                print(f"     Strategy: {signal.wall_street_signal.get('strategy', 'unknown')}")
                print(f"     Action: {signal.wall_street_signal.get('action', 'unknown')}")
                print(f"     Confidence: {signal.composite_confidence:.3f}")
                print(f"     Risk Score: {signal.risk_score:.3f}")
                print(f"     Priority: {signal.execution_priority}")
        
        # Test execution
        if signals:
            print("\n6. Testing Signal Execution...")
            execution_result = await bridge.execute_integrated_signal(signals[0])
            print(f"   Execution Result: {execution_result}")
        
        # Final summary
        final_summary = bridge.get_integration_summary()
        print(f"\n7. Final Summary:")
        print(f"   Total Signals Generated: {final_summary['orchestration_state']['total_signals_generated']}")
        print(f"   Signals Generated Today: {final_summary['orchestration_state']['signals_generated_today']}")
        print(f"   Average Signal Confidence: {final_summary['orchestration_state']['average_signal_confidence']:.3f}")
        
        print("\n🎉 Integration Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        logger.error(f"Integration test error: {e}")
        return False


if __name__ == "__main__":
    # Run the integration test
    success = asyncio.run(test_integration())
    
    if success:
        print("\n✅ All systems operational and ready for deployment!")
    else:
        print("\n❌ System requires additional configuration.") 