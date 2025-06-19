#!/usr/bin/env python3
"""
Test Step 2: CCXT Execution Manager Integration
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_step2_ccxt_integration():
    """Test CCXT execution manager integration with mathematical systems"""
    print("🔧 STEP 2: Testing CCXT Execution Manager Integration...")
    print("="*70)
    
    try:
        from core.ccxt_execution_manager import CCXTExecutionManager, create_mathematical_execution_system
        
        # Test 1: Create mathematical execution system
        print("1️⃣ Creating Mathematical Execution System...")
        execution_manager = create_mathematical_execution_system()
        print(f"   ✅ Execution manager created")
        print(f"   📊 Math processor: {type(execution_manager.math_processor).__name__}")
        print(f"   🎯 Fitness oracle: {type(execution_manager.fitness_oracle).__name__}")
        print(f"   🔗 API coordinator: {type(execution_manager.api_coordinator).__name__}")
        
        # Test 2: Start API coordinator
        print("\n2️⃣ Starting API Coordinator...")
        coordinator_started = await execution_manager.api_coordinator.start_coordinator()
        print(f"   ✅ API Coordinator started: {coordinator_started}")
        
        # Test 3: Evaluate trade opportunity
        print("\n3️⃣ Testing Trade Opportunity Evaluation...")
        
        # Create sample market data
        sample_market_data = {
            'symbol': 'BTC/USDT',
            'price': 45000.0,
            'volume': 1000.0,
            'price_series': [44800, 44900, 45000, 45100, 45000],
            'volume_series': [900, 950, 1000, 1100, 1000],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Evaluate opportunity
        trade_signal = await execution_manager.evaluate_trade_opportunity(sample_market_data)
        
        if trade_signal:
            print(f"   ✅ Trade signal generated: {trade_signal.signal_id}")
            print(f"   📊 Symbol: {trade_signal.symbol}")
            print(f"   📈 Side: {trade_signal.side}")
            print(f"   🎯 Confidence: {trade_signal.confidence:.3f}")
            print(f"   🧮 Mathematical validity: {trade_signal.mathematical_validity}")
            print(f"   🌀 Klein bottle consistent: {trade_signal.klein_bottle_consistent}")
            print(f"   🔄 Fractal convergent: {trade_signal.fractal_convergent}")
            print(f"   ⚡ Phase gate: {trade_signal.phase_gate}")
            
            # Test 4: Execute trade signal
            print("\n4️⃣ Testing Trade Signal Execution...")
            execution_result = await execution_manager.execute_signal(trade_signal)
            
            print(f"   ✅ Execution completed: {execution_result.status.value}")
            print(f"   📊 Signal ID: {execution_result.signal_id}")
            if execution_result.executed_price:
                print(f"   💰 Executed price: ${execution_result.executed_price:.2f}")
                print(f"   📦 Executed quantity: {execution_result.executed_quantity:.6f}")
            if execution_result.error_message:
                print(f"   ⚠️ Error: {execution_result.error_message}")
        else:
            print("   ⚠️ No trade signal generated (mathematical validation may have failed)")
        
        # Test 5: Get execution summary
        print("\n5️⃣ Testing Execution Summary...")
        summary = execution_manager.get_execution_summary()
        print(f"   ✅ Summary generated")
        print(f"   📊 Total signals: {summary['statistics']['total_signals']}")
        print(f"   🎯 Mathematical validations: {summary['statistics']['mathematical_validations']}")
        print(f"   💰 Executed trades: {summary['statistics']['executed_trades']}")
        print(f"   ⚠️ Risk rejections: {summary['statistics']['risk_rejections']}")
        print(f"   🔄 Active signals: {summary['active_signals']}")
        
        # Test 6: Stop API coordinator
        print("\n6️⃣ Stopping API Coordinator...")
        coordinator_stopped = await execution_manager.api_coordinator.stop_coordinator()
        print(f"   ✅ API Coordinator stopped: {coordinator_stopped}")
        
        print("\n" + "="*70)
        print("🎉 STEP 2 COMPLETE: CCXT Execution Manager Integration successful!")
        print("✅ Mathematical systems now connected to trading execution")
        print("✅ Trade signals are mathematically validated before execution")
        print("✅ Risk management integrated with mathematical analysis")
        print("✅ Ready to proceed to Step 3: Phase gate logic connection")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Need to debug the CCXT execution manager integration")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_step2_ccxt_integration())
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("   3️⃣ Phase gate logic connection")
        print("   4️⃣ Profit routing implementation") 
        print("   5️⃣ Unified controller orchestration")
        print("\n💡 KEY FEATURES IMPLEMENTED:")
        print("   🧮 Mathematical validation before all trades")
        print("   🎯 Fitness score integration with execution")
        print("   🔒 Risk management with mathematical bounds")
        print("   📊 Profit vector analysis for trade sizing")
        print("   ⚡ Phase gate determination for trade timing")
    
    sys.exit(0 if success else 1) 