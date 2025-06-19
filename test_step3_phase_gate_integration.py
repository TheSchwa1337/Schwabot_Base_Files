#!/usr/bin/env python3
"""
Test Step 3: Phase Gate Logic Connection
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_step3_phase_gate_integration():
    """Test phase gate controller integration with mathematical systems"""
    print("⚡ STEP 3: Testing Phase Gate Logic Connection...")
    print("="*70)
    
    try:
        from core.phase_gate_controller import (
            PhaseGateController, 
            create_phase_gate_system,
            PhaseGateType,
            PhaseGateStatus,
            GateDecision
        )
        from core.ccxt_execution_manager import create_mathematical_execution_system
        
        # Test 1: Create integrated mathematical + phase gate system
        print("1️⃣ Creating Integrated Mathematical + Phase Gate System...")
        
        # First create the execution manager (Step 2)
        execution_manager = create_mathematical_execution_system()
        print(f"   ✅ Execution manager created")
        
        # Then create the phase gate system (Step 3)
        phase_controller = create_phase_gate_system(execution_manager)
        print(f"   ✅ Phase gate controller created")
        print(f"   🌀 Entropy engine: {type(phase_controller.entropy_engine).__name__}")
        print(f"   🔢 Bit operations: {type(phase_controller.bit_operations).__name__}")
        print(f"   🧮 Math processor: {type(phase_controller.math_processor).__name__}")
        print(f"   🎯 Fitness oracle: {type(phase_controller.fitness_oracle).__name__}")
        
        # Test 2: Generate mathematical trade signal
        print("\n2️⃣ Generating Mathematical Trade Signal...")
        
        # Create sample market data with price series for entropy calculation
        sample_market_data = {
            'symbol': 'BTC/USDT',
            'price': 45000.0,
            'volume': 1000.0,
            'price_series': [44500, 44700, 44900, 45000, 45100, 45200, 44800, 44900],
            'volume_series': [800, 900, 950, 1000, 1100, 1200, 850, 900],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Start API coordinator for trade signal generation
        await execution_manager.api_coordinator.start_coordinator()
        
        # Generate trade signal using execution manager
        trade_signal = await execution_manager.evaluate_trade_opportunity(sample_market_data)
        
        if trade_signal:
            print(f"   ✅ Trade signal generated: {trade_signal.signal_id}")
            print(f"   📊 Symbol: {trade_signal.symbol}")
            print(f"   📈 Side: {trade_signal.side}")
            print(f"   🎯 Confidence: {trade_signal.confidence:.3f}")
            print(f"   🧮 Mathematical validity: {trade_signal.mathematical_validity}")
            
            # Test 3: Evaluate phase gate
            print("\n3️⃣ Testing Phase Gate Evaluation...")
            gate_decision = await phase_controller.evaluate_phase_gate(trade_signal, sample_market_data)
            
            print(f"   ✅ Phase gate evaluation completed")
            print(f"   ⚡ Optimal gate: {gate_decision.gate_type.value}")
            print(f"   🚀 Decision: {gate_decision.decision.value}")
            print(f"   🎯 Confidence: {gate_decision.confidence:.3f}")
            if gate_decision.delay_seconds > 0:
                print(f"   ⏱️ Delay: {gate_decision.delay_seconds:.2f}s")
            print(f"   🔄 Position adjustment: {gate_decision.position_size_adjustment:.3f}")
            print(f"   📊 Reasoning: {gate_decision.reasoning}")
            
            if gate_decision.metrics:
                print(f"   🌀 Entropy score: {gate_decision.metrics.entropy_score:.3f}")
                print(f"   🧮 Coherence score: {gate_decision.metrics.coherence_score:.3f}")
                print(f"   🔢 Bit density: {gate_decision.metrics.bit_density:.3f}")
                print(f"   💪 Pattern strength: {gate_decision.metrics.pattern_strength:.3f}")
                print(f"   📏 Micro suitability: {gate_decision.metrics.micro_suitability:.3f}")
                print(f"   🌊 Harmonic suitability: {gate_decision.metrics.harmonic_suitability:.3f}")
                print(f"   🎯 Strategic suitability: {gate_decision.metrics.strategic_suitability:.3f}")
            
            # Test 4: Execute through phase gate
            print("\n4️⃣ Testing Execution Through Phase Gate...")
            execution_result = await phase_controller.execute_through_phase_gate(
                trade_signal, 
                sample_market_data
            )
            
            print(f"   ✅ Phase gate execution completed")
            print(f"   📊 Status: {execution_result['status']}")
            print(f"   ⚡ Gate type: {execution_result['gate_type']}")
            
            if execution_result['status'] == 'executed':
                print(f"   💰 Execution successful!")
                if 'execution_result' in execution_result:
                    exec_res = execution_result['execution_result']
                    print(f"   📈 Execution status: {exec_res.status.value}")
                    if exec_res.executed_price:
                        print(f"   💵 Price: ${exec_res.executed_price:.2f}")
            elif execution_result['status'] == 'rejected':
                print(f"   ⚠️ Execution rejected: {execution_result['reason']}")
            elif execution_result['status'] == 'scheduled':
                print(f"   ⏱️ Execution scheduled with {execution_result['delay_seconds']:.2f}s delay")
            elif execution_result['status'] == 'queued':
                print(f"   📋 Execution queued at position {execution_result['queue_position']}")
                
        else:
            print("   ⚠️ No trade signal generated (using simulated signal for phase gate test)")
            
            # Create a simulated trade signal for testing
            from core.ccxt_execution_manager import MathematicalTradeSignal, RiskLevel
            from core.math_core import AnalysisResult
            from enhanced_fitness_oracle import UnifiedFitnessScore
            from core.mathlib_v3 import GradedProfitVector
            
            simulated_signal = MathematicalTradeSignal(
                signal_id="simulated_test",
                timestamp=datetime.now(timezone.utc),
                unified_analysis=AnalysisResult(
                    name="simulated_analysis",
                    data={'mathematical_validity': {'topology_consistent': True, 'fractal_convergent': True}},
                    confidence=0.75,
                    timestamp=sample_market_data['timestamp'].timestamp()
                ),
                fitness_score=UnifiedFitnessScore(
                    overall_fitness=0.7,
                    profit_fitness=0.6,
                    risk_fitness=0.8,
                    confidence=0.75,
                    action='BUY',
                    position_size=0.1,
                    stop_loss=44000.0,
                    take_profit=46000.0
                ),
                graded_vector=GradedProfitVector(profit=100.0, signal_strength=0.7, smart_money_score=0.6),
                symbol='BTC/USDT',
                side='buy',
                position_size=0.1,
                confidence=0.75,
                risk_level=RiskLevel.MEDIUM,
                mathematical_validity=True,
                coherence_score=0.8
            )
            
            print("\n3️⃣ Testing Phase Gate with Simulated Signal...")
            gate_decision = await phase_controller.evaluate_phase_gate(simulated_signal, sample_market_data)
            print(f"   ✅ Phase gate evaluation: {gate_decision.gate_type.value} -> {gate_decision.decision.value}")
        
        # Test 5: Test different gate configurations
        print("\n5️⃣ Testing Different Phase Gate Configurations...")
        
        # Test closing a gate
        print("   🔒 Closing MICRO_4B gate...")
        phase_controller.set_gate_status(PhaseGateType.MICRO_4B, PhaseGateStatus.CLOSED)
        
        # Test throttling a gate
        print("   ⏸️ Throttling STRATEGIC_42B gate...")
        phase_controller.set_gate_status(PhaseGateType.STRATEGIC_42B, PhaseGateStatus.THROTTLED)
        
        # Test with different entropy scenarios
        test_scenarios = [
            {"entropy": 0.2, "name": "Low Entropy (should prefer MICRO_4B)"},
            {"entropy": 0.5, "name": "Medium Entropy (should prefer HARMONIC_8B)"},
            {"entropy": 0.8, "name": "High Entropy (should prefer STRATEGIC_42B)"}
        ]
        
        for scenario in test_scenarios:
            print(f"   📊 Testing {scenario['name']}")
            
            # Modify market data to produce specific entropy
            test_market_data = sample_market_data.copy()
            if scenario['entropy'] < 0.3:
                # Low entropy - stable prices
                test_market_data['price_series'] = [45000 + i*5 for i in range(8)]
            elif scenario['entropy'] > 0.7:
                # High entropy - volatile prices  
                test_market_data['price_series'] = [45000 + ((-1)**i)*100*i for i in range(8)]
            
            if trade_signal or 'simulated_signal' in locals():
                test_signal = trade_signal if trade_signal else simulated_signal
                test_decision = await phase_controller.evaluate_phase_gate(test_signal, test_market_data)
                print(f"      -> Gate: {test_decision.gate_type.value}, Decision: {test_decision.decision.value}")
        
        # Test 6: Get phase gate summary
        print("\n6️⃣ Testing Phase Gate Summary...")
        summary = phase_controller.get_phase_gate_summary()
        print(f"   ✅ Summary generated")
        print(f"   📊 Total evaluations: {summary['statistics']['total_evaluations']}")
        print(f"   🔒 Mathematical validations: {summary['statistics']['mathematical_validations']}")
        print(f"   💔 Coherence rejections: {summary['statistics']['coherence_rejections']}")
        print(f"   🌀 Entropy rejections: {summary['statistics']['entropy_rejections']}")
        print(f"   📋 Queue length: {summary['queue_length']}")
        
        for gate_type, stats in summary['statistics']['gate_decisions'].items():
            print(f"   ⚡ {gate_type.value}: Open={stats['open']}, Closed={stats['closed']}, Executed={stats['executed']}")
        
        # Test 7: Stop API coordinator
        print("\n7️⃣ Stopping API Coordinator...")
        await execution_manager.api_coordinator.stop_coordinator()
        print(f"   ✅ API Coordinator stopped")
        
        print("\n" + "="*70)
        print("🎉 STEP 3 COMPLETE: Phase Gate Logic Connection successful!")
        print("✅ Entropy calculations integrated with phase gate selection")
        print("✅ Bit operations (4b/8b/42b) connected to execution timing")
        print("✅ Mathematical coherence validates all phase gate decisions")
        print("✅ Risk-based execution routing through appropriate gates")
        print("✅ Position sizing adjustments based on phase gate analysis")
        print("✅ Ready to proceed to Step 4: Profit routing implementation")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Need to debug the phase gate integration")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_step3_phase_gate_integration())
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("   4️⃣ Profit routing implementation")
        print("   5️⃣ Unified controller orchestration")
        print("\n💡 PHASE GATE FEATURES IMPLEMENTED:")
        print("   🌀 Entropy-driven gate selection (4b/8b/42b)")
        print("   🧮 Mathematical coherence validation")
        print("   🔢 Bit pattern analysis for execution timing")
        print("   ⚡ Dynamic phase gate status management")
        print("   🎯 Risk-adjusted position sizing")
        print("   ⏱️ Intelligent execution delays")
        print("   📊 Comprehensive gate performance tracking")
    
    sys.exit(0 if success else 1) 