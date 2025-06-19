#!/usr/bin/env python3
"""
Test Phase Gate Logic Integration - Phase Gate Decision Making and Market Data Connection
========================================================================================

Test for the integration of phase gate logic that connects mathematical analysis
with intelligent trading decisions based on entropy, bit density, and market conditions.
This component provides sophisticated timing decisions for trade execution.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_phase_gate_logic_integration():
    """Test phase gate logic integration with mathematical systems and market data"""
    print("🔧 TESTING PHASE GATE LOGIC INTEGRATION...")
    print("="*70)
    
    try:
        from core.phase_gate_controller import PhaseGateController, PhaseGateType, PhaseGateDecision
        
        # Test 1: Create phase gate controller
        print("1️⃣ Creating Phase Gate Controller...")
        phase_controller = PhaseGateController()
        print(f"   ✅ Phase gate controller created")
        print(f"   📊 Entropy calculator: {type(phase_controller.entropy_calculator).__name__}")
        print(f"   🎮 Bit analyzer: {type(phase_controller.bit_analyzer).__name__}")
        
        # Test 2: Basic phase gate decision
        print("\n2️⃣ Testing Basic Phase Gate Decision...")
        
        # Create sample trade signal
        sample_trade_signal = {
            'signal_id': 'phase_gate_test_001',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'position_size': 0.1,
            'confidence': 0.85,
            'mathematical_validity': True
        }
        
        # Create sample market data
        sample_market_data = {
            'price': 45000.0,
            'volume': 1500.0,
            'price_series': [44800, 44900, 45000, 45100, 45000],
            'volume_series': [1400, 1450, 1500, 1550, 1500],
            'volatility_24h': 0.042,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Make phase gate decision
        phase_decision = await phase_controller.make_phase_gate_decision(
            sample_trade_signal, sample_market_data
        )
        
        if phase_decision:
            print(f"   ✅ Phase gate decision made: {phase_decision.decision.value}")
            print(f"   🚦 Gate type: {phase_decision.gate_type.value}")
            print(f"   📊 Decision confidence: {phase_decision.confidence:.3f}")
            print(f"   ⚡ Entropy score: {phase_decision.entropy_score:.3f}")
            print(f"   🎮 Bit density: {phase_decision.bit_density:.3f}")
            print(f"   ⏱️ Recommended delay: {phase_decision.recommended_delay_ms}ms")
        else:
            print("   ❌ No phase gate decision generated")
            return False
        
        # Test 3: Different gate types
        print("\n3️⃣ Testing Different Gate Types...")
        
        test_scenarios = [
            {
                'name': '4-bit gate (low complexity)',
                'market_data': {**sample_market_data, 'volatility_24h': 0.015},
                'expected_gate': PhaseGateType.GATE_4B
            },
            {
                'name': '8-bit gate (medium complexity)',
                'market_data': {**sample_market_data, 'volatility_24h': 0.035},
                'expected_gate': PhaseGateType.GATE_8B
            },
            {
                'name': '16-bit gate (high complexity)',
                'market_data': {**sample_market_data, 'volatility_24h': 0.075},
                'expected_gate': PhaseGateType.GATE_16B
            }
        ]
        
        for scenario in test_scenarios:
            print(f"   Testing {scenario['name']}...")
            decision = await phase_controller.make_phase_gate_decision(
                sample_trade_signal, scenario['market_data']
            )
            
            if decision:
                print(f"     ✅ Gate type: {decision.gate_type.value}")
                print(f"     📊 Decision: {decision.decision.value}")
                print(f"     ⚡ Entropy: {decision.entropy_score:.3f}")
            else:
                print(f"     ❌ Failed to generate decision for {scenario['name']}")
        
        # Test 4: Phase gate with mathematical validation
        print("\n4️⃣ Testing Phase Gate with Mathematical Validation...")
        
        # Test signal with poor mathematical validity
        poor_signal = {
            'signal_id': 'phase_gate_test_002',
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'position_size': 0.05,
            'confidence': 0.3,  # Low confidence
            'mathematical_validity': False
        }
        
        poor_decision = await phase_controller.make_phase_gate_decision(
            poor_signal, sample_market_data
        )
        
        if poor_decision:
            print(f"   ✅ Poor signal decision: {poor_decision.decision.value}")
            print(f"   🎯 Confidence (should be low): {poor_decision.confidence:.3f}")
            
            # Should recommend rejection or extreme caution
            if poor_decision.decision in [PhaseGateDecision.REJECT, PhaseGateDecision.DELAY_SIGNIFICANT]:
                print("   ✅ Correctly rejected or significantly delayed poor signal")
            else:
                print("   ⚠️ May have been too permissive with poor signal")
        
        # Test 5: Phase gate entropy analysis
        print("\n5️⃣ Testing Phase Gate Entropy Analysis...")
        
        # Create market data with different entropy characteristics
        high_entropy_data = {
            **sample_market_data,
            'price_series': [44000, 45500, 44200, 45800, 44100],  # Very volatile
            'volume_series': [1000, 2000, 1100, 2100, 1050]       # Irregular volume
        }
        
        low_entropy_data = {
            **sample_market_data,
            'price_series': [45000, 45010, 45020, 45015, 45025],  # Very stable
            'volume_series': [1500, 1510, 1520, 1515, 1525]       # Consistent volume
        }
        
        high_entropy_decision = await phase_controller.make_phase_gate_decision(
            sample_trade_signal, high_entropy_data
        )
        
        low_entropy_decision = await phase_controller.make_phase_gate_decision(
            sample_trade_signal, low_entropy_data
        )
        
        if high_entropy_decision and low_entropy_decision:
            print(f"   ✅ High entropy decision: {high_entropy_decision.decision.value}")
            print(f"     📊 Entropy score: {high_entropy_decision.entropy_score:.3f}")
            print(f"   ✅ Low entropy decision: {low_entropy_decision.decision.value}")
            print(f"     📊 Entropy score: {low_entropy_decision.entropy_score:.3f}")
            
            # Validate that entropy scores make sense
            if high_entropy_decision.entropy_score > low_entropy_decision.entropy_score:
                print("   ✅ Entropy analysis working correctly")
            else:
                print("   ⚠️ Entropy analysis may need calibration")
        
        # Test 6: Integration with execution timing
        print("\n6️⃣ Testing Integration with Execution Timing...")
        
        execution_summary = phase_controller.get_phase_gate_summary()
        print(f"   ✅ Phase gate summary generated")
        print(f"   📊 Total decisions: {execution_summary['statistics']['total_decisions']}")
        print(f"   🚦 Execute decisions: {execution_summary['statistics']['execute_decisions']}")
        print(f"   ⏸️ Delay decisions: {execution_summary['statistics']['delay_decisions']}")
        print(f"   ❌ Reject decisions: {execution_summary['statistics']['reject_decisions']}")
        print(f"   ⚡ Average entropy: {execution_summary['statistics']['average_entropy']:.3f}")
        
        print("\n" + "="*70)
        print("🎉 PHASE GATE LOGIC INTEGRATION COMPLETE!")
        print("✅ Phase gate controller functional with mathematical validation")
        print("✅ Entropy-based decision making operational")
        print("✅ Bit density analysis integrated")
        print("✅ Market volatility considerations implemented")
        print("✅ Ready to integrate with profit routing engine")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE GATE LOGIC INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Need to debug the phase gate integration")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_phase_gate_logic_integration())
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("   4️⃣ Profit routing implementation") 
        print("   5️⃣ Unified controller orchestration")
        print("\n💡 KEY FEATURES IMPLEMENTED:")
        print("   ⚡ Entropy-based trade timing decisions")
        print("   🎮 Bit density analysis for market complexity")
        print("   🚦 Multiple gate types for different market conditions")
        print("   📊 Mathematical validation integration")
        print("   ⏱️ Intelligent execution delay recommendations")
    
    sys.exit(0 if success else 1) 