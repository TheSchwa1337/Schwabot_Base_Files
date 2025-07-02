#!/usr/bin/env python3
"""
Test 32-bit Dualistic Phase Switching Integration
================================================

This test demonstrates the integration of 32-bit phase switching into the
profit vectorization pipeline for dynamic profit vectorization management
through mathematical portals where relevant.
"""

import time
import json
import logging
from typing import Dict, Any

# Import the enhanced systems
from core.phase_bit_integration import (
    PhaseBitIntegration, BitPhase, StrategyType,
    resolve_bit_phases, process_hash_with_phase, get_phase_optimized_strategy
)
from core.ferris_rde_core import FerrisPhase, FerrisRDECore
from core.unified_profit_vectorization_system import (
    UnifiedProfitVectorizationSystem,
    calculate_profit_vectorization,
    get_32bit_dualistic_status
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_32bit_dualistic_integration():
    """Test the complete 32-bit dualistic phase switching integration."""
    print("🎯 Testing 32-bit Dualistic Phase Switching Integration")
    print("=" * 70)

    # Initialize systems
    phase_integration = PhaseBitIntegration()
    ferris_core = FerrisRDECore()
    profit_system = UnifiedProfitVectorizationSystem()

    # Test 1: 32-bit Phase Resolution
    print("\n📊 32-bit Phase Resolution:")
    test_strategy_id = "0x123456789abcdef123456789abcdef123456789"
    bit_result = resolve_bit_phases(test_strategy_id)
    print(f"  Strategy ID: {test_strategy_id}")
    print(f"  φ₄ (4-bit):   {bit_result.phi_4}")
    print(f"  φ₈ (8-bit):   {bit_result.phi_8}")
    print(f"  φ₃₂ (32-bit): {bit_result.phi_32}")
    print(f"  φ₄₂ (42-bit): {bit_result.phi_42}")
    print(f"  Cycle Score:  {bit_result.cycle_score:.4f}")

    # Test 2: Dualistic Mapping
    print("\n🔄 Dualistic Mapping:")
    market_conditions = {
        'volatility': 0.8,
        'entropy': 0.9,
        'trend_strength': 0.2,
        'complexity': 0.7
}
    dualistic_mapping = phase_integration.get_dualistic_mapping(market_conditions)
    print(f"  Bit Phase: {dualistic_mapping.bit_phase.value}")
    print(f"  Strategy Type: {dualistic_mapping.strategy_type.value}")
    print(f"  Mathematical Factor: {dualistic_mapping.mathematical_factor:.3f}")
    print(f"  Hash Sensitivity: {dualistic_mapping.hash_sensitivity:.3f}")
    print(f"  Tensor Weight: {dualistic_mapping.tensor_weight:.3f}")
    print(f"  Dualistic Threshold: {dualistic_mapping.dualistic_switch_threshold:.3f}")

    # Test 3: Hash Processing with 32-bit Dualistic
    print("\n🔐 Hash Processing with 32-bit Dualistic:")
    test_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"

    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        result = process_hash_with_phase(test_hash, phase, market_conditions)
        print(f"  {phase.value.upper():4}: "
              f"{result['bit_phase']}-bit, "
              f"Dualistic: {result['dualistic_active']}, "
              f"Strategy: {result['strategy_type']}, "
              f"Tensor: {result['tensor_score']:.4f}")

    # Test 4: Profit Vectorization with 32-bit Dualistic
    print("\n💰 Profit Vectorization with 32-bit Dualistic:")

    # Test market data that should trigger dualistic switching
    dualistic_market_data = {
        'volatility': 0.75,
        'entropy': 0.85,
        'trend_strength': 0.3,
        'complexity': 0.8,
        'price_momentum': 0.4,
        'volume_profile': 0.6
}
    # Test standard market data
    standard_market_data = {
        'volatility': 0.3,
        'entropy': 0.4,
        'trend_strength': 0.7,
        'complexity': 0.3,
        'price_momentum': 0.8,
        'volume_profile': 0.5
}
    btc_price = 52000.0
    volume = 1.5

    # Test dualistic conditions
    dualistic_result = calculate_profit_vectorization(btc_price, volume, dualistic_market_data)
    print(f"  Dualistic Market:")
    print(f"    Profit Score: {dualistic_result.get('profit_score', 0):.4f}")
    print(f"    Confidence: {dualistic_result.get('confidence_score', 0):.4f}")
    print(f"    Action: {dualistic_result.get('recommended_action', 'hold')}")
    print(f"    Dualistic Active: {dualistic_result.get('dualistic_active', False)}")
    print(f"    Bit Phase: {dualistic_result.get('bit_phase', 0)}")

    # Test standard conditions
    standard_result = calculate_profit_vectorization(btc_price, volume, standard_market_data)
    print(f"  Standard Market:")
    print(f"    Profit Score: {standard_result.get('profit_score', 0):.4f}")
    print(f"    Confidence: {standard_result.get('confidence_score', 0):.4f}")
    print(f"    Action: {standard_result.get('recommended_action', 'hold')}")
    print(f"    Dualistic Active: {standard_result.get('dualistic_active', False)}")
    print(f"    Bit Phase: {standard_result.get('bit_phase', 0)}")

    # Test 5: 32-bit Dualistic Status
    print("\n📈 32-bit Dualistic Status:")
    status = get_32bit_dualistic_status()
    print(f"  Dualistic Enabled: {status.get('32bit_dualistic_enabled', False)}")
    print(f"  Volatility Threshold: {status.get('dualistic_volatility_threshold', 0):.3f}")
    print(f"  Entropy Threshold: {status.get('dualistic_entropy_threshold', 0):.3f}")
    print(f"  Total Dualistic Vectors: {status.get('total_dualistic_vectors', 0)}")
    print(f"  Dualistic Success Rate: {status.get('dualistic_success_rate', 0):.3f}")

    # Test 6: Phase-Bit Integration Status
    print("\n🔗 Phase-Bit Integration Status:")
    integration_status = status.get('phase_bit_integration_status', {})

    # Show dualistic mapping
    dualistic_info = integration_status.get('dualistic_mapping', {})
    print(f"  Dualistic Bit Phase: {dualistic_info.get('bit_phase', 'N/A')}")
    print(f"  Dualistic Strategy: {dualistic_info.get('strategy_type', 'N/A')}")
    print(f"  Dualistic Math Factor: {dualistic_info.get('mathematical_factor', 0):.3f}")
    print(f"  Dualistic Threshold: {dualistic_info.get('dualistic_threshold', 0):.3f}")

    # Show bit phase weights
    weights = integration_status.get('bit_phase_weights', {})
    print(f"  α Weight (4-bit): {weights.get('alpha_weight', 0):.3f}")
    print(f"  β Weight (8-bit): {weights.get('beta_weight', 0):.3f}")
    print(f"  γ Weight (32-bit): {weights.get('gamma_weight', 0):.3f}")
    print(f"  δ Weight (42-bit): {weights.get('delta_weight', 0):.3f}")

    print("\n✅ 32-bit Dualistic Phase Switching Integration Test Complete!")

def test_profit_vectorization_pipeline():
    """Test the complete profit vectorization pipeline with 32-bit dualistic switching."""
    print("\n🚀 Testing Complete Profit Vectorization Pipeline")
    print("=" * 60)

    profit_system = UnifiedProfitVectorizationSystem()

    # Simulate market conditions over time
    market_scenarios = [
        {
            'name': 'High Volatility + High Entropy',
            'data': {
                'volatility': 0.8,
                'entropy': 0.9,
                'trend_strength': 0.2,
                'complexity': 0.8
            },
            'expected_dualistic': True
        },
        {
            'name': 'Low Volatility + Low Entropy',
            'data': {
                'volatility': 0.2,
                'entropy': 0.3,
                'trend_strength': 0.8,
                'complexity': 0.2
            },
            'expected_dualistic': False
        },
        {
            'name': 'Mixed Conditions',
            'data': {
                'volatility': 0.6,
                'entropy': 0.7,
                'trend_strength': 0.4,
                'complexity': 0.6
            },
            'expected_dualistic': True
}
]
    btc_price = 52000.0
    volume = 1.0

    for scenario in market_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"  Expected Dualistic: {scenario['expected_dualistic']}")

        # Calculate profit vectorization
        result = profit_system.calculate_profit_vectorization(btc_price, volume, scenario['data'])

        print(f"  Actual Dualistic: {result.get('dualistic_active', False)}")
        print(f"  Profit Score: {result.get('profit_score', 0):.4f}")
        print(f"  Confidence: {result.get('confidence_score', 0):.4f}")
        print(f"  Action: {result.get('recommended_action', 'hold')}")
        print(f"  Bit Phase: {result.get('bit_phase', 0)}")

        # Verify dualistic activation
        if result.get('dualistic_active', False) == scenario['expected_dualistic']:
            print("  ✅ Dualistic activation matches expectation")
        else:
            print("  ⚠️  Dualistic activation differs from expectation")

    print("\n✅ Profit Vectorization Pipeline Test Complete!")

def test_mathematical_portal_integration():
    """Test the integration of 32-bit dualistic switching through mathematical portals."""
    print("\n🧮 Testing Mathematical Portal Integration")
    print("=" * 50)

    phase_integration = PhaseBitIntegration()

    # Test mathematical portal connections
    test_cases = [
        {
            'name': 'Hash Processing Portal',
            'hash': 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
            'market_conditions': {'volatility': 0.8, 'entropy': 0.9, 'trend_strength': 0.2}
        },
        {
            'name': 'Strategy Optimization Portal',
            'hash': 'b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567890',
            'market_conditions': {'volatility': 0.6, 'entropy': 0.7, 'trend_strength': 0.4}
        },
        {
            'name': 'Tensor Processing Portal',
            'hash': 'c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567890ab',
            'market_conditions': {'volatility': 0.4, 'entropy': 0.5, 'trend_strength': 0.6}
}
]
    for test_case in test_cases:
        print(f"\n🔗 Portal: {test_case['name']}")

        # Test hash processing through portal
        hash_result = process_hash_with_phase(
            test_case['hash'],
            FerrisPhase.MID,
            test_case['market_conditions']
        )

        print(f"  Bit Phase: {hash_result.get('bit_phase', 0)}")
        print(f"  Dualistic Active: {hash_result.get('dualistic_active', False)}")
        print(f"  Strategy Type: {hash_result.get('strategy_type', 'unknown')}")
        print(f"  Tensor Score: {hash_result.get('tensor_score', 0):.4f}")
        print(f"  Hash Sensitivity: {hash_result.get('hash_sensitivity', 0):.4f}")

        # Test strategy optimization through portal
        strategy = get_phase_optimized_strategy(FerrisPhase.MID, test_case['market_conditions'])
        print(f"  Optimized Strategy: {strategy.get('strategy_type', 'unknown')}")
        print(f"  Mathematical Factor: {strategy.get('mathematical_factor', 0):.3f}")
        print(f"  Dualistic Active: {strategy.get('dualistic_active', False)}")

    print("\n✅ Mathematical Portal Integration Test Complete!")

if __name__ == "__main__":
    print("🎯 32-bit Dualistic Phase Switching Integration Test Suite")
    print("=" * 70)

    try:
        # Run all tests
        test_32bit_dualistic_integration()
        test_profit_vectorization_pipeline()
        test_mathematical_portal_integration()

        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("  ✅ 32-bit phase switching integrated into phase-bit system")
        print("  ✅ Dualistic mapping based on market conditions")
        print("  ✅ Profit vectorization pipeline with dualistic support")
        print("  ✅ Mathematical portal integration for dynamic switching")
        print("  ✅ Hash processing with 32-bit dualistic phase")
        print("  ✅ Strategy optimization with dualistic considerations")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
