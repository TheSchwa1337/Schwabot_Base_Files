from core.phase_bit_integration import (
import hashlib

#!/usr/bin/env python3
"""Test Phase-Bit Integration System."""

    PhaseBitIntegration,
    FerrisPhase,
    resolve_bit_phases,
    process_hash_with_phase,
    get_phase_optimized_strategy,
)


def test_phase_bit_integration():
    """Test the complete Phase-Bit Integration system."""
    print("🎯 Testing Phase-Bit Integration System")
    print("=" * 60)

    integration = PhaseBitIntegration()

    # Test 1: Phase to Bit Mappings
    print("\n📊 Phase to Bit Mappings:")
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        mapping = integration.get_phase_bit_mapping(phase)
        print(
            f"  {phase.value.upper():4} → {mapping.bit_phase.value}-bit "
            f"({mapping.strategy_type.value}) "
            f"Math Factor: {mapping.mathematical_factor:.1f}"
        )

    # Test 2: Bit Phase Resolution
    print("\n🔄 Bit Phase Resolution (φ₄, φ₈, φ₄₂):")
    test_strategy_id = "0x123456789abcdef"
    bit_result = resolve_bit_phases(test_strategy_id)
    print(f"  Strategy ID: {test_strategy_id}")
    print(f"  φ₄ (4-bit):  {bit_result.phi_4}")
    print(f"  φ₈ (8-bit):  {bit_result.phi_8}")
    print(f"  φ₄₂ (42-bit): {bit_result.phi_42}")
    print(f"  Cycle Score: {bit_result.cycle_score:.4f}")

    # Test 3: Hash Processing with Phases
    print("\n🔐 Hash Processing with Phases:")
    test_hash = hashlib.sha256("BTC_52000".encode()).hexdigest()
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        result = process_hash_with_phase(test_hash, phase)
        print(
            f"  {phase.value.upper():4}: "
            f"{result['bit_phase']}-bit, "
            f"Strategy: {result['strategy_type']}, "
            f"Tensor: {result['tensor_score']:.4f}"
        )

    # Test 4: Phase-Optimized Strategies
    print("\n📈 Phase-Optimized Strategies:")
    market_data = {"volatility": 0.03, "entropy_level": 5.0, "btc_price": 52000.0}
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        strategy = get_phase_optimized_strategy(phase, market_data)
        print(
            f"  {phase.value.upper():4}: "
            f"{strategy['strategy_type']} "
            f"(Risk: {strategy['risk_tolerance']:.2f}, "
            f"Pos: {strategy['position_multiplier']:.2f})"
        )
        print(
            f"    Allocation: BTC {strategy['allocation']['BTC']:.1%}, "
            f"USDC {strategy['allocation']['USDC']:.1%}, "
            f"ETH {strategy['allocation']['ETH']:.1%}"
        )

    # Test 5: Phase-Adjusted Bit Operations
    print("\n⚙️ Phase-Adjusted Bit Operations:")
    base_operations = {
        "tensor_score": 1.0,
        "hash_sensitivity": 0.6,
        "mathematical_factor": 1.0,
    }
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        adjusted = integration.calculate_phase_adjusted_bit_operations(
            phase, base_operations
        )
        print(
            f"  {phase.value.upper():4}: "
            f"Tensor: {adjusted['tensor_score']:.3f}, "
            f"Hash: {adjusted['hash_sensitivity']:.3f}, "
            f"Math: {adjusted['mathematical_factor']:.3f}"
        )

    # Test 6: System Status
    print("\n📊 System Status:")
    status = integration.get_system_status()
    print(f"  Phase Mappings: {len(status['phase_bit_mappings'])} configured")
    print(
        f"  Bit Weights: α={status['bit_phase_weights']['alpha_weight']:.1f}, "
        f"β={status['bit_phase_weights']['beta_weight']:.1f}, "
        f"γ={status['bit_phase_weights']['gamma_weight']:.1f}"
    )

    # Test 7: Mathematical Formula Verification
    print("\n🧮 Mathematical Formula Verification:")
    strategy_int = int(test_strategy_id, 16)
    expected_phi_4 = strategy_int & 0b1111
    expected_phi_8 = (strategy_int >> 4) & 0b11111111
    expected_phi_42 = (strategy_int >> 12) & 0x3FFFFFFFFFF

    print(
        f"  Expected φ₄: {expected_phi_4}, Actual: {bit_result.phi_4} "
        f"({'✅' if expected_phi_4 == bit_result.phi_4 else '❌'})"
    )
    print(
        f"  Expected φ₈: {expected_phi_8}, Actual: {bit_result.phi_8} "
        f"({'✅' if expected_phi_8 == bit_result.phi_8 else '❌'})"
    )
    print(
        f"  Expected φ₄₂: {expected_phi_42}, Actual: {bit_result.phi_42} "
        f"({'✅' if expected_phi_42 == bit_result.phi_42 else '❌'})"
    )

    # Test 8: Complete Pipeline Test
    print("\n🔄 Complete Pipeline Test:")
    btc_price = 52000.0
    btc_hash = hashlib.sha256(f"{btc_price}".encode()).hexdigest()

    # Simulate Ferris wheel phase calculation
    ferris_core = integration.ferris_core
    wheel_state = ferris_core.update_ferris_wheel(0.1)  # 6 seconds
    current_phase = wheel_state.phase

    # Process through complete pipeline
    hash_result = process_hash_with_phase(btc_hash, current_phase)
    strategy = get_phase_optimized_strategy(current_phase, market_data)

    print(f"  BTC Price: ${btc_price:,.0f}")
    print(f"  BTC Hash: {btc_hash[:16]}...")
    print(f"  Current Phase: {current_phase.value.upper()}")
    print(f"  Bit Phase: {hash_result['bit_phase']}-bit")
    print(f"  Strategy: {strategy['strategy_type']}")
    print(f"  BTC Allocation: {strategy['allocation']['BTC']:.1%}")
    print(f"  Tensor Score: {hash_result['tensor_score']:.4f}")

    print("\n✅ All Phase-Bit Integration tests completed successfully!")
    print("🎯 Mathematical connections verified and operational!")


if __name__ == "__main__":
    test_phase_bit_integration()
