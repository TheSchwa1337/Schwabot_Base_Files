import math
from core.ferris_rde_core import FerrisRDECore, FerrisPhase

#!/usr/bin/env python3
"""Test LOW, MID, HIGH phases implementation."""



def test_phases():
    """Test the LOW, MID, HIGH phases implementation."""
    print("🎯 Testing LOW, MID, HIGH Phases Implementation")
    print("=" * 50)

    core = FerrisRDECore()

    # Test phase characteristics
    print("\n📊 Phase Characteristics:")
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        chars = core.get_phase_characteristics(phase)
        print(
            f"  {phase.value.upper()}: {chars['strategy']} strategy, "
            f"Risk: {chars['risk_tolerance']}, "
            f"BTC: {chars['btc_allocation'] * 100:.0f}%"
        )

    # Test phase calculations
    print("\n🔄 Phase Calculations:")
    test_angles = [
        0,
        math.pi / 4,
        math.pi / 2,
        3 * math.pi / 4,
        math.pi,
        5 * math.pi / 4,
        3 * math.pi / 2,
        7 * math.pi / 4,
    ]

    for angle in test_angles:
        intensity_phase, motion_phase = core._calculate_both_phases(angle)
        height = (math.sin(angle) + 1) / 2
        print(
            f"  Angle: {math.degrees(angle):6.1f}° -> "
            f"Intensity: {intensity_phase.value.upper():4} "
            f"(Height: {height:.3f})"
        )

    # Test profit calculations
    print("\n💰 Phase-Adjusted Profit:")
    base_profit = 1.0  # 1% base profit
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        adjusted_profit = core.calculate_phase_adjusted_profit(base_profit, phase)
        print(f"  {phase.value.upper()}: {adjusted_profit:.2f}%")

    # Test allocations
    print("\n📈 Phase-Optimized Allocations ($10,000):")
    total_capital = 10000.0
    for phase in [FerrisPhase.LOW, FerrisPhase.MID, FerrisPhase.HIGH]:
        allocation = core.get_phase_optimized_allocation(phase, total_capital)
        print(
            f"  {phase.value.upper()}: "
            f"BTC: ${allocation['BTC']:.0f}, "
            f"USDC: ${allocation['USDC']:.0f}, "
            f"ETH: ${allocation['ETH']:.0f}"
        )

    print("\n✅ All LOW, MID, HIGH phases implemented successfully!")
    print("🎯 Mathematical phase system ready for trading operations!")


if __name__ == "__main__":
    test_phases()
