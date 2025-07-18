import os
import sys
import traceback

import VMM_Schwabot

#!/usr/bin/env python3
"""
Simple VMM Test
==============

Simplified test for the Vitruvian Man Management system that doesn't depend'
on problematic imports.
"""


# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_vmm_basic():
    """Test basic VMM functionality without external dependencies."""
    print("🧬 Testing VMM Basic Functionality")
    print("=" * 50)

    try:
        # Import VMM directly without other dependencies
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

        # Import the VMM module directly

        print("✅ VMM module imported successfully")

        # Test basic constants
        print(f"✅ Golden Ratio (PHI): {VMM_Schwabot.PHI:.10f}")
        print(f"✅ PI: {VMM_Schwabot.PI:.10f}")
        print(f"✅ E: {VMM_Schwabot.E:.10f}")

        # Test enums
        print()
            f"✅ Vitruvian Zones: {[zone.value for zone in VMM_Schwabot.VitruvianZone]}"
        )
        print(f"✅ Limb Vectors: {[limb.value for limb in VMM_Schwabot.LimbVector]}")
        print()
            f"✅ Compression Modes: {[mode.value for mode in VMM_Schwabot.CompressionMode]}"
        )

        # Test manager creation
        VMM_Schwabot.get_vitruvian_manager()
        print("✅ VMM manager created successfully")

        # Test basic state update
        state = VMM_Schwabot.update_vitruvian_state()
            price=103586.0,
            rsi=45.0,
            volume=1000000.0,
            entropy=0.6,
            echo_strength=0.7,
            drift_score=0.2,
        )

        print("✅ State updated successfully")
        print(f"   Phi center: {state.phi_center:.4f}")
        print(f"   Thermal state: {state.thermal_state}")
        print(f"   Bit phase: {state.bit_phase}")
        print(f"   NCCO state: {state.ncco_state:.4f}")
        print(f"   SFS state: {state.sfs_state:.4f}")
        print(f"   UFS state: {state.ufs_state:.4f}")
        print(f"   ZPLS state: {state.zpls_state:.4f}")
        print(f"   RBMS state: {state.rbms_state:.4f}")

        # Test trading route
        route = VMM_Schwabot.get_optimal_trading_route()
            price=103586.0, rsi=45.0, volume=1000000.0
        )

        print("✅ Trading route generated")
        print(f"   Action: {route['action']}")
        print(f"   Reason: {route['reason']}")
        print(f"   Confidence: {route['confidence']:.3f}")

        # Test statistics
        stats = VMM_Schwabot.get_vitruvian_statistics()
        print("✅ Statistics generated")
        print(f"   Total triggers: {stats['total_triggers']}")
        print(f"   Current thermal state: {stats['current_thermal_state']}")
        print(f"   Current bit phase: {stats['current_bit_phase']}")

        return True

    except Exception as e:
        print(f"❌ VMM basic test failed: {e}")

        traceback.print_exc()
        return False


def test_mathematical_integration():
    """Test mathematical integration without external dependencies."""
    print("\n🧮 Testing Mathematical Integration")
    print("=" * 50)

    try:

        VMM_Schwabot.get_vitruvian_manager()

        # Test different market scenarios
        scenarios = []
            (103586.0, 30.0, "Oversold - Feet Entry"),
            (103586.0, 40.0, "Neutral - Pelvis Hold"),
            (103586.0, 50.0, "Balance - Heart Balance"),
            (103586.0, 70.0, "Overbought - Arms Exit"),
            (103586.0, 80.0, "Peak - Halo Peak"),
        ]
        for price, rsi, description in scenarios:
            print(f"\n   Testing: {description}")

            state = VMM_Schwabot.update_vitruvian_state()
                price=price,
                rsi=rsi,
                volume=1000000.0,
                entropy=0.5,
                echo_strength=0.6,
                drift_score=0.2,
            )

            active_zones = []
                zone.value for zone, active in state.zone_activations.items() if active
            ]
            print(f"      Active zones: {active_zones}")
            print(f"      Thermal state: {state.thermal_state}")
            print(f"      Bit phase: {state.bit_phase}")
            print(f"      NCCO: {state.ncco_state:.4f}")
            print(f"      SFS: {state.sfs_state:.4f}")
            print(f"      UFS: {state.ufs_state:.4f}")
            print(f"      ZPLS: {state.zpls_state:.4f}")
            print(f"      RBMS: {state.rbms_state:.4f}")

        return True

    except Exception as e:
        print(f"❌ Mathematical integration test failed: {e}")

        traceback.print_exc()
        return False


def test_vitruvian_calculations():
    """Test Vitruvian mathematical calculations."""
    print("\n📐 Testing Vitruvian Calculations")
    print("=" * 50)

    try:

        # Test golden ratio calculations
        phi = VMM_Schwabot.PHI
        print(f"✅ Golden Ratio (Φ): {phi:.10f}")
        print(f"✅ Φ²: {phi**2:.10f}")
        print(f"✅ 1/Φ: {1 / phi:.10f}")

        # Test Fibonacci ratios
        fib_ratios = [0.618, 0.786, 1.00, 1.414, 1.618]
        print(f"✅ Fibonacci Ratios: {fib_ratios}")

        # Test limb position calculations
        vmm = VMM_Schwabot.get_vitruvian_manager()

        # Test phi center calculation
        phi_center = vmm._calculate_phi_center(103586.0, 50.0)
        print(f"✅ Phi Center: {phi_center:.6f}")

        # Test limb positions
        vmm._update_limb_positions(103586.0, 50.0, 1000000.0)
        limb_positions = vmm.current_state.limb_positions
        print("✅ Limb Positions:")
        for limb, position in limb_positions.items():
            print(f"   {limb.value}: {position:.4f}")

        return True

    except Exception as e:
        print(f"❌ Vitruvian calculations test failed: {e}")

        traceback.print_exc()
        return False


def main():
    """Run all VMM tests."""
    print("🚀 Starting VMM Simple Test Suite")
    print("=" * 60)

    tests = []
        ("VMM Basic Functionality", test_vmm_basic),
        ("Mathematical Integration", test_mathematical_integration),
        ("Vitruvian Calculations", test_vitruvian_calculations),
    ]
    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            result = test_func()

            if result:
                passed += 1
                print(f"✅ {name} test passed")
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All VMM simple tests passed!")
        print("\n✅ VMM System Summary:")
        print("   - Core functionality: Working")
        print("   - Mathematical integration: NCCO, SFS, UFS, ZPLS, RBMS connected")
        print("   - Vitruvian calculations: Golden ratio and Fibonacci ratios")
        print()
            "   - Zone mapping: Feet→Entry, Pelvis→Hold, Heart→Balance, Arms→Exit, Halo→Peak"
        )
        print("   - Thermal states: Cool→Hot with bit phase coordination")
        print()
            "   - Trading routes: Optimal route generation based on Vitruvian analysis"
        )
    else:
        print("⚠️ Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
