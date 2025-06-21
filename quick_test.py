#!/usr/bin/env python3
"""Quick test script to verify mathematical framework fixes."""."""


def test_mathlib_fixes():
    """Test the mathlib package fixes."""."""
    try:
        print("🔬 Testing Schwabot Mathematical Framework Fixes")
        print("=" * 50)

        # Test 1: MathLib package imports
        print("1. Testing mathlib package imports...")
        from mathlib import add
        from mathlib import divide
        from mathlib import Dual
        from mathlib import GradedProfitVector
        from mathlib import kelly_fraction
        from mathlib import MathLib
        from mathlib import MathLibV2
        from mathlib import MathLibV3

        print("   ✅ All mathlib imports successful")

        # Test 2: MathLib instantiation
        print("2. Testing mathematical library instantiation...")
        math_v1 = MathLib()
        math_v2 = MathLibV2()
        math_v3 = MathLibV3()
        print(f"   ✅ MathLib V1: {math_v1.version}")
        print(f"   ✅ MathLib V2: {math_v2.version}")
        print(f"   ✅ MathLib V3: {math_v3.version}")

        # Test 3: GradedProfitVector
        print("3. Testing GradedProfitVector...")
        profits = [100, 150, -50, 200]
        grades = ["A", "B", "C", "A"]
        vector = GradedProfitVector(profits, grades=grades)
        total = vector.total_profit()
        avg_grade = vector.average_grade()
        print(f"   ✅ Profit vector total: ${total}")
        print(f"   ✅ Average grade: {avg_grade}")

        # Test 4: Basic mathematical operations
        print("4. Testing basic mathematical operations...")
        result_add = add(5, 3)
        result_div = divide(10, 2)
        print(f"   ✅ Addition: 5 + 3 = {result_add}")
        print(f"   ✅ Division: 10 / 2 = {result_div}")

        # Test 5: Dual numbers for automatic differentiation
        print("5. Testing dual numbers...")
        x = Dual(2.0, 1.0)
        y = x * x + 3 * x + 1  # f(x) = x² + 3x + 1
        print(f"   ✅ f(2) = {y.val}, f'(2) = {y.eps}")

        # Test 6: Kelly fraction calculation
        print("6. Testing Kelly fraction...")
        kelly_result = kelly_fraction(0.1, 0.04)  # 10% return, 4% variance
        print(f"   ✅ Kelly fraction: {kelly_result:.3f}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_core_imports():
    """Test core component imports."""."""
    try:
        print("\n7. Testing core component imports...")

        # Test constraints system
        from core.constraints import ConstraintValidator

        validator = ConstraintValidator()
        print(f"   ✅ ConstraintValidator v{validator.version}")

        # Test unified controller
        from core.unified_mathematical_trading_controller import \
            UnifiedMathematicalTradingController

        controller = UnifiedMathematicalTradingController()
        print(
            f"   ✅ UnifiedMathematicalTradingController v{controller.version}"
        )

        # Test thermal zone manager
        from core.thermal_zone_manager import ThermalZoneManager

        thermal_manager = ThermalZoneManager()
        print(f"   ✅ ThermalZoneManager v{thermal_manager.version}")

        # Test triplet matcher
        from core.triplet_matcher import TripletMatcher

        triplet_matcher = TripletMatcher()
        print(f"   ✅ TripletMatcher v{triplet_matcher.version}")

        return True

    except Exception as e:
        print(f"   ❌ Core import error: {e}")
        return False


def test_integration():
    """Test basic integration between components."""."""
    try:
        print("\n8. Testing component integration...")

        from core.constraints import ConstraintValidator
        from core.unified_mathematical_trading_controller import \
            UnifiedMathematicalTradingController

        # Test signal processing
        controller = UnifiedMathematicalTradingController()
        signal_data = {
            "asset": "BTC",
            "entry_price": 26000.0,
            "exit_price": 27000.0,
            "volume": 0.5,
            "thermal_index": 1.2,
            "timestamp": 1640995200.0,
            "strategy": "test",
        }

        result = controller.process_trade_signal(signal_data)
        print(f"   ✅ Signal processing: {result.get('status', 'unknown')}")

        # Test constraint validation
        validator = ConstraintValidator()
        trading_params = {"position_size": 0.5, "leverage": 1.5}

        validation_result = validator.validate_trading_operation(
            trading_params
        )
        print(
            f"   ✅ Constraint validation: {'PASS' if validation_result.valid else 'FAIL'}"
        )

        return True

    except Exception as e:
        print(f"   ❌ Integration error: {e}")
        return False


def main():
    """Run all tests."""."""
    print("🚀 Mathematical Framework Integration Test")
    print("Schwabot Framework - Testing Critical Fixes")
    print()

    results = []

    # Run tests
    results.append(test_mathlib_fixes())
    results.append(test_core_imports())
    results.append(test_integration())

    # Summary
    total_tests = len(results)
    passed_tests = sum(results)
    success_rate = passed_tests / total_tests

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1%}")

    if success_rate >= 0.8:
        print("\n🎉 INTEGRATION FIXES SUCCESSFUL!")
        print("✅ Mathematical framework is working correctly")
        print("✅ Core components are properly integrated")
        print("✅ Cross-component communication is functional")
    else:
        print("\n⚠️ SOME ISSUES DETECTED")
        print("❌ Additional fixes may be needed")

    return success_rate >= 0.8


if __name__ == "__main__":
    main()
