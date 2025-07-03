        from core.unified_math_system import UnifiedMathSystem
        from core.dual_unicore_handler import DualUnicoreHandler
        from core.dualistic_thought_engines import DualisticThoughtEngines
        from core.phase_bit_integration import PhaseBitIntegration
        from core.price_precision_utils import format_price
        from core.unified_api_coordinator import ExchangeType, UnifiedApiCoordinator
        from core.unified_profit_vectorization_system import (
import sys
import traceback

#!/usr/bin/env python3
"""
Test script to verify all core module imports and basic functionality.
"""



def test_imports():
    """Test all core module imports."""

    print("🔍 Testing Core Module Imports...")
    print("=" * 50)

    # Test 1: Basic imports
    try:
        print("✅ Basic core imports successful")
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

    # Test 2: Phase bit integration
    try:
        print("✅ Phase bit integration import successful")
    except Exception as e:
        print(f"❌ Phase bit integration import failed: {e}")
        return False

    # Test 3: Profit vectorization
    try:
        print("✅ Profit vectorization import successful")
    except Exception as e:
        print(f"❌ Profit vectorization import failed: {e}")
        return False

    # Test 4: Dual unicore handler
    try:
        print("✅ Dual unicore handler import successful")
    except Exception as e:
        print(f"❌ Dual unicore handler import failed: {e}")
        return False

    # Test 5: Dualistic thought engines
    try:
        print("✅ Dualistic thought engines import successful")
    except Exception as e:
        print(f"❌ Dualistic thought engines import failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_functionality():
    """Test basic functionality of core modules."""

    print("\n🧪 Testing Core Module Functionality...")
    print("=" * 50)

    try:
        # Test UnifiedMathSystem

        math_system = UnifiedMathSystem()
        result = math_system.add(10.5, 5.2)
        print(f"✅ Math system add: {result}")

        # Test price formatting

        formatted_price = format_price(50321.123456789, decimals=6)
        print(f"✅ Price formatting: {formatted_price}")

        # Test exchange types

        coordinator = UnifiedApiCoordinator()
        coordinator.add_exchange(ExchangeType.COINBASE, {"api_key": "test"})
        print(f"✅ API coordinator: {len(coordinator.exchanges)} exchanges")

        # Test phase bit integration

        phase_system = PhaseBitIntegration()
        result = phase_system.resolve_bit_phase("test_data", "auto")
        print(f"✅ Phase bit integration: {result.bit_phase.value}")

        # Test profit vectorization
            UnifiedProfitVectorizationSystem,
        )

        profit_system = UnifiedProfitVectorizationSystem()
        sample_data = {"time_series": [100, 105, 98, 110], "total_profit": 110}
        result = profit_system.vectorize_profit_stream(sample_data)
        print(f"✅ Profit vectorization: efficiency {result.profit_efficiency:.3f}")

        # Test dual core handler

        handler = DualUnicoreHandler(max_workers=2)
        math_result = handler.execute_mathematical_operation(
            "mean", [1.0, 2.0, 3.0, 4.0]
        )
        handler.shutdown()
        print(f"✅ Dual core handler: mean = {math_result['result']:.2f}")

        # Test dualistic thought engines

        engines = DualisticThoughtEngines()

        sample_market_data = {
            "rsi": 45.0,
            "macd_signal": 0.005,
            "volume_change": 0.15,
            "current_price": 61500.0,
            "moving_average": 61200.0,
            "previous_close": 61300.0,
            "price_history": [61000, 61100, 61200, 61300, 61400, 61500],
            "volume_history": [100, 110, 105, 120, 115, 125],
            "phase_data": [0.5, 0.6, 0.4, 0.7],
            "volatility": 0.4,
            "sentiment_score": 0.6,
            "performance_delta": 0.02,
            "actual_profit": 50.0,
            "consensus_signal": "hold",
        }
        thought_result = engines.process_market_data(sample_market_data)
        print(
            f"✅ Dualistic engines: {thought_result.decision} (confidence: {thought_result.confidence:.3f})"
        )

        performance = engines.get_engine_performance()
        print(f"✅ Engine performance: {performance['success_rate']:.2%} success rate")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Starting Core Module Tests")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n❌ Import tests failed. Cannot proceed with functionality tests.")
        return False

    # Test functionality
    functionality_ok = test_functionality()

    print("\n📊 Test Results Summary")
    print("=" * 50)
    print(f"Imports: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    print(f"Functionality: {'✅ PASSED' if functionality_ok else '❌ FAILED'}")

    if imports_ok and functionality_ok:
        print("\n🎉 All tests passed! Core modules are working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
