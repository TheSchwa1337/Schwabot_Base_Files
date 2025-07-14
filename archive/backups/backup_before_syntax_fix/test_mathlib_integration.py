import logging
import time

import numpy as np

from core.mathlib_v4 import MathLibV4
from core.matrix_math_utils import analyze_price_matrix
from core.unified_math_system import UnifiedMathSystem

#!/usr/bin/env python3
"""
Test MathLibV4 Integration
==========================

Simple test to verify that MathLibV4 is properly integrated
into the Schwabot pipeline and working correctly.
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_mathlib_v4_standalone():
    """Test MathLibV4 standalone functionality."""
    print("🧮 Testing MathLibV4 Standalone...")

    try:

        ml4 = MathLibV4(precision=64)

        # Test data
        test_data = {}
            "prices": [50000, 50001, 50002, 50001, 50003, 50005, 50004, 50006],
            "volumes": [1000, 1200, 800, 1100, 900, 1300, 950, 1100],
            "timestamps": [time.time() - i for i in range(8, 0, -1)],
        }

        result = ml4.calculate_dlt_metrics(test_data)

        if "error" not in result:
            print("✅ MathLibV4 standalone test PASSED")
            print(f"   Pattern Hash: {result['pattern_hash'][:10]}...")
            print(f"   Triplet Lock: {result['triplet_lock']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Warp Factor: {result['warp_factor']:.3f}")
            return True
        else:
            print(f"❌ MathLibV4 standalone test FAILED: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ MathLibV4 standalone test FAILED: {e}")
        return False


def test_unified_math_integration():
    """Test MathLibV4 integration with UnifiedMathSystem."""
    print("\n🔗 Testing Unified Math System Integration...")

    try:

        ums = UnifiedMathSystem()

        # Test data
        test_data = {}
            "prices": [50000, 50001, 50002, 50001, 50003, 50005, 50004, 50006],
            "volumes": [1000, 1200, 800, 1100, 900, 1300, 950, 1100],
        }

        result = ums.dlt_analysis(test_data)

        if result["status"] == "success":
            print("✅ Unified Math System integration test PASSED")
            dlt_metrics = result["dlt_metrics"]
            print(f"   Pattern Hash: {dlt_metrics['pattern_hash'][:10]}...")
            print(f"   Triplet Lock: {dlt_metrics['triplet_lock']}")
            print(f"   Confidence: {dlt_metrics['confidence']:.3f}")
            print(f"   Warp Factor: {dlt_metrics['warp_factor']:.3f}")
            return True
        else:
            print(f"❌ Unified Math System integration test FAILED: {result}")
            return False

    except Exception as e:
        print(f"❌ Unified Math System integration test FAILED: {e}")
        return False


def test_demo_integration():
    """Test that the demo can import all components without errors."""
    print("\n🎯 Testing Demo Integration...")

    try:
        # Test imports

        print("✅ All core components imported successfully")

        # Test MathLibV4 initialization
        ml4 = MathLibV4()
        print(f"✅ MathLibV4 initialized: v{ml4.version.value}")

        # Test matrix math utils

        test_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        matrix_result = analyze_price_matrix(test_matrix)
        print()
            f"✅ Matrix math utils working: {matrix_result.get('stability_score', 'N/A'):.3f}"
        )

        return True

    except Exception as e:
        print(f"❌ Demo integration test FAILED: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🧠 MathLibV4 Integration Test Suite")
    print("=" * 50)

    tests = []
        test_mathlib_v4_standalone,
        test_unified_math_integration,
        test_demo_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests PASSED! MathLibV4 is properly integrated.")
        return True
    else:
        print("⚠️  Some tests FAILED. Check the integration.")
        return False


if __name__ == "__main__":
    main()
