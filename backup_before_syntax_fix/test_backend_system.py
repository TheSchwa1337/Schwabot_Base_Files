#!/usr/bin/env python3
"""
Comprehensive test script to verify the backend system is working across all core files.
"""

import sys
import traceback
from pathlib import Path


def test_backend_system():
    """Test the backend system functionality."""
    print("=== Testing Backend System ===\n")

    try:
        # Test backend_math.py
        from core.backend_math import get_backend, is_gpu, backend_info

        xp = get_backend()
        info = backend_info()

        print(f"✓ Backend system loaded successfully")
        print(f"  Backend: {info['backend']}")
        print(f"  GPU Accelerated: {info['accelerated']}")
        print(f"  Force CPU: {info['force_cpu']}")

        # Test basic array operations
        arr = xp.array([1, 2, 3, 4, 5], dtype=xp.float32)
        result = xp.sum(arr)
        print(f"  Array sum test: {result} ✓")

        # Test mathematical operations
        result = xp.sin(xp.pi / 2)
        print(f"  Math operation test: {result:.6f} ✓")

        return True

    except Exception as e:
        print(f"✗ Backend system test failed: {e}")
        traceback.print_exc()
        return False


def test_core_files():
    """Test core files that have been patched with backend_math."""
    print("\n=== Testing Core Files ===\n")

    core_files = [
        "core.backend_math",
        "core.fractal_core",
        "core.strategy_consensus_router",
        "core.qsc_enhanced_profit_allocator",
        "core.quantum_mathematical_bridge",
        "core.chrono_recursive_logic_function",
        "core.risk_manager",
        "core.zpe_zbe_core",
        "core.advanced_tensor_algebra",
        "core.tensor_weight_memory",
        "core.zpe_core",
    ]

    results = []

    for module_name in core_files:
        try:
            __import__(module_name)
            print(f"✓ {module_name} - Import successful")
            results.append((module_name, True))
        except Exception as e:
            print(f"✗ {module_name} - Import failed: {e}")
            results.append((module_name, False))

    return results


def test_backend_consistency():
    """Test that all files use the same backend consistently."""
    print("\n=== Testing Backend Consistency ===\n")

    try:
        from core.backend_math import get_backend, backend_info
        from core.fractal_core import xp as fractal_xp
        from core.quantum_mathematical_bridge import xp as quantum_xp
        from core.zpe_core import xp as zpe_xp

        main_xp = get_backend()
        info = backend_info()

        # Check that all modules use the same backend
        backends = [
            ("main", main_xp),
            ("fractal_core", fractal_xp),
            ("quantum_mathematical_bridge", quantum_xp),
            ("zpe_core", zpe_xp),
        ]

        all_consistent = True
        for name, backend in backends:
            if backend is main_xp:
                print(f"✓ {name} - Backend consistent")
            else:
                print(f"✗ {name} - Backend inconsistent")
                all_consistent = False

        if all_consistent:
            print(f"\n🎉 All modules use consistent backend: {info['backend']}")
        else:
            print(f"\n⚠️ Backend inconsistency detected")

        return all_consistent

    except Exception as e:
        print(f"✗ Backend consistency test failed: {e}")
        return False


def test_math_operations():
    """Test that math operations work correctly with the backend."""
    print("\n=== Testing Math Operations ===\n")

    try:
        from core.backend_math import get_backend

        xp = get_backend()

        # Test array operations
        arr1 = xp.array([1, 2, 3], dtype=xp.float32)
        arr2 = xp.array([4, 5, 6], dtype=xp.float32)

        # Basic operations
        sum_result = xp.sum(arr1)
        dot_result = xp.dot(arr1, arr2)
        mean_result = xp.mean(arr1)

        print(f"✓ Array sum: {sum_result}")
        print(f"✓ Dot product: {dot_result}")
        print(f"✓ Array mean: {mean_result}")

        # Test mathematical functions
        sin_result = xp.sin(xp.pi / 2)
        exp_result = xp.exp(1.0)
        log_result = xp.log(xp.e)

        print(f"✓ Sin(π/2): {sin_result:.6f}")
        print(f"✓ Exp(1): {exp_result:.6f}")
        print(f"✓ Log(e): {log_result:.6f}")

        # Test array creation and manipulation
        zeros = xp.zeros((3, 3))
        ones = xp.ones((2, 2))
        random_arr = xp.random.rand(5)

        print(f"✓ Zeros array shape: {zeros.shape}")
        print(f"✓ Ones array shape: {ones.shape}")
        print(f"✓ Random array shape: {random_arr.shape}")

        return True

    except Exception as e:
        print(f"✗ Math operations test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧠 Schwabot Backend System Test Suite\n")
    print("=" * 50)

    # Test backend system
    backend_ok = test_backend_system()

    # Test core files
    core_results = test_core_files()

    # Test backend consistency
    consistency_ok = test_backend_consistency()

    # Test math operations
    math_ok = test_math_operations()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    core_success = sum(1 for _, success in core_results if success)
    core_total = len(core_results)

    print(f"Backend System: {'✓ PASS' if backend_ok else '✗ FAIL'}")
    print(f"Core Files: {core_success}/{core_total} passed")
    print(f"Backend Consistency: {'✓ PASS' if consistency_ok else '✗ FAIL'}")
    print(f"Math Operations: {'✓ PASS' if math_ok else '✗ FAIL'}")

    if backend_ok and core_success == core_total and consistency_ok and math_ok:
        print("\n🎉 All tests passed! Backend system is fully functional.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
