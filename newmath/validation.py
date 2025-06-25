from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
NEWMATH VALIDATION
=================

Comprehensive validation and testing framework for the new math library.
Clean implementation for testing all mathematical components.
"""

from core.unified_math_system import unified_math
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def run_basic_tests() -> bool:
    """
    Run basic validation tests for the new math library.
    
    Returns:
        True if all basic tests pass
    """
    try:
        safe_print("🔬 Running NewMath Basic Validation Tests...")
        
        # Test tensor operations
        from . import tensor_ops
        A = np.random.random((2, 3))
        B = np.random.random((3, 2))
        result = tensor_ops.tensor_contraction(A, B)
        tensor_test = result.shape == (2, 2)
        safe_print(f"✓ Tensor contraction: {'PASS' if tensor_test else 'FAIL'}")
        
        # Test profit math
        from . import profit_math
        prices = np.array([100.0, 101.5, 99.8, 102.3])
        derivatives = profit_math.profit_derivative(prices)
        profit_test = len(derivatives) == len(prices) - 1
        safe_print(f"✓ Profit derivative: {'PASS' if profit_test else 'FAIL'}")
        
        # Test entropy calculations
        from . import entropy_calc
        entropy = entropy_calc.calculate_entropy(1000.0, 0.1)
        entropy_test = entropy > 0
        safe_print(f"✓ Entropy calculation: {'PASS' if entropy_test else 'FAIL'}")
        
        # Test hash vectors
        from . import hash_vectors
        hash_vec = hash_vectors.generate_hash_vector(100.0, 2.5, 42)
        hash_test = len(hash_vec) == 64
        safe_print(f"✓ Hash generation: {'PASS' if hash_test else 'FAIL'}")
        
        all_passed = tensor_test and profit_test and entropy_test and hash_test
        safe_print(f"🎯 Basic Tests: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        return all_passed
        
    except Exception as e:
        safe_print(f"❌ Basic tests failed: {e}")
        return False


def run_full_tests() -> Dict[str, Any]:
    """
    Run comprehensive validation tests.
    
    Returns:
        Dictionary with detailed test results
    """
    safe_print("🧮 Running NewMath Comprehensive Validation Suite...")
    safe_print("=" * 60)
    
    start_time = time.time()
    results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": []
    }
    
    # Test each module
    test_modules = [
        ("Tensor Operations", _test_tensor_operations),
        ("Profit Mathematics", _test_profit_math),
        ("Entropy Calculations", _test_entropy_calc),
        ("Hash Vectors", _test_hash_vectors),
        ("Matrix Utils", _test_matrix_utils),
        ("Render Engine", _test_render_engine)
    ]
    
    for module_name, test_func in test_modules:
        try:
            test_start = time.time()
            module_result = test_func()
            test_time = time.time() - test_start
            
            results["total_tests"] += 1
            if module_result["passed"]:
                results["passed_tests"] += 1
                safe_print(f"✅ {module_name}: PASSED ({test_time:.3f}s)")
            else:
                results["failed_tests"] += 1
                safe_print(f"❌ {module_name}: FAILED - {module_result.get('error', 'Unknown error')}")
            
            results["test_details"].append({
                "module": module_name,
                "passed": module_result["passed"],
                "execution_time": test_time,
                "details": module_result
            })
            
        except Exception as e:
            results["total_tests"] += 1
            results["failed_tests"] += 1
            safe_print(f"❌ {module_name}: ERROR - {str(e)}")
            results["test_details"].append({
                "module": module_name,
                "passed": False,
                "execution_time": 0,
                "error": str(e)
            })
    
    total_time = time.time() - start_time
    success_rate = (
        (results["passed_tests"] / results["total_tests"]) * 100
        if results["total_tests"] > 0 else 0
    )

    safe_print("\n" + "=" * 60)
    safe_print("📊 COMPREHENSIVE TEST SUMMARY")
    safe_print(f"Total Tests: {results['total_tests']}")
    safe_print(f"Passed: {results['passed_tests']}")
    safe_print(f"Failed: {results['failed_tests']}")
    safe_print(f"Success Rate: {success_rate:.1f}%")
    safe_print(f"Total Time: {total_time:.3f} seconds")

    results["success_rate"] = success_rate
    results["total_time"] = total_time

    return results


def _test_tensor_operations() -> Dict[str, Any]:
    """Test tensor operations module."""
    try:
        from . import tensor_ops

        # Test tensor contraction
        A = np.random.random((3, 4))
        B = np.random.random((4, 5))
        result = tensor_ops.tensor_contraction(A, B)
        contraction_ok = result.shape == (3, 5)

        # Test bit phase operations
        phi_4, phi_8, phi_42 = tensor_ops.bit_phase_operations(0x12345)
        bit_phase_ok = all(isinstance(x, int) for x in [phi_4, phi_8, phi_42])

        # Test similarity
        tensor_a = np.random.random((2, 3))
        tensor_b = np.random.random((2, 3))
        similarity = tensor_ops.tensor_similarity(tensor_a, tensor_b)
        similarity_ok = 0.0 <= similarity <= 1.0

        all_passed = contraction_ok and bit_phase_ok and similarity_ok
        return {
            "passed": all_passed,
            "contraction": contraction_ok,
            "bit_phase": bit_phase_ok,
            "similarity": similarity_ok
        }

    except Exception as e:
        return {"passed": False, "error": str(e)}


def _test_profit_math() -> Dict[str, Any]:
    """Test profit mathematics module."""
    try:
        from . import profit_math

        # Test profit derivative
        prices = np.array([100.0, 101.5, 99.8, 102.3, 98.7])
        derivatives = profit_math.profit_derivative(prices)
        derivative_ok = len(derivatives) == len(prices) - 1

        # Test trade execution logic
        should_trade = profit_math.should_execute_trade(2.5, 2.0)
        trade_logic_ok = should_trade is True

        # Test momentum calculation
        momentum = profit_math.profit_momentum(prices, window=3)
        momentum_ok = len(momentum) == len(prices)

        all_passed = derivative_ok and trade_logic_ok and momentum_ok
        return {
            "passed": all_passed,
            "derivative": derivative_ok,
            "trade_logic": trade_logic_ok,
            "momentum": momentum_ok
        }

    except Exception as e:
        return {"passed": False, "error": str(e)}


def _test_entropy_calc() -> Dict[str, Any]:
    """Test entropy calculations module."""
    try:
        from . import entropy_calc

        # Test basic entropy
        entropy = entropy_calc.calculate_entropy(1000.0, 0.1)
        entropy_ok = entropy > 0

        # Test entropy trigger
        trigger = entropy_calc.entropy_trigger(50.0, entropy)
        trigger_ok = isinstance(trigger, float)

        # Test volume entropy
        volumes = np.array([1000, 1200, 800, 1500])
        prices = np.array([100, 101, 99, 102])
        vol_entropy = entropy_calc.volume_entropy(volumes, prices)
        vol_entropy_ok = len(vol_entropy) == len(volumes)

        all_passed = entropy_ok and trigger_ok and vol_entropy_ok
        return {
            "passed": all_passed,
            "entropy": entropy_ok,
            "trigger": trigger_ok,
            "volume_entropy": vol_entropy_ok
        }

    except Exception as e:
        return {"passed": False, "error": str(e)}


def _test_hash_vectors() -> Dict[str, Any]:
    """Test hash vectors module."""
    try:
        from . import hash_vectors

        # Test hash generation
        hash_vec = hash_vectors.generate_hash_vector(100.0, 2.5, 42)
        hash_gen_ok = len(hash_vec) == 64

        # Test similarity
        hash_a = "abc123"
        hash_b = "abc456"
        similarity = hash_vectors.hash_similarity_score(hash_a, hash_b)
        similarity_ok = 0.0 <= similarity <= 1.0

        # Test memory encoding
        data = np.array([1.0, 2.0, 3.0])
        encoded = hash_vectors.memory_encoding(data)
        encoding_ok = len(encoded) == len(data)

        all_passed = hash_gen_ok and similarity_ok and encoding_ok
        return {
            "passed": all_passed,
            "generation": hash_gen_ok,
            "similarity": similarity_ok,
            "encoding": encoding_ok
        }

    except Exception as e:
        return {"passed": False, "error": str(e)}


def _test_matrix_utils() -> Dict[str, Any]:
    """Test matrix utilities module."""
    try:
        from . import matrix_utils

        # Test safe multiplication
        A = np.random.random((3, 4))
        B = np.random.random((4, 2))
        result, info = matrix_utils.safe_matrix_multiply(A, B)
        multiply_ok = info["success"] and result.shape == (3, 2)

        # Test condition check
        matrix = np.random.random((3, 3))
        condition = matrix_utils.condition_check(matrix)
        condition_ok = "shape" in condition

        all_passed = multiply_ok and condition_ok
        return {"passed": all_passed, "multiplication": multiply_ok, "condition": condition_ok}

    except Exception as e:
        return {"passed": False, "error": str(e)}


def _test_render_engine() -> Dict[str, Any]:
    """Test render engine module."""
    try:
        from . import render_engine

        # Test price line rendering
        prices = [100.0, 101.5, 99.8, 102.3]
        price_result = render_engine.render_price_line(prices)
        price_ok = "points" in price_result and len(price_result["points"]) == len(prices)

        # Test function plotting
        func_values = [np.unified_math.sin(x / 10) for x in range(20)]
        func_result = render_engine.plot_function(func_values)
        func_ok = "points" in func_result

        all_passed = price_ok and func_ok
        return {"passed": all_passed, "price_rendering": price_ok, "function_plotting": func_ok}

    except Exception as e:
        return {"passed": False, "error": str(e)}


# Export main functions
__all__ = [
    'run_basic_tests',
    'run_full_tests'
] 