from core.unified_math_system import unified_math
import numpy as np
from core.gpu_cpu_calculation_bridge import (
from core.gpu_offload_manager import GPUOffloadManager
from dataclasses import dataclass
from typing import Dict, List, Any
from utils.safe_print import safe_print, warn, error, success
import logging
import sys
import time

#!/usr/bin/env python3
"""
Connectivity and Compatibility Test Suite
========================================

Comprehensive test suite to validate the connectivity and compatibility
improvements between GPU and CPU systems, ensuring consistent functionality
as in the legacy system.
"""


# Add the project root to the path
sys.path.insert(0, ".")

    get_gpu_cpu_bridge,
    ExecutionPath,
    ThermalState,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data structure."""

    test_name: str
    success: bool
    execution_time_ms: float
    gpu_result: Any
    cpu_result: Any
    consistency_score: float
    error_message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConnectivityCompatibilityTester:
    """Test suite for connectivity and compatibility validation."""

    def __init__(self):
        self.bridge = get_gpu_cpu_bridge()
        self.gpu_manager = GPUOffloadManager()
        self.test_results: List[TestResult] = []

    def run_all_tests():-> Dict[str, Any]:
        """Run all connectivity and compatibility tests."""
        safe_print("🔗 Starting Connectivity and Compatibility Test Suite")
        safe_print("=" * 60)

        test_functions = [
            self.test_calculation_consistency,
            self.test_thermal_state_integration,
            self.test_error_recovery,
            self.test_memory_management,
            self.test_legacy_compatibility,
            self.test_performance_monitoring,
            self.test_gpu_cpu_fallback,
            self.test_matrix_operations,
            self.test_wave_entropy,
            self.test_tensor_scores,
        ]
        results = {}
        for test_func in test_functions:
            try:
                test_name = test_func.__name__
                safe_print(f"\n🧪 Running {test_name}...")

                start_time = time.time()
                result = test_func()
                execution_time = (time.time() - start_time) * 1000

                results[test_name] = {
                    "success": result,
                    "execution_time_ms": execution_time,
                }
                if result:
                    success(f"✅ {test_name} PASSED ({execution_time:.2f}ms)")
                else:
                    error(f"❌ {test_name} FAILED ({execution_time:.2f}ms)")

            except Exception as e:
                error(f"❌ {test_func.__name__} ERROR: {e}")
                results[test_func.__name__] = {
                    "success": False,
                    "execution_time_ms": 0,
                    "error": str(e),
                }
        self._print_summary(results)
        return results

    def test_calculation_consistency():-> bool:
        """Test that GPU and CPU calculations produce consistent results."""
        try:
            # Test matrix multiplication
            test_matrix = np.random.rand(10, 10)

            # Execute on both GPU and CPU
            gpu_result = self.bridge.execute_calculation(
                "matrix_multiply", test_matrix, force_path=ExecutionPath.GPU_ONLY
            )
            cpu_result = self.bridge.execute_calculation(
                "matrix_multiply", test_matrix, force_path=ExecutionPath.CPU_ONLY
            )

            # Validate consistency
            if not gpu_result.success or not cpu_result.success:
                return False

            # Check if results are consistent
            validation = self.bridge.consistency_validator.validate_calculation(
                gpu_result.result, cpu_result.result, "matrix_multiply"
            )

            self.test_results.append(
                TestResult(
                    test_name="calculation_consistency",
                    success=validation.is_consistent,
                    execution_time_ms=gpu_result.execution_time_ms
                    + cpu_result.execution_time_ms,
                    gpu_result=gpu_result.result,
                    cpu_result=cpu_result.result,
                    consistency_score=validation.is_consistent,
                    metadata={"max_difference": validation.max_difference},
                )
            )

            return validation.is_consistent

        except Exception as e:
            logger.error(f"Error in calculation consistency test: {e}")
            return False

    def test_thermal_state_integration():-> bool:
        """Test thermal state management integration."""
        try:
            # Test thermal state updates
            thermal_manager = self.bridge.thermal_manager

            # Simulate different temperatures
            test_temperatures = [45.0, 65.0, 75.0, 85.0]
            expected_states = [
                ThermalState.COOL,
                ThermalState.WARM,
                ThermalState.HOT,
                ThermalState.CRITICAL,
            ]

            for temp, expected_state in zip(test_temperatures, expected_states):
                actual_state = thermal_manager.update_thermal_state(temp)
                if actual_state != expected_state:
                    logger.error(
                        f"Thermal state mismatch: expected {expected_state}, got {actual_state}"
                    )
                    return False

            # Test calculation strategy based on thermal state
            np.random.rand(100, 100)

            # Cool state should prefer GPU
            thermal_manager.update_thermal_state(45.0)
            cool_strategy = thermal_manager.get_calculation_strategy(
                "matrix_multiply", 10000
            )

            # Critical state should prefer CPU
            thermal_manager.update_thermal_state(85.0)
            critical_strategy = thermal_manager.get_calculation_strategy(
                "matrix_multiply", 10000
            )

            success = (
                cool_strategy == ExecutionPath.GPU_ONLY
                and critical_strategy == ExecutionPath.CPU_ONLY
            )

            self.test_results.append(
                TestResult(
                    test_name="thermal_state_integration",
                    success=success,
                    execution_time_ms=0.0,
                    gpu_result=cool_strategy,
                    cpu_result=critical_strategy,
                    consistency_score=1.0 if success else 0.0,
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in thermal state integration test: {e}")
            return False

    def test_error_recovery():-> bool:
        """Test error recovery mechanisms."""
        try:
            # Test GPU fallback when GPU is not available
            test_data = np.random.rand(5, 5)

            # Force CPU fallback
            result = self.bridge.execute_calculation(
                "matrix_multiply", test_data, force_path=ExecutionPath.CPU_ONLY
            )

            # Test with invalid data
            invalid_data = None
            error_result = self.bridge.execute_calculation(
                "matrix_multiply", invalid_data, force_path=ExecutionPath.FALLBACK
            )

            # Both should handle errors gracefully
            success = result.success and not error_result.success

            self.test_results.append(
                TestResult(
                    test_name="error_recovery",
                    success=success,
                    execution_time_ms=result.execution_time_ms,
                    gpu_result=result.success,
                    cpu_result=error_result.success,
                    consistency_score=1.0 if success else 0.0,
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in error recovery test: {e}")
            return False

    def test_memory_management():-> bool:
        """Test memory management functionality."""
        try:
            # Test GPU memory usage tracking
            gpu_memory_before = self.gpu_manager._get_gpu_memory_usage()

            # Perform some operations
            test_matrices = [np.random.rand(50, 50) for _ in range(5)]
            for matrix in test_matrices:
                self.gpu_manager.matrix_operation_gpu([matrix], "multiply")

            gpu_memory_after = self.gpu_manager._get_gpu_memory_usage()

            # Memory should be tracked (even if not available)
            memory_tracked = gpu_memory_after >= gpu_memory_before

            self.test_results.append(
                TestResult(
                    test_name="memory_management",
                    success=memory_tracked,
                    execution_time_ms=0.0,
                    gpu_result=gpu_memory_before,
                    cpu_result=gpu_memory_after,
                    consistency_score=1.0 if memory_tracked else 0.0,
                )
            )

            return memory_tracked

        except Exception as e:
            logger.error(f"Error in memory management test: {e}")
            return False

    def test_legacy_compatibility():-> bool:
        """Test legacy system compatibility."""
        try:
            # Test that unified math system still works
            test_data = np.random.rand(10, 10)

            # Test basic operations
            add_result = unified_math.add(1.0, 2.0)
            multiply_result = unified_math.multiply(3.0, 4.0)
            matrix_result = unified_math.matrix_multiply(test_data, test_data)

            # All should work as expected
            success = (
                add_result == 3.0
                and multiply_result == 12.0
                and matrix_result is not None
            )

            self.test_results.append(
                TestResult(
                    test_name="legacy_compatibility",
                    success=success,
                    execution_time_ms=0.0,
                    gpu_result=add_result,
                    cpu_result=multiply_result,
                    consistency_score=1.0 if success else 0.0,
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in legacy compatibility test: {e}")
            return False

    def test_performance_monitoring():-> bool:
        """Test performance monitoring functionality."""
        try:
            # Get initial performance stats
            initial_stats = self.bridge.get_performance_stats()

            # Perform some operations
            test_data = np.random.rand(20, 20)
            for _ in range(5):
                self.bridge.execute_calculation("matrix_multiply", test_data)

            # Get updated performance stats
            updated_stats = self.bridge.get_performance_stats()

            # Stats should be updated
            stats_updated = (
                updated_stats["total_calculations"]
                > initial_stats["total_calculations"]
            )

            self.test_results.append(
                TestResult(
                    test_name="performance_monitoring",
                    success=stats_updated,
                    execution_time_ms=0.0,
                    gpu_result=initial_stats["total_calculations"],
                    cpu_result=updated_stats["total_calculations"],
                    consistency_score=1.0 if stats_updated else 0.0,
                )
            )

            return stats_updated

        except Exception as e:
            logger.error(f"Error in performance monitoring test: {e}")
            return False

    def test_gpu_cpu_fallback():-> bool:
        """Test GPU to CPU fallback mechanisms."""
        try:
            # Test bit phase resolution
            test_hashes = ["a1b2c3d4", "e5f6g7h8", "i9j0k1l2"]

            # Test GPU path
            gpu_phases = self.gpu_manager.resolve_bit_phase_gpu(test_hashes, "8bit")

            # Test CPU fallback
            cpu_phases = self.gpu_manager._resolve_bit_phase_cpu(test_hashes, "8bit")

            # Results should be consistent
            phases_consistent = gpu_phases == cpu_phases

            self.test_results.append(
                TestResult(
                    test_name="gpu_cpu_fallback",
                    success=phases_consistent,
                    execution_time_ms=0.0,
                    gpu_result=gpu_phases,
                    cpu_result=cpu_phases,
                    consistency_score=1.0 if phases_consistent else 0.0,
                )
            )

            return phases_consistent

        except Exception as e:
            logger.error(f"Error in GPU-CPU fallback test: {e}")
            return False

    def test_matrix_operations():-> bool:
        """Test matrix operations consistency."""
        try:
            test_matrix = np.random.rand(15, 15)

            # Test different matrix operations
            operations = ["multiply", "eigenvalues", "transpose"]

            for operation in operations:
                # GPU result
                gpu_result = self.gpu_manager.matrix_operation_gpu(
                    [test_matrix], operation
                )

                # CPU result
                cpu_result = self.gpu_manager._matrix_operation_cpu(
                    [test_matrix], operation
                )

                # Validate consistency
                if len(gpu_result) != len(cpu_result):
                    return False

                for gpu_res, cpu_res in zip(gpu_result, cpu_result):
                    if gpu_res is None or cpu_res is None:
                        continue

                    if gpu_res.shape != cpu_res.shape:
                        return False

                    # Check if results are close (within tolerance)
                    if not np.allclose(gpu_res, cpu_res, rtol=1e-5, atol=1e-8):
                        return False

            self.test_results.append(
                TestResult(
                    test_name="matrix_operations",
                    success=True,
                    execution_time_ms=0.0,
                    gpu_result="consistent",
                    cpu_result="consistent",
                    consistency_score=1.0,
                )
            )

            return True

        except Exception as e:
            logger.error(f"Error in matrix operations test: {e}")
            return False

    def test_wave_entropy():-> bool:
        """Test wave entropy calculation consistency."""
        try:
            test_sequences = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]

            # GPU calculation
            gpu_entropies = self.gpu_manager.wave_entropy_gpu(test_sequences)

            # CPU calculation
            cpu_entropies = self.gpu_manager._wave_entropy_cpu(test_sequences)

            # Results should be consistent
            entropies_consistent = len(gpu_entropies) == len(cpu_entropies)

            if entropies_consistent:
                for gpu_ent, cpu_ent in zip(gpu_entropies, cpu_entropies):
                    if abs(gpu_ent - cpu_ent) > 1e-6:
                        entropies_consistent = False
                        break

            self.test_results.append(
                TestResult(
                    test_name="wave_entropy",
                    success=entropies_consistent,
                    execution_time_ms=0.0,
                    gpu_result=gpu_entropies,
                    cpu_result=cpu_entropies,
                    consistency_score=1.0 if entropies_consistent else 0.0,
                )
            )

            return entropies_consistent

        except Exception as e:
            logger.error(f"Error in wave entropy test: {e}")
            return False

    def test_tensor_scores():-> bool:
        """Test tensor score calculation consistency."""
        try:
            entry_prices = [100.0, 200.0, 300.0]
            current_prices = [110.0, 190.0, 320.0]
            phases = [1, 2, 3]

            # GPU calculation
            gpu_scores = self.gpu_manager.tensor_score_gpu(
                entry_prices, current_prices, phases
            )

            # CPU calculation
            cpu_scores = self.gpu_manager._tensor_score_cpu(
                entry_prices, current_prices, phases
            )

            # Results should be consistent
            scores_consistent = len(gpu_scores) == len(cpu_scores)

            if scores_consistent:
                for gpu_score, cpu_score in zip(gpu_scores, cpu_scores):
                    if abs(gpu_score - cpu_score) > 1e-6:
                        scores_consistent = False
                        break

            self.test_results.append(
                TestResult(
                    test_name="tensor_scores",
                    success=scores_consistent,
                    execution_time_ms=0.0,
                    gpu_result=gpu_scores,
                    cpu_result=cpu_scores,
                    consistency_score=1.0 if scores_consistent else 0.0,
                )
            )

            return scores_consistent

        except Exception as e:
            logger.error(f"Error in tensor scores test: {e}")
            return False

    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        safe_print("\n" + "=" * 60)
        safe_print("📊 CONNECTIVITY AND COMPATIBILITY TEST SUMMARY")
        safe_print("=" * 60)

        total_tests = len(results)
        passed_tests = sum(
            1 for result in results.values() if result.get("success", False)
        )
        failed_tests = total_tests - passed_tests

        safe_print(f"Total Tests: {total_tests}")
        safe_print(f"Passed: {passed_tests} ✅")
        safe_print(f"Failed: {failed_tests} ❌")
        safe_print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        if failed_tests > 0:
            safe_print("\n❌ Failed Tests:")
            for test_name, result in results.items():
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown error")
                    safe_print(f"  - {test_name}: {error_msg}")

        # Print detailed results
        safe_print("\n📋 Detailed Results:")
        for test_result in self.test_results:
            status = "✅ PASS" if test_result.success else "❌ FAIL"
            safe_print(f"  {status} {test_result.test_name}")
            if test_result.metadata:
                for key, value in test_result.metadata.items():
                    safe_print(f"    {key}: {value}")

        # Overall assessment
        if passed_tests == total_tests:
            success(
                "\n🎉 ALL TESTS PASSED! System connectivity and compatibility verified."
            )
        elif passed_tests >= total_tests * 0.8:
            warn(
                f"\n⚠️  MOST TESTS PASSED ({passed_tests}/{total_tests}). Minor issues detected."
            )
        else:
            error(
                f"\n🚨 MANY TESTS FAILED ({failed_tests}/{total_tests}). Critical issues detected."
            )


def main():
    """Main test execution function."""
    try:
        tester = ConnectivityCompatibilityTester()
        results = tester.run_all_tests()

        # Return exit code based on results
        total_tests = len(results)
        passed_tests = sum(
            1 for result in results.values() if result.get("success", False)
        )

        if passed_tests == total_tests:
            return 0  # All tests passed
        elif passed_tests >= total_tests * 0.8:
            return 1  # Most tests passed
        else:
            return 2  # Many tests failed

    except Exception as e:
        error(f"Test suite execution failed: {e}")
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
