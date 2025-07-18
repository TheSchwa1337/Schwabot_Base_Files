import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from core.gpu_offload_manager import GPUOffloadManager
from core.unified_math_system import unified_math
from utils.safe_print import error, safe_print, success, warn

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

from core.gpu_cpu_bridge import (
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

    def run_all_tests(self) -> Dict[str, Any]:
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

    def test_calculation_consistency(self) -> bool:
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

    def test_thermal_state_integration(self) -> bool:
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
            test_matrix = np.random.rand(100, 100)

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

    def test_error_recovery(self) -> bool:
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

    def test_memory_management(self) -> bool:
        """Test memory management functionality."""
        try:
            # Test GPU memory usage tracking
            gpu_memory_before = self.gpu_manager._get_gpu_memory_usage()

            # Perform some operations
            test_matrices = [np.random.rand(50, 50) for _ in range(5)]
            for matrix in test_matrices:
                self.gpu_manager.matrix_operation_gpu([matrix], "multiply")

            gpu_memory_after = self.gpu_manager._get_gpu_memory_usage()

            # Check if memory usage is tracked
            memory_tracked = gpu_memory_after > gpu_memory_before

            # Test memory cleanup
            self.gpu_manager.cleanup_gpu_memory()
            gpu_memory_cleanup = self.gpu_manager._get_gpu_memory_usage()

            # Memory should be reduced after cleanup
            memory_cleanup_works = gpu_memory_cleanup < gpu_memory_after

            success = memory_tracked and memory_cleanup_works

            self.test_results.append(
                TestResult(
                    test_name="memory_management",
                    success=success,
                    execution_time_ms=0.0,
                    gpu_result=gpu_memory_after,
                    cpu_result=gpu_memory_cleanup,
                    consistency_score=1.0 if success else 0.0,
                    metadata={
                        "memory_before": gpu_memory_before,
                        "memory_after": gpu_memory_after,
                        "memory_cleanup": gpu_memory_cleanup,
                    },
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in memory management test: {e}")
            return False

    def test_legacy_compatibility(self) -> bool:
        """Test compatibility with legacy systems."""
        try:
            # Test legacy API compatibility
            legacy_data = {"legacy_format": True, "data": [1, 2, 3, 4, 5]}

            # Convert to new format
            converted_data = self.bridge.convert_legacy_format(legacy_data)

            # Test backward compatibility
            backward_compatible = self.bridge.is_backward_compatible(converted_data)

            success = backward_compatible

            self.test_results.append(
                TestResult(
                    test_name="legacy_compatibility",
                    success=success,
                    execution_time_ms=0.0,
                    gpu_result=converted_data,
                    cpu_result=backward_compatible,
                    consistency_score=1.0 if success else 0.0,
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in legacy compatibility test: {e}")
            return False

    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring capabilities."""
        try:
            # Test performance metrics collection
            start_time = time.time()
            test_matrix = np.random.rand(100, 100)

            # Perform operation with monitoring
            result = self.bridge.execute_calculation(
                "matrix_multiply", test_matrix, enable_monitoring=True
            )

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000

            # Check if performance metrics are available
            metrics_available = hasattr(result, "performance_metrics")
            if metrics_available:
                metrics = result.performance_metrics
                has_gpu_metrics = "gpu_utilization" in metrics
                has_cpu_metrics = "cpu_utilization" in metrics
                has_memory_metrics = "memory_usage" in metrics
            else:
                has_gpu_metrics = has_cpu_metrics = has_memory_metrics = False

            success = metrics_available and has_gpu_metrics and has_cpu_metrics and has_memory_metrics

            self.test_results.append(
                TestResult(
                    test_name="performance_monitoring",
                    success=success,
                    execution_time_ms=execution_time,
                    gpu_result=has_gpu_metrics,
                    cpu_result=has_cpu_metrics,
                    consistency_score=1.0 if success else 0.0,
                    metadata={"metrics_available": metrics_available},
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in performance monitoring test: {e}")
            return False

    def test_gpu_cpu_fallback(self) -> bool:
        """Test GPU to CPU fallback mechanisms."""
        try:
            # Test automatic fallback when GPU is unavailable
            test_data = np.random.rand(10, 10)

            # Simulate GPU unavailability
            self.bridge.simulate_gpu_unavailable()

            # Execute calculation (should fallback to CPU)
            result = self.bridge.execute_calculation(
                "matrix_multiply", test_data, force_path=ExecutionPath.AUTO
            )

            # Should succeed with CPU fallback
            fallback_success = result.success and result.execution_path == ExecutionPath.CPU_ONLY

            # Restore GPU availability
            self.bridge.restore_gpu_availability()

            # Test with GPU available
            gpu_result = self.bridge.execute_calculation(
                "matrix_multiply", test_data, force_path=ExecutionPath.AUTO
            )

            gpu_success = gpu_result.success and gpu_result.execution_path == ExecutionPath.GPU_ONLY

            success = fallback_success and gpu_success

            self.test_results.append(
                TestResult(
                    test_name="gpu_cpu_fallback",
                    success=success,
                    execution_time_ms=result.execution_time_ms + gpu_result.execution_time_ms,
                    gpu_result=fallback_success,
                    cpu_result=gpu_success,
                    consistency_score=1.0 if success else 0.0,
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in GPU-CPU fallback test: {e}")
            return False

    def test_matrix_operations(self) -> bool:
        """Test matrix operations on both GPU and CPU."""
        try:
            # Test various matrix operations
            test_matrix_a = np.random.rand(20, 20)
            test_matrix_b = np.random.rand(20, 20)

            operations = ["multiply", "add", "subtract", "transpose"]
            results = {}

            for operation in operations:
                # GPU operation
                gpu_result = self.bridge.execute_calculation(
                    f"matrix_{operation}", [test_matrix_a, test_matrix_b], force_path=ExecutionPath.GPU_ONLY
                )

                # CPU operation
                cpu_result = self.bridge.execute_calculation(
                    f"matrix_{operation}", [test_matrix_a, test_matrix_b], force_path=ExecutionPath.CPU_ONLY
                )

                # Validate consistency
                if gpu_result.success and cpu_result.success:
                    validation = self.bridge.consistency_validator.validate_calculation(
                        gpu_result.result, cpu_result.result, f"matrix_{operation}"
                    )
                    results[operation] = validation.is_consistent
                else:
                    results[operation] = False

            # All operations should be consistent
            all_consistent = all(results.values())

            self.test_results.append(
                TestResult(
                    test_name="matrix_operations",
                    success=all_consistent,
                    execution_time_ms=0.0,
                    gpu_result=results,
                    cpu_result=all_consistent,
                    consistency_score=sum(results.values()) / len(results),
                    metadata={"operation_results": results},
                )
            )

            return all_consistent

        except Exception as e:
            logger.error(f"Error in matrix operations test: {e}")
            return False

    def test_wave_entropy(self) -> bool:
        """Test wave entropy calculations."""
        try:
            # Test wave entropy calculation
            test_signal = np.random.rand(1000)

            # GPU calculation
            gpu_result = self.bridge.execute_calculation(
                "wave_entropy", test_signal, force_path=ExecutionPath.GPU_ONLY
            )

            # CPU calculation
            cpu_result = self.bridge.execute_calculation(
                "wave_entropy", test_signal, force_path=ExecutionPath.CPU_ONLY
            )

            # Validate consistency
            if gpu_result.success and cpu_result.success:
                validation = self.bridge.consistency_validator.validate_calculation(
                    gpu_result.result, cpu_result.result, "wave_entropy"
                )
                success = validation.is_consistent
            else:
                success = False

            self.test_results.append(
                TestResult(
                    test_name="wave_entropy",
                    success=success,
                    execution_time_ms=gpu_result.execution_time_ms + cpu_result.execution_time_ms,
                    gpu_result=gpu_result.result if gpu_result.success else None,
                    cpu_result=cpu_result.result if cpu_result.success else None,
                    consistency_score=validation.is_consistent if success else 0.0,
                    metadata={"max_difference": validation.max_difference if success else None},
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in wave entropy test: {e}")
            return False

    def test_tensor_scores(self) -> bool:
        """Test tensor score calculations."""
        try:
            # Test tensor score calculation
            test_tensor = np.random.rand(10, 10, 10)

            # GPU calculation
            gpu_result = self.bridge.execute_calculation(
                "tensor_score", test_tensor, force_path=ExecutionPath.GPU_ONLY
            )

            # CPU calculation
            cpu_result = self.bridge.execute_calculation(
                "tensor_score", test_tensor, force_path=ExecutionPath.CPU_ONLY
            )

            # Validate consistency
            if gpu_result.success and cpu_result.success:
                validation = self.bridge.consistency_validator.validate_calculation(
                    gpu_result.result, cpu_result.result, "tensor_score"
                )
                success = validation.is_consistent
            else:
                success = False

            self.test_results.append(
                TestResult(
                    test_name="tensor_scores",
                    success=success,
                    execution_time_ms=gpu_result.execution_time_ms + cpu_result.execution_time_ms,
                    gpu_result=gpu_result.result if gpu_result.success else None,
                    cpu_result=cpu_result.result if cpu_result.success else None,
                    consistency_score=validation.is_consistent if success else 0.0,
                    metadata={"max_difference": validation.max_difference if success else None},
                )
            )

            return success

        except Exception as e:
            logger.error(f"Error in tensor scores test: {e}")
            return False

    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        safe_print("\n" + "=" * 60)
        safe_print("📊 CONNECTIVITY AND COMPATIBILITY TEST SUMMARY")
        safe_print("=" * 60)

        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get("success", False))
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
        passed_tests = sum(1 for result in results.values() if result.get("success", False))

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
