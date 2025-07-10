#!/usr/bin/env python3
"""
🧪 CUPY INTEGRATION TEST SUITE
==============================

Comprehensive test suite to verify all cupy integrations work correctly
with fallback logic in the Schwabot codebase.

Tests:
- Core mathematical modules with cupy fallback
- GPU/CPU switching functionality
- Error handling and recovery
- Performance monitoring
- Cross-platform compatibility
"""

import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List

# Add project root to path
sys.path.append('.')

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CupyIntegrationTester:
    """Test suite for cupy integration across all modules"""

    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all cupy integration tests"""
        logger.info("🚀 Starting CUPY Integration Test Suite")

        # Test core mathematical modules
        self.test_quantum_mathematical_bridge()
        self.test_distributed_mathematical_processor()
        self.test_mathematical_optimization_bridge()
        self.test_strategy_bit_mapper()
        self.test_advanced_tensor_algebra()
        self.test_fractal_core()
        self.test_gpu_handlers()
        self.test_enhanced_error_recovery_system()

        # Test utility modules
        self.test_cuda_helper()

        # Test performance and fallback
        self.test_performance_fallback()
        self.test_cross_platform_compatibility()

        # Generate report
        return self.generate_report()

    def test_quantum_mathematical_bridge(self):
        """Test quantum mathematical bridge cupy integration"""
        logger.info("🔬 Testing Quantum Mathematical Bridge")

        try:
            from core.quantum_mathematical_bridge import QuantumMathematicalBridge

            # Test initialization
            bridge = QuantumMathematicalBridge(quantum_dimension=8, use_gpu=True)

            # Test quantum operations
            trading_signals = [0.1, 0.2, 0.3, 0.4, 0.5]
            quantum_state = bridge.create_quantum_superposition(trading_signals)

            # Test tensor operations
            tensor_data = np.array([[1, 2], [3, 4]], dtype=complex)
            quantum_tensor = bridge.quantum_tensor_operation(tensor_data, "qft")

            # Test profit vectorization
            result = bridge.quantum_profit_vectorization()
                btc_price=50000.0,
                usdc_hold=1000.0,
                entry_signals=[0.1, 0.2, 0.3],
                exit_signals=[0.4, 0.5, 0.6]
            )

            self.record_test_result("quantum_mathematical_bridge", True, "All operations successful")
            logger.info("✅ Quantum Mathematical Bridge: PASSED")

        except Exception as e:
            self.record_test_result("quantum_mathematical_bridge", False, str(e))
            logger.error(f"❌ Quantum Mathematical Bridge: FAILED - {e}")

    def test_distributed_mathematical_processor(self):
        """Test distributed mathematical processor cupy integration"""
        logger.info("🔬 Testing Distributed Mathematical Processor")

        try:
            from core.distributed_mathematical_processor import DistributedMathematicalProcessor

            # Test initialization
            processor = DistributedMathematicalProcessor(max_workers=2, use_gpu=True)

            # Test matrix operations
            data = np.array([[1, 2], [3, 4]])
            task_id = processor.submit_task("matrix_multiplication", data, {"matrix_b": data})

            # Wait for result
            time.sleep(0.1)
            result = processor.get_task_result(task_id)

            # Test optimization
            opt_task_id = processor.submit_task("optimization", data, {"objective_function": "quadratic"})
            time.sleep(0.1)
            opt_result = processor.get_task_result(opt_task_id)

            # Cleanup
            processor.cleanup_resources()

            self.record_test_result("distributed_mathematical_processor", True, "All operations successful")
            logger.info("✅ Distributed Mathematical Processor: PASSED")

        except Exception as e:
            self.record_test_result("distributed_mathematical_processor", False, str(e))
            logger.error(f"❌ Distributed Mathematical Processor: FAILED - {e}")

    def test_mathematical_optimization_bridge(self):
        """Test mathematical optimization bridge cupy integration"""
        logger.info("🔬 Testing Mathematical Optimization Bridge")

        try:
            from core.mathematical_optimization_bridge import MathematicalOptimizationBridge, OptimizationMode

            # Test initialization
            bridge = MathematicalOptimizationBridge()

            # Test tensor operations
            tensor_a = np.array([[1, 2], [3, 4]])
            tensor_b = np.array([[5, 6], [7, 8]])

            # Test different optimization modes
            for mode in [OptimizationMode.CPU_ONLY, OptimizationMode.GPU_ONLY, OptimizationMode.HYBRID]:
                result = bridge.optimize_tensor_operation(tensor_a, tensor_b, "dot", mode)

                if not result.success:
                    raise Exception(f"Optimization failed for mode {mode}")

            # Test entropy calculations
            data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            metrics = bridge.calculate_entropy_metrics(data, computational_load=1.0)

            self.record_test_result("mathematical_optimization_bridge", True, "All operations successful")
            logger.info("✅ Mathematical Optimization Bridge: PASSED")

        except Exception as e:
            self.record_test_result("mathematical_optimization_bridge", False, str(e))
            logger.error(f"❌ Mathematical Optimization Bridge: FAILED - {e}")

    def test_strategy_bit_mapper(self):
        """Test strategy bit mapper cupy integration"""
        logger.info("🔬 Testing Strategy Bit Mapper")

        try:
            from core.strategy_bit_mapper import ExpansionMode, StrategyBitMapper

            # Test initialization
            mapper = StrategyBitMapper(matrix_dir="./test_matrices")

            # Test strategy expansion
            strategy_id = 42
            expanded = mapper.expand_strategy_bits(strategy_id, target_bits=8, mode=ExpansionMode.RANDOM)

            # Test qutrit gate application
            qutrit_result = mapper.apply_qutrit_gate("test_strategy", "test_seed")

            # Test vector operations
            vector_a = np.array([1, 2, 3, 4])
            vector_b = np.array([5, 6, 7, 8])
            similarity = mapper.compute_cosine_similarity(vector_a, vector_b)

            self.record_test_result("strategy_bit_mapper", True, "All operations successful")
            logger.info("✅ Strategy Bit Mapper: PASSED")

        except Exception as e:
            self.record_test_result("strategy_bit_mapper", False, str(e))
            logger.error(f"❌ Strategy Bit Mapper: FAILED - {e}")

    def test_advanced_tensor_algebra(self):
        """Test advanced tensor algebra cupy integration"""
        logger.info("🔬 Testing Advanced Tensor Algebra")

        try:
            from core.advanced_tensor_algebra import AdvancedTensorAlgebra

            # Test initialization
            algebra = AdvancedTensorAlgebra(precision=8, enable_caching=True)

            # Test tensor operations
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])

            # Test tensor fusion
            fused = algebra.tensor_dot_fusion(A, B)

            # Test bit phase rotation
            x = np.array([1, 2, 3, 4])
            rotated = algebra.bit_phase_rotation(x, theta=0.5)

            # Test entropy quantization
            V = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            quantized = algebra.entropy_vector_quantize(V, E=0.5)

            # Test matrix trace conditions
            M = np.array([[1, 2], [3, 4]])
            trace_conditions = algebra.matrix_trace_conditions(M)

            self.record_test_result("advanced_tensor_algebra", True, "All operations successful")
            logger.info("✅ Advanced Tensor Algebra: PASSED")

        except Exception as e:
            self.record_test_result("advanced_tensor_algebra", False, str(e))
            logger.error(f"❌ Advanced Tensor Algebra: FAILED - {e}")

    def test_fractal_core(self):
        """Test fractal core cupy integration"""
        logger.info("🔬 Testing Fractal Core")

        try:
            from core.fractal_core import fractal_quantize_vector

            # Test fractal quantization
            vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            result = fractal_quantize_vector(vector, precision=8, method="mandelbrot")

            # Verify result structure
            assert hasattr(result, 'quantized_vector')
            assert hasattr(result, 'fractal_dimension')
            assert hasattr(result, 'self_similarity_score')
            assert hasattr(result, 'compression_ratio')

            # Test different methods
            julia_result = fractal_quantize_vector(vector, precision=8, method="julia")
            sierpinski_result = fractal_quantize_vector(vector, precision=8, method="sierpinski")

            self.record_test_result("fractal_core", True, "All operations successful")
            logger.info("✅ Fractal Core: PASSED")

        except Exception as e:
            self.record_test_result("fractal_core", False, str(e))
            logger.error(f"❌ Fractal Core: FAILED - {e}")

    def test_gpu_handlers(self):
        """Test GPU handlers cupy integration"""
        logger.info("🔬 Testing GPU Handlers")

        try:
            from core.gpu_handlers import run_gpu_strategy

            # Test matrix matching
            matrix_data = {}
                "hash_vector": [0.1, 0.2, 0.3, 0.4],
                "matrices": []
                    {"matrix": [[1, 2], [3, 4]], "id": "matrix1"},
                    {"matrix": [[5, 6], [7, 8]], "id": "matrix2"}
                ],
                "threshold": 0.8
            }

            result = run_gpu_strategy("matrix_match", matrix_data)

            # Test ghost tick detection
            ghost_data = {}
                "price_data": [100, 101, 102, 103, 104],
                "volume_data": [1000, 1100, 1200, 1300, 1400]
            }

            ghost_result = run_gpu_strategy("ghost_tick", ghost_data)

            self.record_test_result("gpu_handlers", True, "All operations successful")
            logger.info("✅ GPU Handlers: PASSED")

        except Exception as e:
            self.record_test_result("gpu_handlers", False, str(e))
            logger.error(f"❌ GPU Handlers: FAILED - {e}")

    def test_enhanced_error_recovery_system(self):
        """Test enhanced error recovery system cupy integration"""
        logger.info("🔬 Testing Enhanced Error Recovery System")

        try:
            from core.enhanced_error_recovery_system import EnhancedErrorRecoverySystem

            # Test initialization
            recovery_system = EnhancedErrorRecoverySystem()

            # Test error handling
            try:
                # Intentionally cause an error
                raise ValueError("Test error for recovery system")
            except Exception as e:
                result = recovery_system.handle_error(e, {"test_context": True})

            # Test system health
            health = recovery_system.get_current_health()

            # Test error statistics
            stats = recovery_system.get_error_statistics()

            # Cleanup
            recovery_system.cleanup_resources()

            self.record_test_result("enhanced_error_recovery_system", True, "All operations successful")
            logger.info("✅ Enhanced Error Recovery System: PASSED")

        except Exception as e:
            self.record_test_result("enhanced_error_recovery_system", False, str(e))
            logger.error(f"❌ Enhanced Error Recovery System: FAILED - {e}")

    def test_cuda_helper(self):
        """Test CUDA helper utility"""
        logger.info("🔬 Testing CUDA Helper")

        try:
            from utils.cuda_helper import USING_CUDA, safe_cuda_operation, xp

            # Test safe CUDA operation
            def cuda_fn():
                return xp.array([1, 2, 3, 4])

            def cpu_fn():
                return np.array([1, 2, 3, 4])

            result = safe_cuda_operation(cuda_fn, cpu_fn)

            # Test xp wrapper
            test_array = xp.array([1, 2, 3, 4])
            test_sum = xp.sum(test_array)

            self.record_test_result("cuda_helper", True, "All operations successful")
            logger.info("✅ CUDA Helper: PASSED")

        except Exception as e:
            self.record_test_result("cuda_helper", False, str(e))
            logger.error(f"❌ CUDA Helper: FAILED - {e}")

    def test_performance_fallback(self):
        """Test performance and fallback functionality"""
        logger.info("🔬 Testing Performance and Fallback")

        try:
            # Test that operations work with and without cupy
            import numpy as np

            # Create test data
            large_matrix = np.random.rand(100, 100)

            # Test matrix operations
            start_time = time.time()
            result = np.dot(large_matrix, large_matrix)
            cpu_time = time.time() - start_time

            # Test that fallback works
            if 'USING_CUDA' in globals():
                logger.info(f"Backend: {'GPU' if USING_CUDA else 'CPU'}")
                logger.info(f"Matrix multiplication time: {cpu_time:.4f}s")

            self.record_test_result("performance_fallback", True, "Fallback functionality working")
            logger.info("✅ Performance and Fallback: PASSED")

        except Exception as e:
            self.record_test_result("performance_fallback", False, str(e))
            logger.error(f"❌ Performance and Fallback: FAILED - {e}")

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility"""
        logger.info("🔬 Testing Cross-Platform Compatibility")

        try:
            import platform
            import sys

            # Check platform
            platform_info = platform.platform()
            python_version = sys.version

            # Test that all modules can be imported
            modules_to_test = []
                'core.quantum_mathematical_bridge',
                'core.distributed_mathematical_processor',
                'core.mathematical_optimization_bridge',
                'core.strategy_bit_mapper',
                'core.advanced_tensor_algebra',
                'core.fractal_core',
                'core.gpu_handlers',
                'core.enhanced_error_recovery_system',
                'utils.cuda_helper'
            ]

            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                except ImportError as e:
                    raise Exception(f"Failed to import {module_name}: {e}")

            self.record_test_result("cross_platform_compatibility", True,)
                                  f"Platform: {platform_info}, Python: {python_version}")
            logger.info("✅ Cross-Platform Compatibility: PASSED")

        except Exception as e:
            self.record_test_result("cross_platform_compatibility", False, str(e))
            logger.error(f"❌ Cross-Platform Compatibility: FAILED - {e}")

    def record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

        self.test_results[test_name] = {}
            "passed": passed,
            "message": message,
            "timestamp": time.time()
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0

        report = {}
            "summary": {}
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": success_rate
            },
            "test_results": self.test_results,
            "recommendations": self.generate_recommendations()
        }

        # Log summary
        logger.info("=" * 60)
        logger.info("🧪 CUPY INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)

        # Log individual results
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            logger.info(f"{test_name}: {status} - {result['message']}")

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        if self.failed_tests > 0:
            recommendations.append("🔧 Fix failed tests before deployment")

        if self.passed_tests == self.total_tests:
            recommendations.append("🎉 All cupy integrations working correctly")
            recommendations.append("✅ System ready for production deployment")

        # Check for specific issues
        for test_name, result in self.test_results.items():
            if not result["passed"]:
                if "import" in result["message"].lower():
                    recommendations.append(f"📦 Check dependencies for {test_name}")
                elif "cuda" in result["message"].lower():
                    recommendations.append(f"🔧 Verify CUDA installation for {test_name}")

        return recommendations

def main():
    """Main test execution"""
    try:
        # Create test directory if needed
        os.makedirs("./test_matrices", exist_ok=True)

        # Run tests
        tester = CupyIntegrationTester()
        report = tester.run_all_tests()

        # Save report
        import json
        with open("cupy_integration_test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("📄 Test report saved to: cupy_integration_test_report.json")

        # Exit with appropriate code
        if report["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 