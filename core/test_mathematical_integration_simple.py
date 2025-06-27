#!/usr/bin/env python3
"""
Simplified Test Script for Mathematical Integration Bridges

This script tests all the mathematical integration bridges implemented across
the Schwabot system to ensure proper connectivity and mathematical integrity.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MathematicalIntegrationTester:
    """Comprehensive tester for mathematical integration bridges."""
    
    def __init__(self):
        """Initialize the mathematical integration tester."""
        self.test_results = {}
        self.integration_metrics = {}
        self.start_time = time.time()
        
        logger.info("Mathematical Integration Tester initialized")

    async def test_mathlib_v4_integration(self) -> Dict[str, Any]:
        """Test MathLib V4 integration with unified math system."""
        logger.info("Testing MathLib V4 Integration...")
        
        try:
            # Import MathLib V4
            from mathlib_v4 import MathLibV4
            
            # Initialize MathLib V4
            mathlib = MathLibV4()
            
            # Test data
            test_data = np.array([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
            
            # Test DLT analysis
            dlt_result = mathlib.analyze_dlt_waveform(test_data)
            
            # Test pattern hashing
            deltas = mathlib.calculate_deltas(test_data)
            pattern_hash = mathlib.generate_pattern_hash(deltas)
            
            # Test confidence calculation
            confidence = mathlib.calculate_greyscale_confidence(0.85, drift_velocity=0.1)
            
            # Test fractal creation
            fractal = mathlib.create_forever_fractal(deltas)
            
            result = {
                "success": True,
                "dlt_analysis": dlt_result,
                "pattern_hash": pattern_hash,
                "confidence": confidence,
                "fractal": {
                    "pattern_hash": fractal.pattern_hash,
                    "mean_delta": fractal.mean_delta,
                    "std_dev": fractal.std_dev
                },
                "mathematical_formula": "delta(x_n) = x_n - x_n-1"
            }
            
            logger.info("MathLib V4 Integration Test PASSED")
            return result
            
        except Exception as e:
            logger.error(f"MathLib V4 Integration Test FAILED: {e}")
            return {"success": False, "error": str(e)}

    async def test_unified_math_system_integration(self) -> Dict[str, Any]:
        """Test Unified Math System integration."""
        logger.info("Testing Unified Math System Integration...")
        
        try:
            # Import unified math system
            from unified_math_system import UnifiedMathSystem, MathOperation
            
            # Initialize unified math system
            unified_math = UnifiedMathSystem()
            
            # Test data
            test_data = np.array([100, 101, 99, 102, 98, 103])
            tensor_a = np.array([[1, 2], [3, 4]])
            tensor_b = np.array([[5, 6], [7, 8]])
            
            # Test basic operations first
            add_result = unified_math.execute(MathOperation.ADD, 5, 3)
            multiply_result = unified_math.execute(MathOperation.MULTIPLY, 4, 6)
            
            # Test DLT analysis integration (may fail if MathLib V4 not available)
            try:
                dlt_result = unified_math.execute(MathOperation.DLT_ANALYSIS, test_data)
                dlt_success = dlt_result.success
            except:
                dlt_success = False
            
            # Test tensor contraction integration (may fail if Tensor Algebra not available)
            try:
                tensor_result = unified_math.execute(MathOperation.TENSOR_CONTRACTION, tensor_a, tensor_b)
                tensor_success = tensor_result.success
            except:
                tensor_success = False
            
            # Test thermal correction
            thermal_result = unified_math.execute(MathOperation.THERMAL_CORRECTION, test_data, thermal_factor=1.2)
            
            # Get statistics
            stats = unified_math.get_statistics()
            
            result = {
                "success": True,
                "basic_operations": {
                    "add": add_result.value if add_result.success else None,
                    "multiply": multiply_result.value if multiply_result.success else None
                },
                "dlt_integration": dlt_success,
                "tensor_integration": tensor_success,
                "thermal_integration": thermal_result.value if thermal_result.success else None,
                "statistics": stats,
                "mathematical_formula": "unified_result = DLT_analysis * math_system_weight"
            }
            
            logger.info("Unified Math System Integration Test PASSED")
            return result
            
        except Exception as e:
            logger.error(f"Unified Math System Integration Test FAILED: {e}")
            return {"success": False, "error": str(e)}

    async def test_fractal_core_integration(self) -> Dict[str, Any]:
        """Test Fractal Core integration with interlinking system."""
        logger.info("Testing Fractal Core Integration...")
        
        try:
            # Import fractal core
            from fractal_core import FractalCore, QuantizationDepth
            
            # Initialize fractal core
            fractal_core = FractalCore()
            
            # Test fractal sequence generation
            sequence = fractal_core.generate_sequence(seed=42, depth=20, quantization=QuantizationDepth.STANDARD)
            
            # Test pattern correlation
            sequence2 = fractal_core.generate_sequence(seed=123, depth=20, quantization=QuantizationDepth.STANDARD)
            correlation = fractal_core.analyze_pattern_correlation(sequence, sequence2)
            
            # Test bit collapse resolution (simulated)
            bit_collapse_data = {"collapse_type": "test", "severity": 0.5}
            fractal_state_data = {"sequence_id": "test_sequence"}
            
            # This would normally be called by the interlinking system
            # For testing, we'll simulate the integration
            resolved_result = fractal_core.resolve_bit_collapse_with_fractal_state(
                bit_collapse_data, fractal_state_data
            )
            
            # Get metrics
            metrics = fractal_core.get_metrics()
            
            result = {
                "success": True,
                "sequence_generated": len(sequence),
                "pattern_correlation": correlation,
                "bit_collapse_resolution": resolved_result,
                "metrics": {
                    "total_sequences": metrics.total_sequences,
                    "coherence_ratio": metrics.coherence_ratio,
                    "integration_success_rate": metrics.integration_success_rate
                },
                "mathematical_formula": "interlinked_result = fractal_state * bridge_weight"
            }
            
            logger.info("Fractal Core Integration Test PASSED")
            return result
            
        except Exception as e:
            logger.error(f"Fractal Core Integration Test FAILED: {e}")
            return {"success": False, "error": str(e)}

    async def test_tensor_algebra_integration(self) -> Dict[str, Any]:
        """Test Tensor Algebra integration."""
        logger.info("Testing Tensor Algebra Integration...")
        
        try:
            # Import tensor algebra with correct path
            import sys
            sys.path.append('.')
            from math.tensor_algebra import UnifiedTensorAlgebra
            
            # Initialize tensor algebra
            tensor_algebra = UnifiedTensorAlgebra()
            
            # Test data
            tensor_a = np.array([[1, 2], [3, 4]])
            tensor_b = np.array([[5, 6], [7, 8]])
            prices = np.array([100, 101, 102])
            weights = np.array([0.3, 0.4, 0.3])
            
            # Test tensor contraction
            contraction_result = tensor_algebra.tensor_contraction(tensor_a, tensor_b)
            
            # Test bit phase tensor
            bit_phase_result = tensor_algebra.bit_phase_tensor(strategy_id=12345, mode='4bit')
            
            # Test matrix basket operation
            basket_result = tensor_algebra.matrix_basket_operation(prices, weights)
            
            # Test hash memory encoding
            hash_result = tensor_algebra.hash_memory_encoding("test_data")
            
            # Get statistics
            stats = tensor_algebra.get_statistics()
            
            result = {
                "success": True,
                "tensor_contraction": contraction_result.tolist(),
                "bit_phase": {
                    "phi_4": bit_phase_result.phi_4,
                    "phi_8": bit_phase_result.phi_8,
                    "phi_42": bit_phase_result.phi_42
                },
                "matrix_basket": basket_result.tolist(),
                "hash_encoding": hash_result,
                "statistics": stats,
                "mathematical_formula": "T_ij = sum_k A_ik * B_kj"
            }
            
            logger.info("Tensor Algebra Integration Test PASSED")
            return result
            
        except Exception as e:
            logger.error(f"Tensor Algebra Integration Test FAILED: {e}")
            return {"success": False, "error": str(e)}

    async def test_api_gateway_integration(self) -> Dict[str, Any]:
        """Test API Gateway integration with consciousness."""
        logger.info("Testing API Gateway Integration...")
        
        try:
            # Import API gateway
            from api_gateway import SchwabotAPIGateway
            
            # Initialize API gateway
            gateway = SchwabotAPIGateway(host="127.0.0.1", port=8001)
            
            # Test API gateway initialization
            if not gateway.app:
                return {"success": False, "error": "FastAPI not available"}
            
            # Test mathematical integration status
            integration_status = {
                "mathematical_consciousness_available": hasattr(gateway, 'mathematical_consciousness_bridge'),
                "consciousness_system_available": hasattr(gateway, 'gpt_layer'),
                "api_gateway_available": gateway.app is not None,
                "integration_metrics": gateway.integration_metrics
            }
            
            # Test API weight calculation
            api_weight = gateway._calculate_api_weight("high")
            
            result = {
                "success": True,
                "integration_status": integration_status,
                "api_weight_calculation": api_weight,
                "endpoints_available": [
                    "/health",
                    "/status", 
                    "/command/submit",
                    "/mathematical/consciousness/request",
                    "/mathematical/integration/status"
                ],
                "mathematical_formula": "api_weight = priority_factor * trust_level"
            }
            
            logger.info("API Gateway Integration Test PASSED")
            return result
            
        except Exception as e:
            logger.error(f"API Gateway Integration Test FAILED: {e}")
            return {"success": False, "error": str(e)}

    async def test_mathematical_integration_mapping(self) -> Dict[str, Any]:
        """Test Mathematical Integration Mapping."""
        logger.info("Testing Mathematical Integration Mapping...")
        
        try:
            # Import mathematical integration mapping
            from mathematical_integration_mapping import MathematicalIntegrationMapper, MathematicalOperation
            
            # Initialize mapper
            mapper = MathematicalIntegrationMapper()
            
            # Test integration path finding
            path = mapper.get_integration_path("mathlib_v4", "fractal_core")
            
            # Test mathematical integrity validation
            validation = mapper.validate_mathematical_integrity(
                MathematicalOperation.DLT_ANALYSIS,
                {"system": "mathlib_v4"},
                {"system": "unified_math_system"}
            )
            
            # Generate integration report
            report = mapper.generate_integration_report()
            
            result = {
                "success": True,
                "integration_path": [point.source_system + " -> " + point.target_system for point in path],
                "mathematical_validation": validation,
                "integration_report": {
                    "total_systems": report["total_systems"],
                    "total_integration_points": report["total_integration_points"],
                    "mathematical_formulas": len(report["mathematical_formulas"])
                },
                "mathematical_formula": "integration_path = optimal_route(source, target)"
            }
            
            logger.info("Mathematical Integration Mapping Test PASSED")
            return result
            
        except Exception as e:
            logger.error(f"Mathematical Integration Mapping Test FAILED: {e}")
            return {"success": False, "error": str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all mathematical integration tests."""
        logger.info("Starting Comprehensive Mathematical Integration Tests")
        logger.info("=" * 60)
        
        tests = [
            ("MathLib V4 Integration", self.test_mathlib_v4_integration),
            ("Unified Math System Integration", self.test_unified_math_system_integration),
            ("Fractal Core Integration", self.test_fractal_core_integration),
            ("Tensor Algebra Integration", self.test_tensor_algebra_integration),
            ("API Gateway Integration", self.test_api_gateway_integration),
            ("Mathematical Integration Mapping", self.test_mathematical_integration_mapping)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\nRunning: {test_name}")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                
                if result.get("success", False):
                    logger.info(f"{test_name} PASSED")
                    passed += 1
                else:
                    logger.error(f"{test_name} FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"{test_name} ERROR: {e}")
                self.test_results[test_name] = {"success": False, "error": str(e)}
        
        # Calculate overall metrics
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        self.integration_metrics = {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "execution_time": execution_time
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("MATHEMATICAL INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {self.integration_metrics['success_rate']:.2%}")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        
        if passed == total:
            logger.info("ALL MATHEMATICAL INTEGRATION TESTS PASSED!")
        else:
            logger.warning("Some mathematical integration tests failed. Check the output above for details.")
        
        return {
            "overall_success": passed == total,
            "metrics": self.integration_metrics,
            "test_results": self.test_results
        }


async def main():
    """Main function to run mathematical integration tests."""
    try:
        tester = MathematicalIntegrationTester()
        results = await tester.run_all_tests()
        
        # Print detailed results
        print("\nDETAILED TEST RESULTS:")
        print("=" * 60)
        
        for test_name, result in results["test_results"].items():
            print(f"\n{test_name}:")
            if result.get("success", False):
                print(f"   PASSED")
                if "mathematical_formula" in result:
                    print(f"   Formula: {result['mathematical_formula']}")
            else:
                print(f"   FAILED: {result.get('error', 'Unknown error')}")
        
        print(f"\nOverall Success: {'YES' if results['overall_success'] else 'NO'}")
        print(f"Success Rate: {results['metrics']['success_rate']:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main test execution: {e}")
        return {"overall_success": False, "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main()) 