# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Mathematical Integration Validator - Schwabot UROS v1.0
=====================================================

Comprehensive validation of all mathematical foundations working together.
Tests the complete mathematical pipeline from bit phase resolution to
hash memory encoding, ensuring all components integrate seamlessly.

Mathematical Pipeline:
1. Bit Phase Resolution → 2. Tensor Contraction → 3. Profit Routing → 4. Entropy Compensation → 5. Hash Memory
"""

import json
import time
import logging
from core.unified_math_system import unified_math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from math.tensor_algebra import UnifiedTensorAlgebra, BitPhaseResult
from tensor_matcher import TensorMatcher
from bit_resolution_engine import BitResolutionEngine
from matrix_mapper import MatrixMapper
from profit_routing_engine import ProfitRoutingEngine
from dlt_waveform_engine import DLTWaveformEngine

logger = logging.getLogger(__name__)

@dataclass
class IntegrationTestResult:
    """Result of integration test."""
    test_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineValidationResult:
    """Result of pipeline validation."""
    pipeline_name: str
    all_tests_passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    test_results: List[IntegrationTestResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MathematicalIntegrationValidator:
    """
    Mathematical integration validator for Schwabot system.
    
    Validates the complete mathematical pipeline:
    1. Bit Phase Resolution (φ₄, φ₈, φ₄₂)
    2. Matrix Basket Tensor Algebra (Tᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ)
    3. Profit Routing Differential Calculus (dP/dt)
    4. Entropy Compensation and Drift Dynamics (E(t) = unified_math.log(V + 1) / (1 + δ))
    5. Hash Memory Vector Encoding (H(t) = SHA256(P_t || ΔP || φ_t))
    """
    
    def __init__(self):
        """Initialize the mathematical integration validator."""
        self.tensor_algebra = UnifiedTensorAlgebra()
        self.tensor_matcher = TensorMatcher()
        self.bit_resolution_engine = BitResolutionEngine()
        self.matrix_mapper = MatrixMapper()
        self.profit_routing_engine = ProfitRoutingEngine()
        self.dlt_waveform_engine = DLTWaveformEngine()
        
        self.validation_results: List[PipelineValidationResult] = []
        
        logger.info("Mathematical Integration Validator initialized")

    def validate_bit_phase_pipeline(self) -> PipelineValidationResult:
        """Validate bit phase resolution pipeline."""
        test_results = []
        start_time = time.time()
        
        try:
            # Test 1: Basic bit phase resolution
            test_start = time.time()
            strategy_id = "0x123456789abcdef"
            bit_result = self.tensor_algebra.resolve_bit_phases(strategy_id)
            
            success = (
                isinstance(bit_result, BitPhaseResult) and
                0 <= bit_result.phi_4 <= 15 and
                0 <= bit_result.phi_8 <= 255 and
                0 <= bit_result.phi_42 <= 0x3FFFFFFFFFF
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Basic Bit Phase Resolution",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "phi_4": bit_result.phi_4,
                    "phi_8": bit_result.phi_8,
                    "phi_42": bit_result.phi_42,
                    "cycle_score": bit_result.cycle_score
                }
            ))
            
            # Test 2: Bit phase consistency across engines
            test_start = time.time()
            bit_engine_result = self.bit_resolution_engine.resolve_bit_phase(strategy_id, "auto")
            tensor_result = self.tensor_matcher.match_tensor(
                strategy_id, 45000.0, 46000.0, {"entropy_level": 4.0}
            )
            
            success = (
                bit_engine_result is not None and
                tensor_result is not None and
                unified_math.abs(bit_result.cycle_score - tensor_result.tensor_score) < 10.0
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Cross-Engine Bit Phase Consistency",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "bit_engine_result": bit_engine_result,
                    "tensor_score": tensor_result.tensor_score if tensor_result else None
                }
            ))
            
            # Test 3: Bit phase mathematical formulas
            test_start = time.time()
            strategy_int = int(strategy_id, 16)
            expected_phi_4 = strategy_int & 0b1111
            expected_phi_8 = (strategy_int >> 4) & 0b11111111
            expected_phi_42 = (strategy_int >> 12) & 0x3FFFFFFFFFF
            
            success = (
                bit_result.phi_4 == expected_phi_4 and
                bit_result.phi_8 == expected_phi_8 and
                bit_result.phi_42 == expected_phi_42
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Bit Phase Mathematical Formulas",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "expected_phi_4": expected_phi_4,
                    "expected_phi_8": expected_phi_8,
                    "expected_phi_42": expected_phi_42,
                    "actual_phi_4": bit_result.phi_4,
                    "actual_phi_8": bit_result.phi_8,
                    "actual_phi_42": bit_result.phi_42
                }
            ))
            
        except Exception as e:
            test_results.append(IntegrationTestResult(
                test_name="Bit Phase Pipeline Exception",
                success=False,
                execution_time=0.0,
                error_message=str(e)
            ))
        
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in test_results if result.success)
        
        return PipelineValidationResult(
            pipeline_name="Bit Phase Resolution Pipeline",
            all_tests_passed=passed_tests == len(test_results),
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            execution_time=total_time,
            test_results=test_results
        )

    def validate_tensor_contraction_pipeline(self) -> PipelineValidationResult:
        """Validate tensor contraction pipeline."""
        test_results = []
        start_time = time.time()
        
        try:
            # Test 1: Basic tensor contraction
            test_start = time.time()
            matrix_a = np.random.random((4, 4))
            matrix_b = np.random.random((4, 4))
            tensor_result = self.tensor_algebra.perform_tensor_contraction(matrix_a, matrix_b)
            
            success = (
                tensor_result is not None and
                tensor_result.tensor_score > 0 and
                tensor_result.contraction_matrix.shape == (4, 4)
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Basic Tensor Contraction",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "tensor_score": tensor_result.tensor_score,
                    "matrix_shape": tensor_result.contraction_matrix.shape
                }
            ))
            
            # Test 2: Matrix basket integration
            test_start = time.time()
            basket_result = self.matrix_mapper.create_matrix_basket(
                "test_basket", 100, 45000.0
            )
            
            success = basket_result is not None
            
            test_results.append(IntegrationTestResult(
                test_name="Matrix Basket Integration",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "basket_id": basket_result.basket_id if basket_result else None
                }
            ))
            
            # Test 3: DLT waveform tensor integration
            test_start = time.time()
            market_data = {"entropy_level": 4.0, "complexity": 0.5}
            dlt_basket = self.dlt_waveform_engine.create_matrix_basket(market_data)
            
            success = dlt_basket is not None and dlt_basket.resonance_score > 0
            
            test_results.append(IntegrationTestResult(
                test_name="DLT Waveform Tensor Integration",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "resonance_score": dlt_basket.resonance_score if dlt_basket else None
                }
            ))
            
        except Exception as e:
            test_results.append(IntegrationTestResult(
                test_name="Tensor Contraction Pipeline Exception",
                success=False,
                execution_time=0.0,
                error_message=str(e)
            ))
        
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in test_results if result.success)
        
        return PipelineValidationResult(
            pipeline_name="Tensor Contraction Pipeline",
            all_tests_passed=passed_tests == len(test_results),
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            execution_time=total_time,
            test_results=test_results
        )

    def validate_profit_routing_pipeline(self) -> PipelineValidationResult:
        """Validate profit routing pipeline."""
        test_results = []
        start_time = time.time()
        
        try:
            # Test 1: Basic profit routing
            test_start = time.time()
            profit_result = self.tensor_algebra.calculate_profit_routing(
                1000.0, 950.0, 1.0, 0.01
            )
            
            success = (
                profit_result is not None and
                profit_result.profit_rate == 50.0 and  # (1000 - 950) / 1.0
                profit_result.execution_trigger == True
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Basic Profit Routing",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "profit_rate": profit_result.profit_rate,
                    "execution_trigger": profit_result.execution_trigger
                }
            ))
            
            # Test 2: Profit routing engine integration
            test_start = time.time()
            delta_trade = self.profit_routing_engine.calculate_delta_trade(
                50000.0, 51000.0
            )
            
            success = delta_trade is not None and delta_trade.delta_profit == 1000.0
            
            test_results.append(IntegrationTestResult(
                test_name="Profit Routing Engine Integration",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "delta_profit": delta_trade.delta_profit if delta_trade else None
                }
            ))
            
            # Test 3: Differential calculus validation
            test_start = time.time()
            # Test dP/dt = (P_t - P_t-1) / Δt
            P_t = 1000.0
            P_t_minus_1 = 950.0
            delta_t = 1.0
            expected_rate = (P_t - P_t_minus_1) / delta_t
            
            calc_result = self.tensor_algebra.calculate_profit_routing(P_t, P_t_minus_1, delta_t)
            success = unified_math.abs(calc_result.profit_rate - expected_rate) < 1e-6
            
            test_results.append(IntegrationTestResult(
                test_name="Differential Calculus Validation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "expected_rate": expected_rate,
                    "actual_rate": calc_result.profit_rate
                }
            ))
            
        except Exception as e:
            test_results.append(IntegrationTestResult(
                test_name="Profit Routing Pipeline Exception",
                success=False,
                execution_time=0.0,
                error_message=str(e)
            ))
        
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in test_results if result.success)
        
        return PipelineValidationResult(
            pipeline_name="Profit Routing Pipeline",
            all_tests_passed=passed_tests == len(test_results),
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            execution_time=total_time,
            test_results=test_results
        )

    def validate_entropy_compensation_pipeline(self) -> PipelineValidationResult:
        """Validate entropy compensation pipeline."""
        test_results = []
        start_time = time.time()
        
        try:
            # Test 1: Basic entropy compensation
            test_start = time.time()
            entropy_result = self.tensor_algebra.calculate_entropy_compensation(
                1000.0, 0.1
            )
            
            success = (
                entropy_result is not None and
                entropy_result.entropy_gate > 0 and
                entropy_result.entropy_gate < 10.0  # Reasonable range
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Basic Entropy Compensation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "entropy_gate": entropy_result.entropy_gate,
                    "adaptive_trigger": entropy_result.adaptive_trigger
                }
            ))
            
            # Test 2: Entropy gate formula validation
            test_start = time.time()
            # Test E(t) = unified_math.log(V + 1) / (1 + δ)
            V = 1000.0
            delta = 0.1
            expected_gate = unified_math.unified_math.log(V + 1) / (1 + delta)
            
            calc_result = self.tensor_algebra.calculate_entropy_compensation(V, delta)
            success = unified_math.abs(calc_result.entropy_gate - expected_gate) < 1e-6
            
            test_results.append(IntegrationTestResult(
                test_name="Entropy Gate Formula Validation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "expected_gate": expected_gate,
                    "actual_gate": calc_result.entropy_gate
                }
            ))
            
            # Test 3: Drift dynamics validation
            test_start = time.time()
            # Test with different drift magnitudes
            low_drift = self.tensor_algebra.calculate_entropy_compensation(1000.0, 0.01)
            high_drift = self.tensor_algebra.calculate_entropy_compensation(1000.0, 0.9)
            
            success = low_drift.entropy_gate > high_drift.entropy_gate  # Lower drift = higher gate
            
            test_results.append(IntegrationTestResult(
                test_name="Drift Dynamics Validation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "low_drift_gate": low_drift.entropy_gate,
                    "high_drift_gate": high_drift.entropy_gate
                }
            ))
            
        except Exception as e:
            test_results.append(IntegrationTestResult(
                test_name="Entropy Compensation Pipeline Exception",
                success=False,
                execution_time=0.0,
                error_message=str(e)
            ))
        
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in test_results if result.success)
        
        return PipelineValidationResult(
            pipeline_name="Entropy Compensation Pipeline",
            all_tests_passed=passed_tests == len(test_results),
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            execution_time=total_time,
            test_results=test_results
        )

    def validate_hash_memory_pipeline(self) -> PipelineValidationResult:
        """Validate hash memory pipeline."""
        test_results = []
        start_time = time.time()
        
        try:
            # Test 1: Basic hash memory encoding
            test_start = time.time()
            bit_result = self.tensor_algebra.resolve_bit_phases("0x123456789abcdef")
            hash_result = self.tensor_algebra.encode_hash_memory(
                1000.0, 50.0, bit_result
            )
            
            success = (
                hash_result is not None and
                len(hash_result.hash_signature) == 64 and  # SHA256 hex length
                hash_result.similarity_score >= 0.0 and
                hash_result.similarity_score <= 1.0
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Basic Hash Memory Encoding",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "hash_signature": hash_result.hash_signature[:16] + "...",
                    "similarity_score": hash_result.similarity_score,
                    "strategy_match": hash_result.strategy_match
                }
            ))
            
            # Test 2: Hash memory formula validation
            test_start = time.time()
            # Test H(t) = SHA256(P_t || ΔP || φ_t)
            P_t = 1000.0
            delta_P = 50.0
            phi_t = bit_result.cycle_score
            
            expected_input = f"{P_t:.8f}||{delta_P:.8f}||{phi_t:.8f}"
            import hashlib
            expected_hash = hashlib.sha256(expected_input.encode()).hexdigest()
            
            success = hash_result.hash_signature == expected_hash
            
            test_results.append(IntegrationTestResult(
                test_name="Hash Memory Formula Validation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "expected_hash": expected_hash[:16] + "...",
                    "actual_hash": hash_result.hash_signature[:16] + "..."
                }
            ))
            
            # Test 3: Memory activation validation
            test_start = time.time()
            # Test multiple hashes for similarity calculation
            hash1 = self.tensor_algebra.encode_hash_memory(1000.0, 50.0, bit_result)
            hash2 = self.tensor_algebra.encode_hash_memory(1001.0, 51.0, bit_result)
            
            success = (
                hash1.memory_activation is not None and
                hash2.memory_activation is not None and
                isinstance(hash1.similarity_score, float)
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Memory Activation Validation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "hash1_activation": hash1.memory_activation,
                    "hash2_activation": hash2.memory_activation,
                    "hash1_similarity": hash1.similarity_score
                }
            ))
            
        except Exception as e:
            test_results.append(IntegrationTestResult(
                test_name="Hash Memory Pipeline Exception",
                success=False,
                execution_time=0.0,
                error_message=str(e)
            ))
        
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in test_results if result.success)
        
        return PipelineValidationResult(
            pipeline_name="Hash Memory Pipeline",
            all_tests_passed=passed_tests == len(test_results),
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            execution_time=total_time,
            test_results=test_results
        )

    def validate_complete_pipeline(self) -> PipelineValidationResult:
        """Validate the complete mathematical pipeline end-to-end."""
        test_results = []
        start_time = time.time()
        
        try:
            # Test 1: Complete unified operation
            test_start = time.time()
            strategy_id = "0x123456789abcdef"
            market_data = {
                'current_profit': 1000.0,
                'previous_profit': 950.0,
                'time_delta': 1.0,
                'volume': 1000.0,
                'drift_magnitude': 0.1
            }
            
            unified_result = self.tensor_algebra.perform_unified_operation(strategy_id, market_data)
            
            success = (
                unified_result is not None and
                'bit_phases' in unified_result and
                'tensor_contraction' in unified_result and
                'profit_routing' in unified_result and
                'entropy_compensation' in unified_result and
                'hash_memory' in unified_result
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Complete Unified Operation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "strategy_match": unified_result.get('hash_memory', {}).get('strategy_match'),
                    "memory_activation": unified_result.get('hash_memory', {}).get('memory_activation')
                }
            ))
            
            # Test 2: Mathematical consistency across pipeline
            test_start = time.time()
            bit_phases = unified_result['bit_phases']
            tensor_contraction = unified_result['tensor_contraction']
            profit_routing = unified_result['profit_routing']
            entropy_compensation = unified_result['entropy_compensation']
            hash_memory = unified_result['hash_memory']
            
            # Check that all components are mathematically consistent
            success = (
                bit_phases['cycle_score'] > 0 and
                tensor_contraction['tensor_score'] > 0 and
                profit_routing['profit_rate'] == 50.0 and  # (1000 - 950) / 1.0
                entropy_compensation['entropy_gate'] > 0 and
                hash_memory['similarity_score'] >= 0.0
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Mathematical Consistency",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "cycle_score": bit_phases['cycle_score'],
                    "tensor_score": tensor_contraction['tensor_score'],
                    "profit_rate": profit_routing['profit_rate'],
                    "entropy_gate": entropy_compensation['entropy_gate'],
                    "similarity_score": hash_memory['similarity_score']
                }
            ))
            
            # Test 3: Pipeline integration validation
            test_start = time.time()
            # Verify that all mathematical components work together
            execution_trigger = profit_routing['execution_trigger']
            adaptive_trigger = entropy_compensation['adaptive_trigger']
            memory_activation = hash_memory['memory_activation']
            
            # All triggers should be boolean values
            success = (
                isinstance(execution_trigger, bool) and
                isinstance(adaptive_trigger, bool) and
                isinstance(memory_activation, bool)
            )
            
            test_results.append(IntegrationTestResult(
                test_name="Pipeline Integration Validation",
                success=success,
                execution_time=time.time() - test_start,
                metadata={
                    "execution_trigger": execution_trigger,
                    "adaptive_trigger": adaptive_trigger,
                    "memory_activation": memory_activation
                }
            ))
            
        except Exception as e:
            test_results.append(IntegrationTestResult(
                test_name="Complete Pipeline Exception",
                success=False,
                execution_time=0.0,
                error_message=str(e)
            ))
        
        total_time = time.time() - start_time
        passed_tests = sum(1 for result in test_results if result.success)
        
        return PipelineValidationResult(
            pipeline_name="Complete Mathematical Pipeline",
            all_tests_passed=passed_tests == len(test_results),
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=len(test_results) - passed_tests,
            execution_time=total_time,
            test_results=test_results
        )

    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete mathematical validation."""
        safe_print("🧮 Running Complete Mathematical Integration Validation...")
        
        # Run all pipeline validations
        pipelines = [
            self.validate_bit_phase_pipeline(),
            self.validate_tensor_contraction_pipeline(),
            self.validate_profit_routing_pipeline(),
            self.validate_entropy_compensation_pipeline(),
            self.validate_hash_memory_pipeline(),
            self.validate_complete_pipeline()
        ]
        
        # Store results
        self.validation_results = pipelines
        
        # Calculate overall statistics
        total_tests = sum(p.total_tests for p in pipelines)
        total_passed = sum(p.passed_tests for p in pipelines)
        total_failed = sum(p.failed_tests for p in pipelines)
        total_time = sum(p.execution_time for p in pipelines)
        
        overall_success = all(p.all_tests_passed for p in pipelines)
        
        # Print results
        safe_print(f"\n📊 Validation Results:")
        safe_print(f"  Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        safe_print(f"  Total Tests: {total_tests}")
        safe_print(f"  Passed: {total_passed}")
        safe_print(f"  Failed: {total_failed}")
        safe_print(f"  Success Rate: {(total_passed/total_tests)*100:.1f}%")
        safe_print(f"  Total Execution Time: {total_time:.2f}s")
        
        safe_print(f"\n📋 Pipeline Results:")
        for pipeline in pipelines:
            status = "✅ PASSED" if pipeline.all_tests_passed else "❌ FAILED"
            safe_print(f"  {pipeline.pipeline_name}: {status} ({pipeline.passed_tests}/{pipeline.total_tests})")
        
        # Return comprehensive results
        return {
            "overall_success": overall_success,
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "failed_tests": total_failed,
            "success_rate": (total_passed/total_tests)*100 if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "pipelines": [
                {
                    "name": p.pipeline_name,
                    "success": p.all_tests_passed,
                    "tests": p.total_tests,
                    "passed": p.passed_tests,
                    "failed": p.failed_tests,
                    "execution_time": p.execution_time,
                    "test_results": [
                        {
                            "name": t.test_name,
                            "success": t.success,
                            "execution_time": t.execution_time,
                            "error": t.error_message,
                            "metadata": t.metadata
                        }
                        for t in p.test_results
                    ]
                }
                for p in pipelines
            ]
        }

    def export_validation_results(self, output_path: str = "mathematical_validation_results.json") -> None:
        """Export validation results to JSON file."""
        try:
            results = self.run_complete_validation()
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Validation results exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting validation results: {e}")


def main():
    """Main function for mathematical integration validation."""
    safe_print("🧮 Mathematical Integration Validator - Schwabot UROS v1.0")
    safe_print("=" * 60)
    
    # Initialize validator
    validator = MathematicalIntegrationValidator()
    
    # Run complete validation
    results = validator.run_complete_validation()
    
    # Export results
    validator.export_validation_results()
    
    # Return exit code based on success
    return 0 if results["overall_success"] else 1


if __name__ == "__main__":
    exit(main()) 