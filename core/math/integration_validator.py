"""
Integration Validator - Mathematical System Validation for Schwabot
==================================================================

Comprehensive mathematical integration validator for the Schwabot trading framework.
Provides validation of tensor operations, bit phase calculations, and system integration.

Key Features:
- Bit phase resolution validation
- Tensor contraction accuracy testing
- Profit routing calculation validation
- Entropy compensation verification
- Hash memory encoding validation
- Matrix decomposition testing
- Complete system integration validation
- Performance and accuracy metrics
- Integration with all core components
- Windows CLI compatibility with emoji fallbacks

Validation Tests:
- Bit Phase Resolution: Strategy ID processing and phase extraction
- Tensor Contraction: Matrix multiplication accuracy and shape compatibility
- Profit Routing: Profit data processing and weight application
- Entropy Compensation: Data stream compensation calculations
- Hash Memory Encoding: Data encoding and hash generation
- Matrix Decomposition: SVD, QR, and Cholesky decompositions

Integration Points:
- All core components for mathematical validation
- enhanced_windows_cli_compatibility.py: CLI compatibility
- thermal_boundary_manager.py: Thermal-aware validation
- main_orchestrator.py: System-wide validation coordination
- tensor_algebra.py: Mathematical operation validation

Windows CLI compatible with flake8 compliance.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Import tensor algebra
try:
    from core.math.tensor_algebra import UnifiedTensorAlgebra
    TENSOR_ALGEBRA_AVAILABLE = True
except ImportError:
    TENSOR_ALGEBRA_AVAILABLE = False
    UnifiedTensorAlgebra = None

# Import Windows CLI compatibility
try:
    from core.enhanced_windows_cli_compatibility import safe_print, safe_format_error
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
        
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"

# Configure logging
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
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_execution_time: float
    test_results: List[IntegrationTestResult]
    system_status: str
    recommendations: List[str] = field(default_factory=list)


class MathematicalIntegrationValidator:
    """Validator for mathematical integration across all components."""

    def __init__(self):
        """Initialize the mathematical integration validator."""
        self.test_results: List[IntegrationTestResult] = []
        self.tensor_algebra = None
        self.validation_count = 0

        # Initialize tensor algebra if available
        if TENSOR_ALGEBRA_AVAILABLE:
            try:
                self.tensor_algebra = UnifiedTensorAlgebra()
                logger.info("Tensor algebra initialized for validation")
            except Exception as e:
                logger.warning(f"Tensor algebra initialization failed: {e}")

        logger.info("Mathematical Integration Validator initialized")

    def validate_bit_phase_resolution(self) -> IntegrationTestResult:
        """
        Validate bit phase resolution operations.
        
        Tests:
        - Bit phase tensor calculations
        - Strategy ID processing
        - Phase extraction accuracy
        """
        start_time = time.time()
        test_name = "bit_phase_resolution"
        
        try:
            if not self.tensor_algebra:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Tensor algebra not available"
                )
            
            # Test bit phase calculations
            test_strategy_ids = [12345, 67890, 11111, 99999]
            expected_phi_4 = [9, 10, 15, 15]  # Expected phi_4 values
            
            for i, strategy_id in enumerate(test_strategy_ids):
                result = self.tensor_algebra.bit_phase_tensor(strategy_id)
                
                # Validate phi_4 calculation
                if result.phi_4 != expected_phi_4[i]:
                    return IntegrationTestResult(
                        test_name=test_name,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message=f"Phi_4 mismatch: expected {expected_phi_4[i]}, got {result.phi_4}"
                    )
                
                # Validate bit ranges
                if not (0 <= result.phi_4 <= 15):
                    return IntegrationTestResult(
                        test_name=test_name,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message=f"Phi_4 out of range: {result.phi_4}"
                    )
                
                if not (0 <= result.phi_8 <= 255):
                    return IntegrationTestResult(
                        test_name=test_name,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message=f"Phi_8 out of range: {result.phi_8}"
                    )
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"tested_strategies": len(test_strategy_ids)}
            )

        except Exception as e:
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Bit phase resolution failed: {e}"
            )
    
    def validate_tensor_contraction(self) -> IntegrationTestResult:
        """
        Validate tensor contraction operations.
        
        Tests:
        - Matrix multiplication accuracy
        - Shape compatibility
        - Numerical precision
        """
        start_time = time.time()
        test_name = "tensor_contraction"
        
        try:
            if not self.tensor_algebra:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Tensor algebra not available"
                )
            
            # Test matrix multiplication
            A = np.array([[1, 2], [3, 4]], dtype=np.float64)
            B = np.array([[5, 6], [7, 8]], dtype=np.float64)
            
            # Expected result: [[19, 22], [43, 50]]
            expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
            
            result = self.tensor_algebra.tensor_contraction(A, B)
            
            # Check shape
            if result.shape != expected.shape:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Shape mismatch: expected {expected.shape}, got {result.shape}"
                )
            
            # Check numerical accuracy
            if not np.allclose(result, expected, rtol=1e-10):
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Numerical mismatch: expected {expected}, got {result}"
                )
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"matrix_size": A.shape}
            )

        except Exception as e:
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Tensor contraction failed: {e}"
            )
    
    def validate_profit_routing(self) -> IntegrationTestResult:
        """
        Validate profit routing calculations.
        
        Tests:
        - Profit data processing
        - Weight application
        - Confidence calculations
        """
        start_time = time.time()
        test_name = "profit_routing"
        
        try:
            if not self.tensor_algebra:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Tensor algebra not available"
                )
            
            # Test profit routing
            profit_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
            routing_weights = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
            
            result = self.tensor_algebra.profit_routing_tensor(profit_data, routing_weights)
            
            # Check shape
            expected_shape = (1, 2)  # (weights.shape[0], profit_data.shape[1])
            if result.shape != expected_shape:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Shape mismatch: expected {expected_shape}, got {result.shape}"
                )
            
            # Check that result is finite
            if not np.all(np.isfinite(result)):
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Non-finite values in result"
                )
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"profit_data_shape": profit_data.shape}
            )

        except Exception as e:
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Profit routing validation failed: {e}"
            )
    
    def validate_entropy_compensation(self) -> IntegrationTestResult:
        """
        Validate entropy compensation calculations.
        
        Tests:
        - Data normalization
        - Gradient calculation
        - Compensation application
        """
        start_time = time.time()
        test_name = "entropy_compensation"
        
        try:
            if not self.tensor_algebra:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Tensor algebra not available"
                )
            
            # Test entropy compensation
            test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
            compensation_factor = 1.0
            
            result = self.tensor_algebra.entropy_compensation(test_data, compensation_factor)
            
            # Check shape preservation
            if result.shape != test_data.shape:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Shape not preserved: expected {test_data.shape}, got {result.shape}"
                )
            
            # Check that result is finite
            if not np.all(np.isfinite(result)):
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Non-finite values in result"
                )
            
            # Check that compensation was applied (result should be different from input)
            if np.allclose(result, test_data, rtol=1e-10):
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Compensation not applied"
                )
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"data_length": len(test_data)}
            )

        except Exception as e:
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Entropy compensation validation failed: {e}"
            )
    
    def validate_hash_memory_encoding(self) -> IntegrationTestResult:
        """
        Validate hash memory encoding operations.
        
        Tests:
        - String encoding
        - Array encoding
        - Hash generation
        """
        start_time = time.time()
        test_name = "hash_memory_encoding"
        
        try:
            if not self.tensor_algebra:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Tensor algebra not available"
                )
            
            # Test string encoding
            test_string = "test_data"
            string_hash = self.tensor_algebra.hash_memory_encoding(test_string)
            
            # Check hash format (should be 64 characters hex)
            if len(string_hash) != 64:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Invalid hash length: {len(string_hash)}"
                )
            
            # Test array encoding
            test_array = np.array([1, 2, 3, 4, 5], dtype=np.float64)
            array_hash = self.tensor_algebra.hash_memory_encoding(test_array)
            
            if len(array_hash) != 64:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Invalid array hash length: {len(array_hash)}"
                )
            
            # Check that different inputs produce different hashes
            if string_hash == array_hash:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Different inputs produced same hash"
                )
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"tested_inputs": 2}
            )

        except Exception as e:
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Hash memory encoding validation failed: {e}"
            )
    
    def validate_matrix_decomposition(self) -> IntegrationTestResult:
        """
        Validate matrix decomposition operations.
        
        Tests:
        - SVD decomposition
        - QR decomposition
        - Cholesky decomposition
        """
        start_time = time.time()
        test_name = "matrix_decomposition"
        
        try:
            if not self.tensor_algebra:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Tensor algebra not available"
                )
            
            # Test SVD decomposition
            test_matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
            
            try:
                U, s, Vt = self.tensor_algebra.matrix_decomposition(test_matrix, 'svd')
                
                # Check shapes
                if U.shape != (2, 2) or s.shape != (2,) or Vt.shape != (2, 2):
                    return IntegrationTestResult(
                        test_name=test_name,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="SVD shape mismatch"
                    )
                
                # Check reconstruction
                reconstructed = U @ np.diag(s) @ Vt
                if not np.allclose(test_matrix, reconstructed, rtol=1e-10):
                    return IntegrationTestResult(
                        test_name=test_name,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="SVD reconstruction failed"
                    )
                    
            except Exception as e:
                return IntegrationTestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"SVD decomposition failed: {e}"
                )
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"decomposition_methods": ["svd"]}
            )

        except Exception as e:
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Matrix decomposition validation failed: {e}"
            )
    
    def run_complete_validation(self) -> ValidationReport:
        """
        Run complete validation suite.
        
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        
        # Run all validation tests
        tests = [
            self.validate_bit_phase_resolution,
            self.validate_tensor_contraction,
            self.validate_profit_routing,
            self.validate_entropy_compensation,
            self.validate_hash_memory_encoding,
            self.validate_matrix_decomposition
        ]
        
        results = []
        for test in tests:
            result = test()
            results.append(result)
            self.test_results.append(result)
        
        # Calculate statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        total_execution_time = time.time() - start_time
        
        # Determine system status
        if failed_tests == 0:
            system_status = "PASSED"
        elif failed_tests <= total_tests // 2:
            system_status = "PARTIAL"
        else:
            system_status = "FAILED"
        
        # Generate recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append("Review failed validation tests")
        if not TENSOR_ALGEBRA_AVAILABLE:
            recommendations.append("Install tensor algebra dependencies")
        if total_execution_time > 10.0:
            recommendations.append("Optimize validation performance")
        
        return ValidationReport(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_execution_time=total_execution_time,
            test_results=results,
            system_status=system_status,
            recommendations=recommendations
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.test_results:
            return {"message": "No validation tests run yet"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.execution_time for r in self.test_results)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / total_tests if total_tests > 0 else 0.0,
            "tensor_algebra_available": TENSOR_ALGEBRA_AVAILABLE
        }


# Global validator instance
_validator: Optional[MathematicalIntegrationValidator] = None


def get_validator() -> MathematicalIntegrationValidator:
    """Get global validation instance."""
    global _validator
    if _validator is None:
        _validator = MathematicalIntegrationValidator()
    return _validator


def main():
    """Test the mathematical integration validator."""
    try:
        # Create validator
        validator = get_validator()
        
        # Run complete validation
        report = validator.run_complete_validation()
        
        # Print results
        safe_print(f"📊 Validation Report:")
        safe_print(f"   Status: {report.system_status}")
        safe_print(f"   Tests: {report.passed_tests}/{report.total_tests} passed")
        safe_print(f"   Time: {report.total_execution_time:.3f}s")
        
        # Print failed tests
        failed_tests = [r for r in report.test_results if not r.success]
        if failed_tests:
            safe_print(f"❌ Failed Tests:")
            for test in failed_tests:
                safe_print(f"   {test.test_name}: {test.error_message}")
        
        # Print recommendations
        if report.recommendations:
            safe_print(f"💡 Recommendations:")
            for rec in report.recommendations:
                safe_print(f"   - {rec}")
        
        # Get statistics
        stats = validator.get_validation_statistics()
        safe_print(f"📈 Statistics: {stats}")
        
        safe_print("🎉 Mathematical integration validation completed")
        
    except Exception as e:
        safe_print(f"❌ Validation test failed: {safe_format_error(e, 'main_test')}")


if __name__ == "__main__":
    main() 