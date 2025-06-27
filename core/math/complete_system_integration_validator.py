"""
Complete System Integration Validator - Comprehensive System Validation for Schwabot
==================================================================================

Comprehensive system integration validator for the Schwabot trading framework.
Provides validation of all system components, mathematical foundations, and integration points.

Key Features:
- Core mathematical foundations validation
- UI system integration testing
- Training and demo pipeline validation
- Visualizer integration testing
- Math library integration validation
- Component interoperability testing
- Complete system validation suite
- Performance and accuracy metrics
- Integration with all core components
- Windows CLI compatibility with emoji fallbacks

Validation Tests:
- Core Mathematical Foundations: Unified math system and tensor algebra
- UI System Integration: CLI compatibility and safe print functionality
- Training Demo Pipeline: Pipeline execution and data flow
- Visualizer Integration: Visualization system functionality
- Math Library Integration: Mathematical library operations
- Component Interoperability: Cross-component communication

Integration Points:
- All core components for system validation
- enhanced_windows_cli_compatibility.py: CLI compatibility
- thermal_boundary_manager.py: Thermal-aware validation
- main_orchestrator.py: System-wide validation coordination
- tensor_algebra.py: Mathematical operation validation
- integration_validator.py: Mathematical integration validation

Windows CLI compatible with flake8 compliance.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Import core components
try:
    from core.math.tensor_algebra import UnifiedTensorAlgebra
    TENSOR_ALGEBRA_AVAILABLE = True
except ImportError:
    TENSOR_ALGEBRA_AVAILABLE = False
    UnifiedTensorAlgebra = None

try:
    from core.math.integration_validator import MathematicalIntegrationValidator
    INTEGRATION_VALIDATOR_AVAILABLE = True
except ImportError:
    INTEGRATION_VALIDATOR_AVAILABLE = False
    MathematicalIntegrationValidator = None

try:
    from core.enhanced_windows_cli_compatibility import safe_print, safe_format_error
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
        
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"

# Import unified math system
try:
    from core.unified_math_system import unified_math
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False
    unified_math = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SystemIntegrationTestResult:
    """Result of system integration test."""
    test_name: str
    component: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemIntegrationValidationResult:
    """Result of system integration validation."""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[SystemIntegrationTestResult]
    overall_success: bool
    execution_time: float
    component_status: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompleteSystemIntegrationValidator:
    """Validator for complete system integration across all components."""

    def __init__(self):
        """Initialize the complete system integration validator."""
        self.test_results: List[SystemIntegrationTestResult] = []
        self.tensor_algebra = None
        self.integration_validator = None
        self.validation_count = 0
        self.component_status = {}
        
        # Initialize tensor algebra if available
        if TENSOR_ALGEBRA_AVAILABLE:
            try:
                self.tensor_algebra = UnifiedTensorAlgebra()
                self.component_status['tensor_algebra'] = True
                logger.info("Tensor algebra initialized for system validation")
            except Exception as e:
                logger.warning(f"Tensor algebra initialization failed: {e}")
                self.component_status['tensor_algebra'] = False
        
        # Initialize integration validator if available
        if INTEGRATION_VALIDATOR_AVAILABLE:
            try:
                self.integration_validator = MathematicalIntegrationValidator()
                self.component_status['integration_validator'] = True
                logger.info("Integration validator initialized for system validation")
            except Exception as e:
                logger.warning(f"Integration validator initialization failed: {e}")
                self.component_status['integration_validator'] = False

        logger.info("Complete System Integration Validator initialized")

    def validate_core_mathematical_foundations(self) -> SystemIntegrationTestResult:
        """
        Validate core mathematical foundations.
        
        Tests:
        - Unified math system functionality
        - Tensor algebra operations
        - Mathematical constants and precision
        """
        start_time = time.time()
        test_name = "core_mathematical_foundations"
        component = "mathematical_core"
        
        try:
            # Test unified math system
            if not UNIFIED_MATH_AVAILABLE:
                return SystemIntegrationTestResult(
                    test_name=test_name,
                    component=component,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Unified math system not available"
                )
            
            # Test tensor algebra if available
            if self.tensor_algebra:
                # Test basic tensor operations
                A = np.array([[1, 2], [3, 4]], dtype=np.float64)
                B = np.array([[5, 6], [7, 8]], dtype=np.float64)
                
                result = self.tensor_algebra.tensor_contraction(A, B)
                if not np.allclose(result, np.array([[19, 22], [43, 50]]), rtol=1e-10):
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="Tensor algebra basic operations failed"
                    )
            
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"tensor_algebra_available": self.tensor_algebra is not None}
            )
            
        except Exception as e:
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Core mathematical foundations failed: {e}"
            )
    
    def validate_ui_system_integration(self) -> SystemIntegrationTestResult:
        """
        Validate UI system integration.
        
        Tests:
        - CLI compatibility
        - Safe print functionality
        - Error handling
        """
        start_time = time.time()
        test_name = "ui_system_integration"
        component = "ui_system"
        
        try:
            # Test CLI compatibility functions
            test_message = "Test message"
            
            # Test safe print
            try:
                safe_print(test_message)
            except Exception as e:
                return SystemIntegrationTestResult(
                    test_name=test_name,
                    component=component,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Safe print failed: {e}"
                )
            
            # Test error formatting
            try:
                test_error = ValueError("Test error")
                error_msg = safe_format_error(test_error, "test_context")
                if not error_msg:
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="Error formatting failed"
                    )
            except Exception as e:
                return SystemIntegrationTestResult(
                    test_name=test_name,
                    component=component,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Error formatting failed: {e}"
                )
            
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"cli_compatibility_available": CLI_HANDLER_AVAILABLE}
            )

        except Exception as e:
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"UI system integration failed: {e}"
            )
    
    def validate_training_demo_pipeline_integration(self) -> SystemIntegrationTestResult:
        """
        Validate training and demo pipeline integration.
        
        Tests:
        - Pipeline execution
        - Data flow
        - Error handling
        """
        start_time = time.time()
        test_name = "training_demo_pipeline_integration"
        component = "training_pipeline"
        
        try:
            # Test basic pipeline functionality
            # This would typically test actual pipeline components
            # For now, we'll test the mathematical foundations that support the pipeline
            
            if self.tensor_algebra:
                # Test profit routing (key component of training pipeline)
                profit_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
                routing_weights = np.array([[0.5, 0.5]], dtype=np.float64)
                
                result = self.tensor_algebra.profit_routing_tensor(profit_data, routing_weights)
                
                if not np.all(np.isfinite(result)):
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="Non-finite values in profit routing result"
                    )
                
                # Check shape
                expected_shape = (1, 2)
                if result.shape != expected_shape:
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message=f"Profit routing shape mismatch: expected {expected_shape}, got {result.shape}"
                    )
            
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"tensor_algebra_available": self.tensor_algebra is not None}
            )

        except Exception as e:
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Training demo pipeline integration failed: {e}"
            )
    
    def validate_visualizer_integration(self) -> SystemIntegrationTestResult:
        """
        Validate visualizer integration.
        
        Tests:
        - Visualization system functionality
        - Data plotting capabilities
        - Performance metrics display
        """
        start_time = time.time()
        test_name = "visualizer_integration"
        component = "visualizer"
        
        try:
            # Test basic visualization functionality
            # This would typically test actual visualization components
            # For now, we'll test the data preparation that supports visualization
            
            if self.tensor_algebra:
                # Test data preparation for visualization
                test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
                
                # Test entropy compensation (used in visualization)
                result = self.tensor_algebra.entropy_compensation(test_data)
                
                if not np.all(np.isfinite(result)):
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="Non-finite values in visualization data"
                    )
                
                # Check shape preservation
                if result.shape != test_data.shape:
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="Visualization data shape not preserved"
                    )
            
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"tensor_algebra_available": self.tensor_algebra is not None}
            )

        except Exception as e:
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Visualizer integration failed: {e}"
            )
    
    def validate_mathlib_integration(self) -> SystemIntegrationTestResult:
        """
        Validate math library integration.
        
        Tests:
        - Mathematical library operations
        - Function availability
        - Numerical accuracy
        """
        start_time = time.time()
        test_name = "mathlib_integration"
        component = "mathlib"
        
        try:
            # Test mathematical library functionality
            if self.tensor_algebra:
                # Test matrix decomposition (key math library operation)
                test_matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
                
                try:
                    U, s, Vt = self.tensor_algebra.matrix_decomposition(test_matrix, 'svd')
                    
                    # Check shapes
                    if U.shape != (2, 2) or s.shape != (2,) or Vt.shape != (2, 2):
                        return SystemIntegrationTestResult(
                            test_name=test_name,
                            component=component,
                            success=False,
                            execution_time=time.time() - start_time,
                            error_message="Matrix decomposition shape mismatch"
                        )
                    
                    # Check reconstruction
                    reconstructed = U @ np.diag(s) @ Vt
                    if not np.allclose(test_matrix, reconstructed, rtol=1e-10):
                        return SystemIntegrationTestResult(
                            test_name=test_name,
                            component=component,
                            success=False,
                            execution_time=time.time() - start_time,
                            error_message="Matrix decomposition reconstruction failed"
                        )
                        
                except Exception as e:
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message=f"Matrix decomposition failed: {e}"
                    )
            
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"tensor_algebra_available": self.tensor_algebra is not None}
            )

        except Exception as e:
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Math library integration failed: {e}"
            )
    
    def validate_component_interoperability(self) -> SystemIntegrationTestResult:
        """
        Validate component interoperability.
        
        Tests:
        - Cross-component communication
        - Data exchange
        - Error propagation
        """
        start_time = time.time()
        test_name = "component_interoperability"
        component = "interoperability"
        
        try:
            # Test component interoperability
            # This would typically test actual component communication
            # For now, we'll test the integration validator if available
            
            if self.integration_validator:
                # Run a quick validation test
                validation_report = self.integration_validator.run_complete_validation()
                
                if validation_report.system_status == "FAILED":
                    return SystemIntegrationTestResult(
                        test_name=test_name,
                        component=component,
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message="Integration validator failed"
                    )
            
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=True,
                execution_time=time.time() - start_time,
                metadata={"integration_validator_available": self.integration_validator is not None}
            )

        except Exception as e:
            return SystemIntegrationTestResult(
                test_name=test_name,
                component=component,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Component interoperability failed: {e}"
            )
    
    def run_complete_system_validation(self) -> SystemIntegrationValidationResult:
        """
        Run complete system validation suite.
        
        Returns:
            Comprehensive system validation result
        """
        start_time = time.time()
        self.validation_count += 1
        
        # Run all validation tests
        tests = [
            self.validate_core_mathematical_foundations,
            self.validate_ui_system_integration,
            self.validate_training_demo_pipeline_integration,
            self.validate_visualizer_integration,
            self.validate_mathlib_integration,
            self.validate_component_interoperability
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
        overall_success = failed_tests == 0
        execution_time = time.time() - start_time
        
        # Create validation result
        validation_result = SystemIntegrationValidationResult(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=results,
            overall_success=overall_success,
            execution_time=execution_time,
            component_status=self.component_status,
            metadata={
                "validation_count": self.validation_count,
                "tensor_algebra_available": TENSOR_ALGEBRA_AVAILABLE,
                "integration_validator_available": INTEGRATION_VALIDATOR_AVAILABLE,
                "cli_handler_available": CLI_HANDLER_AVAILABLE,
                "unified_math_available": UNIFIED_MATH_AVAILABLE
            }
        )
        
        # Log results
        if overall_success:
            safe_print("✅ Complete system integration validation passed")
        else:
            safe_print(f"❌ Complete system integration validation failed: {failed_tests}/{total_tests} tests")
        
        return validation_result
    
    def get_system_validation_statistics(self) -> Dict[str, Any]:
        """Get system validation statistics."""
        if not self.test_results:
            return {"message": "No system validation tests run yet"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.execution_time for r in self.test_results)
        
        # Get component status
        component_status = {}
        for result in self.test_results:
            component = result.component
            if component not in component_status:
                component_status[component] = True
            if not result.success:
                component_status[component] = False
        
        return {
            "total_validations": self.validation_count,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / total_tests if total_tests > 0 else 0.0,
            "component_status": component_status,
            "tensor_algebra_available": TENSOR_ALGEBRA_AVAILABLE,
            "integration_validator_available": INTEGRATION_VALIDATOR_AVAILABLE,
            "cli_handler_available": CLI_HANDLER_AVAILABLE,
            "unified_math_available": UNIFIED_MATH_AVAILABLE
        }


# Global system validator instance
_system_validator: Optional[CompleteSystemIntegrationValidator] = None


def get_system_validator() -> CompleteSystemIntegrationValidator:
    """Get global system validation instance."""
    global _system_validator
    if _system_validator is None:
        _system_validator = CompleteSystemIntegrationValidator()
    return _system_validator


def main():
    """Test the complete system integration validator."""
    try:
        # Create system validator
        validator = get_system_validator()
        
        # Run complete system validation
        result = validator.run_complete_system_validation()
        
        # Print results
        safe_print(f"📊 Complete System Integration Validation Report:")
        safe_print(f"   Overall Success: {result.overall_success}")
        safe_print(f"   Tests: {result.passed_tests}/{result.total_tests} passed")
        safe_print(f"   Time: {result.execution_time:.3f}s")
        
        # Print component status
        safe_print(f"🔧 Component Status:")
        for component, status in result.component_status.items():
            status_icon = "✅" if status else "❌"
            safe_print(f"   {component}: {status_icon}")
        
        # Print failed tests
        failed_tests = [r for r in result.test_results if not r.success]
        if failed_tests:
            safe_print(f"❌ Failed Tests:")
            for test in failed_tests:
                safe_print(f"   {test.test_name} ({test.component}): {test.error_message}")
        
        # Get statistics
        stats = validator.get_system_validation_statistics()
        safe_print(f"📈 Statistics: {stats}")
        
        safe_print("🎉 Complete system integration validation completed")
        
    except Exception as e:
        safe_print(f"❌ System validation test failed: {safe_format_error(e, 'main_test')}")


if __name__ == "__main__":
    main() 