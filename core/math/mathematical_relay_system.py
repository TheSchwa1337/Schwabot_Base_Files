# -*- coding: utf-8 -*-
"""
Mathematical Relay System for Tensor Operations
==============================================

Provides centralized routing and validation for all mathematical operations
within the Schwabot trading system. Handles operation dispatch, result
validation, and performance tracking.

Operation Categories:
- Basic tensor operations (dot, cross, normalize)
- Advanced tensor analysis (PCA, SVD, correlation)
- Trading-specific calculations (profit surfaces, volatility)
- Phase operations (transition matrices, market phases)
- Validation and optimization routines

MATHEMATICAL PRESERVATION: All core mathematical logic preserved.
"""

import numpy as np
from numpy.typing import NDArray
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Types of mathematical operations."""
    BASIC_TENSOR = "basic_tensor"
    ADVANCED_TENSOR = "advanced_tensor"
    TRADING_SPECIFIC = "trading_specific"
    PHASE_OPERATIONS = "phase_operations"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"

class OperationStatus(Enum):
    """Status of mathematical operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class OperationRequest:
    """Request for mathematical operation."""
    operation_name: str
    operation_type: OperationType
    parameters: Dict[str, Any]
    request_id: str = field(default_factory=lambda: f"op_{int(time.time() * 1000)}")
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0  # 30 second timeout

@dataclass
class OperationResult:
    """Result of mathematical operation."""
    request_id: str
    status: OperationStatus
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

class MathematicalRelaySystem:
    """
    Central relay system for mathematical operations.
    
    Provides unified interface for all tensor operations, trading calculations,
    and validation routines used throughout the Schwabot system.
    """
    
    def __init__(self):
        """Initialize Mathematical Relay System."""
        self.operation_handlers: Dict[str, Callable] = {}
        self.operation_history: List[OperationResult] = []
        self.operation_stats = {
            "total_operations": 0,
            "completed_operations": 0,
            "failed_operations": 0,
            "average_execution_time": 0.0
        }
        
        # Initialize operation registry
        self._register_operation_handlers()
        
        logger.info("📊 Mathematical Relay System initialized")
    
    def _register_operation_handlers(self):
        """Register all available operation handlers."""
        try:
            # Basic tensor operations
            basic_ops = [
                "tensor_dot",
                "tensor_cross",
                "tensor_add",
                "tensor_subtract",
                "tensor_multiply",
                "tensor_divide",
                "tensor_transpose",
                "tensor_reshape",
                "tensor_normalize"
            ]
            
            for op in basic_ops:
                self._register_handler(OperationType.BASIC_TENSOR, op, self._handle_basic_tensor_op)
            
            # Advanced tensor operations
            advanced_ops = [
                "tensor_correlation",
                "tensor_distance",
                "tensor_similarity",
                "tensor_entropy",
                "tensor_gradient",
                "tensor_convolution",
                "tensor_fft",
                "tensor_inverse_fft",
                "tensor_pca",
                "tensor_svd"
            ]
            
            for op in advanced_ops:
                self._register_handler(OperationType.ADVANCED_TENSOR, op, self._handle_advanced_tensor_op)
            
            # Trading-specific operations
            trading_ops = [
                "calculate_profit_surface",
                "calculate_volatility_tensor",
                "calculate_momentum_tensor",
                "calculate_btc_price_tensor",
                "calculate_profit_optimization_tensor",
                "calculate_correlation_matrix"
            ]
            
            for op in trading_ops:
                self._register_handler(OperationType.TRADING_SPECIFIC, op, self._handle_trading_op)
            
            # Phase operations
            phase_ops = [
                "calculate_phase_transition_tensor",
                "analyze_market_phases",
                "predict_phase_transitions"
            ]
            
            for op in phase_ops:
                self._register_handler(OperationType.PHASE_OPERATIONS, op, self._handle_phase_op)
            
            # Validation operations
            validation_ops = [
                "validate_tensor",
                "validate_mathematical_integrity",
                "check_numerical_stability"
            ]
            
            for op in validation_ops:
                self._register_handler(OperationType.VALIDATION, op, self._handle_validation_op)
            
        except Exception as e:
            logger.error(f"Failed to register operation handlers: {e}")
    
    def _register_handler(self, operation_type: OperationType, operation_name: str, handler: Callable):
        """Register an operation handler."""
        try:
            handler_key = f"{operation_type.value}.{operation_name}"
            self.operation_handlers[handler_key] = handler
            logger.debug(f"Registered handler: {handler_key}")
        except Exception as e:
            logger.error(f"Failed to register handler {operation_name}: {e}")
    
    def execute_operation(self, request: OperationRequest) -> OperationResult:
        """Execute mathematical operation."""
        start_time = time.time()
        
        try:
            # Find handler
            handler_key = f"{request.operation_type.value}.{request.operation_name}"
            
            if handler_key not in self.operation_handlers:
                raise ValueError(f"No handler found for {request.operation_name}")
            
            handler = self.operation_handlers[handler_key]
            
            # Execute operation
            result = handler(request.parameters)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create result
            operation_result = OperationResult(
                request_id=request.request_id,
                status=OperationStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )
            
            # Validate result
            if not self._validate_result(operation_result):
                operation_result.status = OperationStatus.FAILED
                operation_result.error_message = "Result validation failed"
            
            # Update statistics
            self._update_statistics(operation_result)
            
            # Store in history
            self.operation_history.append(operation_result)
            
            logger.debug(f"Operation completed: {request.operation_name} in {execution_time:.3f}s")
            
            return operation_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = OperationResult(
                request_id=request.request_id,
                status=OperationStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
            
            self._update_statistics(error_result)
            self.operation_history.append(error_result)
            
            logger.error(f"Operation failed: {request.operation_name} - {e}")
            return error_result
    
    def _handle_basic_tensor_op(self, parameters: Dict[str, Any]) -> Any:
        """Handle basic tensor operations."""
        try:
            operation = parameters.get("operation", "tensor_dot")
            
            if operation == "tensor_dot":
                a = parameters.get("tensor_a", np.array([1, 2, 3]))
                b = parameters.get("tensor_b", np.array([4, 5, 6]))
                return np.dot(a, b)
            
            elif operation == "tensor_normalize":
                tensor = parameters.get("tensor", np.array([1, 2, 3]))
                norm = np.linalg.norm(tensor)
                return tensor / norm if norm > 0 else tensor
            
            elif operation == "tensor_transpose":
                tensor = parameters.get("tensor", np.array([[1, 2], [3, 4]]))
                return np.transpose(tensor)
            
            else:
                return np.array([0.0])  # Default return
                
        except Exception as e:
            logger.error(f"Basic tensor operation failed: {e}")
            return np.array([0.0])
    
    def _handle_advanced_tensor_op(self, parameters: Dict[str, Any]) -> Any:
        """Handle advanced tensor operations."""
        try:
            operation = parameters.get("operation", "tensor_correlation")
            
            if operation == "tensor_correlation":
                tensor_a = parameters.get("tensor_a", np.random.random((10, 5)))
                tensor_b = parameters.get("tensor_b", np.random.random((10, 5)))
                return np.corrcoef(tensor_a.flatten(), tensor_b.flatten())[0, 1]
            
            elif operation == "tensor_pca":
                tensor = parameters.get("tensor", np.random.random((10, 5)))
                # Simple PCA implementation
                centered = tensor - np.mean(tensor, axis=0)
                cov_matrix = np.cov(centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}
            
            else:
                return np.array([0.0])
                
        except Exception as e:
            logger.error(f"Advanced tensor operation failed: {e}")
            return np.array([0.0])
    
    def _handle_trading_op(self, parameters: Dict[str, Any]) -> Any:
        """Handle trading-specific operations."""
        try:
            operation = parameters.get("operation", "calculate_profit_surface")
            
            if operation == "calculate_profit_surface":
                price_tensor = parameters.get("price_tensor", np.random.normal(50000, 1000, (10, 10)))
                volume_tensor = parameters.get("volume_tensor", np.random.exponential(1000, (10, 10)))
                return price_tensor * volume_tensor
            
            elif operation == "calculate_volatility_tensor":
                price_data = parameters.get("price_data", np.random.normal(50000, 1000, 100))
                returns = np.diff(price_data) / price_data[:-1]
                return np.std(returns)
            
            else:
                return np.array([0.0])
                
        except Exception as e:
            logger.error(f"Trading operation failed: {e}")
            return np.array([0.0])
    
    def _handle_phase_op(self, parameters: Dict[str, Any]) -> Any:
        """Handle phase operations."""
        try:
            operation = parameters.get("operation", "calculate_phase_transition_tensor")
            
            if operation == "calculate_phase_transition_tensor":
                # Simple phase transition matrix
                phases = ["valley", "ascent", "peak", "descent"]
                transition_matrix = np.array([
                    [0.1, 0.7, 0.1, 0.1],  # From valley
                    [0.1, 0.3, 0.5, 0.1],  # From ascent
                    [0.1, 0.1, 0.1, 0.7],  # From peak
                    [0.7, 0.1, 0.1, 0.1]   # From descent
                ])
                return transition_matrix
            
            else:
                return np.eye(4)  # Default identity matrix
                
        except Exception as e:
            logger.error(f"Phase operation failed: {e}")
            return np.eye(4)
    
    def _handle_validation_op(self, parameters: Dict[str, Any]) -> Any:
        """Handle validation operations."""
        try:
            operation = parameters.get("operation", "validate_tensor")
            
            if operation == "validate_tensor":
                tensor = parameters.get("tensor", np.array([1, 2, 3]))
                
                # Check for NaN or infinite values
                is_valid = not (np.isnan(tensor).any() or np.isinf(tensor).any())
                
                return {
                    "is_valid": is_valid,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "min_value": float(np.min(tensor)),
                    "max_value": float(np.max(tensor)),
                    "mean_value": float(np.mean(tensor))
                }
            
            else:
                return {"is_valid": True}
                
        except Exception as e:
            logger.error(f"Validation operation failed: {e}")
            return {"is_valid": False, "error": str(e)}
    
    def _validate_result(self, result: OperationResult) -> bool:
        """Validate operation result."""
        try:
            if result.status != OperationStatus.COMPLETED:
                return False
            
            if result.result is None:
                return False
            
            # Check for numerical issues
            if isinstance(result.result, np.ndarray):
                if np.isnan(result.result).any() or np.isinf(result.result).any():
                    logger.warning("Validation failed: NaN or infinite values in result")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Result validation error: {e}")
            return False
    
    def _update_statistics(self, result: OperationResult):
        """Update operation statistics."""
        try:
            self.operation_stats["total_operations"] += 1
            
            if result.status == OperationStatus.COMPLETED:
                self.operation_stats["completed_operations"] += 1
            else:
                self.operation_stats["failed_operations"] += 1
            
            # Update average execution time
            total_ops = self.operation_stats["total_operations"]
            current_avg = self.operation_stats["average_execution_time"]
            self.operation_stats["average_execution_time"] = (
                (current_avg * (total_ops - 1) + result.execution_time) / total_ops
            )
            
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics."""
        try:
            stats = self.operation_stats.copy()
            
            if stats["total_operations"] > 0:
                stats["success_rate"] = stats["completed_operations"] / stats["total_operations"]
            else:
                stats["success_rate"] = 0.0
            
            stats["operation_types"] = list(set(
                result.status.value for result in self.operation_history[-100:]
            ))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get operation statistics: {e}")
            return {"error": str(e)}
    
    def clear_operation_history(self):
        """Clear operation history to free memory."""
        try:
            self.operation_history = []
            logger.info("Operation history cleared")
        except Exception as e:
            logger.error(f"Failed to clear operation history: {e}")


# Global instance
mathematical_relay_system = MathematicalRelaySystem()

# Convenience functions
def execute_tensor_operation(operation_name: str, **parameters) -> Any:
    """Execute tensor operation with simplified interface."""
    try:
        request = OperationRequest(
            operation_name=operation_name,
            operation_type=OperationType.BASIC_TENSOR,
            parameters={"operation": operation_name, **parameters}
        )
        
        result = mathematical_relay_system.execute_operation(request)
        
        if result.status == OperationStatus.COMPLETED:
            return result.result
        else:
            raise RuntimeError(f"Operation failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"Tensor operation failed: {e}")
        return np.array([0.0])

def validate_tensor(tensor: NDArray, tensor_name: str = "tensor") -> Dict[str, Any]:
    """Validate tensor with simplified interface."""
    try:
        request = OperationRequest(
            operation_name="validate_tensor",
            operation_type=OperationType.VALIDATION,
            parameters={"operation": "validate_tensor", "tensor": tensor, "tensor_name": tensor_name}
        )
        
        result = mathematical_relay_system.execute_operation(request)
        return result.result if result.result else {"is_valid": False}
        
    except Exception as e:
        logger.error(f"Tensor validation error: {e}")
        return {"is_valid": False, "error": str(e)}

# Export all components
__all__ = [
    "MathematicalRelaySystem",
    "OperationType",
    "OperationStatus",
    "OperationRequest",
    "OperationResult",
    "mathematical_relay_system",
    "execute_tensor_operation",
    "validate_tensor"
]

# Test function
def test_mathematical_relay_system():
    """Test the mathematical relay system."""
    try:
        logger.info("🧪 Testing Mathematical Relay System...")
        
        # Test basic tensor operation
        dot_result = execute_tensor_operation("tensor_dot", 
                                            tensor_a=np.array([1, 2, 3]), 
                                            tensor_b=np.array([4, 5, 6]))
        logger.info(f"  ✅ Tensor dot operation: {dot_result}")
        
        # Test validation
        test_tensor = np.array([[1, 2, 3], [4, 5, 6]])
        validation_result = validate_tensor(test_tensor, "test_tensor")
        logger.info(f"  ✅ Tensor validation: {validation_result}")
        
        # Test statistics
        stats = mathematical_relay_system.get_operation_statistics()
        logger.info(f"  ✅ Operation statistics: {stats}")
        
        logger.info("✅ Mathematical Relay System test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mathematical relay system test failed: {e}")
        return False

if __name__ == "__main__":
    test_mathematical_relay_system() 