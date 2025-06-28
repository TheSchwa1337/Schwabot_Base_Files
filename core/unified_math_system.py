# -*- coding: utf-8 -*-
"""
Unified Math System for Schwabot Trading
========================================

Provides a unified interface for all mathematical operations used throughout
the trading system, including basic arithmetic, statistical functions,
linear algebra, and specialized trading calculations.

Mathematical Foundation:
- Tensor operations for multi-dimensional analysis
- Statistical computations for market analysis  
- Linear algebra for portfolio optimization
- Specialized BTC and crypto calculations

Windows CLI compatible with comprehensive error handling.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# Import tensor algebra with fallback
try:
    from .math.tensor_algebra.unified_tensor_algebra import UnifiedTensorAlgebra
    TENSOR_ALGEBRA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Tensor algebra not available: {e}")
    TENSOR_ALGEBRA_AVAILABLE = False

# Safe print functions for cross-platform compatibility
def safe_print(message: str) -> None:
    """Safe print for cross-platform compatibility."""
    try:
        print(message)
    except Exception:
        pass

def info(message: str) -> None:
    """Info level message."""
    print(f"[INFO] {message}")

def warn(message: str) -> None:
    """Warning level message."""
    print(f"[WARN] {message}")

def error(message: str) -> None:
    """Error level message."""
    print(f"[ERROR] {message}")


class MathOperation(Enum):
    """Mathematical operation types."""
    
    # Basic arithmetic
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    SQRT = "sqrt"
    LOG = "log"
    EXP = "exp"
    
    # Trigonometric
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"
    
    # Statistical
    ABS = "abs"
    MAX = "max"
    MIN = "min"
    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    CORRELATION = "correlation"
    COVARIANCE = "covariance"
    
    # Linear algebra
    DOT_PRODUCT = "dot_product"
    CROSS_PRODUCT = "cross_product"
    MATRIX_MULTIPLY = "matrix_multiply"
    INVERSE = "inverse"
    DETERMINANT = "determinant"
    EIGENVALUES = "eigenvalues"
    EIGENVECTORS = "eigenvectors"
    SVD = "svd"
    QR = "qr"
    LU = "lu"
    CHOLESKY = "cholesky"
    
    # Trading specific
    HASH_RATE = "hash_rate"
    DIFFICULTY_ADJUST = "difficulty_adjust"
    BLOCK_REWARD = "block_reward"
    PROFIT_VECTOR = "profit_vector"
    TIER_NAVIGATION = "tier_navigation"
    ENTRY_EXIT_OPTIMIZATION = "entry_exit_optimization"
    DLT_ANALYSIS = "dlt_analysis"
    TENSOR_CONTRACTION = "tensor_contraction"
    THERMAL_CORRECTION = "thermal_correction"


class UnifiedMathSystem:
    """Unified mathematical system for trading operations."""
    
    def __init__(self, precision: int = 64):
        """Initialize the unified math system."""
        self.precision = precision
        self.tensor_algebra = UnifiedTensorAlgebra() if TENSOR_ALGEBRA_AVAILABLE else None
        
        # Integration metrics
        self.integration_metrics = {
            "dlt_analysis_calls": 0,
            "tensor_operation_calls": 0,
            "thermal_correction_calls": 0,
            "integration_success_rate": 0.0
        }
        
        safe_print(f"Unified Math System initialized with precision {precision}")
        logger.info(f"Unified Math System initialized with precision {precision}")
    
    def execute_operation(self, operation: MathOperation, *args, **kwargs) -> Any:
        """Execute a mathematical operation."""
        try:
            if operation == MathOperation.ADD:
                return self.add(*args)
            elif operation == MathOperation.SUBTRACT:
                return self.subtract(*args)
            elif operation == MathOperation.MULTIPLY:
                return self.multiply(*args)
            elif operation == MathOperation.DIVIDE:
                return self.divide(*args)
            elif operation == MathOperation.POWER:
                return self.power(*args)
            elif operation == MathOperation.SQRT:
                return self.sqrt(*args)
            elif operation == MathOperation.LOG:
                return self.log(*args)
            elif operation == MathOperation.EXP:
                return self.exp(*args)
            elif operation == MathOperation.SIN:
                return self.sin(*args)
            elif operation == MathOperation.COS:
                return self.cos(*args)
            elif operation == MathOperation.TAN:
                return self.tan(*args)
            elif operation == MathOperation.ABS:
                return self.abs(*args)
            elif operation == MathOperation.MAX:
                return self.max(*args)
            elif operation == MathOperation.MIN:
                return self.min(*args)
            elif operation == MathOperation.MEAN:
                return self.mean(*args)
            elif operation == MathOperation.STD:
                return self.std(*args)
            elif operation == MathOperation.VAR:
                return self.var(*args)
            elif operation == MathOperation.DOT_PRODUCT:
                return self.dot_product(*args)
            elif operation == MathOperation.MATRIX_MULTIPLY:
                return self.matrix_multiply(*args)
            elif operation == MathOperation.EIGENVALUES:
                return self.eigenvalues(*args)
            elif operation == MathOperation.SVD:
                return self.svd(*args)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            error(f"Mathematical operation failed: {operation.value} - {e}")
            logger.error(f"Mathematical operation failed: {operation.value} - {e}")
            raise
    
    # Basic arithmetic operations
    def add(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Addition operation."""
        return np.add(a, b)
    
    def subtract(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Subtraction operation."""
        return np.subtract(a, b)
    
    def multiply(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Multiplication operation."""
        return np.multiply(a, b)
    
    def divide(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Division operation."""
        return np.divide(a, b)
    
    def power(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Power operation."""
        return np.power(a, b)
    
    def sqrt(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Square root operation."""
        return np.sqrt(a)
    
    def log(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Natural logarithm operation."""
        return np.log(a)
    
    def exp(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Exponential operation."""
        return np.exp(a)
    
    # Trigonometric functions
    def sin(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Sine function."""
        return np.sin(a)
    
    def cos(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cosine function."""
        return np.cos(a)
    
    def tan(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Tangent function."""
        return np.tan(a)
    
    # Statistical functions
    def abs(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Absolute value."""
        return np.abs(a)
    
    def max(self, a: Union[float, np.ndarray], axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Maximum value."""
        return np.max(a, axis=axis)
    
    def min(self, a: Union[float, np.ndarray], axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Minimum value."""
        return np.min(a, axis=axis)
    
    def mean(self, a: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Mean value."""
        return np.mean(a, axis=axis)
    
    def std(self, a: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Standard deviation."""
        return np.std(a, axis=axis)
    
    def var(self, a: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Variance."""
        return np.var(a, axis=axis)
    
    # Linear algebra operations
    def dot_product(self, a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """Dot product of two arrays."""
        return np.dot(a, b)
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""
        return np.matmul(a, b)
    
    def eigenvalues(self, a: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of a matrix."""
        return np.linalg.eigvals(a)
    
    def eigenvectors(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of a matrix."""
        return np.linalg.eig(a)
    
    def svd(self, a: np.ndarray, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular value decomposition."""
        return np.linalg.svd(a, full_matrices=full_matrices)
    
    # Integration functions
    def dlt_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform DLT (Distributed Ledger Technology) analysis."""
        try:
            self.integration_metrics["dlt_analysis_calls"] += 1
            
            # Import MathLib V4 for DLT analysis
            try:
                from .mathlib_v4 import MathLibV4
                mathlib = MathLibV4()
                mathlib_result = mathlib.calculate_dlt_metrics(data)
            except ImportError:
                raise ValueError("MathLib V4 not available for DLT analysis")
            
            if "error" in mathlib_result:
                raise ValueError(f"MathLib V4 DLT analysis failed: {mathlib_result['error']}")
            
            return {
                "status": "success",
                "dlt_metrics": mathlib_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            error(f"DLT analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        return self.integration_metrics.copy()


# Global instance
unified_math = UnifiedMathSystem()

# Export key functions and classes
__all__ = [
    "UnifiedMathSystem",
    "MathOperation", 
    "unified_math",
    "TENSOR_ALGEBRA_AVAILABLE"
]