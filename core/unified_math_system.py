"""
Unified Mathematical System

This module provides a centralized mathematical system that consolidates all mathematical
operations used throughout the trading system, ensuring consistency and proper integration
with the unified mathematical framework.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Import safe print for CLI compatibility
try:
    from utils.safe_print import safe_print, safe_math, info, warn, error
except ImportError:
    # Fallback for when utils is not available
def safe_print(*args, **kwargs):
    print(*args, **kwargs)
def safe_math(*args, **kwargs):
    print(*args, **kwargs)
def info(*args, **kwargs):
    print(*args, **kwargs)
def warn(*args, **kwargs):
    print(*args, **kwargs)
def error(*args, **kwargs):
    print(*args, **kwargs)

class MathOperation(Enum):
    """Enumeration of mathematical operations."""
ADD = "add"
SUBTRACT = "subtract"
MULTIPLY = "multiply"
DIVIDE = "divide"
POWER = "power"
SQRT = "sqrt"
LOG = "log"
EXP = "exp"
SIN = "sin"
COS = "cos"
TAN = "tan"
ABS = "abs"
MAX = "max"
MIN = "min"
MEAN = "mean"
STD = "std"
VAR = "var"
CORRELATION = "correlation"
COVARIANCE = "covariance"
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

@dataclass
class MathResult:
    """Container for mathematical operation results."""
value: Any
operation: MathOperation
inputs: List[Any]
metadata: Dict[str, Any]
success: bool
error_message: Optional[str] = None

class UnifiedMathSystem:
    """
Unified mathematical system for the trading platform.

This class provides a centralized interface for all mathematical operations,
    ensuring consistency and proper error handling across the system.
"""

    def __init__(self, precision: int = 8, use_safe_print: bool = True):
        """
Initialize the unified math system.

Args:
precision: Number of decimal places for floating point operations
use_safe_print: Whether to use safe print for CLI compatibility
"""
self.precision = precision
self.use_safe_print = use_safe_print
self.operation_history = []
self.error_count = 0
self.success_count = 0

        # Set numpy precision
np.set_printoptions(precision=precision)

        # Initialize logging
self.logger = logging.getLogger(__name__)

info("Unified Math System initialized")

    def _log_operation(self, operation: MathOperation, inputs: List[Any],
                      result: Any, success: bool, error_msg: Optional[str] = None) -> None:
"""Log mathematical operations for debugging and auditing."""
log_entry = {
'operation': operation.value,
'inputs': inputs,
'result': result,
'success': success,
'error_message': error_msg,
'timestamp': np.datetime64('now')
        }

self.operation_history.append(log_entry)

        if success:
self.success_count += 1
            if self.use_safe_print:
safe_math(f"{operation.value}: {result}")
        else:
self.error_count += 1
            if self.use_safe_print:
error(f"Math error in {operation.value}: {error_msg}")

    def _validate_inputs(self, inputs: List[Any], expected_types: List[type]) -> bool:
        """Validate input types for mathematical operations."""
        if len(inputs) != len(expected_types):
            return False

        for input_val, expected_type in zip(inputs, expected_types):
            if not isinstance(input_val, expected_type):
                return False

        return True

    def _safe_operation(self, operation_func, inputs: List[Any],
                       operation: MathOperation) -> MathResult:
"""Safely execute a mathematical operation with error handling."""
        try:
result = operation_func(*inputs)

            # Round floating point results to specified precision
            if isinstance(result, (float, np.floating)):
                result = round(result, self.precision)
            elif isinstance(result, np.ndarray) and result.dtype.kind in 'fc':
                result = np.round(result, self.precision)

math_result = MathResult(
                value=result,
operation=operation,
inputs=inputs,
metadata={'precision': self.precision},
success=True


self._log_operation(operation, inputs, result, True)
            return math_result

        except Exception as e:
error_msg = f"Operation {operation.value} failed: {str(e)}"
            math_result = MathResult(
                value=None,
operation=operation,
inputs=inputs,
metadata={'error': str(e)},
                success=False,
error_message=error_msg


self._log_operation(operation, inputs, None, False, error_msg)
            return math_result

    # Basic arithmetic operations
    def add(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Add two numbers or arrays."""
        return self._safe_operation(lambda x, y: x + y, [a, b], MathOperation.ADD)

    def subtract(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Subtract two numbers or arrays."""
        return self._safe_operation(lambda x, y: x - y, [a, b], MathOperation.SUBTRACT)

    def multiply(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Multiply two numbers or arrays."""
        return self._safe_operation(lambda x, y: x * y, [a, b], MathOperation.MULTIPLY)

    def divide(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Divide two numbers or arrays."""
        if isinstance(b, (int, float)) and b == 0:
            return MathResult(None, MathOperation.DIVIDE, [a, b], {}, False, "Division by zero")
        return self._safe_operation(lambda x, y: x / y, [a, b], MathOperation.DIVIDE)

    def power(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> MathResult:
        """Raise a to the power of b."""
        return self._safe_operation(lambda x, y: x ** y, [a, b], MathOperation.POWER)

    def sqrt(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate square root."""
        if isinstance(a, (int, float)) and a < 0:
            return MathResult(None, MathOperation.SQRT, [a], {}, False, "Negative number under square root")
        return self._safe_operation(lambda x: np.sqrt(x), [a], MathOperation.SQRT)

    def log(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate natural logarithm."""
        if isinstance(a, (int, float)) and a <= 0:
            return MathResult(None, MathOperation.LOG, [a], {}, False, "Non-positive number for logarithm")
        return self._safe_operation(lambda x: np.log(x), [a], MathOperation.LOG)

    def exp(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate exponential."""
        return self._safe_operation(lambda x: np.exp(x), [a], MathOperation.EXP)

    def sin(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate sine."""
        return self._safe_operation(lambda x: np.sin(x), [a], MathOperation.SIN)

    def cos(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate cosine."""
        return self._safe_operation(lambda x: np.cos(x), [a], MathOperation.COS)

    def tan(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate tangent."""
        return self._safe_operation(lambda x: np.tan(x), [a], MathOperation.TAN)

    # Statistical operations
    def mean(self, a: Union[List, np.ndarray]) -> MathResult:
        """Calculate mean."""
        return self._safe_operation(lambda x: np.mean(x), [a], MathOperation.MEAN)

    def std(self, a: Union[List, np.ndarray], ddof: int = 1) -> MathResult:
        """Calculate standard deviation."""
        return self._safe_operation(lambda x: np.std(x, ddof=ddof), [a], MathOperation.STD)

    def var(self, a: Union[List, np.ndarray], ddof: int = 1) -> MathResult:
        """Calculate variance."""
        return self._safe_operation(lambda x: np.var(x, ddof=ddof), [a], MathOperation.VAR)

    def correlation(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> MathResult:
        """Calculate correlation coefficient."""
        return self._safe_operation(lambda x, y: np.corrcoef(x, y)[0, 1], [a, b], MathOperation.CORRELATION)

    def covariance(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> MathResult:
        """Calculate covariance."""
        return self._safe_operation(lambda x, y: np.cov(x, y)[0, 1], [a, b], MathOperation.COVARIANCE)

    # Linear algebra operations
    def dot_product(self, a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> MathResult:
        """Calculate dot product."""
        return self._safe_operation(lambda x, y: np.dot(x, y), [a, b], MathOperation.DOT_PRODUCT)

    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> MathResult:
        """Multiply matrices."""
        return self._safe_operation(lambda x, y: np.matmul(x, y), [a, b], MathOperation.MATRIX_MULTIPLY)

    def inverse(self, a: np.ndarray) -> MathResult:
        """Calculate matrix inverse."""
        return self._safe_operation(lambda x: np.linalg.inv(x), [a], MathOperation.INVERSE)

    def determinant(self, a: np.ndarray) -> MathResult:
        """Calculate matrix determinant."""
        return self._safe_operation(lambda x: np.linalg.det(x), [a], MathOperation.DETERMINANT)

    def eigenvalues(self, a: np.ndarray) -> MathResult:
        """Calculate eigenvalues."""
        return self._safe_operation(lambda x: np.linalg.eigvals(x), [a], MathOperation.EIGENVALUES)

    def eigenvectors(self, a: np.ndarray) -> MathResult:
        """Calculate eigenvalues and eigenvectors."""
        return self._safe_operation(lambda x: np.linalg.eig(x), [a], MathOperation.EIGENVECTORS)

    def svd(self, a: np.ndarray) -> MathResult:
        """Calculate singular value decomposition."""
        return self._safe_operation(lambda x: np.linalg.svd(x), [a], MathOperation.SVD)

    # Utility operations
    def abs(self, a: Union[float, np.ndarray]) -> MathResult:
        """Calculate absolute value."""
        return self._safe_operation(lambda x: np.abs(x), [a], MathOperation.ABS)

    def max(self, a: Union[List, np.ndarray]) -> MathResult:
        """Find maximum value."""
        return self._safe_operation(lambda x: np.max(x), [a], MathOperation.MAX)

    def min(self, a: Union[List, np.ndarray]) -> MathResult:
        """Find minimum value."""
        return self._safe_operation(lambda x: np.min(x), [a], MathOperation.MIN)

    # Financial calculations
    def calculate_returns(self, prices: List[float]) -> MathResult:
        """Calculate percentage returns from price series."""
        if len(prices) < 2:
            return MathResult(None, MathOperation.DIVIDE, [prices], {}, False, "Need at least 2 prices")

        try:
returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
                else:
returns.append(0.0)

            return MathResult(
                value=returns,
operation=MathOperation.DIVIDE,
inputs=[prices],
metadata={'type': 'returns'},
success=True

        except Exception as e:
            return MathResult(None, MathOperation.DIVIDE, [prices], {}, False, str(e))

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> MathResult:
        """Calculate Sharpe ratio."""
        if not returns:
            return MathResult(None, MathOperation.DIVIDE, [returns], {}, False, "Empty returns list")

        try:
mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                return MathResult(None, MathOperation.DIVIDE, [returns], {}, False, "Zero standard deviation")

sharpe = (mean_return - risk_free_rate) / std_return

            return MathResult(
                value=sharpe,
operation=MathOperation.DIVIDE,
inputs=[returns, risk_free_rate],
metadata={'type': 'sharpe_ratio'},
success=True

        except Exception as e:
            return MathResult(None, MathOperation.DIVIDE, [returns], {}, False, str(e))

    def calculate_max_drawdown(self, prices: List[float]) -> MathResult:
        """Calculate maximum drawdown."""
        if not prices:
            return MathResult(None, MathOperation.MIN, [prices], {}, False, "Empty prices list")

        try:
peak = prices[0]
max_dd = 0.0

            for price in prices:
                if price > peak:
peak = price
dd = (peak - price) / peak
                max_dd = max(max_dd, dd)

            return MathResult(
                value=max_dd,
operation=MathOperation.MIN,
inputs=[prices],
metadata={'type': 'max_drawdown'},
success=True

        except Exception as e:
            return MathResult(None, MathOperation.MIN, [prices], {}, False, str(e))

    def calculate_volatility(self, returns: List[float], window: int = 252) -> MathResult:
        """Calculate rolling volatility."""
        if len(returns) < window:
            return MathResult(None, MathOperation.STD, [returns], {}, False, f"Need at least {window} returns")

        try:
volatilities = []
            for i in range(window, len(returns) + 1):
                window_returns = returns[i-window:i]
vol = np.std(window_returns) * np.sqrt(252)  # Annualized
                volatilities.append(vol)

            return MathResult(
                value=volatilities,
operation=MathOperation.STD,
inputs=[returns, window],
metadata={'type': 'rolling_volatility', 'window': window},
success=True

        except Exception as e:
            return MathResult(None, MathOperation.STD, [returns], {}, False, str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
'total_operations': len(self.operation_history),
            'success_count': self.success_count,
'error_count': self.error_count,
'success_rate': self.success_count / max(len(self.operation_history), 1),
            'precision': self.precision
}

    def clear_history(self) -> None:
        """Clear operation history."""
self.operation_history.clear()
        self.error_count = 0
self.success_count = 0

# Create global instance
unified_math = UnifiedMathSystem()
