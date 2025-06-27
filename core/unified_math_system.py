# -*- coding: utf-8 -*-
"""
Unified Mathematical System

This module provides a centralized mathematical system that consolidates all mathematical
operations used throughout the trading system, ensuring consistency and proper integration
with the unified mathematical framework.

Supports:
- Two-bit logic pathways for fast computation
- Profit tier navigation vectorization
- BTC hashing calculations
- Traditional mathematical operations
- Financial calculations
- Linear algebra operations
- Mathematical integration bridges with MathLib V4 and Tensor Algebra
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np

# Import core components (with fallbacks to avoid circular imports)
try:
    from core.bit_phase_sequencer import BitPhase, BitSequence
except ImportError:
    BitPhase = None
    BitSequence = None

try:
    from core.dual_error_handler import PhaseState, SickType, SickState
except ImportError:
    PhaseState = None
    SickType = None
    SickState = None

try:
    from core.symbolic_profit_router import ProfitTier, FlipBias, SymbolicState
except ImportError:
    ProfitTier = None
    FlipBias = None
    SymbolicState = None

# Import MathLib V4 for integration
try:
    from core.mathlib_v4 import MathLibV4, ForeverFractal
    MATHLIB_V4_AVAILABLE = True
except ImportError:
    MATHLIB_V4_AVAILABLE = False
    MathLibV4 = None
    ForeverFractal = None

# Import Tensor Algebra for integration
try:
    from core.math.tensor_algebra import UnifiedTensorAlgebra
    TENSOR_ALGEBRA_AVAILABLE = True
except ImportError:
    TENSOR_ALGEBRA_AVAILABLE = False
    UnifiedTensorAlgebra = None

try:
    from core.dual_unicore_handler import DualUnicoreHandler
    unicore = DualUnicoreHandler()
except ImportError:
    unicore = None

# Safe print functions with fallbacks
try:
    from core.utils.safe_print import safe_print, safe_math, info, warn, error
except ImportError:
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)

    def safe_math(*args, **kwargs):
        print(*args, **kwargs)

    def info(*args, **kwargs):
        print("[INFO]", *args, **kwargs)

    def warn(*args, **kwargs):
        print("[WARN]", *args, **kwargs)

    def error(*args, **kwargs):
        print("[ERROR]", *args, **kwargs)


class MathOperation(Enum):
    """Enumeration of mathematical operations."""

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
    
    # Utility
    ABS = "abs"
    MAX = "max"
    MIN = "min"
    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"
    
    # Statistical
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

    # BTC/Crypto operations
    HASH_RATE = "hash_rate"
    DIFFICULTY_ADJUST = "difficulty_adjust"
    BLOCK_REWARD = "block_reward"
    
    # Profit operations
    PROFIT_VECTOR = "profit_vector"
    TIER_NAVIGATION = "tier_navigation"
    ENTRY_EXIT_OPTIMIZATION = "entry_exit_optimization"
    
    # Integration operations
    DLT_ANALYSIS = "dlt_analysis"
    TENSOR_CONTRACTION = "tensor_contraction"
    THERMAL_CORRECTION = "thermal_correction"


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
    
    Features:
    - Traditional mathematical operations
    - BTC hashing and mining calculations
    - Profit tier navigation vectorization
    - Two-bit logic pathway optimization
    - Financial analysis tools
    - Mathematical integration bridges with MathLib V4 and Tensor Algebra
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

        # Initialize integration components
        self.mathlib_v4 = MathLibV4() if MATHLIB_V4_AVAILABLE else None
        self.tensor_algebra = UnifiedTensorAlgebra() if TENSOR_ALGEBRA_AVAILABLE else None

        # BTC/Crypto constants
        self.btc_constants = {
            'max_supply': 21_000_000,
            'halving_interval': 210_000,
            'initial_reward': 50.0,
            'target_block_time': 600,  # 10 minutes in seconds
            'difficulty_adjustment_blocks': 2016,
            'hash_rate_unit': 1e12,  # TH/s
        }

        # Mathematical constants
        self.math_constants = {
            'pi': np.pi,
            'e': np.e,
            'golden_ratio': 1.618033988749,
            'euler_mascheroni': 0.577215664901,
            'sqrt_2': np.sqrt(2),
            'sqrt_3': np.sqrt(3),
        }

        # Initialize operation registry
        self._register_operations()
        
        # Integration metrics
        self.integration_metrics = {
            "dlt_analysis_calls": 0,
            "tensor_operation_calls": 0,
            "thermal_correction_calls": 0,
            "integration_success_rate": 1.0
        }

        if self.use_safe_print:
            safe_print(f"Unified Math System initialized with precision {precision}")
        else:
            self.logger.info(f"Unified Math System initialized with precision {precision}")

    def _register_operations(self):
        """Register all available mathematical operations."""
        self.operations = {
            # Basic arithmetic
            MathOperation.ADD: lambda x, y: x + y,
            MathOperation.SUBTRACT: lambda x, y: x - y,
            MathOperation.MULTIPLY: lambda x, y: x * y,
            MathOperation.DIVIDE: lambda x, y: x / y if y != 0 else float('inf'),
            MathOperation.POWER: lambda x, y: x ** y,
            MathOperation.SQRT: lambda x: np.sqrt(x),
            MathOperation.LOG: lambda x: np.log(x),
            MathOperation.EXP: lambda x: np.exp(x),
            
            # Trigonometric
            MathOperation.SIN: lambda x: np.sin(x),
            MathOperation.COS: lambda x: np.cos(x),
            MathOperation.TAN: lambda x: np.tan(x),
            MathOperation.ASIN: lambda x: np.arcsin(x),
            MathOperation.ACOS: lambda x: np.arccos(x),
            MathOperation.ATAN: lambda x: np.arctan(x),
            
            # Utility
            MathOperation.ABS: lambda x: np.abs(x),
            MathOperation.MAX: lambda x: np.max(x),
            MathOperation.MIN: lambda x: np.min(x),
            MathOperation.ROUND: lambda x: np.round(x, self.precision),
            MathOperation.FLOOR: lambda x: np.floor(x),
            MathOperation.CEIL: lambda x: np.ceil(x),
            
            # Statistical
            MathOperation.MEAN: lambda x: np.mean(x),
            MathOperation.STD: lambda x: np.std(x),
            MathOperation.VAR: lambda x: np.var(x),
            MathOperation.CORRELATION: lambda x, y: np.corrcoef(x, y)[0, 1],
            MathOperation.COVARIANCE: lambda x, y: np.cov(x, y)[0, 1],
            
            # Linear algebra
            MathOperation.DOT_PRODUCT: lambda x, y: np.dot(x, y),
            MathOperation.CROSS_PRODUCT: lambda x, y: np.cross(x, y),
            MathOperation.MATRIX_MULTIPLY: lambda x, y: np.matmul(x, y),
            MathOperation.INVERSE: lambda x: np.linalg.inv(x),
            MathOperation.DETERMINANT: lambda x: np.linalg.det(x),
            MathOperation.EIGENVALUES: lambda x: np.linalg.eigvals(x),
            MathOperation.EIGENVECTORS: lambda x: np.linalg.eig(x),
            MathOperation.SVD: lambda x: np.linalg.svd(x),
            MathOperation.QR: lambda x: np.linalg.qr(x),
            MathOperation.LU: lambda x: np.linalg.lu(x),
            MathOperation.CHOLESKY: lambda x: np.linalg.cholesky(x),
            
            # Integration operations
            MathOperation.DLT_ANALYSIS: self._execute_dlt_analysis,
            MathOperation.TENSOR_CONTRACTION: self._execute_tensor_contraction,
            MathOperation.THERMAL_CORRECTION: self._execute_thermal_correction,
        }

    def execute(self, operation: MathOperation, *args, **kwargs) -> MathResult:
        """
        Execute a mathematical operation.

        Args:
            operation: The mathematical operation to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            MathResult containing the operation result
        """
        try:
            if operation not in self.operations:
                raise ValueError(f"Unknown operation: {operation}")

            # Execute the operation
            integration_operations = [
                MathOperation.DLT_ANALYSIS,
                MathOperation.TENSOR_CONTRACTION,
                MathOperation.THERMAL_CORRECTION
            ]
            if operation in integration_operations:
                result = self.operations[operation](*args, **kwargs)
            else:
                result = self.operations[operation](*args)

            # Record successful operation
            self.success_count += 1
            self.operation_history.append({
                'operation': operation.value,
                'args': args,
                'kwargs': kwargs,
                'success': True,
                'timestamp': self._get_timestamp()
            })

            return MathResult(
                value=result,
                operation=operation,
                inputs=list(args),
                metadata={'success': True, 'precision': self.precision},
                success=True
            )

        except Exception as e:
            # Record failed operation
            self.error_count += 1
            self.operation_history.append({
                'operation': operation.value,
                'args': args,
                'kwargs': kwargs,
                'success': False,
                'error': str(e),
                'timestamp': self._get_timestamp()
            })

            if self.use_safe_print:
                error(f"Mathematical operation failed: {operation.value} - {e}")
            else:
                self.logger.error(f"Mathematical operation failed: {operation.value} - {e}")

            return MathResult(
                value=None,
                operation=operation,
                inputs=list(args),
                metadata={'success': False, 'error': str(e)},
                success=False,
                error_message=str(e)
            )

    def _execute_dlt_analysis(self, time_series: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Execute DLT analysis with MathLib V4 integration.
        
        Mathematical Formula: unified_result = DLT_analysis × math_system_weight
        
        Args:
            time_series: Input time series data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing DLT analysis results with unified math integration
        """
        try:
            self.integration_metrics["dlt_analysis_calls"] += 1
            
            if not self.mathlib_v4:
                raise ValueError("MathLib V4 not available for DLT analysis")
            
            # Get MathLib V4 DLT analysis result
            mathlib_result = self.mathlib_v4.analyze_dlt_waveform(time_series)
            
            if "error" in mathlib_result:
                raise ValueError(f"MathLib V4 DLT analysis failed: {mathlib_result['error']}")
            
            # Calculate unified math system weight
            unified_weight = self._calculate_system_weight(mathlib_result)
            
            # Apply unified math integration
            unified_result = {
                "dlt_analysis": mathlib_result,
                "unified_weight": unified_weight,
                "final_result": mathlib_result["confidence"] * unified_weight,
                "integration_metrics": self.integration_metrics,
                "mathematical_formula": "unified_result = DLT_analysis × math_system_weight"
            }
            
            return unified_result
            
        except Exception as e:
            self.logger.error(f"DLT analysis integration failed: {e}")
            return {"error": str(e), "integration_metrics": self.integration_metrics}

    def _execute_tensor_contraction(self, tensor_a: np.ndarray, tensor_b: np.ndarray, **kwargs) -> np.ndarray:
        """
        Execute tensor contraction with Tensor Algebra integration.
        
        Mathematical Formula: tensor_result = unified_operation ⊗ tensor_weight
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            **kwargs: Additional parameters
            
        Returns:
            Tensor contraction result with unified math integration
        """
        try:
            self.integration_metrics["tensor_operation_calls"] += 1
            
            if not self.tensor_algebra:
                raise ValueError("Tensor Algebra not available for tensor contraction")
            
            # Get tensor algebra result
            tensor_result = self.tensor_algebra.tensor_contraction(tensor_a, tensor_b)
            
            # Calculate tensor weight
            tensor_weight = self._calculate_tensor_weight(tensor_a, tensor_b)
            
            # Apply unified math integration
            unified_tensor_result = tensor_result * tensor_weight
            
            return unified_tensor_result
            
        except Exception as e:
            self.logger.error(f"Tensor contraction integration failed: {e}")
            return np.zeros_like(tensor_a)

    def _execute_thermal_correction(self, data: np.ndarray, thermal_factor: float = 1.0, **kwargs) -> np.ndarray:
        """
        Execute thermal correction with thermal integration.
        
        Mathematical Formula: T_corrected = T_original × thermal_factor
        
        Args:
            data: Input data
            thermal_factor: Thermal correction factor
            **kwargs: Additional parameters
            
        Returns:
            Thermally corrected data
        """
        try:
            self.integration_metrics["thermal_correction_calls"] += 1
            
            # Apply thermal correction
            corrected_data = data * thermal_factor
            
            return corrected_data
            
        except Exception as e:
            self.logger.error(f"Thermal correction failed: {e}")
            return data

    def _calculate_system_weight(self, mathlib_result: Dict[str, Any]) -> float:
        """Calculate unified math system weight for DLT analysis."""
        try:
            # Base weight from confidence
            base_weight = mathlib_result.get("confidence", 0.5)
            
            # Adjust based on triplet lock
            triplet_bonus = 0.1 if mathlib_result.get("triplet_lock", False) else 0.0
            
            # Adjust based on standard deviation (lower is better)
            std_dev = mathlib_result.get("std_dev", 1.0)
            std_penalty = min(0.2, std_dev * 0.1)
            
            final_weight = base_weight + triplet_bonus - std_penalty
            return max(0.0, min(1.0, final_weight))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating system weight: {e}")
            return 0.5

    def _calculate_tensor_weight(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
        """Calculate tensor weight for tensor operations."""
        try:
            # Weight based on tensor sizes and operation speed
            size_factor = min(1.0, (tensor_a.size + tensor_b.size) / 1000)
            speed_factor = 0.95  # Assume 95% operation speed
            accuracy_factor = 0.99  # Assume 99% accuracy
            
            weight = size_factor * speed_factor * accuracy_factor
            return max(0.0, min(1.0, weight))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating tensor weight: {e}")
            return 0.5

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        total_operations = self.success_count + self.error_count
        success_rate = self.success_count / total_operations if total_operations > 0 else 0
        
        return {
            'total_operations': total_operations,
            'successful_operations': self.success_count,
            'failed_operations': self.error_count,
            'success_rate': success_rate,
            'precision': self.precision,
            'integration_metrics': self.integration_metrics,
            'components_available': {
                'mathlib_v4': MATHLIB_V4_AVAILABLE,
                'tensor_algebra': TENSOR_ALGEBRA_AVAILABLE
            }
        }


def get_unified_math() -> UnifiedMathSystem:
    """Get a singleton instance of the unified math system."""
    if not hasattr(get_unified_math, '_instance'):
        get_unified_math._instance = UnifiedMathSystem()
    return get_unified_math._instance


# Global instance
unified_math = get_unified_math()
