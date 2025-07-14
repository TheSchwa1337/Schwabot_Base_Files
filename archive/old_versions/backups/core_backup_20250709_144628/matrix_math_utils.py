"""Module for Schwabot trading system."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

"""
Matrix Math Utilities for Schwabot Trading System.

Provides advanced matrix and linear-algebra functions used by
back-testing and self-corrective engines.

    Key Features:
    1. Covariance & correlation matrix calculation
    2. Eigenvalue & condition-number diagnostics
    3. Simple risk-parity weight generator
    4. Matrix stability scoring for dynamic risk controls

    All public helpers are pure functions and NumPy-based so they can be
    unit-tested in isolation.

        CUDA Integration:
        - GPU-accelerated matrix utilities with automatic CPU fallback
        - Performance monitoring and optimization
        - Cross-platform compatibility (Windows, macOS, Linux)
        """

        # CUDA Integration with Fallback
            try:
            import cupy as cp

            USING_CUDA = True
            _backend = 'cupy (GPU)'
            xp = cp
                except ImportError:
                import numpy as cp  # fallback to numpy

                USING_CUDA = False
                _backend = 'numpy (CPU)'
                xp = cp

                # Log backend status
                logger = logging.getLogger(__name__)
                    if USING_CUDA:
                    logger.info("⚡ Matrix Math Utils using GPU acceleration: {0}".format(_backend))
                        else:
                        logger.info("🔄 Matrix Math Utils using CPU fallback: {0}".format(_backend))


                        @dataclass
                            class MatrixResult:
    """Class for Schwabot trading functionality."""
                            """Class for Schwabot trading functionality."""
                            """Result container for matrix operations."""

                            result: Any
                            operation: str
                            timestamp: float
                            metadata: Dict[str, Any] = field(default_factory=dict)


                                class MatrixMathUtils:
    """Class for Schwabot trading functionality."""
                                """Class for Schwabot trading functionality."""
                                """
                                Advanced matrix mathematical utilities for trading calculations.

                                Implements matrix operations, eigenvalue analysis, and matrix-based
                                trading metrics with GPU acceleration support.
                                """

                                    def __init__(self) -> None:
                                    """Initialize the matrix math utilities."""
                                    self.operation_history: List[MatrixResult] = []
                                    self.matrix_cache: Dict[str, Any] = {}

                                        def matrix_multiply(self, matrix_a: xp.ndarray, matrix_b: xp.ndarray) -> xp.ndarray:
                                        """
                                        Multiply two matrices with GPU acceleration.

                                            Args:
                                            matrix_a: First matrix
                                            matrix_b: Second matrix

                                                Returns:
                                                Result matrix
                                                """
                                                    try:
                                                        if matrix_a.shape[1] != matrix_b.shape[0]:
                                                    raise ValueError("Matrix dimensions incompatible for multiplication")

                                                    result = xp.dot(matrix_a, matrix_b)

                                                    self._log_operation(
                                                    "matrix_multiply",
                                                    result,
                                                    {
                                                    "matrix_a_shape": matrix_a.shape,
                                                    "matrix_b_shape": matrix_b.shape,
                                                    "result_shape": result.shape,
                                                    },
                                                    )
                                                return result

                                                    except Exception as e:
                                                    logger.error("Error in matrix multiplication: {0}".format(e))
                                                return xp.array([])

                                                    def matrix_inverse(self, matrix: xp.ndarray) -> xp.ndarray:
                                                    """
                                                    Calculate matrix inverse with GPU acceleration.

                                                        Args:
                                                        matrix: Input matrix

                                                            Returns:
                                                            Inverse matrix
                                                            """
                                                                try:
                                                                    if matrix.shape[0] != matrix.shape[1]:
                                                                raise ValueError("Matrix must be square for inverse")

                                                                # Check if matrix is invertible
                                                                det = xp.linalg.det(matrix)
                                                                    if abs(det) < 1e-10:
                                                                    logger.warning("Matrix is nearly singular, using pseudo-inverse")
                                                                    result = xp.linalg.pinv(matrix)
                                                                        else:
                                                                        result = xp.linalg.inv(matrix)

                                                                        self._log_operation("matrix_inverse", result, {"matrix_shape": matrix.shape, "determinant": det})
                                                                    return result

                                                                        except Exception as e:
                                                                        logger.error("Error in matrix inverse: {0}".format(e))
                                                                    return xp.array([])

                                                                        def eigenvalue_decomposition(self, matrix: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
                                                                        """
                                                                        Perform eigenvalue decomposition.

                                                                            Args:
                                                                            matrix: Input matrix

                                                                                Returns:
                                                                                Tuple of (eigenvalues, eigenvectors)
                                                                                """
                                                                                    try:
                                                                                        if matrix.shape[0] != matrix.shape[1]:
                                                                                    raise ValueError("Matrix must be square for eigenvalue decomposition")

                                                                                    eigenvalues, eigenvectors = xp.linalg.eig(matrix)

                                                                                    self._log_operation(
                                                                                    "eigenvalue_decomposition",
                                                                                    (eigenvalues, eigenvectors),
                                                                                    {"matrix_shape": matrix.shape, "num_eigenvalues": len(eigenvalues)},
                                                                                    )
                                                                                return eigenvalues, eigenvectors

                                                                                    except Exception as e:
                                                                                    logger.error("Error in eigenvalue decomposition: {0}".format(e))
                                                                                return xp.array([]), xp.array([])

                                                                                    def singular_value_decomposition(self, matrix: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
                                                                                    """
                                                                                    Perform singular value decomposition.

                                                                                        Args:
                                                                                        matrix: Input matrix

                                                                                            Returns:
                                                                                            Tuple of (U, S, V) matrices
                                                                                            """
                                                                                                try:
                                                                                                U, S, V = xp.linalg.svd(matrix)

                                                                                                self._log_operation("svd", (U, S, V), {"matrix_shape": matrix.shape, "num_singular_values": len(S)})
                                                                                            return U, S, V

                                                                                                except Exception as e:
                                                                                                logger.error("Error in SVD: {0}".format(e))
                                                                                            return xp.array([]), xp.array([]), xp.array([])

                                                                                                def matrix_rank(self, matrix: xp.ndarray, tolerance: float = 1e-10) -> int:
                                                                                                """
                                                                                                Calculate matrix rank.

                                                                                                    Args:
                                                                                                    matrix: Input matrix
                                                                                                    tolerance: Tolerance for singular values

                                                                                                        Returns:
                                                                                                        Matrix rank
                                                                                                        """
                                                                                                            try:
                                                                                                            S = xp.linalg.svd(matrix, compute_uv=False)
                                                                                                            rank = xp.sum(S > tolerance)

                                                                                                            self._log_operation(
                                                                                                            "matrix_rank",
                                                                                                            rank,
                                                                                                            {"matrix_shape": matrix.shape, "tolerance": tolerance, "singular_values": S},
                                                                                                            )
                                                                                                        return int(rank)

                                                                                                            except Exception as e:
                                                                                                            logger.error("Error calculating matrix rank: {0}".format(e))
                                                                                                        return 0

                                                                                                            def matrix_condition_number(self, matrix: xp.ndarray) -> float:
                                                                                                            """
                                                                                                            Calculate matrix condition number.

                                                                                                                Args:
                                                                                                                matrix: Input matrix

                                                                                                                    Returns:
                                                                                                                    Condition number
                                                                                                                    """
                                                                                                                        try:
                                                                                                                        S = xp.linalg.svd(matrix, compute_uv=False)
                                                                                                                        condition_number = xp.max(S) / xp.min(S)

                                                                                                                        self._log_operation(
                                                                                                                        "matrix_condition_number",
                                                                                                                        condition_number,
                                                                                                                        {"matrix_shape": matrix.shape, "singular_values": S},
                                                                                                                        )
                                                                                                                    return float(condition_number)

                                                                                                                        except Exception as e:
                                                                                                                        logger.error("Error calculating condition number: {0}".format(e))
                                                                                                                    return float('inf')

                                                                                                                        def matrix_norm(self, matrix: xp.ndarray, norm_type: str = 'frobenius') -> float:
                                                                                                                        """
                                                                                                                        Calculate matrix norm.

                                                                                                                            Args:
                                                                                                                            matrix: Input matrix
                                                                                                                            norm_type: Type of norm ('frobenius', 'spectral', 'nuclear')

                                                                                                                                Returns:
                                                                                                                                Matrix norm
                                                                                                                                """
                                                                                                                                    try:
                                                                                                                                        if norm_type == 'frobenius':
                                                                                                                                        norm = xp.linalg.norm(matrix, 'fro')
                                                                                                                                            elif norm_type == 'spectral':
                                                                                                                                            norm = xp.linalg.norm(matrix, 2)
                                                                                                                                                elif norm_type == 'nuclear':
                                                                                                                                                S = xp.linalg.svd(matrix, compute_uv=False)
                                                                                                                                                norm = xp.sum(S)
                                                                                                                                                    else:
                                                                                                                                                raise ValueError("Unknown norm type: {0}".format(norm_type))

                                                                                                                                                self._log_operation("matrix_norm", norm, {"matrix_shape": matrix.shape, "norm_type": norm_type})
                                                                                                                                            return float(norm)

                                                                                                                                                except Exception as e:
                                                                                                                                                logger.error("Error calculating matrix norm: {0}".format(e))
                                                                                                                                            return 0.0

                                                                                                                                                def matrix_trace(self, matrix: xp.ndarray) -> float:
                                                                                                                                                """
                                                                                                                                                Calculate matrix trace.

                                                                                                                                                    Args:
                                                                                                                                                    matrix: Input matrix

                                                                                                                                                        Returns:
                                                                                                                                                        Matrix trace
                                                                                                                                                        """
                                                                                                                                                            try:
                                                                                                                                                                if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                            raise ValueError("Matrix must be square for trace calculation")

                                                                                                                                                            trace = xp.trace(matrix)

                                                                                                                                                            self._log_operation("matrix_trace", trace, {"matrix_shape": matrix.shape})
                                                                                                                                                        return float(trace)

                                                                                                                                                            except Exception as e:
                                                                                                                                                            logger.error("Error calculating matrix trace: {0}".format(e))
                                                                                                                                                        return 0.0

                                                                                                                                                            def matrix_determinant(self, matrix: xp.ndarray) -> float:
                                                                                                                                                            """
                                                                                                                                                            Calculate matrix determinant.

                                                                                                                                                                Args:
                                                                                                                                                                matrix: Input matrix

                                                                                                                                                                    Returns:
                                                                                                                                                                    Matrix determinant
                                                                                                                                                                    """
                                                                                                                                                                        try:
                                                                                                                                                                            if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                        raise ValueError("Matrix must be square for determinant calculation")

                                                                                                                                                                        det = xp.linalg.det(matrix)

                                                                                                                                                                        self._log_operation("matrix_determinant", det, {"matrix_shape": matrix.shape})
                                                                                                                                                                    return float(det)

                                                                                                                                                                        except Exception as e:
                                                                                                                                                                        logger.error("Error calculating matrix determinant: {0}".format(e))
                                                                                                                                                                    return 0.0

                                                                                                                                                                        def matrix_power(self, matrix: xp.ndarray, power: int) -> xp.ndarray:
                                                                                                                                                                        """
                                                                                                                                                                        Calculate matrix power.

                                                                                                                                                                            Args:
                                                                                                                                                                            matrix: Input matrix
                                                                                                                                                                            power: Power to raise matrix to

                                                                                                                                                                                Returns:
                                                                                                                                                                                Matrix raised to power
                                                                                                                                                                                """
                                                                                                                                                                                    try:
                                                                                                                                                                                        if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                                    raise ValueError("Matrix must be square for power calculation")

                                                                                                                                                                                        if power == 0:
                                                                                                                                                                                        result = xp.eye(matrix.shape[0])
                                                                                                                                                                                            elif power == 1:
                                                                                                                                                                                            result = matrix
                                                                                                                                                                                                elif power > 1:
                                                                                                                                                                                                result = xp.linalg.matrix_power(matrix, power)
                                                                                                                                                                                                    else:
                                                                                                                                                                                                    # Negative power: matrix^(-n) = (matrix^(-1))^n
                                                                                                                                                                                                    result = xp.linalg.matrix_power(xp.linalg.inv(matrix), -power)

                                                                                                                                                                                                    self._log_operation("matrix_power", result, {"matrix_shape": matrix.shape, "power": power})
                                                                                                                                                                                                return result

                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                    logger.error("Error calculating matrix power: {0}".format(e))
                                                                                                                                                                                                return xp.array([])

                                                                                                                                                                                                    def matrix_exponential(self, matrix: xp.ndarray) -> xp.ndarray:
                                                                                                                                                                                                    """
                                                                                                                                                                                                    Calculate matrix exponential.

                                                                                                                                                                                                        Args:
                                                                                                                                                                                                        matrix: Input matrix

                                                                                                                                                                                                            Returns:
                                                                                                                                                                                                            Matrix exponential
                                                                                                                                                                                                            """
                                                                                                                                                                                                                try:
                                                                                                                                                                                                                    if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                                                                raise ValueError("Matrix must be square for exponential calculation")

                                                                                                                                                                                                                result = xp.linalg.expm(matrix)

                                                                                                                                                                                                                self._log_operation("matrix_exponential", result, {"matrix_shape": matrix.shape})
                                                                                                                                                                                                            return result

                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                logger.error("Error calculating matrix exponential: {0}".format(e))
                                                                                                                                                                                                            return xp.array([])

                                                                                                                                                                                                                def matrix_logarithm(self, matrix: xp.ndarray) -> xp.ndarray:
                                                                                                                                                                                                                """
                                                                                                                                                                                                                Calculate matrix logarithm.

                                                                                                                                                                                                                    Args:
                                                                                                                                                                                                                    matrix: Input matrix

                                                                                                                                                                                                                        Returns:
                                                                                                                                                                                                                        Matrix logarithm
                                                                                                                                                                                                                        """
                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                                                                            raise ValueError("Matrix must be square for logarithm calculation")

                                                                                                                                                                                                                            result = xp.linalg.logm(matrix)

                                                                                                                                                                                                                            self._log_operation("matrix_logarithm", result, {"matrix_shape": matrix.shape})
                                                                                                                                                                                                                        return result

                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                            logger.error("Error calculating matrix logarithm: {0}".format(e))
                                                                                                                                                                                                                        return xp.array([])

                                                                                                                                                                                                                            def matrix_sqrt(self, matrix: xp.ndarray) -> xp.ndarray:
                                                                                                                                                                                                                            """
                                                                                                                                                                                                                            Calculate matrix square root.

                                                                                                                                                                                                                                Args:
                                                                                                                                                                                                                                matrix: Input matrix

                                                                                                                                                                                                                                    Returns:
                                                                                                                                                                                                                                    Matrix square root
                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                            if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                                                                                        raise ValueError("Matrix must be square for square root calculation")

                                                                                                                                                                                                                                        result = xp.linalg.sqrtm(matrix)

                                                                                                                                                                                                                                        self._log_operation("matrix_sqrt", result, {"matrix_shape": matrix.shape})
                                                                                                                                                                                                                                    return result

                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                        logger.error("Error calculating matrix square root: {0}".format(e))
                                                                                                                                                                                                                                    return xp.array([])

                                                                                                                                                                                                                                        def matrix_pseudo_inverse(self, matrix: xp.ndarray) -> xp.ndarray:
                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                        Calculate matrix pseudo-inverse (Moore-Penrose inverse).

                                                                                                                                                                                                                                            Args:
                                                                                                                                                                                                                                            matrix: Input matrix

                                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                                Pseudo-inverse matrix
                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                    result = xp.linalg.pinv(matrix)

                                                                                                                                                                                                                                                    self._log_operation(
                                                                                                                                                                                                                                                    "matrix_pseudo_inverse",
                                                                                                                                                                                                                                                    result,
                                                                                                                                                                                                                                                    {"matrix_shape": matrix.shape, "result_shape": result.shape},
                                                                                                                                                                                                                                                    )
                                                                                                                                                                                                                                                return result

                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                    logger.error("Error calculating matrix pseudo-inverse: {0}".format(e))
                                                                                                                                                                                                                                                return xp.array([])

                                                                                                                                                                                                                                                    def matrix_cholesky_decomposition(self, matrix: xp.ndarray) -> xp.ndarray:
                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                    Perform Cholesky decomposition.

                                                                                                                                                                                                                                                        Args:
                                                                                                                                                                                                                                                        matrix: Input matrix (must be positive definite)

                                                                                                                                                                                                                                                            Returns:
                                                                                                                                                                                                                                                            Lower triangular matrix L
                                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                    if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                                                                                                                raise ValueError("Matrix must be square for Cholesky decomposition")

                                                                                                                                                                                                                                                                result = xp.linalg.cholesky(matrix)

                                                                                                                                                                                                                                                                self._log_operation("matrix_cholesky", result, {"matrix_shape": matrix.shape})
                                                                                                                                                                                                                                                            return result

                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                logger.error("Error in Cholesky decomposition: {0}".format(e))
                                                                                                                                                                                                                                                            return xp.array([])

                                                                                                                                                                                                                                                                def matrix_qr_decomposition(self, matrix: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                Perform QR decomposition.

                                                                                                                                                                                                                                                                    Args:
                                                                                                                                                                                                                                                                    matrix: Input matrix

                                                                                                                                                                                                                                                                        Returns:
                                                                                                                                                                                                                                                                        Tuple of (Q, R) matrices
                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                            Q, R = xp.linalg.qr(matrix)

                                                                                                                                                                                                                                                                            self._log_operation("matrix_qr", (Q, R), {"matrix_shape": matrix.shape})
                                                                                                                                                                                                                                                                        return Q, R

                                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                            logger.error("Error in QR decomposition: {0}".format(e))
                                                                                                                                                                                                                                                                        return xp.array([]), xp.array([])

                                                                                                                                                                                                                                                                            def matrix_lu_decomposition(self, matrix: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
                                                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                                            Perform LU decomposition.

                                                                                                                                                                                                                                                                                Args:
                                                                                                                                                                                                                                                                                matrix: Input matrix

                                                                                                                                                                                                                                                                                    Returns:
                                                                                                                                                                                                                                                                                    Tuple of (P, L, U) matrices
                                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                            if matrix.shape[0] != matrix.shape[1]:
                                                                                                                                                                                                                                                                                        raise ValueError("Matrix must be square for LU decomposition")

                                                                                                                                                                                                                                                                                        P, L, U = xp.linalg.lu(matrix)

                                                                                                                                                                                                                                                                                        self._log_operation("matrix_lu", (P, L, U), {"matrix_shape": matrix.shape})
                                                                                                                                                                                                                                                                                    return P, L, U

                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                        logger.error("Error in LU decomposition: {0}".format(e))
                                                                                                                                                                                                                                                                                    return xp.array([]), xp.array([]), xp.array([])

                                                                                                                                                                                                                                                                                        def _log_operation(self, operation: str, result: Any, metadata: Dict[str, Any]) -> None:
                                                                                                                                                                                                                                                                                        """Log a matrix operation for debugging and analysis."""
                                                                                                                                                                                                                                                                                        matrix_result = MatrixResult(result=result, operation=operation, timestamp=time.time(), metadata=metadata)
                                                                                                                                                                                                                                                                                        self.operation_history.append(matrix_result)

                                                                                                                                                                                                                                                                                        # Cache result
                                                                                                                                                                                                                                                                                        cache_key = f"{operation}_{hash(str(metadata))}"
                                                                                                                                                                                                                                                                                        self.matrix_cache[cache_key] = result

                                                                                                                                                                                                                                                                                            def get_operation_history(self) -> List[MatrixResult]:
                                                                                                                                                                                                                                                                                            """Get operation history."""
                                                                                                                                                                                                                                                                                        return self.operation_history.copy()

                                                                                                                                                                                                                                                                                            def clear_cache(self) -> None:
                                                                                                                                                                                                                                                                                            """Clear the matrix cache."""
                                                                                                                                                                                                                                                                                            self.matrix_cache.clear()
                                                                                                                                                                                                                                                                                            logger.info("Matrix cache cleared")


    def tensor_contraction(self, tensor_a, tensor_b, axes=None):
        """C_ij = Σ_k A_ik * B_kj"""
        try:
            a = np.array(tensor_a)
            b = np.array(tensor_b)
            return np.tensordot(a, b, axes=axes)
        except:
            return np.zeros((1, 1))

                                                                                                                                                                                                                                                                                                def get_statistics(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                                                """Get matrix operation statistics."""
                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                        if not self.operation_history:
                                                                                                                                                                                                                                                                                                    return {"error": "No operation history available"}

                                                                                                                                                                                                                                                                                                    # Calculate statistics by operation type
                                                                                                                                                                                                                                                                                                    operation_counts = {}
                                                                                                                                                                                                                                                                                                    operation_times = {}

                                                                                                                                                                                                                                                                                                        for op in self.operation_history:
                                                                                                                                                                                                                                                                                                        op_type = op.operation
                                                                                                                                                                                                                                                                                                        operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

                                                                                                                                                                                                                                                                                                            if op_type not in operation_times:
                                                                                                                                                                                                                                                                                                            operation_times[op_type] = []
                                                                                                                                                                                                                                                                                                            operation_times[op_type].append(op.timestamp)

                                                                                                                                                                                                                                                                                                            # Calculate average times by operation type
                                                                                                                                                                                                                                                                                                            operation_avg_times = {}
                                                                                                                                                                                                                                                                                                                for op_type, times in operation_times.items():
                                                                                                                                                                                                                                                                                                                    if len(times) > 1:
                                                                                                                                                                                                                                                                                                                    intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
                                                                                                                                                                                                                                                                                                                    operation_avg_times[op_type] = xp.mean(intervals)
                                                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                                        operation_avg_times[op_type] = 0.0

                                                                                                                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                                                                                                                    "total_operations": len(self.operation_history),
                                                                                                                                                                                                                                                                                                                    "operation_counts": operation_counts,
                                                                                                                                                                                                                                                                                                                    "operation_avg_times": operation_avg_times,
                                                                                                                                                                                                                                                                                                                    "cache_size": len(self.matrix_cache),
                                                                                                                                                                                                                                                                                                                    "last_operation_time": (self.operation_history[-1].timestamp if self.operation_history else 0),
                                                                                                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                                        logger.error("Error getting statistics: {0}".format(e))
                                                                                                                                                                                                                                                                                                                    return {"error": str(e)}


                                                                                                                                                                                                                                                                                                                        def create_matrix_math_utils() -> MatrixMathUtils:
                                                                                                                                                                                                                                                                                                                        """Factory function to create a matrix math utils instance."""
                                                                                                                                                                                                                                                                                                                    return MatrixMathUtils()


                                                                                                                                                                                                                                                                                                                    # Example usage and testing
                                                                                                                                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                        # Configure logging
                                                                                                                                                                                                                                                                                                                        logging.basicConfig(level=logging.INFO)

                                                                                                                                                                                                                                                                                                                        # Create matrix math utils
                                                                                                                                                                                                                                                                                                                        matrix_utils = create_matrix_math_utils()

                                                                                                                                                                                                                                                                                                                        print("=== Testing Matrix Math Utils ===")

                                                                                                                                                                                                                                                                                                                        # Test matrix operations
                                                                                                                                                                                                                                                                                                                        matrix_a = xp.array([[1, 2], [3, 4]], dtype=float)
                                                                                                                                                                                                                                                                                                                        matrix_b = xp.array([[5, 6], [7, 8]], dtype=float)

                                                                                                                                                                                                                                                                                                                        # Test matrix multiplication
                                                                                                                                                                                                                                                                                                                        result = matrix_utils.matrix_multiply(matrix_a, matrix_b)
                                                                                                                                                                                                                                                                                                                        print("Matrix multiplication result:\n{0}".format(result))

                                                                                                                                                                                                                                                                                                                        # Test matrix inverse
                                                                                                                                                                                                                                                                                                                        inverse = matrix_utils.matrix_inverse(matrix_a)
                                                                                                                                                                                                                                                                                                                        print("Matrix inverse:\n{0}".format(inverse))

                                                                                                                                                                                                                                                                                                                        # Test eigenvalue decomposition
                                                                                                                                                                                                                                                                                                                        eigenvalues, eigenvectors = matrix_utils.eigenvalue_decomposition(matrix_a)
                                                                                                                                                                                                                                                                                                                        print("Eigenvalues: {0}".format(eigenvalues))
                                                                                                                                                                                                                                                                                                                        print("Eigenvectors:\n{0}".format(eigenvectors))

                                                                                                                                                                                                                                                                                                                        # Test matrix norm
                                                                                                                                                                                                                                                                                                                        norm = matrix_utils.matrix_norm(matrix_a, 'frobenius')
                                                                                                                                                                                                                                                                                                                        print("Frobenius norm: {0}".format(norm))

                                                                                                                                                                                                                                                                                                                        # Test matrix determinant
                                                                                                                                                                                                                                                                                                                        det = matrix_utils.matrix_determinant(matrix_a)
                                                                                                                                                                                                                                                                                                                        print("Determinant: {0}".format(det))

                                                                                                                                                                                                                                                                                                                        # Get statistics
                                                                                                                                                                                                                                                                                                                        stats = matrix_utils.get_statistics()
                                                                                                                                                                                                                                                                                                                        print("\nMatrix Statistics:")
                                                                                                                                                                                                                                                                                                                        print("Total operations: {0}".format(stats.get("total_operations", 0)))
                                                                                                                                                                                                                                                                                                                        print("Operation counts: {0}".format(stats.get("operation_counts", {})))

                                                                                                                                                                                                                                                                                                                        print("Matrix Math Utils test completed")
