import scipy as sp

# -*- coding: utf-8 -*-
"""
Unified Tensor Algebra
======================

Advanced tensor algebra operations for mathematical trading calculations.
Provides high-performance tensor operations, matrix manipulations, and
mathematical foundations for the Schwabot trading system.

Mathematical Foundation:
    - Tensor Operations: T(A,B) = A ⊗ B (tensor product)
    - Matrix Decomposition: A = UsumVᵀ (SVD)
    - Eigenvalue Analysis: Av = lambdav
    - Tensor Contraction: C = sumᵢⱼ Aᵢⱼ Bᵢⱼ
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class UnifiedTensorAlgebra:
    """
    Unified tensor algebra system for advanced mathematical operations.

    Provides comprehensive tensor operations, matrix algebra, and
    mathematical utilities for trading system calculations.
    """

    def __init__(self, precision: str = "float64"):
        """Initialize the tensor algebra system."""
        self.precision = precision
        self.default_dtype = getattr(np, precision)

        # Performance metrics
        self.operation_count = 0
        self.total_computation_time = 0.0

        logger.info(f"Unified Tensor Algebra initialized with {precision} precision")

    def tensor_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute tensor product of two tensors.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            Tensor product result
        """
        try:
            self.operation_count += 1
            result = np.tensordot(a, b, axes=0)
            return result.astype(self.default_dtype)
        except Exception as e:
            logger.error(f"Tensor product failed: {e}")
            raise

    def matrix_decomposition(self, matrix: np.ndarray, method: str = "svd") -> Dict[str, np.ndarray]:
        """
        Perform matrix decomposition using specified method.

        Args:
            matrix: Input matrix
            method: Decomposition method ("svd", "qr", "lu", "cholesky")

        Returns:
            Dictionary containing decomposition components
        """
        try:
            self.operation_count += 1

            if method == "svd":
                U, s, Vt = np.linalg.svd(matrix)
                return {"U": U, "s": s, "Vt": Vt}

            elif method == "qr":
                Q, R = np.linalg.qr(matrix)
                return {"Q": Q, "R": R}

            elif method == "lu":
                # Simple LU decomposition using scipy if available
                try:
                    from scipy.linalg import lu

                    P, L, U = lu(matrix)
                    return {"P": P, "L": L, "U": U}
                except ImportError:
                    # Fallback to SVD
                    logger.warning("SciPy not available, falling back to SVD")
                    return self.matrix_decomposition(matrix, "svd")

            elif method == "cholesky":
                L = np.linalg.cholesky(matrix)
                return {"L": L}

            else:
                raise ValueError(f"Unknown decomposition method: {method}")

        except Exception as e:
            logger.error(f"Matrix decomposition failed: {e}")
            raise

    def eigenvalue_analysis(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform eigenvalue and eigenvector analysis.

        Args:
            matrix: Input square matrix

        Returns:
            Dictionary containing eigenvalues and eigenvectors
        """
        try:
            self.operation_count += 1
            eigenvalues, eigenvectors = np.linalg.eig(matrix)

            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            return {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "condition_number": np.abs(eigenvalues[0] / eigenvalues[-1]) if len(eigenvalues) > 1 else 1.0,
            }

        except Exception as e:
            logger.error(f"Eigenvalue analysis failed: {e}")
            raise

    def tensor_contraction(
        self, tensor_a: np.ndarray, tensor_b: np.ndarray, axes: Union[int, Tuple[int, int]]
    ) -> np.ndarray:
        """
        Perform tensor contraction operation.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            axes: Axes to contract over

        Returns:
            Contracted tensor result
        """
        try:
            self.operation_count += 1
            result = np.tensordot(tensor_a, tensor_b, axes=axes)
            return result.astype(self.default_dtype)
        except Exception as e:
            logger.error(f"Tensor contraction failed: {e}")
            raise

    def matrix_inverse(self, matrix: np.ndarray, method: str = "direct") -> np.ndarray:
        """
        Compute matrix inverse using specified method.

        Args:
            matrix: Input square matrix
            method: Inversion method ("direct", "pseudo", "svd")

        Returns:
            Inverted matrix
        """
        try:
            self.operation_count += 1

            if method == "direct":
                return np.linalg.inv(matrix)

            elif method == "pseudo":
                return np.linalg.pinv(matrix)

            elif method == "svd":
                # SVD-based pseudo-inverse for numerical stability
                U, s, Vt = np.linalg.svd(matrix)
                # Use tolerance for singular values
                tolerance = np.finfo(s.dtype).eps * max(matrix.shape) * s[0]
                s_inv = np.where(s > tolerance, 1.0 / s, 0.0)
                return Vt.T @ np.diag(s_inv) @ U.T

            else:
                raise ValueError(f"Unknown inversion method: {method}")

        except Exception as e:
            logger.error(f"Matrix inversion failed: {e}")
            raise

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve linear system Ax = b.

        Args:
            A: Coefficient matrix
            b: Right-hand side vector/matrix

        Returns:
            Solution vector/matrix
        """
        try:
            self.operation_count += 1
            return np.linalg.solve(A, b)
        except Exception as e:
            logger.error(f"Linear system solve failed: {e}")
            # Fallback to least squares
            logger.warning("Falling back to least squares solution")
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def compute_norm(self, tensor: np.ndarray, norm_type: Union[str, int, float] = "fro") -> float:
        """
        Compute various norms of a tensor.

        Args:
            tensor: Input tensor
            norm_type: Type of norm ("fro", "nuc", 1, 2, -1, -2, np.inf, -np.inf)

        Returns:
            Norm value
        """
        try:
            self.operation_count += 1
            return float(np.linalg.norm(tensor, ord=norm_type))
        except Exception as e:
            logger.error(f"Norm computation failed: {e}")
            raise

    def tensor_reshape(self, tensor: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reshape tensor to new dimensions.

        Args:
            tensor: Input tensor
            new_shape: Target shape

        Returns:
            Reshaped tensor
        """
        try:
            self.operation_count += 1
            return tensor.reshape(new_shape)
        except Exception as e:
            logger.error(f"Tensor reshape failed: {e}")
            raise

    def kronecker_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Kronecker product of two matrices.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Kronecker product result
        """
        try:
            self.operation_count += 1
            return np.kron(a, b).astype(self.default_dtype)
        except Exception as e:
            logger.error(f"Kronecker product failed: {e}")
            raise

    def trace_operation(self, matrix: np.ndarray, offset: int = 0) -> float:
        """
        Compute trace of a matrix.

        Args:
            matrix: Input matrix
            offset: Diagonal offset

        Returns:
            Trace value
        """
        try:
            self.operation_count += 1
            return float(np.trace(matrix, offset=offset))
        except Exception as e:
            logger.error(f"Trace operation failed: {e}")
            raise

    def determinant(self, matrix: np.ndarray) -> float:
        """
        Compute determinant of a square matrix.

        Args:
            matrix: Input square matrix

        Returns:
            Determinant value
        """
        try:
            self.operation_count += 1
            return float(np.linalg.det(matrix))
        except Exception as e:
            logger.error(f"Determinant computation failed: {e}")
            raise

    def matrix_rank(self, matrix: np.ndarray, tolerance: Optional[float] = None) -> int:
        """
        Compute rank of a matrix.

        Args:
            matrix: Input matrix
            tolerance: Tolerance for rank computation

        Returns:
            Matrix rank
        """
        try:
            self.operation_count += 1
            return int(np.linalg.matrix_rank(matrix, tol=tolerance))
        except Exception as e:
            logger.error(f"Matrix rank computation failed: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get tensor algebra performance metrics."""
        return {
            "operation_count": self.operation_count,
            "total_computation_time": self.total_computation_time,
            "average_operation_time": (self.total_computation_time / max(self.operation_count, 1)),
            "precision": self.precision,
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.operation_count = 0
        self.total_computation_time = 0.0


# Example usage
if __name__ == "__main__":
    print("Unified Tensor Algebra Demonstration")
    print("=" * 40)

    # Initialize system
    tensor_algebra = UnifiedTensorAlgebra()

    # Test matrix operations
    A = np.random.rand(4, 4).astype(np.float64)
    B = np.random.rand(4, 4).astype(np.float64)

    # Test decomposition
    decomp = tensor_algebra.matrix_decomposition(A, "svd")
    print(f"SVD decomposition shapes: U={decomp['U'].shape}, s={decomp['s'].shape}, Vt={decomp['Vt'].shape}")

    # Test eigenvalue analysis
    eigen_result = tensor_algebra.eigenvalue_analysis(A)
    print(f"Eigenvalues: {eigen_result['eigenvalues'][:3]}")  # Show first 3
    print(f"Condition number: {eigen_result['condition_number']:.2f}")

    # Test tensor operations
    tensor_prod = tensor_algebra.tensor_product(A, B)
    print(f"Tensor product shape: {tensor_prod.shape}")

    # Test matrix inverse
    A_inv = tensor_algebra.matrix_inverse(A, "svd")
    identity_check = A @ A_inv
    identity_error = np.linalg.norm(identity_check - np.eye(4))
    print(f"Inverse accuracy (should be ~0): {identity_error:.6f}")

    # Test norms
    frobenius_norm = tensor_algebra.compute_norm(A, "fro")
    print(f"Frobenius norm: {frobenius_norm:.3f}")

    # Show performance metrics
    metrics = tensor_algebra.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Operations performed: {metrics['operation_count']}")
    print(f"  Precision: {metrics['precision']}")
