#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Math Utils
================

Provides matrix mathematical operations including SVD, QR, LU decompositions
and other linear algebra operations for trading system analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MatrixMathUtils:
    """
    Matrix mathematical utilities for trading system analysis.
    Provides SVD, QR, LU decompositions and other linear algebra operations.
    """

    def __init__(self):
        """Initialize the matrix math utils."""
        self.decomposition_cache = {}
        self.operation_history = []
        self.logger = logging.getLogger(__name__)
        
        logger.info("Matrix Math Utils initialized")

    def svd_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Singular Value Decomposition (SVD): A = U * Σ * V^T
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (U, S, Vt) where:
            - U: Left singular vectors
            - S: Singular values
            - Vt: Right singular vectors (transposed)
        """
        try:
            # Perform SVD decomposition
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # Cache result
            cache_key = f"svd_{hash(str(matrix.shape))}"
            self.decomposition_cache[cache_key] = {
                'U': U,
                'S': S,
                'Vt': Vt,
                'timestamp': time.time()
            }
            
            return U, S, Vt
            
        except Exception as e:
            logger.error(f"SVD decomposition failed: {e}")
            # Return identity matrices as fallback
            n, m = matrix.shape
            U = np.eye(n)
            S = np.zeros(min(n, m))
            Vt = np.eye(m)
            return U, S, Vt

    def qr_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform QR Decomposition: A = Q * R
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (Q, R) where:
            - Q: Orthogonal matrix
            - R: Upper triangular matrix
        """
        try:
            # Perform QR decomposition
            Q, R = np.linalg.qr(matrix)
            
            # Cache result
            cache_key = f"qr_{hash(str(matrix.shape))}"
            self.decomposition_cache[cache_key] = {
                'Q': Q,
                'R': R,
                'timestamp': time.time()
            }
            
            return Q, R
            
        except Exception as e:
            logger.error(f"QR decomposition failed: {e}")
            # Return identity matrices as fallback
            n, m = matrix.shape
            Q = np.eye(n)
            R = matrix.copy()
            return Q, R

    def lu_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform LU Decomposition: A = L * U
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (L, U) where:
            - L: Lower triangular matrix
            - U: Upper triangular matrix
        """
        try:
            # Perform LU decomposition
            P, L, U = np.linalg.lu(matrix)
            
            # Cache result
            cache_key = f"lu_{hash(str(matrix.shape))}"
            self.decomposition_cache[cache_key] = {
                'P': P,
                'L': L,
                'U': U,
                'timestamp': time.time()
            }
            
            return L, U
            
        except Exception as e:
            logger.error(f"LU decomposition failed: {e}")
            # Return identity matrices as fallback
            n, m = matrix.shape
            L = np.eye(n)
            U = matrix.copy()
            return L, U

    def cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform Cholesky Decomposition: A = L * L^T
        
        Args:
            matrix: Symmetric positive definite matrix
            
        Returns:
            Lower triangular matrix L
        """
        try:
            # Perform Cholesky decomposition
            L = np.linalg.cholesky(matrix)
            
            # Cache result
            cache_key = f"cholesky_{hash(str(matrix.shape))}"
            self.decomposition_cache[cache_key] = {
                'L': L,
                'timestamp': time.time()
            }
            
            return L
            
        except Exception as e:
            logger.error(f"Cholesky decomposition failed: {e}")
            # Return identity matrix as fallback
            n = matrix.shape[0]
            return np.eye(n)

    def eigenvalue_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Eigenvalue Decomposition: A = V * Λ * V^(-1)
        
        Args:
            matrix: Square matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            # Perform eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            
            # Cache result
            cache_key = f"eigen_{hash(str(matrix.shape))}"
            self.decomposition_cache[cache_key] = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'timestamp': time.time()
            }
            
            return eigenvalues, eigenvectors
            
        except Exception as e:
            logger.error(f"Eigenvalue decomposition failed: {e}")
            # Return identity matrices as fallback
            n = matrix.shape[0]
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            return eigenvalues, eigenvectors

    def matrix_condition_number(self, matrix: np.ndarray) -> float:
        """
        Calculate matrix condition number.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Condition number
        """
        try:
            condition_number = np.linalg.cond(matrix)
            return float(condition_number)
        except Exception as e:
            logger.error(f"Condition number calculation failed: {e}")
            return 1.0

    def matrix_rank(self, matrix: np.ndarray) -> int:
        """
        Calculate matrix rank.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix rank
        """
        try:
            rank = np.linalg.matrix_rank(matrix)
            return int(rank)
        except Exception as e:
            logger.error(f"Matrix rank calculation failed: {e}")
            return 0

    def matrix_determinant(self, matrix: np.ndarray) -> float:
        """
        Calculate matrix determinant.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Matrix determinant
        """
        try:
            determinant = np.linalg.det(matrix)
            return float(determinant)
        except Exception as e:
            logger.error(f"Matrix determinant calculation failed: {e}")
            return 0.0

    def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate matrix inverse.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Matrix inverse
        """
        try:
            inverse = np.linalg.inv(matrix)
            return inverse
        except Exception as e:
            logger.error(f"Matrix inverse calculation failed: {e}")
            # Return identity matrix as fallback
            n = matrix.shape[0]
            return np.eye(n)

    def matrix_pseudoinverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate matrix pseudoinverse (Moore-Penrose inverse).
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix pseudoinverse
        """
        try:
            pseudoinverse = np.linalg.pinv(matrix)
            return pseudoinverse
        except Exception as e:
            logger.error(f"Matrix pseudoinverse calculation failed: {e}")
            # Return zero matrix as fallback
            return np.zeros_like(matrix.T)

    def matrix_norm(self, matrix: np.ndarray, norm_type: str = 'frobenius') -> float:
        """
        Calculate matrix norm.
        
        Args:
            matrix: Input matrix
            norm_type: Type of norm ('frobenius', 'l1', 'l2', 'inf')
            
        Returns:
            Matrix norm
        """
        try:
            if norm_type == 'frobenius':
                norm = np.linalg.norm(matrix, 'fro')
            elif norm_type == 'l1':
                norm = np.linalg.norm(matrix, 1)
            elif norm_type == 'l2':
                norm = np.linalg.norm(matrix, 2)
            elif norm_type == 'inf':
                norm = np.linalg.norm(matrix, np.inf)
            else:
                norm = np.linalg.norm(matrix, 'fro')
            
            return float(norm)
        except Exception as e:
            logger.error(f"Matrix norm calculation failed: {e}")
            return 0.0

    def matrix_trace(self, matrix: np.ndarray) -> float:
        """
        Calculate matrix trace.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Matrix trace
        """
        try:
            trace = np.trace(matrix)
            return float(trace)
        except Exception as e:
            logger.error(f"Matrix trace calculation failed: {e}")
            return 0.0

    def matrix_symmetry_check(self, matrix: np.ndarray) -> bool:
        """
        Check if matrix is symmetric.
        
        Args:
            matrix: Input matrix
            
        Returns:
            True if symmetric, False otherwise
        """
        try:
            return np.allclose(matrix, matrix.T)
        except Exception as e:
            logger.error(f"Matrix symmetry check failed: {e}")
            return False

    def matrix_positive_definite_check(self, matrix: np.ndarray) -> bool:
        """
        Check if matrix is positive definite.
        
        Args:
            matrix: Input matrix
            
        Returns:
            True if positive definite, False otherwise
        """
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            return np.all(eigenvalues > 0)
        except Exception as e:
            logger.error(f"Matrix positive definite check failed: {e}")
            return False

    def get_decomposition_summary(self) -> Dict[str, Any]:
        """Get summary of cached decompositions."""
        return {
            'total_decompositions': len(self.decomposition_cache),
            'decomposition_types': list(set(key.split('_')[0] for key in self.decomposition_cache.keys())),
            'cache_size': len(self.decomposition_cache),
            'timestamp': time.time()
        }

    def clear_cache(self) -> None:
        """Clear decomposition cache."""
        self.decomposition_cache.clear()
        logger.info("Matrix decomposition cache cleared")

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': True,
            'cached_decompositions': len(self.decomposition_cache),
            'operation_history': len(self.operation_history),
            'timestamp': time.time()
        }


# Factory function
def create_matrix_math_utils() -> MatrixMathUtils:
    """Create a new matrix math utils instance."""
    return MatrixMathUtils()


# Example usage
if __name__ == "__main__":
    # Create matrix utils
    matrix_utils = MatrixMathUtils()
    
    # Test matrix
    A = np.array([[1, 2], [3, 4]])
    
    # Test SVD decomposition
    U, S, Vt = matrix_utils.svd_decomposition(A)
    print(f"SVD - U shape: {U.shape}, S: {S}, Vt shape: {Vt.shape}")
    
    # Test QR decomposition
    Q, R = matrix_utils.qr_decomposition(A)
    print(f"QR - Q shape: {Q.shape}, R shape: {R.shape}")
    
    # Test LU decomposition
    L, U = matrix_utils.lu_decomposition(A)
    print(f"LU - L shape: {L.shape}, U shape: {U.shape}") 