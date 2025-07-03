# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, Tuple

import numpy as np
from utils.safe_print import safe_print

from core.unified_math_system import unified_math
# from dual_unicore_handler import DualUnicoreHandler

# Initialize Unicode handler
# unicore = DualUnicoreHandler()
logger = logging.getLogger(__name__)


class MatrixFaultResolver:
    def __init__(self, max_condition_number=1000, epsilon=1e-12):
        self.max_condition_number = max_condition_number
        self.epsilon = epsilon

    def check_matrix_validity(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Checks the validity of a matrix for common numerical operations."""
        try:
            if matrix.size == 0:
                return {
                    "valid": False,
                    "error": "Empty matrix",
                    "fixes": ["provide_data"],
                }

            if not np.isfinite(matrix).all():
                return {
                    "valid": False,
                    "error": "Non-finite values",
                    "fixes": ["remove_nan", "interpolate"],
                }

            if matrix.ndim != 2:
                return {
                    "valid": False,
                    "error": "Not a 2D matrix",
                    "fixes": ["reshape", "flatten"],
                }

            # Check condition number for square matrices
            if matrix.shape[0] == matrix.shape[1]:
                try:
                    cond_num = np.linalg.cond(matrix)
                    if cond_num > self.max_condition_number:
                        return {
                            "valid": False,
                            "error": "Ill-conditioned matrix",
                            "fixes": ["regularize", "svd"],
                        }
                except np.linalg.LinAlgError:
                    return {
                        "valid": False,
                        "error": "Singular matrix, cannot compute condition number",
                        "fixes": ["regularize", "svd"],
                    }

            return {"valid": True, "error": None, "fixes": []}

        except Exception as e:
            logger.error(f"Matrix validity check failed: {e}")
            return {"valid": False, "error": str(e), "fixes": ["fallback"]}

    def resolve_singular_matrix(
        self, matrix: np.ndarray, regularization: float = 1e-6
    ) -> np.ndarray:
        """Resolves a singular matrix by adding a small regularization term."""
        try:
            return matrix + np.eye(matrix.shape[0]) * regularization
        except Exception as e:
            logger.error(f"Singular matrix resolution failed: {e}")
            return (
                np.eye(matrix.shape[0])
                if matrix.shape[0] == matrix.shape[1]
                else np.zeros_like(matrix)
            )

    def resolve_nan_values(
        self, matrix: np.ndarray, method: str = "zero"
    ) -> np.ndarray:
        """Resolves NaN values in a matrix."""
        try:
            if method == "zero":
                return np.nan_to_num(matrix, nan=0.0)
            elif method == "mean":
                col_mean = np.nanmean(matrix, axis=0)
                inds = np.where(np.isnan(matrix))
                matrix[inds] = np.take(col_mean, inds[1])
                return matrix
            else:  # fallback to zero
                return np.nan_to_num(matrix, nan=0.0)
        except Exception as e:
            logger.error(f"NaN resolution failed: {e}")
            return np.zeros_like(matrix)

    def resolve_matrix_multiplication_fault(
        self, A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resolves faults in matrix multiplication."""
        resolution_info = {}
        try:
            if A.shape[1] != B.shape[0]:
                # Try transposing B
                if A.shape[1] == B.shape[1]:
                    B = B.T
                    resolution_info = {"method": "transpose_B", "success": True}
                # Try transposing A
                elif A.shape[0] == B.shape[0]:
                    A = A.T
                    resolution_info = {"method": "transpose_A", "success": True}
                else:
                    # Use broadcasting or fallback
                    min_dim = unified_math.min(A.shape[1], B.shape[0])
                    A_truncated = A[:, :min_dim]
                    B_truncated = B[:min_dim, :]
                    result = unified_math.dot_product(A_truncated, B_truncated)
                    resolution_info = {
                        "method": "dimension_truncation",
                        "success": True,
                    }
                    return result, resolution_info

            # Perform multiplication
            result = unified_math.dot_product(A, B)
            resolution_info.setdefault("method", "normal")
            resolution_info.setdefault("success", True)
            return result, resolution_info

        except Exception as e:
            logger.error(f"Matrix multiplication fault resolution failed: {e}")
            # Return fallback result
            fallback_shape = (
                (A.shape[0], B.shape[1]) if A.ndim == 2 and B.ndim == 2 else (1, 1)
            )
            fallback_result = np.zeros(fallback_shape)
            resolution_info = {"method": "fallback", "success": False, "error": str(e)}
            return fallback_result, resolution_info

    def resolve_eigenvalue_fault(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Resolves faults in eigenvalue computation."""
        resolution_info = {}
        try:
            # Check for symmetry - use specialized solver if symmetric
            if np.allclose(matrix, matrix.T, rtol=1e-10):
                eigenvals, eigenvecs = np.linalg.eigh(matrix)
                resolution_info = {"method": "symmetric", "success": True}
                return eigenvals, eigenvecs, resolution_info

            # General eigenvalue computation
            eigenvals, eigenvecs = unified_math.eigenvectors(matrix)
            resolution_info = {"method": "general", "success": True}
            return eigenvals, eigenvecs, resolution_info
        except np.linalg.LinAlgError:
            try:
                U, s, Vh = unified_math.svd(matrix)
                resolution_info = {"method": "svd_fallback", "success": True}
                return s, U, resolution_info
            except Exception as e_svd:
                logger.error(f"SVD fallback for eigenvalue computation failed: {e_svd}")
                n = matrix.shape[0]
                return (
                    np.zeros(n),
                    np.eye(n),
                    {"method": "fallback", "success": False, "error": str(e_svd)},
                )
        except Exception as e:
            logger.error(f"Eigenvalue computation fault resolution failed: {e}")
            # Return fallback
            n = matrix.shape[0]
            fallback_vals = np.zeros(n)
            fallback_vecs = np.eye(n)
            resolution_info = {"method": "fallback", "success": False, "error": str(e)}
            return fallback_vals, fallback_vecs, resolution_info

    def resolve_inversion_fault(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resolves faults in matrix inversion."""
        resolution_info = {}
        try:
            # Check determinant
            det = unified_math.determinant(matrix)
            if unified_math.abs(det) < self.epsilon:
                # Matrix is singular, use pseudo-inverse
                pseudo_inv = np.linalg.pinv(matrix)
                resolution_info = {"method": "pseudo_inverse_singular", "success": True}
                return pseudo_inv, resolution_info

            # Normal inversion
            inv_matrix = unified_math.inverse(matrix)
            resolution_info = {"method": "normal", "success": True}
            return inv_matrix, resolution_info
        except np.linalg.LinAlgError:
            try:
                pseudo_inv = np.linalg.pinv(matrix)
                resolution_info = {"method": "pseudo_inverse", "success": True}
                return pseudo_inv, resolution_info
            except Exception as e_pinv:
                logger.error(f"Pseudo-inverse fallback for inversion failed: {e_pinv}")
                n = matrix.shape[0] if matrix.ndim == 2 else 1
                return np.eye(n), {
                    "method": "identity_fallback",
                    "success": False,
                    "error": str(e_pinv),
                }
        except Exception as e:
            logger.error(f"Matrix inversion fault resolution failed: {e}")
            # Return identity as fallback
            n = matrix.shape[0] if matrix.ndim == 2 else 1
            fallback = np.eye(n)
            resolution_info = {
                "method": "identity_fallback",
                "success": False,
                "error": str(e),
            }
            return fallback, resolution_info


# Global instance for easy import
matrix_resolver = MatrixFaultResolver()


# Convenience functions for main pipeline
def check_matrix_validity(matrix: np.ndarray) -> Dict[str, Any]:
    """Convenience function to check matrix validity."""
    return matrix_resolver.check_matrix_validity(matrix)


def resolve_singular_matrix(
    matrix: np.ndarray, regularization: float = 1e-6
) -> np.ndarray:
    """Convenience function to resolve a singular matrix."""
    return matrix_resolver.resolve_singular_matrix(matrix, regularization)


def resolve_nan_values(matrix: np.ndarray, method: str = "zero") -> np.ndarray:
    """Convenience function to resolve NaN values in a matrix."""
    return matrix_resolver.resolve_nan_values(matrix, method)


def main():
    """Main function for testing matrix fault resolver."""
    safe_print("Matrix Fault Resolver - Mathematical Error Recovery System")

    # Test singular matrix resolution
    singular_matrix = np.array([[1, 1], [1, 1]], dtype=np.float64)
    resolved = resolve_singular_matrix(singular_matrix)
    safe_print(f"Singular matrix resolved: shape {resolved.shape}")

    # Test NaN resolution
    nan_matrix = np.array([[1.0, np.nan], [2.0, 3.0]])
    resolved_nan = resolve_nan_values(nan_matrix, method="mean")
    safe_print(f"NaN values resolved: {resolved_nan}")

    safe_print("Matrix Fault Resolver test completed successfully")


if __name__ == "__main__":
    main()
