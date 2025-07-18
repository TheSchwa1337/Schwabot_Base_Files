#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Math Utils - Advanced Matrix Operations
=============================================

Advanced matrix mathematical utilities for Schwabot trading system.
Provides SVD, QR, LU decompositions and other linear algebra operations.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def analyze_price_matrix(price_matrix: np.ndarray) -> Dict[str, Any]:
"""
Analyze a 2-D matrix of prices or returns.

Args:
price_matrix: Input matrix of shape (N, M) where N is number of samples, M is number of assets

Returns:
Dictionary containing comprehensive matrix analysis
"""
try:
if price_matrix.ndim != 2:
raise ValueError("Input must be a 2D matrix")

n_samples, n_assets = price_matrix.shape

# Basic statistics
mean_prices = np.mean(price_matrix, axis=0)
std_prices = np.std(price_matrix, axis=0)
min_prices = np.min(price_matrix, axis=0)
max_prices = np.max(price_matrix, axis=0)

# Calculate returns if we have enough data
returns = None
if n_samples > 1:
returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]

# Correlation matrix
correlation_matrix = np.corrcoef(price_matrix.T)

# Covariance matrix
covariance_matrix = np.cov(price_matrix.T)

# SVD analysis
try:
U, S, Vt = np.linalg.svd(price_matrix, full_matrices=False)
svd_analysis = {
"singular_values": S.tolist(),
"rank": np.sum(S > 1e-10),
"condition_number": S[0] / S[-1] if len(S) > 1 else 1.0,
}
except Exception as e:
logger.warning(f"SVD analysis failed: {e}")
svd_analysis = {"error": str(e)}

# Eigenvalue analysis
try:
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
eigen_analysis = {
"eigenvalues": eigenvalues.tolist(),
"largest_eigenvalue": float(np.max(np.real(eigenvalues))),
"eigenvalue_ratio": float(
np.max(np.real(eigenvalues)) / np.min(np.real(eigenvalues))
)
if np.min(np.real(eigenvalues)) > 0
else 0.0,
}
except Exception as e:
logger.warning(f"Eigenvalue analysis failed: {e}")
eigen_analysis = {"error": str(e)}

# Volatility analysis
volatility_analysis = {}
if returns is not None:
volatility_analysis = {
"mean_volatility": float(np.mean(np.std(returns, axis=0))),
"volatility_range": [
float(np.min(np.std(returns, axis=0))),
float(np.max(np.std(returns, axis=0))),
],
"total_volatility": float(np.sqrt(np.sum(np.var(returns, axis=0)))),
}

return {
"matrix_shape": price_matrix.shape,
"basic_stats": {
"mean_prices": mean_prices.tolist(),
"std_prices": std_prices.tolist(),
"min_prices": min_prices.tolist(),
"max_prices": max_prices.tolist(),
},
"correlation_matrix": correlation_matrix.tolist(),
"covariance_matrix": covariance_matrix.tolist(),
"svd_analysis": svd_analysis,
"eigen_analysis": eigen_analysis,
"volatility_analysis": volatility_analysis,
"matrix_rank": int(np.linalg.matrix_rank(price_matrix)),
"determinant": float(np.linalg.det(correlation_matrix))
if correlation_matrix.shape[0] == correlation_matrix.shape[1]
else None,
"analysis_timestamp": time.time(),
}

except Exception as e:
logger.error(f"Price matrix analysis failed: {e}")
return {
"error": str(e),
"matrix_shape": price_matrix.shape
if hasattr(price_matrix, "shape")
else None,
"analysis_timestamp": time.time(),
}


def risk_parity_weights(
covariance_matrix: np.ndarray, target_volatility: float = 0.1
) -> Dict[str, Any]:
"""
Calculate risk parity weights for portfolio optimization.

Args:
covariance_matrix: Asset covariance matrix
target_volatility: Target portfolio volatility

Returns:
Dictionary containing risk parity weights and analysis
"""
try:
n_assets = covariance_matrix.shape[0]

# Initialize equal weights
weights = np.ones(n_assets) / n_assets

# Risk parity optimization using iterative approach
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
# Calculate current portfolio volatility
portfolio_variance = weights.T @ covariance_matrix @ weights
portfolio_volatility = np.sqrt(portfolio_variance)

# Calculate individual asset contributions to portfolio risk
asset_risk_contributions = (
(covariance_matrix @ weights) * weights / portfolio_volatility
)

# Check if risk contributions are equal (within tolerance)
risk_contribution_std = np.std(asset_risk_contributions)
if risk_contribution_std < tolerance:
break

# Update weights to equalize risk contributions
target_risk_contribution = portfolio_volatility / n_assets
weight_adjustments = target_risk_contribution / (
covariance_matrix @ weights
)
weights = weights * weight_adjustments

# Normalize weights
weights = weights / np.sum(weights)

# Calculate final metrics
final_portfolio_variance = weights.T @ covariance_matrix @ weights
final_portfolio_volatility = np.sqrt(final_portfolio_variance)

# Scale to target volatility if specified
if target_volatility > 0:
scaling_factor = target_volatility / final_portfolio_volatility
weights = weights * scaling_factor
final_portfolio_volatility = target_volatility

# Calculate individual asset risk contributions
asset_risk_contributions = (
(covariance_matrix @ weights) * weights / final_portfolio_volatility
)

return {
"weights": weights.tolist(),
"portfolio_volatility": float(final_portfolio_volatility),
"asset_risk_contributions": asset_risk_contributions.tolist(),
"risk_contribution_std": float(np.std(asset_risk_contributions)),
"convergence_iterations": iteration + 1,
"weights_sum": float(np.sum(weights)),
"target_volatility": target_volatility,
"calculation_timestamp": time.time(),
}

except Exception as e:
logger.error(f"Risk parity calculation failed: {e}")
return {
"error": str(e),
"weights": [1.0 / n_assets] * n_assets if "n_assets" in locals() else None,
"calculation_timestamp": time.time(),
}


class MatrixMathUtils:
"""
Matrix mathematical utilities for trading system analysis.
Provides SVD, QR, LU decompositions and other linear algebra operations.
"""

def __init__(self) -> None:
"""Initialize the matrix math utils."""
self.decomposition_cache = {}
self.operation_history = []
self.logger = logging.getLogger(__name__)

logger.info("Matrix Math Utils initialized")

def svd_decomposition(
self, matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

# Cache the result
matrix_hash = hash(matrix.tobytes())
self.decomposition_cache[f"svd_{matrix_hash}"] = {
"U": U,
"S": S,
"Vt": Vt,
"timestamp": time.time(),
}

self.operation_history.append(
{
"operation": "svd_decomposition",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return U, S, Vt

except Exception as e:
self.logger.error(f"SVD decomposition failed: {e}")
raise

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
Q, R = np.linalg.qr(matrix)

# Cache the result
matrix_hash = hash(matrix.tobytes())
self.decomposition_cache[f"qr_{matrix_hash}"] = {
"Q": Q,
"R": R,
"timestamp": time.time(),
}

self.operation_history.append(
{
"operation": "qr_decomposition",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return Q, R

except Exception as e:
self.logger.error(f"QR decomposition failed: {e}")
raise

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
P, L, U = np.linalg.lu(matrix)

# Cache the result
matrix_hash = hash(matrix.tobytes())
self.decomposition_cache[f"lu_{matrix_hash}"] = {
"P": P,
"L": L,
"U": U,
"timestamp": time.time(),
}

self.operation_history.append(
{
"operation": "lu_decomposition",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return L, U

except Exception as e:
self.logger.error(f"LU decomposition failed: {e}")
raise

def cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
"""
Perform Cholesky Decomposition: A = L * L^T

Args:
matrix: Symmetric positive definite matrix

Returns:
Lower triangular matrix L
"""
try:
L = np.linalg.cholesky(matrix)

# Cache the result
matrix_hash = hash(matrix.tobytes())
self.decomposition_cache[f"cholesky_{matrix_hash}"] = {
"L": L,
"timestamp": time.time(),
}

self.operation_history.append(
{
"operation": "cholesky_decomposition",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return L

except Exception as e:
self.logger.error(f"Cholesky decomposition failed: {e}")
raise

def eigenvalue_decomposition(
self, matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
"""
Perform Eigenvalue Decomposition: A = V * Λ * V^(-1)

Args:
matrix: Square matrix

Returns:
Tuple of (eigenvalues, eigenvectors) where:
- eigenvalues: Array of eigenvalues
- eigenvectors: Matrix of eigenvectors (columns)
"""
try:
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Cache the result
matrix_hash = hash(matrix.tobytes())
self.decomposition_cache[f"eigen_{matrix_hash}"] = {
"eigenvalues": eigenvalues,
"eigenvectors": eigenvectors,
"timestamp": time.time(),
}

self.operation_history.append(
{
"operation": "eigenvalue_decomposition",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return eigenvalues, eigenvectors

except Exception as e:
self.logger.error(f"Eigenvalue decomposition failed: {e}")
raise

def matrix_determinant(self, matrix: np.ndarray) -> float:
"""
Calculate matrix determinant.

Args:
matrix: Square matrix

Returns:
Determinant value
"""
try:
det = np.linalg.det(matrix)

self.operation_history.append(
{
"operation": "matrix_determinant",
"matrix_shape": matrix.shape,
"determinant": det,
"timestamp": time.time(),
}
)

return float(det)

except Exception as e:
self.logger.error(f"Matrix determinant calculation failed: {e}")
raise

def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
"""
Calculate matrix inverse.

Args:
matrix: Square matrix

Returns:
Inverse matrix
"""
try:
inverse = np.linalg.inv(matrix)

self.operation_history.append(
{
"operation": "matrix_inverse",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return inverse

except Exception as e:
self.logger.error(f"Matrix inverse calculation failed: {e}")
raise

def matrix_pseudoinverse(self, matrix: np.ndarray) -> np.ndarray:
"""
Calculate matrix pseudoinverse (Moore-Penrose inverse).

Args:
matrix: Input matrix

Returns:
Pseudoinverse matrix
"""
try:
pseudoinverse = np.linalg.pinv(matrix)

self.operation_history.append(
{
"operation": "matrix_pseudoinverse",
"matrix_shape": matrix.shape,
"timestamp": time.time(),
}
)

return pseudoinverse

except Exception as e:
self.logger.error(f"Matrix pseudoinverse calculation failed: {e}")
raise

def matrix_norm(self, matrix: np.ndarray, norm_type: str = "frobenius") -> float:
"""
Calculate matrix norm.

Args:
matrix: Input matrix
norm_type: Type of norm ('frobenius', 'spectral', 'nuclear')

Returns:
Norm value
"""
try:
if norm_type == "frobenius":
norm = np.linalg.norm(matrix, "fro")
elif norm_type == "spectral":
norm = np.linalg.norm(matrix, 2)
elif norm_type == "nuclear":
S = np.linalg.svd(matrix, compute_uv=False)
norm = np.sum(S)
else:
raise ValueError(f"Unknown norm type: {norm_type}")

self.operation_history.append(
{
"operation": "matrix_norm",
"matrix_shape": matrix.shape,
"norm_type": norm_type,
"norm_value": norm,
"timestamp": time.time(),
}
)

return float(norm)

except Exception as e:
self.logger.error(f"Matrix norm calculation failed: {e}")
raise

def matrix_trace(self, matrix: np.ndarray) -> float:
"""
Calculate matrix trace.

Args:
matrix: Square matrix

Returns:
Trace value
"""
try:
trace = np.trace(matrix)

self.operation_history.append(
{
"operation": "matrix_trace",
"matrix_shape": matrix.shape,
"trace_value": trace,
"timestamp": time.time(),
}
)

return float(trace)

except Exception as e:
self.logger.error(f"Matrix trace calculation failed: {e}")
raise

def matrix_symmetry_check(self, matrix: np.ndarray) -> bool:
"""
Check if matrix is symmetric.

Args:
matrix: Input matrix

Returns:
True if symmetric, False otherwise
"""
try:
is_symmetric = np.allclose(matrix, matrix.T)

self.operation_history.append(
{
"operation": "matrix_symmetry_check",
"matrix_shape": matrix.shape,
"is_symmetric": is_symmetric,
"timestamp": time.time(),
}
)

return bool(is_symmetric)

except Exception as e:
self.logger.error(f"Matrix symmetry check failed: {e}")
raise

def matrix_positive_definite_check(self, matrix: np.ndarray) -> bool:
"""
Check if matrix is positive definite.

Args:
matrix: Symmetric matrix

Returns:
True if positive definite, False otherwise
"""
try:
# Check if symmetric first
if not self.matrix_symmetry_check(matrix):
return False

# Check if all eigenvalues are positive
eigenvalues = np.linalg.eigvals(matrix)
is_positive_definite = np.all(eigenvalues > 0)

self.operation_history.append(
{
"operation": "matrix_positive_definite_check",
"matrix_shape": matrix.shape,
"is_positive_definite": is_positive_definite,
"timestamp": time.time(),
}
)

return bool(is_positive_definite)

except Exception as e:
self.logger.error(f"Matrix positive definite check failed: {e}")
raise

def get_decomposition_summary(self) -> Dict[str, Any]:
"""Get summary of cached decompositions."""
return {
"cache_size": len(self.decomposition_cache),
"operation_count": len(self.operation_history),
"cache_keys": list(self.decomposition_cache.keys()),
}

def clear_cache(self) -> None:
"""Clear the decomposition cache."""
self.decomposition_cache.clear()
self.logger.info("Matrix decomposition cache cleared")

def get_status(self) -> Dict[str, Any]:
"""Get status information."""
return {
"initialized": True,
"cache_size": len(self.decomposition_cache),
"operation_count": len(self.operation_history),
"last_operation": self.operation_history[-1]
if self.operation_history
else None,
}


def create_matrix_math_utils() -> MatrixMathUtils:
"""Create and return a MatrixMathUtils instance."""
return MatrixMathUtils()
