#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor Recursion Solver with XP Backend
=======================================

Advanced tensor recursion solver for complex mathematical operations
with GPU/CPU compatibility via XP backend.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from core.backend_math import get_backend, is_gpu

xp = get_backend()

# Log backend status
logger = logging.getLogger(__name__)
if is_gpu():
    logger.info("⚡ Tensor Recursion Solver using GPU acceleration: CuPy (GPU)")
else:
    logger.info("🔄 Tensor Recursion Solver using CPU fallback: NumPy (CPU)")


@dataclass
class RecursionResult:
    """Result of tensor recursion operation."""

    solution: xp.ndarray
    convergence_iterations: int
    residual_norm: float
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def recursive_tensor_match(a: xp.ndarray, b: xp.ndarray) -> float:
    """
    Compute cosine similarity across recursive vector bands.

    Args:
        a: First tensor/vector
        b: Second tensor/vector

    Returns:
        Cosine similarity score
    """
    try:
        # Ensure vectors are the same length
        min_len = min(len(a), len(b))
        a_trimmed = a[:min_len]
        b_trimmed = b[:min_len]

        # Compute dot product
        dot_product = xp.dot(a_trimmed, b_trimmed)

        # Compute norms
        norm_a = xp.linalg.norm(a_trimmed)
        norm_b = xp.linalg.norm(b_trimmed)

        # Avoid division by zero
        denominator = norm_a * norm_b + 1e-8

        return float(dot_product / denominator)

    except Exception as e:
        logger.error(f"Error in recursive tensor match: {e}")
        return 0.0


def normalize_tensor(tensor: xp.ndarray) -> xp.ndarray:
    """
    Normalize tensor using L2 norm.

    Args:
        tensor: Input tensor

    Returns:
        Normalized tensor
    """
    try:
        norm = xp.linalg.norm(tensor)
        return tensor / (norm + 1e-8)

    except Exception as e:
        logger.error(f"Error normalizing tensor: {e}")
        return tensor


def compute_tensor_resonance(field_a: xp.ndarray, field_b: xp.ndarray) -> float:
    """
    Compute resonance between two tensor fields.

    Args:
        field_a: First tensor field
        field_b: Second tensor field

    Returns:
        Resonance score
    """
    try:
        # Ensure fields have same shape
        if field_a.shape != field_b.shape:
            # Reshape to match if possible
            min_size = min(field_a.size, field_b.size)
            field_a_flat = field_a.flatten()[:min_size]
            field_b_flat = field_b.flatten()[:min_size]
        else:
            field_a_flat = field_a.flatten()
            field_b_flat = field_b.flatten()

        # Compute mean absolute difference
        resonance = float(xp.mean(xp.abs(field_a_flat - field_b_flat)))

        return resonance

    except Exception as e:
        logger.error(f"Error computing tensor resonance: {e}")
        return 0.0


def solve_tensor_recursion(
    initial_tensor: xp.ndarray,
    recursion_function: callable,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> RecursionResult:
    """
    Solve tensor recursion problem.

    Args:
        initial_tensor: Starting tensor
        recursion_function: Function that defines the recursion
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        RecursionResult with solution and metadata
    """
    try:
        start_time = time.time()
        current_tensor = initial_tensor.copy()

        for iteration in range(max_iterations):
            # Apply recursion function
            next_tensor = recursion_function(current_tensor)

            # Check convergence
            residual = xp.linalg.norm(next_tensor - current_tensor)

            if residual < tolerance:
                computation_time = time.time() - start_time

                return RecursionResult(
                    solution=next_tensor,
                    convergence_iterations=iteration + 1,
                    residual_norm=float(residual),
                    computation_time=computation_time,
                    metadata={
                        "converged": True,
                        "tolerance": tolerance,
                        "max_iterations": max_iterations,
                    },
                )

            current_tensor = next_tensor

        # If we reach here, didn't converge
        computation_time = time.time() - start_time

        return RecursionResult(
            solution=current_tensor,
            convergence_iterations=max_iterations,
            residual_norm=float(xp.linalg.norm(current_tensor)),
            computation_time=computation_time,
            metadata={"converged": False, "tolerance": tolerance, "max_iterations": max_iterations},
        )

    except Exception as e:
        logger.error(f"Error in tensor recursion solver: {e}")
        return RecursionResult(
            solution=xp.array([]),
            convergence_iterations=0,
            residual_norm=float('inf'),
            computation_time=0.0,
            metadata={"error": str(e)},
        )


def compute_tensor_eigenvalues(tensor: xp.ndarray) -> xp.ndarray:
    """
    Compute eigenvalues of a tensor using XP backend.

    Args:
        tensor: Input tensor (2D matrix)

    Returns:
        Array of eigenvalues
    """
    try:
        # Ensure tensor is 2D
        if tensor.ndim > 2:
            # Flatten to 2D if needed
            tensor_2d = tensor.reshape(tensor.shape[0], -1)
        else:
            tensor_2d = tensor

        # Compute eigenvalues
        eigenvalues = xp.linalg.eigvals(tensor_2d)

        return eigenvalues

    except Exception as e:
        logger.error(f"Error computing tensor eigenvalues: {e}")
        return xp.array([])


def tensor_svd_decomposition(tensor: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Perform SVD decomposition of a tensor.

    Args:
        tensor: Input tensor (2D matrix)

    Returns:
        Tuple of (U, S, V) matrices
    """
    try:
        # Ensure tensor is 2D
        if tensor.ndim > 2:
            tensor_2d = tensor.reshape(tensor.shape[0], -1)
        else:
            tensor_2d = tensor

        # Perform SVD
        U, S, V = xp.linalg.svd(tensor_2d)

        return U, S, V

    except Exception as e:
        logger.error(f"Error in tensor SVD decomposition: {e}")
        return xp.array([]), xp.array([]), xp.array([])


def export_tensor_safely(tensor: xp.ndarray) -> xp.ndarray:
    """
    Safely export tensor for plotting or external use.

    Args:
        tensor: Input tensor (CuPy or NumPy)

    Returns:
        NumPy array (safe for external libraries)
    """
    return tensor.get() if hasattr(tensor, 'get') else tensor


# Example usage functions
def test_tensor_recursion():
    """Test the tensor recursion solver."""
    # Create test data
    initial_tensor = xp.random.rand(10, 10)

    # Define a simple recursion function
    def simple_recursion(tensor):
        return 0.5 * (tensor + xp.eye(tensor.shape[0]))

    # Solve recursion
    result = solve_tensor_recursion(initial_tensor, simple_recursion)

    logger.info("Tensor recursion test completed:")
    logger.info(f"Converged: {result.metadata.get('converged', False)}")
    logger.info(f"Iterations: {result.convergence_iterations}")
    logger.info(f"Residual norm: {result.residual_norm:.6f}")
    logger.info(f"Computation time: {result.computation_time:.4f}s")

    return result


if __name__ == "__main__":
    # Run test
    test_result = test_tensor_recursion()
    print("Tensor recursion solver test completed successfully!")
