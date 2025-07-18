#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Tensor Algebra - Tensor Field Unification & Recursive Math Collapse Engine
=================================================================================

Implements Nexus mathematics for full rank-2 and rank-3 tensor unification:
- Canonical Collapse Tensor: T_collapse = Σᵢ₌₀ⁿ Aᵢ⋅∇²(φᵢ) + γ⋅Δ_shift
- Fourier-Tensor Dual Transform: F_tensor = ∫_Rⁿ T(x)e^(-2πiξ⋅x)dx
- Governs matrix field recursion + math routing, core to strategy hash-building
- The fallback tensor for galileo_tensor_field cycles into this
- Confirmed use in strategy_bit_mapper.py to perform hash comparison + cosine similarity on SHA-derived pattern matrices
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import linalg, optimize, signal
from scipy.spatial.distance import cosine
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn, fft2


logger = logging.getLogger(__name__)


class TensorRank(Enum):
    """Tensor rank definitions."""

    RANK_0 = "rank_0"  # Scalar
    RANK_1 = "rank_1"  # Vector
    RANK_2 = "rank_2"  # Matrix
    RANK_3 = "rank_3"  # 3D tensor
    RANK_N = "rank_n"  # N-dimensional tensor


class TensorOperation(Enum):
    """Tensor operation types."""

    CONTRACTION = "contraction"  # Tensor contraction
    PRODUCT = "product"  # Tensor product
    DECOMPOSITION = "decomposition"  # Eigenvalue decomposition
    TRANSFORM = "transform"  # Fourier transform
    COLLAPSE = "collapse"  # Tensor collapse
    NORM = "norm"  # Tensor norm


@dataclass
class TensorResult:
    """Result of tensor operation."""

    timestamp: float
    operation: TensorOperation
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    result_data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollapseResult:
    """Result of tensor collapse operation."""

    timestamp: float
    collapse_tensor: np.ndarray
    fourier_transform: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    norm: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedTensorAlgebra:
    """
    Unified Tensor Algebra - Tensor Field Unification & Recursive Math Collapse Engine

    Implements the Nexus mathematics for full rank-2 and rank-3 tensor unification:
    - Canonical Collapse Tensor: T_collapse = Σᵢ₌₀ⁿ Aᵢ⋅∇²(φᵢ) + γ⋅Δ_shift
    - Fourier-Tensor Dual Transform: F_tensor = ∫_Rⁿ T(x)e^(-2πiξ⋅x)dx
    - Governs matrix field recursion + math routing, core to strategy hash-building
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Unified Tensor Algebra."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.initialized = False

        # Tensor parameters
        self.max_rank = self.config.get("max_rank", 3)
        self.collapse_threshold = self.config.get("collapse_threshold", 0.1)
        self.fourier_resolution = self.config.get("fourier_resolution", 64)
        self.gamma_shift = self.config.get("gamma_shift", 0.1)

        # Operation tracking
        self.tensor_results: List[TensorResult] = []
        self.collapse_results: List[CollapseResult] = []

        self._initialize_algebra()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Unified Tensor Algebra."""
        return {
            "max_rank": 3,  # Maximum tensor rank
            "collapse_threshold": 0.1,  # Collapse detection threshold
            "fourier_resolution": 64,  # Fourier transform resolution
            "gamma_shift": 0.1,  # Gamma shift parameter
            "eigenvalue_threshold": 1e-6,  # Eigenvalue threshold
            "norm_threshold": 1e-8,  # Norm threshold
            "contraction_axes": None,  # Default contraction axes
        }

    def _initialize_algebra(self) -> None:
        """Initialize the tensor algebra system."""
        try:
            self.logger.info("Initializing Unified Tensor Algebra...")

            # Validate parameters
            if not (1 <= self.max_rank <= 5):
                raise ValueError("max_rank must be between 1 and 5")
            if not (0.0 <= self.collapse_threshold <= 1.0):
                raise ValueError("collapse_threshold must be between 0.0 and 1.0")

            # Initialize tensor operations
            self.operation_count = 0
            self.total_operations = 0

            self.initialized = True
            self.logger.info(
                "[SUCCESS] Unified Tensor Algebra initialized successfully"
            )

        except Exception as e:
            self.logger.error(f"[FAIL] Error initializing Unified Tensor Algebra: {e}")
            self.initialized = False

    def compute_canonical_collapse_tensor(
        self,
        A_components: List[np.ndarray],
        phi_components: List[np.ndarray],
        delta_shift: float = None,
    ) -> np.ndarray:
        """
        Compute canonical collapse tensor: T_collapse = Σᵢ₌₀ⁿ Aᵢ⋅∇²(φᵢ) + γ⋅Δ_shift

        Args:
        A_components: List of A coefficient matrices
        phi_components: List of φ component tensors
        delta_shift: Δ_shift parameter (optional)

        Returns:
        Canonical collapse tensor
        """
        try:
            if not A_components or not phi_components:
                raise ValueError("A_components and phi_components must not be empty")

            if len(A_components) != len(phi_components):
                raise ValueError(
                    "A_components and phi_components must have the same length"
                )

            # Initialize collapse tensor
            first_phi = phi_components[0]
            collapse_tensor = np.zeros_like(first_phi)

            # Compute Σᵢ₌₀ⁿ Aᵢ⋅∇²(φᵢ)
            for i, (A_i, phi_i) in enumerate(zip(A_components, phi_components)):
                # Compute Laplacian ∇²(φᵢ)
                laplacian = self._compute_laplacian(phi_i)

                # Multiply by Aᵢ and add to collapse tensor
                if A_i.ndim == 0:  # Scalar
                    collapse_tensor += A_i * laplacian
                else:  # Matrix
                    collapse_tensor += A_i @ laplacian

            # Add γ⋅Δ_shift term
            if delta_shift is not None:
                gamma_term = self.gamma_shift * delta_shift
                if isinstance(gamma_term, (int, float)):
                    collapse_tensor += gamma_term
                else:
                    collapse_tensor += gamma_term * np.ones_like(collapse_tensor)

            return collapse_tensor

        except Exception as e:
            self.logger.error(f"Error computing canonical collapse tensor: {e}")
            return np.array([])

    def _compute_laplacian(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇² of a tensor.

        Args:
        tensor: Input tensor

        Returns:
        Laplacian of the tensor
        """
        try:
            if tensor.ndim == 1:
                # 1D Laplacian: ∇²f = d²f/dx²
                laplacian = np.zeros_like(tensor)
                for i in range(1, len(tensor) - 1):
                    laplacian[i] = tensor[i + 1] - 2 * tensor[i] + tensor[i - 1]
                return laplacian
            elif tensor.ndim == 2:
                # 2D Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²
                laplacian = np.zeros_like(tensor)
                for i in range(1, tensor.shape[0] - 1):
                    for j in range(1, tensor.shape[1] - 1):
                        laplacian[i, j] = (
                            tensor[i + 1, j]
                            + tensor[i - 1, j]
                            + tensor[i, j + 1]
                            + tensor[i, j - 1]
                            - 4 * tensor[i, j]
                        )
                return laplacian
            else:
                # For higher dimensions, use finite difference approximation
                return self._compute_laplacian_nd(tensor)

        except Exception as e:
            self.logger.error(f"Error computing Laplacian: {e}")
            return np.zeros_like(tensor)

    def _compute_laplacian_nd(self, tensor: np.ndarray) -> np.ndarray:
        """Compute Laplacian for N-dimensional tensor using finite differences."""
        try:
            laplacian = np.zeros_like(tensor)

            # For each dimension
            for dim in range(tensor.ndim):
                # Create slice objects for shifting
                slices_plus = [slice(None)] * tensor.ndim
                slices_minus = [slice(None)] * tensor.ndim
                slices_plus[dim] = slice(1, None)
                slices_minus[dim] = slice(None, -1)

                # Add contribution from this dimension
                laplacian[tuple(slices_plus)] += tensor[tuple(slices_plus)]
                laplacian[tuple(slices_minus)] += tensor[tuple(slices_minus)]
                laplacian -= 2 * tensor

            return laplacian

        except Exception as e:
            self.logger.error(f"Error computing N-dimensional Laplacian: {e}")
            return np.zeros_like(tensor)

    def compute_fourier_tensor_dual_transform(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute Fourier-Tensor Dual Transform: F_tensor = ∫_Rⁿ T(x)e^(-2πiξ⋅x)dx

        Args:
        tensor: Input tensor

        Returns:
        Fourier transform of the tensor
        """
        try:
            if tensor.ndim == 1:
                return fft(tensor)
            elif tensor.ndim == 2:
                return fft2(tensor)
            else:
                return fftn(tensor)

        except Exception as e:
            self.logger.error(f"Error computing Fourier transform: {e}")
            return np.array([])

    def tensor_product(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> np.ndarray:
        """
        Compute tensor product of two tensors.

        Args:
        tensor_a: First tensor
        tensor_b: Second tensor

        Returns:
        Tensor product
        """
        try:
            return np.tensordot(tensor_a, tensor_b, axes=0)
        except Exception as e:
            self.logger.error(f"Error computing tensor product: {e}")
            return np.array([])

    def tensor_contraction(self, tensor: np.ndarray, axes: Tuple[int, int]) -> np.ndarray:
        """
        Compute tensor contraction along specified axes.

        Args:
        tensor: Input tensor
        axes: Axes to contract

        Returns:
        Contracted tensor
        """
        try:
            return np.tensordot(tensor, tensor, axes=axes)
        except Exception as e:
            self.logger.error(f"Error computing tensor contraction: {e}")
            return np.array([])

    def compute_eigenvalue_decomposition(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalue decomposition of a tensor.

        Args:
        tensor: Input tensor

        Returns:
        Tuple of (eigenvalues, eigenvectors)
        """
        try:
            if tensor.ndim == 2:
                eigenvalues, eigenvectors = linalg.eig(tensor)
                return eigenvalues, eigenvectors
            else:
                # For higher dimensions, flatten and compute
                shape = tensor.shape
                flattened = tensor.reshape(-1, shape[-1])
                eigenvalues, eigenvectors = linalg.eig(flattened)
                return eigenvalues, eigenvectors.reshape(shape)

        except Exception as e:
            self.logger.error(f"Error computing eigenvalue decomposition: {e}")
            return np.array([]), np.array([])

    def compute_tensor_norm(self, tensor: np.ndarray, norm_type: str = "frobenius") -> float:
        """
        Compute tensor norm.

        Args:
        tensor: Input tensor
        norm_type: Type of norm ("frobenius", "spectral", "nuclear")

        Returns:
        Tensor norm
        """
        try:
            if norm_type == "frobenius":
                return np.linalg.norm(tensor, ord="fro")
            elif norm_type == "spectral":
                return np.linalg.norm(tensor, ord=2)
            elif norm_type == "nuclear":
                return np.sum(np.abs(tensor))
            else:
                return np.linalg.norm(tensor)

        except Exception as e:
            self.logger.error(f"Error computing tensor norm: {e}")
            return 0.0

    def get_algebra_summary(self) -> Dict[str, Any]:
        """Get algebra summary statistics."""
        try:
            return {
                "initialized": self.initialized,
                "max_rank": self.max_rank,
                "collapse_threshold": self.collapse_threshold,
                "fourier_resolution": self.fourier_resolution,
                "gamma_shift": self.gamma_shift,
                "operation_count": self.operation_count,
                "total_operations": self.total_operations,
                "tensor_results_count": len(self.tensor_results),
                "collapse_results_count": len(self.collapse_results),
                "config": self.config,
            }
        except Exception as e:
            self.logger.error(f"Error getting algebra summary: {e}")
            return {"error": str(e)}


# Factory function
def create_unified_tensor_algebra(
    config: Optional[Dict[str, Any]] = None
) -> UnifiedTensorAlgebra:
    """Create a Unified Tensor Algebra instance."""
    return UnifiedTensorAlgebra(config) 