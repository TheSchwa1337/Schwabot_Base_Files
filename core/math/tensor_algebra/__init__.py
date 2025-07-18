#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Tensor Algebra for Advanced Mathematical Operations

This module provides sophisticated tensor operations including:
- Tensor contractions and decompositions
- Eigenvalue analysis
- Fourier transform operations
- Cosine similarity calculations
- Signal processing and decision making
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UnifiedTensorAlgebra:
    """
    Unified Tensor Algebra for Advanced Mathematical Operations.

    Provides sophisticated tensor operations including:
    - Tensor contractions and decompositions
    - Eigenvalue analysis
    - Fourier transform operations
    - Cosine similarity calculations
    - Mathematical decision making
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Unified Tensor Algebra."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration parameters
        self.max_rank = self.config.get("max_rank", 3)
        self.collapse_threshold = self.config.get("collapse_threshold", 0.1)
        self.fourier_resolution = self.config.get("fourier_resolution", 64)
        self.gamma_shift = self.config.get("gamma_shift", 0.1)
        self.eigenvalue_threshold = self.config.get("eigenvalue_threshold", 1e-6)
        self.norm_threshold = self.config.get("norm_threshold", 1e-8)

        self.initialized = True
        self.operation_count = 0

        self.logger.info("Unified Tensor Algebra initialized")

    def perform_tensor_operation(
        self, operation: str, tensors: List[np.ndarray]
    ) -> np.ndarray:
        """Perform tensor operation on given tensors."""
        try:
            if operation == "contraction":
                return self._tensor_contraction(tensors)
            elif operation == "decomposition":
                return self._tensor_decomposition(tensors[0])
            elif operation == "fourier":
                return self._fourier_transform(tensors[0])
            else:
                self.logger.warning(f"Unknown tensor operation: {operation}")
                return tensors[0] if tensors else np.array([])

        except Exception as e:
            self.logger.error(f"Error in tensor operation {operation}: {e}")
            return np.array([])

    def eigenvalue_decomposition(
        self, tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform eigenvalue decomposition of tensor."""
        try:
            # Ensure tensor is 2D for eigenvalue decomposition
            if tensor.ndim == 1:
                tensor_2d = tensor.reshape(-1, 1)
            elif tensor.ndim > 2:
                tensor_2d = tensor.reshape(tensor.shape[0], -1)
            else:
                tensor_2d = tensor

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(tensor_2d)

            # Filter out very small eigenvalues
            mask = np.abs(eigenvalues) > self.eigenvalue_threshold
            eigenvalues = eigenvalues[mask]
            eigenvectors = eigenvectors[:, mask]

            self.operation_count += 1
            return eigenvalues, eigenvectors

        except Exception as e:
            self.logger.error(f"Error in eigenvalue decomposition: {e}")
            return np.array([]), np.array([])

    def tensor_norm(self, tensor: np.ndarray) -> float:
        """Calculate tensor norm."""
        try:
            norm = np.linalg.norm(tensor)
            self.operation_count += 1
            return float(norm)
        except Exception as e:
            self.logger.error(f"Error calculating tensor norm: {e}")
            return 0.0

    def compute_cosine_similarity(
        self, tensor1: np.ndarray, tensor2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two tensors."""
        try:
            # Flatten tensors for similarity calculation
            flat1 = tensor1.flatten()
            flat2 = tensor2.flatten()

            # Ensure same length
            min_len = min(len(flat1), len(flat2))
            flat1 = flat1[:min_len]
            flat2 = flat2[:min_len]

            # Calculate cosine similarity
            dot_product = np.dot(flat1, flat2)
            norm1 = np.linalg.norm(flat1)
            norm2 = np.linalg.norm(flat2)

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                self.operation_count += 1
                return float(similarity)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error computing cosine similarity: {e}")
            return 0.0

    def compute_fourier_tensor_dual_transform(self, tensor: np.ndarray) -> np.ndarray:
        """Compute Fourier transform of tensor."""
        try:
            # Flatten tensor for Fourier transform
            flat_tensor = tensor.flatten()

            # Pad to fourier_resolution if needed
            if len(flat_tensor) < self.fourier_resolution:
                padded = np.pad(
                    flat_tensor, (0, self.fourier_resolution - len(flat_tensor))
                )
            else:
                padded = flat_tensor[: self.fourier_resolution]

            # Compute Fourier transform
            fourier_transform = np.fft.fft(padded)

            # Apply gamma shift
            if self.gamma_shift > 0:
                fourier_transform *= np.exp(
                    -self.gamma_shift * np.arange(len(fourier_transform))
                )

            self.operation_count += 1
            return fourier_transform

        except Exception as e:
            self.logger.error(f"Error in Fourier transform: {e}")
            return np.array([])

    def _tensor_contraction(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Perform tensor contraction."""
        try:
            if len(tensors) < 2:
                return tensors[0] if tensors else np.array([])

            result = tensors[0]
            for tensor in tensors[1:]:
                # Simple tensor contraction
                if result.ndim == 1 and tensor.ndim == 1:
                    result = np.outer(result, tensor)
                else:
                    result = np.tensordot(result, tensor, axes=1)

            return result

        except Exception as e:
            self.logger.error(f"Error in tensor contraction: {e}")
            return np.array([])

    def _tensor_decomposition(self, tensor: np.ndarray) -> np.ndarray:
        """Perform tensor decomposition."""
        try:
            # SVD decomposition
            if tensor.ndim == 2:
                U, S, Vt = np.linalg.svd(tensor)
                return S  # Return singular values
            else:
                # For higher dimensional tensors, flatten first
                flat_tensor = tensor.reshape(tensor.shape[0], -1)
                U, S, Vt = np.linalg.svd(flat_tensor)
                return S

        except Exception as e:
            self.logger.error(f"Error in tensor decomposition: {e}")
            return np.array([])

    def _fourier_transform(self, tensor: np.ndarray) -> np.ndarray:
        """Perform Fourier transform."""
        try:
            return self.compute_fourier_tensor_dual_transform(tensor)
        except Exception as e:
            self.logger.error(f"Error in Fourier transform: {e}")
            return np.array([])

    def get_operation_count(self) -> int:
        """Get total number of operations performed."""
        return self.operation_count

    def reset_operation_count(self) -> None:
        """Reset operation count."""
        self.operation_count = 0

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(new_config)
        self.logger.info("Configuration updated")


# Global instance for easy access
unified_tensor_algebra = UnifiedTensorAlgebra()
