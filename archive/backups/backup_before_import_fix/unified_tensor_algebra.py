#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Tensor Algebra - Tensor Field Unification & Recursive Math Collapse Engine

Implements Nexus mathematics for full rank-2 and rank-3 tensor unification:
- Canonical Collapse Tensor: T_collapse = Σᵢ₌₀ⁿ Aᵢ⋅∇²(φᵢ) + γ⋅Δ_shift
- Fourier-Tensor Dual Transform: F_tensor = ∫_Rⁿ T(x)e^(-2πiξ⋅x)dx
- Governs matrix field recursion + math routing, core to strategy hash-building
- The fallback tensor for galileo_tensor_field cycles into this
- Confirmed use in strategy_bit_mapper.py to perform hash comparison + cosine similarity on SHA-derived pattern matrices
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, optimize, signal
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn
from scipy.spatial.distance import cosine

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
    CONTRACTION = "contraction"      # Tensor contraction
    PRODUCT = "product"             # Tensor product
    DECOMPOSITION = "decomposition" # Eigenvalue decomposition
    TRANSFORM = "transform"         # Fourier transform
    COLLAPSE = "collapse"           # Tensor collapse
    NORM = "norm"                   # Tensor norm


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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Unified Tensor Algebra."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        
        # Tensor parameters
        self.max_rank = self.config.get('max_rank', 3)
        self.collapse_threshold = self.config.get('collapse_threshold', 0.1)
        self.fourier_resolution = self.config.get('fourier_resolution', 64)
        self.gamma_shift = self.config.get('gamma_shift', 0.1)
        
        # Operation tracking
        self.tensor_results: List[TensorResult] = []
        self.collapse_results: List[CollapseResult] = []
        
        self._initialize_algebra()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Unified Tensor Algebra."""
        return {
            'max_rank': 3,              # Maximum tensor rank
            'collapse_threshold': 0.1,  # Collapse detection threshold
            'fourier_resolution': 64,   # Fourier transform resolution
            'gamma_shift': 0.1,         # Gamma shift parameter
            'eigenvalue_threshold': 1e-6,  # Eigenvalue threshold
            'norm_threshold': 1e-8,     # Norm threshold
            'contraction_axes': None,   # Default contraction axes
        }
    
    def _initialize_algebra(self):
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
            self.logger.info("[SUCCESS] Unified Tensor Algebra initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error initializing Unified Tensor Algebra: {e}")
            self.initialized = False
    
    def compute_canonical_collapse_tensor(self, A_components: List[np.ndarray], 
                                        phi_components: List[np.ndarray], 
                                        delta_shift: float = None) -> np.ndarray:
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
                raise ValueError("A_components and phi_components must have the same length")
            
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
                # 1D Laplacian: d²/dx²
                return np.gradient(np.gradient(tensor))
            elif tensor.ndim == 2:
                # 2D Laplacian: ∂²/∂x² + ∂²/∂y²
                grad_x = np.gradient(tensor, axis=1)
                grad_y = np.gradient(tensor, axis=0)
                lapl_x = np.gradient(grad_x, axis=1)
                lapl_y = np.gradient(grad_y, axis=0)
                return lapl_x + lapl_y
            else:
                # Higher dimensional Laplacian
                laplacian = np.zeros_like(tensor)
                for axis in range(tensor.ndim):
                    grad = np.gradient(tensor, axis=axis)
                    lapl = np.gradient(grad, axis=axis)
                    laplacian += lapl
                return laplacian
                
        except Exception as e:
            self.logger.error(f"Error computing Laplacian: {e}")
            return np.zeros_like(tensor)
    
    def compute_fourier_tensor_dual_transform(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute Fourier-tensor dual transform: F_tensor = ∫_Rⁿ T(x)e^(-2πiξ⋅x)dx
        
        Args:
            tensor: Input tensor
            
        Returns:
            Fourier transform of the tensor
        """
        try:
            # Use scipy's n-dimensional FFT
            if tensor.ndim == 1:
                fourier_transform = fft(tensor)
            elif tensor.ndim == 2:
                fourier_transform = fft2(tensor)
            else:
                fourier_transform = fftn(tensor)
            
            return fourier_transform
            
        except Exception as e:
            self.logger.error(f"Error computing Fourier tensor dual transform: {e}")
            return np.array([])
    
    def tensor_contraction(self, tensor_a: np.ndarray, tensor_b: np.ndarray, 
                          axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        Perform tensor contraction.
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            axes: Contraction axes (optional)
            
        Returns:
            Contracted tensor
        """
        try:
            # Use numpy's tensordot for tensor contraction
            if axes is None:
                # Default contraction: sum over all matching dimensions
                result = np.tensordot(tensor_a, tensor_b)
            else:
                result = np.tensordot(tensor_a, tensor_b, axes=axes)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing tensor contraction: {e}")
            return np.array([])
    
    def tensor_product(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> np.ndarray:
        """
        Perform tensor product (outer product).
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            
        Returns:
            Tensor product
        """
        try:
            # Use numpy's outer product for tensor product
            result = np.outer(tensor_a, tensor_b)
            
            # Reshape to maintain tensor structure
            if tensor_a.ndim > 1 or tensor_b.ndim > 1:
                result = result.reshape(tensor_a.shape + tensor_b.shape)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing tensor product: {e}")
            return np.array([])
    
    def eigenvalue_decomposition(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition: A x = λ x
        
        Args:
            tensor: Input tensor (must be square matrix)
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            # Ensure tensor is 2D for eigenvalue decomposition
            if tensor.ndim != 2:
                raise ValueError("Eigenvalue decomposition requires 2D tensor")
            
            if tensor.shape[0] != tensor.shape[1]:
                raise ValueError("Eigenvalue decomposition requires square matrix")
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = linalg.eigh(tensor)
            
            return eigenvalues, eigenvectors
            
        except Exception as e:
            self.logger.error(f"Error performing eigenvalue decomposition: {e}")
            return np.array([]), np.array([])
    
    def tensor_norm(self, tensor: np.ndarray, norm_type: str = 'frobenius') -> float:
        """
        Compute tensor norm: ||T||
        
        Args:
            tensor: Input tensor
            norm_type: Type of norm ('frobenius', 'spectral', 'nuclear')
            
        Returns:
            Tensor norm
        """
        try:
            if norm_type == 'frobenius':
                # Frobenius norm: ||T||_F = sqrt(Σ Tᵢⱼ²)
                return np.sqrt(np.sum(np.abs(tensor)**2))
            elif norm_type == 'spectral':
                # Spectral norm: ||T||_2 = max singular value
                if tensor.ndim == 2:
                    return linalg.svd(tensor, compute_uv=False).max()
                else:
                    raise ValueError("Spectral norm requires 2D tensor")
            elif norm_type == 'nuclear':
                # Nuclear norm: ||T||_* = sum of singular values
                if tensor.ndim == 2:
                    return linalg.svd(tensor, compute_uv=False).sum()
                else:
                    raise ValueError("Nuclear norm requires 2D tensor")
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
                
        except Exception as e:
            self.logger.error(f"Error computing tensor norm: {e}")
            return 0.0
    
    def compute_cosine_similarity(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
        """
        Compute cosine similarity between tensors.
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            
        Returns:
            Cosine similarity
        """
        try:
            # Flatten tensors for cosine similarity
            flat_a = tensor_a.flatten()
            flat_b = tensor_b.flatten()
            
            # Compute cosine similarity
            similarity = 1 - cosine(flat_a, flat_b)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def perform_tensor_operation(self, operation: TensorOperation, 
                               tensors: List[np.ndarray], 
                               parameters: Dict[str, Any] = None) -> TensorResult:
        """
        Perform a tensor operation and track the result.
        
        Args:
            operation: Type of operation to perform
            tensors: List of input tensors
            parameters: Operation parameters
            
        Returns:
            Tensor operation result
        """
        try:
            parameters = parameters or {}
            
            # Record input shapes
            input_shapes = [tensor.shape for tensor in tensors]
            
            # Perform operation based on type
            if operation == TensorOperation.CONTRACTION:
                axes = parameters.get('axes')
                result_data = self.tensor_contraction(tensors[0], tensors[1], axes)
                
            elif operation == TensorOperation.PRODUCT:
                result_data = self.tensor_product(tensors[0], tensors[1])
                
            elif operation == TensorOperation.DECOMPOSITION:
                eigenvalues, eigenvectors = self.eigenvalue_decomposition(tensors[0])
                result_data = np.concatenate([eigenvalues, eigenvectors.flatten()])
                
            elif operation == TensorOperation.TRANSFORM:
                result_data = self.compute_fourier_tensor_dual_transform(tensors[0])
                
            elif operation == TensorOperation.COLLAPSE:
                A_components = parameters.get('A_components', [np.eye(tensors[0].shape[0])])
                phi_components = tensors
                delta_shift = parameters.get('delta_shift', 0.0)
                result_data = self.compute_canonical_collapse_tensor(A_components, phi_components, delta_shift)
                
            elif operation == TensorOperation.NORM:
                norm_type = parameters.get('norm_type', 'frobenius')
                result_data = np.array([self.tensor_norm(tensors[0], norm_type)])
                
            else:
                raise ValueError(f"Unknown tensor operation: {operation}")
            
            # Create result
            result = TensorResult(
                timestamp=time.time(),
                operation=operation,
                input_shape=input_shapes,
                output_shape=result_data.shape,
                result_data=result_data
            )
            
            # Store result
            self.tensor_results.append(result)
            self.operation_count += 1
            self.total_operations += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing tensor operation: {e}")
            return TensorResult(
                timestamp=time.time(),
                operation=operation,
                input_shape=[],
                output_shape=(),
                result_data=np.array([])
            )
    
    def get_algebra_summary(self) -> Dict[str, Any]:
        """Get comprehensive algebra summary."""
        if not self.tensor_results:
            return {'status': 'no_operations'}
        
        # Compute operation statistics
        operation_counts = {}
        for operation in TensorOperation:
            operation_counts[operation.value] = sum(1 for r in self.tensor_results if r.operation == operation)
        
        # Compute tensor rank distribution
        rank_distribution = {}
        for result in self.tensor_results:
            for input_shape in result.input_shape:
                rank = len(input_shape)
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        return {
            'total_operations': self.total_operations,
            'operation_distribution': operation_counts,
            'rank_distribution': rank_distribution,
            'max_rank': self.max_rank,
            'initialized': self.initialized,
            'tensor_results_count': len(self.tensor_results),
            'collapse_results_count': len(self.collapse_results)
        }
    
    def set_max_rank(self, max_rank: int):
        """Set the maximum tensor rank."""
        if 1 <= max_rank <= 5:
            self.max_rank = max_rank
            self.logger.info(f"Maximum tensor rank set to: {max_rank}")
        else:
            self.logger.warning(f"Invalid max_rank: {max_rank}, must be between 1 and 5")


# Factory function
def create_unified_tensor_algebra(config: Optional[Dict[str, Any]] = None) -> UnifiedTensorAlgebra:
    """Create a Unified Tensor Algebra instance."""
    return UnifiedTensorAlgebra(config)
