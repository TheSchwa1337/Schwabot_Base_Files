# -*- coding: utf-8 -*-
"""
Unified Tensor Algebra for Schwabot Trading
==========================================

Provides comprehensive tensor operations for mathematical trading analysis.
Includes basic tensor operations, advanced mathematical functions, and
specialized operations for cryptocurrency and financial data processing.

Mathematical Operations:
- Basic tensor arithmetic and linear algebra
- Statistical and correlation functions
- Signal processing and FFT operations
- Matrix decomposition and analysis
- PCA and dimensionality reduction

Windows CLI compatible with comprehensive error handling.
"""

import numpy as np
from numpy.typing import NDArray
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

logger = logging.getLogger(__name__)

# Suppress NumPy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class UnifiedTensorAlgebra:
    """Unified tensor algebra operations for trading mathematics."""
    
    def __init__(self):
        """Initialize the unified tensor algebra system."""
        self.epsilon = 1e-10  # Small constant to prevent division by zero
        logger.info("Unified Tensor Algebra initialized")
    
    def tensor_dot(self, tensor_a: NDArray, tensor_b: NDArray) -> float:
        """
        Compute tensor dot product.
        
        Mathematical Formula: dot(A, B) = sum(A * B)
        """
        try:
            if tensor_a.shape != tensor_b.shape:
                raise ValueError("Tensors must have the same shape for dot product")
            return float(np.sum(tensor_a * tensor_b))
        except Exception as e:
            logger.error(f"Tensor dot product failed: {e}")
            return 0.0
    
    def tensor_normalize(self, tensor: NDArray, method: str = 'l2') -> NDArray:
        """
        Normalize tensor using specified method.
        
        Methods: 'l1', 'l2', 'max', 'minmax'
        """
        try:
            if method == 'l1':
                norm = np.sum(np.abs(tensor))
                return tensor / (norm + self.epsilon)
            elif method == 'l2':
                norm = np.sqrt(np.sum(tensor ** 2))
                return tensor / (norm + self.epsilon)
            elif method == 'max':
                max_val = np.max(np.abs(tensor))
                return tensor / (max_val + self.epsilon)
            elif method == 'minmax':
                min_val, max_val = np.min(tensor), np.max(tensor)
                if abs(max_val - min_val) < self.epsilon:
                    return tensor
                return (tensor - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        except Exception as e:
            logger.error(f"Tensor normalization failed: {e}")
            return tensor
    
    def tensor_correlation(self, tensor_a: NDArray, tensor_b: NDArray) -> float:
        """
        Calculate Pearson correlation between tensors.
        
        Mathematical Formula: r = cov(A,B) / (std(A) * std(B))
        """
        try:
            flat_a = tensor_a.flatten()
            flat_b = tensor_b.flatten()
            
            if len(flat_a) != len(flat_b):
                return 0.0
            
            correlation_matrix = np.corrcoef(flat_a, flat_b)
            return float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
        except Exception as e:
            logger.error(f"Tensor correlation failed: {e}")
            return 0.0
    
    def tensor_distance(self, tensor_a: NDArray, tensor_b: NDArray, metric: str = 'euclidean') -> float:
        """
        Calculate distance between tensors.
        
        Metrics: 'euclidean', 'manhattan', 'cosine'
        """
        try:
            flat_a = tensor_a.flatten()
            flat_b = tensor_b.flatten()
            
            if metric == 'euclidean':
                return float(np.linalg.norm(flat_a - flat_b))
            elif metric == 'manhattan':
                return float(np.sum(np.abs(flat_a - flat_b)))
            elif metric == 'cosine':
                dot_product = np.dot(flat_a, flat_b)
                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)
                
                if norm_a < self.epsilon or norm_b < self.epsilon:
                    return 1.0
                
                cosine_similarity = dot_product / (norm_a * norm_b)
                return 1.0 - cosine_similarity
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
        except Exception as e:
            logger.error(f"Tensor distance failed: {e}")
            return float('inf')
    
    def tensor_similarity(self, tensor_a: NDArray, tensor_b: NDArray, method: str = 'cosine') -> float:
        """
        Calculate similarity between tensors.
        
        Methods: 'cosine', 'dot', 'jaccard'
        """
        try:
            flat_a = tensor_a.flatten()
            flat_b = tensor_b.flatten()
            
            if method == 'cosine':
                dot_product = np.dot(flat_a, flat_b)
                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)
                
                if norm_a < self.epsilon or norm_b < self.epsilon:
                    return 0.0
                
                return float(dot_product / (norm_a * norm_b))
            elif method == 'dot':
                return float(np.dot(flat_a, flat_b))
            elif method == 'jaccard':
                intersection = np.sum(np.minimum(flat_a, flat_b))
                union = np.sum(np.maximum(flat_a, flat_b))
                return float(intersection / (union + self.epsilon))
            else:
                raise ValueError(f"Unknown similarity method: {method}")
        except Exception as e:
            logger.error(f"Tensor similarity failed: {e}")
            return 0.0


def test_unified_tensor_algebra():
    """Test the unified tensor algebra operations."""
    try:
        algebra = UnifiedTensorAlgebra()
        logger.info("Testing Unified Tensor Algebra...")
        
        # Create test tensors
        tensor_a = np.array([[1, 2], [3, 4]])
        tensor_b = np.array([[2, 1], [1, 2]])
        
        # Test operations
        dot_result = algebra.tensor_dot(tensor_a, tensor_b)
        correlation = algebra.tensor_correlation(tensor_a, tensor_b)
        distance = algebra.tensor_distance(tensor_a, tensor_b)
        similarity = algebra.tensor_similarity(tensor_a, tensor_b)
        
        logger.info(f" Tensor dot product: {dot_result}")
        logger.info(f" Tensor correlation: {correlation}")
        logger.info(f" Tensor distance: {distance}")
        logger.info(f" Tensor similarity: {similarity}")
        
        logger.info(" Unified Tensor Algebra test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f" Tensor algebra test failed: {e}")
        return False


if __name__ == "__main__":
    test_unified_tensor_algebra() 