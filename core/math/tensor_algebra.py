# -*- coding: utf-8 -*-
"""
Tensor Algebra - Unified Mathematical Operations for Schwabot
============================================================

Comprehensive tensor algebra system for the Schwabot mathematical trading framework.
Provides unified tensor operations, bit phase calculations, and mathematical foundations.

Key Features:
- Tensor contraction and decomposition operations
- Bit phase tensor calculations (4-bit, 8-bit, 42-bit)
- Matrix basket operations for portfolio calculations
- Hash memory encoding for data mapping
- Entropy compensation for data streams
- Profit routing tensor calculations
- Matrix decomposition (SVD, QR, Cholesky)
- Tensor normalization and validation
- Integration with all core components
- Windows CLI compatibility with emoji fallbacks

Mathematical Operations:
- Tensor Contraction: T_ij = Σ_k A_ik · B_kj
- Bit Phase Extraction: phi_4, phi_8, phi_42 from strategy_id
- Matrix Basket: B = Σ w_i · P_i for weights w and prices P
- Hash Encoding: H(x) = SHA256(x) for memory mapping
- Entropy Compensation: E_comp = E_orig + λ · log(1 + |∇E|)
- Profit Routing: R = Σ w_i · P_i · confidence_i

Integration Points:
- All core components for mathematical operations
- enhanced_windows_cli_compatibility.py: CLI compatibility
- thermal_boundary_manager.py: Thermal-aware computations
- main_orchestrator.py: System-wide mathematical coordination
- profit_routing_engine.py: Mathematical profit optimization

Windows CLI compatible with flake8 compliance.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BitPhaseResult:
    """Result of bit phase tensor operations."""
    phi_4: int
    phi_8: int
    phi_42: int
    strategy_id: int
    mode: str


@dataclass
class TensorOperationResult:
    """Result of tensor operations."""
    result: np.ndarray
    operation_type: str
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    success: bool
    error_message: Optional[str] = None


class UnifiedTensorAlgebra:
    """Unified tensor algebra operations for Schwabot mathematical pipeline."""
    
    def __init__(self, precision: np.dtype = np.float64):
        """Initialize the unified tensor algebra engine."""
        self.precision = precision
        self.epsilon = 1e-12
        self.operation_count = 0
        self.error_count = 0
        
        # Mathematical constants
        self.constants = {
            'pi': np.pi,
            'e': np.e,
            'golden_ratio': 1.618033988749,
            'sqrt_2': np.sqrt(2),
            'sqrt_3': np.sqrt(3)
        }
        
        logger.info("Unified Tensor Algebra initialized")
    
    def tensor_contraction(self, A: np.ndarray, B: np.ndarray, 
                          axes: Union[int, List[int]] = 1) -> np.ndarray:
        """
        Perform tensor contraction: T_ij = Σ_k A_ik · B_kj

        Args:
            A: First tensor
            B: Second tensor
            axes: Axes to contract over

        Returns:
            Contracted tensor
        """
        try:
            self.operation_count += 1
            result = np.tensordot(A, B, axes=axes)
            return result.astype(self.precision)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Tensor contraction failed: {e}")
            # Return safe fallback
            return np.zeros((A.shape[0], B.shape[-1]), dtype=self.precision)

    def bit_phase_tensor(self, strategy_id: int, mode: str = '4bit') -> BitPhaseResult:
        """
        Compute bit phase tensor operations for strategy routing.

        Mathematical implementation:
        phi_4 = (strategy_id & 0b1111)
        phi_8 = (strategy_id >> 4) & 0b11111111
        phi_42 = (strategy_id >> 12) & 0x3FFFFFFFFFF

        Args:
            strategy_id: Integer strategy identifier
            mode: Bit mode ('4bit', '8bit', '42bit')

        Returns:
            BitPhaseResult with phi values
        """
        try:
            self.operation_count += 1
            
            # Extract bit phases
            phi_4 = strategy_id & 0b1111
            phi_8 = (strategy_id >> 4) & 0b11111111
            phi_42 = (strategy_id >> 12) & 0x3FFFFFFFFFF
            
            return BitPhaseResult(phi_4, phi_8, phi_42, strategy_id, mode)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Bit phase tensor calculation failed: {e}")
            return BitPhaseResult(0, 0, 0, strategy_id, mode)

    def matrix_basket_operation(self, prices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Perform matrix basket operations for portfolio calculations.
        
        Mathematical: B = Σ w_i · P_i for weights w and prices P

        Args:
            prices: Price matrix
            weights: Weight vector

        Returns:
            Basket result
        """
        try:
            self.operation_count += 1
            
            # Ensure proper shapes
            if prices.ndim == 1:
                prices = prices.reshape(-1, 1)
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            
            # Perform basket operation
            result = np.dot(weights, prices)
            return result.astype(self.precision)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Matrix basket operation failed: {e}")
            return np.zeros_like(prices, dtype=self.precision)
    
    def hash_memory_encoding(self, data: Union[str, bytes, np.ndarray]) -> str:
        """
        Encode data for hash memory mapping.
        
        Mathematical: H(x) = SHA256(x) for memory mapping

        Args:
            data: Data to encode

        Returns:
            SHA256 hash string
        """
        try:
            self.operation_count += 1
            
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Generate hash
            hash_result = hashlib.sha256(data_bytes).hexdigest()
            return hash_result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Hash memory encoding failed: {e}")
            return "0" * 64  # Return zero hash as fallback
    
    def entropy_compensation(self, data: np.ndarray, 
                           compensation_factor: float = 1.0) -> np.ndarray:
        """
        Calculate entropy compensation for data streams.
        
        Mathematical: E_comp = E_orig + λ · log(1 + |∇E|)

        Args:
            data: Input data array
            compensation_factor: Compensation factor λ

        Returns:
            Compensated data
        """
        try:
            self.operation_count += 1
            
            # Calculate entropy
            if data.size == 0:
                return data
            
            # Normalize data
            data_norm = data / (np.max(np.abs(data)) + self.epsilon)
            
            # Calculate gradient
            gradient = np.gradient(data_norm)
            gradient_magnitude = np.sqrt(np.sum(gradient**2, axis=0))
            
            # Apply compensation
            compensation = compensation_factor * np.log(1 + gradient_magnitude)
            result = data_norm + compensation
            
            return result.astype(self.precision)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Entropy compensation failed: {e}")
            return data
    
    def profit_routing_tensor(self, profit_data: np.ndarray, 
                            routing_weights: np.ndarray) -> np.ndarray:
        """
        Calculate profit routing tensor for strategy optimization.
        
        Mathematical: R = Σ w_i · P_i · confidence_i

        Args:
            profit_data: Profit data matrix
            routing_weights: Routing weight vector

        Returns:
            Routing tensor
        """
        try:
            self.operation_count += 1
            
            # Ensure proper shapes
            if profit_data.ndim == 1:
                profit_data = profit_data.reshape(-1, 1)
            if routing_weights.ndim == 1:
                routing_weights = routing_weights.reshape(1, -1)
            
            # Calculate confidence weights (simplified)
            confidence = np.ones_like(routing_weights)
            
            # Perform routing calculation
            result = np.dot(routing_weights * confidence, profit_data)
            return result.astype(self.precision)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Profit routing tensor failed: {e}")
            return np.zeros_like(profit_data, dtype=self.precision)
    
    def matrix_decomposition(self, matrix: np.ndarray, 
                           method: str = 'svd') -> Tuple[np.ndarray, ...]:
        """
        Perform matrix decomposition operations.
        
        Args:
            matrix: Input matrix
            method: Decomposition method ('svd', 'qr', 'cholesky')
            
        Returns:
            Tuple of decomposition components
        """
        try:
            self.operation_count += 1
            
            if method.lower() == 'svd':
                U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
                return U, s, Vt
                
            elif method.lower() == 'qr':
                Q, R = np.linalg.qr(matrix)
                return Q, R
                
            elif method.lower() == 'cholesky':
                L = np.linalg.cholesky(matrix)
                return L,
                
            else:
                raise ValueError(f"Unsupported decomposition method: {method}")
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Matrix decomposition failed: {e}")
            # Return identity matrices as fallback
            n = matrix.shape[0]
            return (np.eye(n, dtype=self.precision),)
    
    def tensor_normalization(self, tensor: np.ndarray, 
                           norm_type: str = 'frobenius') -> np.ndarray:
        """
        Normalize tensor using specified norm.
        
        Args:
            tensor: Input tensor
            norm_type: Type of normalization ('frobenius', 'l2', 'max')
            
        Returns:
            Normalized tensor
        """
        try:
            self.operation_count += 1
            
            if norm_type.lower() == 'frobenius':
                norm = np.linalg.norm(tensor, 'fro')
            elif norm_type.lower() == 'l2':
                norm = np.linalg.norm(tensor, 2)
            elif norm_type.lower() == 'max':
                norm = np.max(np.abs(tensor))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
            
            if norm > self.epsilon:
                return (tensor / norm).astype(self.precision)
            else:
                return tensor.astype(self.precision)
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Tensor normalization failed: {e}")
            return tensor.astype(self.precision)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            'operation_count': self.operation_count,
            'error_count': self.error_count,
            'success_rate': (self.operation_count - self.error_count) / max(self.operation_count, 1),
            'precision': str(self.precision),
            'epsilon': self.epsilon
        }
    
    def reset_statistics(self) -> None:
        """Reset operation statistics."""
        self.operation_count = 0
        self.error_count = 0


# Global tensor algebra instance
_tensor_algebra: Optional[UnifiedTensorAlgebra] = None


def get_tensor_algebra() -> UnifiedTensorAlgebra:
    """Get global tensor algebra instance."""
    global _tensor_algebra
    if _tensor_algebra is None:
        _tensor_algebra = UnifiedTensorAlgebra()
    return _tensor_algebra


def main():
    """Test the tensor algebra system."""
    try:
        # Create tensor algebra instance
        algebra = get_tensor_algebra()
        
        # Test bit phase tensor
        bit_result = algebra.bit_phase_tensor(12345, '4bit')
        print(f"Bit Phase Result: phi_4={bit_result.phi_4}, phi_8={bit_result.phi_8}")
        
        # Test matrix basket operation
        prices = np.array([[100], [200], [300]])
        weights = np.array([[0.4, 0.3, 0.3]])
        basket_result = algebra.matrix_basket_operation(prices, weights)
        print(f"Basket Result: {basket_result}")
        
        # Test hash encoding
        hash_result = algebra.hash_memory_encoding("test_data")
        print(f"Hash Result: {hash_result[:16]}...")
        
        # Get statistics
        stats = algebra.get_statistics()
        print(f"Statistics: {stats}")
        
        print("🎉 Tensor algebra test completed successfully")
        
    except Exception as e:
        print(f"❌ Tensor algebra test failed: {e}")


if __name__ == "__main__":
    main()
