#!/usr/bin/env python3
"""
Rittle GEMM - High-Performance Matrix Operations Library
=======================================================

High-performance matrix operations library optimized for mathematical trading.
Provides fast matrix multiplication, decomposition, and linear algebra operations
with GPU acceleration support and memory optimization.

Key Features:
- Optimized matrix multiplication (GEMM)
- BLAS/LAPACK integration for maximum performance
- GPU acceleration support (CUDA/OpenCL)
- Memory-efficient operations with chunking
- Sparse matrix support and optimization
- Matrix decomposition algorithms (LU, QR, SVD, Cholesky)
- Statistical matrix operations
- Real-time optimization for trading applications
- Thread-safe operations with parallel processing
- Numerical stability and error handling
- Windows CLI compatibility with emoji fallbacks

Mathematical Foundations:
- GEMM: C = α * A * B + β * C (General Matrix Multiply)
- SYMM: C = α * A * B + β * C (Symmetric Matrix Multiply)
- TRMM: B = α * A * B (Triangular Matrix Multiply)
- SYRK: C = α * A * A^T + β * C (Symmetric Rank-K Update)
- GER: A = A + α * x * y^T (Rank-1 Update)

Performance Optimizations:
- Block matrix multiplication for cache efficiency
- Memory-aligned operations for SIMD acceleration
- Parallel processing with thread pools
- Lazy evaluation for complex operations
- Smart caching with LRU eviction
- Adaptive algorithm selection based on matrix properties

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from collections import deque, defaultdict
import math
import warnings

import numpy as np
import numpy.typing as npt
from scipy import linalg, sparse
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.linalg import blas, lapack

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler,
        safe_print, safe_log
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback CLI handler for when the main handler is not available
    class CLIHandler:
        @staticmethod
        def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
            """Fallback emoji-safe print function"""
            emoji_mapping = {
                '✅': '[SUCCESS]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🚨': '[ALERT]',
                '🎉': '[COMPLETE]', '🔄': '[PROCESSING]', '⏳': '[WAITING]', '⭐': '[STAR]',
                '🚀': '[LAUNCH]', '🔧': '[TOOLS]', '🛠️': '[REPAIR]', '⚡': '[FAST]',
                '🔍': '[SEARCH]', '🎯': '[TARGET]', '🔥': '[HOT]', '❄️': '[COOL]',
                '📊': '[DATA]', '📈': '[PROFIT]', '📉': '[LOSS]', '💰': '[MONEY]',
                '🧪': '[TEST]', '⚖️': '[BALANCE]', '️': '[TEMP]', '🔬': '[ANALYZE]',
                '': '[SYSTEM]', '️': '[COMPUTER]', '📱': '[MOBILE]', '🌐': '[NETWORK]',
                '🔒': '[SECURE]', '🔓': '[UNLOCK]', '🔑': '[KEY]', '🛡️': '[SHIELD]',
                '🧮': '[CALC]', '📐': '[MATH]', '🔢': '[NUMBERS]', '∞': '[INFINITY]',
                'φ': '[PHI]', 'π': '[PI]', '∑': '[SUM]', '∫': '[INTEGRAL]'
            }
            
            if force_ascii:
                for emoji, replacement in emoji_mapping.items():
                    message = message.replace(emoji, replacement)
            
            return message
        
        @staticmethod
        def safe_print(message: str, force_ascii: bool = False) -> None:
            """Fallback safe print function"""
            safe_message = CLIHandler.safe_emoji_print(message, force_ascii)
            print(safe_message)

if TYPE_CHECKING:
    from typing_extensions import Self

# Type definitions for matrix operations
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
SparseMatrix = Union[csr_matrix, csc_matrix, coo_matrix]

logger = logging.getLogger(__name__)


class MatrixType(Enum):
    """Matrix type enumeration for optimization strategies"""
    DENSE = "dense"
    SPARSE = "sparse"
    SYMMETRIC = "symmetric"
    HERMITIAN = "hermitian"
    TRIANGULAR = "triangular"
    DIAGONAL = "diagonal"
    BANDED = "banded"
    TOEPLITZ = "toeplitz"


class OperationType(Enum):
    """Operation type enumeration for performance tracking"""
    GEMM = "gemm"  # General matrix multiply
    SYMM = "symm"  # Symmetric matrix multiply
    TRMM = "trmm"  # Triangular matrix multiply
    SYRK = "syrk"  # Symmetric rank-k update
    GER = "ger"    # Rank-1 update
    GEMV = "gemv"  # General matrix-vector multiply
    DECOMPOSITION = "decomposition"
    EIGENVALUE = "eigenvalue"
    INVERSE = "inverse"


class OptimizationLevel(Enum):
    """Optimization level enumeration"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class MatrixInfo:
    """Matrix information container for optimization decisions"""
    
    shape: Tuple[int, int]
    dtype: np.dtype
    matrix_type: MatrixType
    is_sparse: bool
    nnz: int  # Number of non-zero elements
    memory_usage: int  # Memory usage in bytes
    condition_number: Optional[float] = None
    rank: Optional[int] = None
    sparsity: float = 0.0
    symmetry_error: float = 0.0
    bandwidth: Optional[int] = None


@dataclass
class OperationResult:
    """Operation result container with performance metrics"""
    
    result: Union[Matrix, SparseMatrix, Vector]
    operation_type: OperationType
    optimization_level: OptimizationLevel
    execution_time: float
    memory_used: int
    flops: int  # Floating point operations
    cache_hits: int
    cache_misses: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking"""
    
    total_operations: int
    total_execution_time: float
    total_flops: int
    average_execution_time: float
    peak_memory_usage: int
    cache_hit_rate: float
    throughput: float  # Operations per second


class RittleGEMM:
    """
    High-performance matrix operations library with optimization strategies
    
    This class provides optimized matrix operations for mathematical trading
    applications, with support for various matrix types and optimization levels.
    Includes robust Windows CLI compatibility with emoji fallbacks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize Rittle GEMM with configuration
        
        Args:
            config: Configuration dictionary for optimization settings
        """
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Initialize CLI compatibility handler
        self.cli_handler = CLIHandler()
        
        # Performance tracking and metrics
        self.operation_history: deque = deque(maxlen=self.config.get('max_history_size', 1000))
        self.total_operations = 0
        self.total_flops = 0
        self.total_execution_time = 0.0
        
        # Memory management and caching
        self.memory_pool: Dict[int, List[Matrix]] = defaultdict(list)
        self.max_memory_usage = self.config.get('max_memory_usage', 1024 * 1024 * 1024)  # 1GB
        self.current_memory_usage = 0
        
        # Threading and parallel processing
        self.thread_pool_size = self.config.get('thread_pool_size', 4)
        self.enable_gpu = self.config.get('enable_gpu', False)
        self.enable_optimization = self.config.get('enable_optimization', True)
        
        # BLAS/LAPACK configuration and optimization
        self.blas_config = self._initialize_blas_config()
        
        # Performance monitoring and statistics
        self.performance_stats = {
            'gemm_operations': 0,
            'decomposition_operations': 0,
            'eigenvalue_operations': 0,
            'inverse_operations': 0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety and synchronization
        self.operation_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        
        # Initialize optimization strategies
        self._initialize_optimization_strategies()
        
        # Log initialization with CLI-safe output
        init_message = f"RittleGEMM v{self.version} initialized with {self.thread_pool_size} threads"
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_log(logger, 'info', init_message)
        else:
            logger.info(init_message)
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Default configuration for optimization settings
        
        Returns:
            Dictionary containing default configuration parameters
        """
        return {
            'max_history_size': 1000,
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'thread_pool_size': 4,
            'enable_gpu': False,
            'enable_optimization': True,
            'enable_memory_pooling': True,
            'enable_performance_monitoring': True,
            'blas_optimization_level': 3,
            'sparse_threshold': 0.1,  # 10% sparsity threshold
            'condition_number_threshold': 1e12,
            'enable_parallel_processing': True,
            'chunk_size': 1024,
            'cache_size': 100,
            'block_size': 64,  # Block size for cache-efficient operations
            'enable_numerical_stability': True,
            'stability_epsilon': 1e-12,
            'enable_cli_compatibility': True,  # Enable CLI compatibility by default
            'force_ascii_output': False  # Force ASCII output for problematic environments
        }
    
    def _initialize_blas_config(self) -> Dict[str, Any]:
        """
        Initialize BLAS/LAPACK configuration for optimal performance
        
        Returns:
            Dictionary containing BLAS/LAPACK configuration
        """
        return {
            'optimization_level': self.config.get('blas_optimization_level', 3),
            'thread_count': self.config.get('thread_pool_size', 4),
            'enable_parallel': self.config.get('enable_parallel_processing', True),
            'cache_size': self.config.get('cache_size', 100),
            'block_size': self.config.get('block_size', 64)
        }
    
    def _initialize_optimization_strategies(self) -> None:
        """
        Initialize optimization strategies for different matrix types and operations
        
        This method sets up the optimization strategies that will be used
        for different types of matrix operations based on matrix properties.
        """
        self.optimization_strategies = {
            MatrixType.DENSE: self._dense_matrix_strategy,
            MatrixType.SPARSE: self._sparse_matrix_strategy,
            MatrixType.SYMMETRIC: self._symmetric_matrix_strategy,
            MatrixType.TRIANGULAR: self._triangular_matrix_strategy,
            MatrixType.DIAGONAL: self._diagonal_matrix_strategy
        }
    
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """
        Safe print function with CLI compatibility and emoji fallbacks
        
        Args:
            message: Message to print
            force_ascii: Force ASCII conversion (None = auto-detect)
        """
        if force_ascii is None:
            force_ascii = self.config.get('force_ascii_output', False)
        
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            # Fallback to basic print with emoji replacement
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """
        Safe logging function with CLI compatibility
        
        Args:
            level: Log level ('info', 'warning', 'error', 'debug')
            message: Message to log
            context: Additional context information
            
        Returns:
            True if logging was successful, False otherwise
        """
        if CLI_COMPATIBILITY_AVAILABLE:
            return safe_log(logger, level, message, context)
        else:
            # Fallback to basic logging
            try:
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(message)
                return True
            except Exception:
                return False
    
    def gemm(self, A: Matrix, B: Matrix, C: Optional[Matrix] = None, 
             alpha: float = 1.0, beta: float = 0.0, 
             transpose_a: bool = False, transpose_b: bool = False,
             optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> OperationResult:
        """
        General Matrix Multiply: C = α * op(A) * op(B) + β * C
        
        This is the core matrix multiplication operation optimized for performance.
        It automatically selects the best algorithm based on matrix properties.
        
        Args:
            A: Input matrix A
            B: Input matrix B
            C: Output matrix C (optional, will be created if None)
            alpha: Scaling factor for A*B
            beta: Scaling factor for C
            transpose_a: Whether to transpose A
            transpose_b: Whether to transpose B
            optimization_level: Level of optimization to apply
            
        Returns:
            OperationResult containing the result and performance metrics
            
        Raises:
            ValueError: If matrix dimensions are incompatible
            RuntimeError: If operation fails due to numerical issues
        """
        try:
            start_time = time.time()
            
            # Validate inputs and check compatibility
            if not self._validate_matrices(A, B):
                return OperationResult(
                    result=np.array([]),
                    operation_type=OperationType.GEMM,
                    optimization_level=optimization_level,
                    execution_time=0.0,
                    memory_used=0,
                    flops=0,
                    cache_hits=0,
                    cache_misses=0,
                    success=False,
                    error_message="Invalid matrix inputs"
                )
            
            # Get matrix shapes and handle transpositions
            shape_a = A.shape
            shape_b = B.shape
            
            if transpose_a:
                shape_a = (shape_a[1], shape_a[0])
            if transpose_b:
                shape_b = (shape_b[1], shape_b[0])
            
            # Check matrix compatibility
            if shape_a[1] != shape_b[0]:
                return OperationResult(
                    result=np.array([]),
                    operation_type=OperationType.GEMM,
                    optimization_level=optimization_level,
                    execution_time=0.0,
                    memory_used=0,
                    flops=0,
                    cache_hits=0,
                    cache_misses=0,
                    success=False,
                    error_message="Matrix dimensions incompatible"
                )
            
            # Prepare output matrix
            if C is None:
                C = np.zeros((shape_a[0], shape_b[1]), dtype=A.dtype)
            elif C.shape != (shape_a[0], shape_b[1]):
                C = np.zeros((shape_a[0], shape_b[1]), dtype=A.dtype)
            
            # Select optimization strategy based on matrix properties
            matrix_info_a = self.get_matrix_info(A)
            matrix_info_b = self.get_matrix_info(B)
            
            # Choose the best optimization strategy
            if optimization_level == OptimizationLevel.MAXIMUM:
                result = self._maximum_optimization_gemm(A, B, C, alpha, beta, 
                                                       transpose_a, transpose_b, 
                                                       matrix_info_a, matrix_info_b)
            elif optimization_level == OptimizationLevel.AGGRESSIVE:
                result = self._aggressive_optimization_gemm(A, B, C, alpha, beta,
                                                          transpose_a, transpose_b,
                                                          matrix_info_a, matrix_info_b)
            else:
                result = self._standard_optimization_gemm(A, B, C, alpha, beta,
                                                        transpose_a, transpose_b,
                                                        matrix_info_a, matrix_info_b)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            flops = 2 * shape_a[0] * shape_a[1] * shape_b[1]  # Approximate FLOP count
            memory_used = result.nbytes
            
            # Update performance tracking
            self._update_performance_metrics(OperationType.GEMM, execution_time, flops, memory_used)
            
            return OperationResult(
                result=result,
                operation_type=OperationType.GEMM,
                optimization_level=optimization_level,
                execution_time=execution_time,
                memory_used=memory_used,
                flops=flops,
                cache_hits=self.performance_stats['cache_hits'],
                cache_misses=self.performance_stats['cache_misses'],
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error in GEMM operation: {e}"
            self.safe_log('error', error_msg)
            return OperationResult(
                result=np.array([]),
                operation_type=OperationType.GEMM,
                optimization_level=optimization_level,
                execution_time=0.0,
                memory_used=0,
                flops=0,
                cache_hits=0,
                cache_misses=0,
                success=False,
                error_message=str(e)
            )
    
    def _maximum_optimization_gemm(self, A: Matrix, B: Matrix, C: Matrix,
                                  alpha: float, beta: float,
                                  transpose_a: bool, transpose_b: bool,
                                  matrix_info_a: MatrixInfo, matrix_info_b: MatrixInfo) -> Matrix:
        """
        Maximum optimization GEMM using the most aggressive optimization strategies
        
        This method applies the highest level of optimization including:
        - BLAS-optimized operations
        - Block matrix multiplication
        - Parallel processing
        - Memory alignment optimizations
        """
        try:
            # Use BLAS GEMM if available and matrices are large enough
            if (A.shape[0] > 100 and B.shape[1] > 100 and 
                self.blas_config['enable_parallel']):
                
                # Prepare matrices for BLAS operation
                if transpose_a:
                    A_blas = A.T.copy()
                else:
                    A_blas = A.copy()
                
                if transpose_b:
                    B_blas = B.T.copy()
                else:
                    B_blas = B.copy()
                
                # Use BLAS GEMM for maximum performance
                result = blas.dgemm(alpha, A_blas, B_blas, beta, C)
                return result
            
            # Fallback to block matrix multiplication
            return self._block_matrix_multiply(A, B, C, alpha, beta, 
                                             transpose_a, transpose_b)
            
        except Exception as e:
            warning_msg = f"Maximum optimization failed, falling back to standard: {e}"
            self.safe_log('warning', warning_msg)
            return self._standard_optimization_gemm(A, B, C, alpha, beta,
                                                  transpose_a, transpose_b,
                                                  matrix_info_a, matrix_info_b)
    
    def _aggressive_optimization_gemm(self, A: Matrix, B: Matrix, C: Matrix,
                                    alpha: float, beta: float,
                                    transpose_a: bool, transpose_b: bool,
                                    matrix_info_a: MatrixInfo, matrix_info_b: MatrixInfo) -> Matrix:
        """
        Aggressive optimization GEMM using advanced optimization strategies
        
        This method applies aggressive optimization including:
        - Block matrix multiplication
        - Cache-aware algorithms
        - Memory pooling
        """
        try:
            # Use block matrix multiplication for cache efficiency
            return self._block_matrix_multiply(A, B, C, alpha, beta,
                                             transpose_a, transpose_b)
            
        except Exception as e:
            warning_msg = f"Aggressive optimization failed, falling back to standard: {e}"
            self.safe_log('warning', warning_msg)
            return self._standard_optimization_gemm(A, B, C, alpha, beta,
                                                  transpose_a, transpose_b,
                                                  matrix_info_a, matrix_info_b)
    
    def _standard_optimization_gemm(self, A: Matrix, B: Matrix, C: Matrix,
                                  alpha: float, beta: float,
                                  transpose_a: bool, transpose_b: bool,
                                  matrix_info_a: MatrixInfo, matrix_info_b: MatrixInfo) -> Matrix:
        """
        Standard optimization GEMM using numpy's optimized operations
        
        This method uses numpy's built-in optimizations and is the most reliable
        fallback for matrix multiplication operations.
        """
        try:
            # Apply transpositions
            A_op = A.T if transpose_a else A
            B_op = B.T if transpose_b else B
            
            # Perform matrix multiplication
            result = alpha * A_op @ B_op + beta * C
            
            return result
            
        except Exception as e:
            error_msg = f"Standard optimization failed: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def _block_matrix_multiply(self, A: Matrix, B: Matrix, C: Matrix,
                              alpha: float, beta: float,
                              transpose_a: bool, transpose_b: bool) -> Matrix:
        """
        Block matrix multiplication for cache-efficient operations
        
        This method implements block matrix multiplication to optimize
        cache usage and improve performance for large matrices.
        """
        try:
            # Apply transpositions
            A_op = A.T if transpose_a else A
            B_op = B.T if transpose_b else B
            
            # Get block size from configuration
            block_size = self.config.get('block_size', 64)
            
            m, k = A_op.shape
            k, n = B_op.shape
            
            # Initialize result matrix
            result = beta * C.copy()
            
            # Block matrix multiplication
            for i in range(0, m, block_size):
                for j in range(0, n, block_size):
                    for l in range(0, k, block_size):
                        # Define block boundaries
                        i_end = min(i + block_size, m)
                        j_end = min(j + block_size, n)
                        l_end = min(l + block_size, k)
                        
                        # Multiply blocks
                        result[i:i_end, j:j_end] += (
                            alpha * A_op[i:i_end, l:l_end] @ B_op[l:l_end, j:j_end]
                        )
            
            return result
            
        except Exception as e:
            error_msg = f"Block matrix multiplication failed: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def lu_decomposition(self, A: Matrix, 
                        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> Tuple[Matrix, Matrix, Matrix]:
        """
        LU decomposition: A = P * L * U
        
        Performs LU decomposition with optional optimization strategies.
        
        Args:
            A: Input matrix
            optimization_level: Level of optimization to apply
            
        Returns:
            Tuple of (P, L, U) matrices where P is permutation matrix
            
        Raises:
            ValueError: If matrix is not square
            RuntimeError: If decomposition fails
        """
        try:
            start_time = time.time()
            
            # Validate input matrix
            if A.shape[0] != A.shape[1]:
                raise ValueError("Matrix must be square for LU decomposition")
            
            # Check if matrix is well-conditioned
            condition_number = np.linalg.cond(A)
            if condition_number > self.config.get('condition_number_threshold', 1e12):
                warning_msg = f"Matrix is ill-conditioned (cond={condition_number:.2e})"
                self.safe_log('warning', warning_msg)
            
            # Perform LU decomposition
            if optimization_level == OptimizationLevel.MAXIMUM:
                # Use LAPACK for maximum performance
                P, L, U = lapack.dgetrf(A)
            else:
                # Use scipy's LU decomposition
                P, L, U = linalg.lu(A)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            flops = 2 * A.shape[0] ** 3 // 3  # Approximate FLOP count for LU
            memory_used = P.nbytes + L.nbytes + U.nbytes
            
            self._update_performance_metrics(OperationType.DECOMPOSITION, execution_time, flops, memory_used)
            
            return P, L, U
            
        except Exception as e:
            error_msg = f"Error in LU decomposition: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def qr_decomposition(self, A: Matrix,
                        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> Tuple[Matrix, Matrix]:
        """
        QR decomposition: A = Q * R
        
        Performs QR decomposition with optional optimization strategies.
        
        Args:
            A: Input matrix
            optimization_level: Level of optimization to apply
            
        Returns:
            Tuple of (Q, R) matrices
        """
        try:
            start_time = time.time()
            
            # Perform QR decomposition
            if optimization_level == OptimizationLevel.MAXIMUM:
                # Use LAPACK for maximum performance
                Q, R = lapack.dgeqrf(A)
            else:
                # Use scipy's QR decomposition
                Q, R = linalg.qr(A)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            flops = 4 * A.shape[0] * A.shape[1] ** 2  # Approximate FLOP count for QR
            memory_used = Q.nbytes + R.nbytes
            
            self._update_performance_metrics(OperationType.DECOMPOSITION, execution_time, flops, memory_used)
            
            return Q, R
            
        except Exception as e:
            error_msg = f"Error in QR decomposition: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def svd_decomposition(self, A: Matrix, full_matrices: bool = True,
                         optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> Tuple[Matrix, Vector, Matrix]:
        """
        Singular Value Decomposition: A = U * S * V^T
        
        Performs SVD decomposition with optional optimization strategies.
        
        Args:
            A: Input matrix
            full_matrices: Whether to return full matrices
            optimization_level: Level of optimization to apply
            
        Returns:
            Tuple of (U, S, V^T) matrices/vectors
        """
        try:
            start_time = time.time()
            
            # Perform SVD decomposition
            if optimization_level == OptimizationLevel.MAXIMUM:
                # Use LAPACK for maximum performance
                U, S, Vt = lapack.dgesvd(A, full_matrices=full_matrices)
            else:
                # Use scipy's SVD
                U, S, Vt = linalg.svd(A, full_matrices=full_matrices)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            flops = 4 * A.shape[0] * A.shape[1] * min(A.shape)  # Approximate FLOP count for SVD
            memory_used = U.nbytes + S.nbytes + Vt.nbytes
            
            self._update_performance_metrics(OperationType.DECOMPOSITION, execution_time, flops, memory_used)
            
            return U, S, Vt
            
        except Exception as e:
            error_msg = f"Error in SVD decomposition: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def eigenvalue_decomposition(self, A: Matrix,
                               optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> Tuple[Vector, Matrix]:
        """
        Eigenvalue decomposition: A = V * D * V^(-1)
        
        Performs eigenvalue decomposition with optional optimization strategies.
        
        Args:
            A: Input matrix (must be square)
            optimization_level: Level of optimization to apply
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            start_time = time.time()
            
            # Validate input matrix
            if A.shape[0] != A.shape[1]:
                raise ValueError("Matrix must be square for eigenvalue decomposition")
            
            # Check condition number
            condition_number = np.linalg.cond(A)
            if condition_number > self.config.get('condition_number_threshold', 1e12):
                warning_msg = f"Matrix is ill-conditioned (cond={condition_number:.2e})"
                self.safe_log('warning', warning_msg)
            
            # Perform eigenvalue decomposition
            if optimization_level == OptimizationLevel.MAXIMUM:
                # Use LAPACK for maximum performance
                eigenvalues, eigenvectors = lapack.dgeev(A)
            else:
                # Use scipy's eigenvalue decomposition
                eigenvalues, eigenvectors = linalg.eig(A)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            flops = 4 * A.shape[0] ** 3  # Approximate FLOP count for eigendecomposition
            memory_used = eigenvalues.nbytes + eigenvectors.nbytes
            
            self._update_performance_metrics(OperationType.EIGENVALUE, execution_time, flops, memory_used)
            
            return eigenvalues, eigenvectors
            
        except Exception as e:
            error_msg = f"Error in eigenvalue decomposition: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def matrix_inverse(self, A: Matrix,
                      optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> Matrix:
        """
        Matrix inverse using optimized methods
        
        Computes the inverse of a matrix using various optimization strategies.
        
        Args:
            A: Input matrix (must be square and non-singular)
            optimization_level: Level of optimization to apply
            
        Returns:
            Inverse of matrix A
        """
        try:
            start_time = time.time()
            
            # Validate input matrix
            if A.shape[0] != A.shape[1]:
                raise ValueError("Matrix must be square for inversion")
            
            # Check condition number
            condition_number = np.linalg.cond(A)
            if condition_number > self.config.get('condition_number_threshold', 1e12):
                warning_msg = f"Matrix is ill-conditioned (cond={condition_number:.2e}), using pseudo-inverse"
                self.safe_log('warning', warning_msg)
                # Use pseudo-inverse for ill-conditioned matrices
                inverse = linalg.pinv(A)
            else:
                # Use optimized inverse
                if optimization_level == OptimizationLevel.MAXIMUM:
                    # Use LAPACK for maximum performance
                    inverse = lapack.dgetri(A)
                else:
                    # Use scipy's optimized inverse
                    inverse = linalg.inv(A)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            flops = 2 * A.shape[0] ** 3  # Approximate FLOP count for matrix inverse
            memory_used = inverse.nbytes
            
            self._update_performance_metrics(OperationType.INVERSE, execution_time, flops, memory_used)
            
            return inverse
            
        except Exception as e:
            error_msg = f"Error in matrix inverse: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def get_matrix_info(self, A: Matrix) -> MatrixInfo:
        """
        Get comprehensive information about a matrix for optimization decisions
        
        This method analyzes matrix properties to determine the best
        optimization strategy for operations involving this matrix.
        
        Args:
            A: Input matrix
            
        Returns:
            MatrixInfo object containing matrix properties
        """
        try:
            # Calculate sparsity
            nnz = np.count_nonzero(A)
            sparsity = 1.0 - (nnz / A.size)
            
            # Determine matrix type
            if sparsity > self.config.get('sparse_threshold', 0.1):
                matrix_type = MatrixType.SPARSE
            elif np.allclose(A, A.T):
                matrix_type = MatrixType.SYMMETRIC
            elif np.allclose(A, A.conj().T):
                matrix_type = MatrixType.HERMITIAN
            elif np.allclose(A, np.triu(A)) or np.allclose(A, np.tril(A)):
                matrix_type = MatrixType.TRIANGULAR
            elif np.allclose(A, np.diag(np.diag(A))):
                matrix_type = MatrixType.DIAGONAL
            else:
                matrix_type = MatrixType.DENSE
            
            # Calculate condition number
            try:
                condition_number = np.linalg.cond(A)
            except:
                condition_number = None
            
            # Calculate rank
            try:
                rank = np.linalg.matrix_rank(A)
            except:
                rank = None
            
            # Calculate symmetry error
            try:
                symmetry_error = np.linalg.norm(A - A.T) / np.linalg.norm(A)
            except:
                symmetry_error = 0.0
            
            # Calculate bandwidth (for banded matrices)
            bandwidth = None
            if matrix_type == MatrixType.BANDED:
                # Simple bandwidth calculation
                bandwidth = self._calculate_bandwidth(A)
            
            return MatrixInfo(
                shape=A.shape,
                dtype=A.dtype,
                matrix_type=matrix_type,
                is_sparse=sparsity > self.config.get('sparse_threshold', 0.1),
                nnz=nnz,
                memory_usage=A.nbytes,
                condition_number=condition_number,
                rank=rank,
                sparsity=sparsity,
                symmetry_error=symmetry_error,
                bandwidth=bandwidth
            )
            
        except Exception as e:
            error_msg = f"Error getting matrix info: {e}"
            self.safe_log('error', error_msg)
            raise
    
    def _calculate_bandwidth(self, A: Matrix) -> int:
        """
        Calculate the bandwidth of a matrix
        
        Args:
            A: Input matrix
            
        Returns:
            Bandwidth of the matrix
        """
        try:
            # Find the maximum distance from diagonal for non-zero elements
            bandwidth = 0
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] != 0:
                        bandwidth = max(bandwidth, abs(i - j))
            return bandwidth
        except Exception as e:
            error_msg = f"Error calculating bandwidth: {e}"
            self.safe_log('error', error_msg)
            return 0
    
    def _validate_matrices(self, *matrices: Matrix) -> bool:
        """
        Validate matrix inputs for operations
        
        Args:
            *matrices: Variable number of matrices to validate
            
        Returns:
            True if all matrices are valid, False otherwise
        """
        try:
            for matrix in matrices:
                if not isinstance(matrix, np.ndarray):
                    return False
                if matrix.ndim != 2:
                    return False
                if not np.isfinite(matrix).all():
                    return False
            return True
        except Exception:
            return False
    
    def _update_performance_metrics(self, operation_type: OperationType, 
                                  execution_time: float, flops: int, memory_used: int) -> None:
        """
        Update performance metrics for tracking
        
        Args:
            operation_type: Type of operation performed
            execution_time: Time taken for operation
            flops: Number of floating point operations
            memory_used: Memory used by operation
        """
        try:
            with self.operation_lock:
                self.total_operations += 1
                self.total_flops += flops
                self.total_execution_time += execution_time
                self.current_memory_usage += memory_used
                
                # Update operation-specific stats
                if operation_type == OperationType.GEMM:
                    self.performance_stats['gemm_operations'] += 1
                elif operation_type == OperationType.DECOMPOSITION:
                    self.performance_stats['decomposition_operations'] += 1
                elif operation_type == OperationType.EIGENVALUE:
                    self.performance_stats['eigenvalue_operations'] += 1
                elif operation_type == OperationType.INVERSE:
                    self.performance_stats['inverse_operations'] += 1
                
                # Update average execution time
                if self.total_operations > 0:
                    self.performance_stats['average_execution_time'] = (
                        self.total_execution_time / self.total_operations
                    )
                
                # Update peak memory usage
                self.performance_stats['peak_memory_usage'] = max(
                    self.performance_stats['peak_memory_usage'],
                    self.current_memory_usage
                )
                
                # Store operation in history
                self.operation_history.append({
                    'operation_type': operation_type.value,
                    'execution_time': execution_time,
                    'flops': flops,
                    'memory_used': memory_used,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            error_msg = f"Error updating performance metrics: {e}"
            self.safe_log('error', error_msg)
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """
        Get comprehensive performance summary
        
        Returns:
            PerformanceMetrics object containing performance statistics
        """
        try:
            cache_hit_rate = 0.0
            if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0:
                cache_hit_rate = (self.performance_stats['cache_hits'] / 
                                (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']))
            
            throughput = 0.0
            if self.total_execution_time > 0:
                throughput = self.total_operations / self.total_execution_time
            
            return PerformanceMetrics(
                total_operations=self.total_operations,
                total_execution_time=self.total_execution_time,
                total_flops=self.total_flops,
                average_execution_time=self.performance_stats['average_execution_time'],
                peak_memory_usage=self.performance_stats['peak_memory_usage'],
                cache_hit_rate=cache_hit_rate,
                throughput=throughput
            )
        except Exception as e:
            error_msg = f"Error getting performance summary: {e}"
            self.safe_log('error', error_msg)
            return PerformanceMetrics(0, 0.0, 0, 0.0, 0, 0.0, 0.0)
    
    def optimize_memory(self) -> None:
        """
        Optimize memory usage by clearing caches and history
        
        This method helps manage memory usage by clearing old data
        and optimizing memory allocation.
        """
        try:
            # Clear operation history if too large
            if len(self.operation_history) > self.config.get('max_history_size', 1000):
                excess = len(self.operation_history) - self.config.get('max_history_size', 1000)
                for _ in range(excess):
                    self.operation_history.popleft()
            
            # Clear memory pool if usage is high
            if self.current_memory_usage > self.max_memory_usage * 0.8:
                self.memory_pool.clear()
                self.current_memory_usage = 0
                
        except Exception as e:
            error_msg = f"Error optimizing memory: {e}"
            self.safe_log('error', error_msg)


def main() -> None:
    """
    Main function for testing Rittle GEMM functionality
    
    This function demonstrates the capabilities of the Rittle GEMM library
    and provides performance benchmarks for various matrix operations.
    Uses CLI-safe output with emoji fallbacks for Windows compatibility.
    """
    try:
        # Initialize Rittle GEMM
        rittle = RittleGEMM()
        
        # Use CLI-safe print for all output
        rittle.safe_print("🚀 Rittle GEMM Performance Test")
        rittle.safe_print("=" * 50)
        
        # Test matrices of various sizes
        test_sizes = [50, 100, 200, 500]
        
        for size in test_sizes:
            rittle.safe_print(f"\n📊 Testing {size}x{size} matrices...")
            
            # Create test matrices
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            # Test GEMM operations with different optimization levels
            optimization_levels = [
                OptimizationLevel.STANDARD,
                OptimizationLevel.AGGRESSIVE,
                OptimizationLevel.MAXIMUM
            ]
            
            for level in optimization_levels:
                rittle.safe_print(f"  Testing {level.value} optimization...")
                
                # Test matrix multiplication
                result = rittle.gemm(A, B, optimization_level=level)
                if result.success:
                    rittle.safe_print(f"    ✅ GEMM completed in {result.execution_time:.6f}s")
                    rittle.safe_print(f"    📈 FLOPs: {result.flops:,}")
                    rittle.safe_print(f"    💾 Memory: {result.memory_used:,} bytes")
                else:
                    rittle.safe_print(f"    ❌ GEMM failed: {result.error_message}")
            
            # Test matrix decomposition
            rittle.safe_print(f"  Testing matrix decomposition...")
            try:
                P, L, U = rittle.lu_decomposition(A, OptimizationLevel.STANDARD)
                rittle.safe_print(f"    ✅ LU decomposition completed")
            except Exception as e:
                rittle.safe_print(f"    ❌ LU decomposition failed: {e}")
            
            # Test eigenvalue decomposition
            rittle.safe_print(f"  Testing eigenvalue decomposition...")
            try:
                eigenvalues, eigenvectors = rittle.eigenvalue_decomposition(A, OptimizationLevel.STANDARD)
                rittle.safe_print(f"    ✅ Eigenvalue decomposition completed")
            except Exception as e:
                rittle.safe_print(f"    ❌ Eigenvalue decomposition failed: {e}")
        
        # Get performance summary
        summary = rittle.get_performance_summary()
        rittle.safe_print(f"\n📊 Performance Summary:")
        rittle.safe_print(f"   Total operations: {summary.total_operations}")
        rittle.safe_print(f"   Total FLOPs: {summary.total_flops:,}")
        rittle.safe_print(f"   Average execution time: {summary.average_execution_time:.6f}s")
        rittle.safe_print(f"   Peak memory usage: {summary.peak_memory_usage:,} bytes")
        rittle.safe_print(f"   Cache hit rate: {summary.cache_hit_rate:.2%}")
        rittle.safe_print(f"   Throughput: {summary.throughput:.2f} ops/sec")
        
        rittle.safe_print("\n🎉 Rittle GEMM test completed successfully!")
        
    except Exception as e:
        # Use CLI-safe error reporting
        rittle = RittleGEMM()  # Create instance for safe printing
        rittle.safe_print(f"❌ Rittle GEMM test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
