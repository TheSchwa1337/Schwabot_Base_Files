#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Fallback Manager for Schwabot Trading System
================================================

Provides intelligent GPU acceleration with automatic CPU fallback.
This module ensures all mathematical operations work regardless of CUDA availability.

Key Features:
- Automatic CuPy/CUDA detection and fallback
- Seamless transition between GPU and CPU operations
- Performance monitoring and optimization
- Cross-platform compatibility
- Mathematical integrity preservation
- Zero-downtime operation during GPU failures
"""

import logging
import os
import platform
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

logger = logging.getLogger(__name__)

# ============================================================================
# GPU DETECTION AND FALLBACK SYSTEM
# ============================================================================

class ComputeBackend(Enum):
    """Available computation backends."""
    CUPY = "cupy"
    NUMPY = "numpy"
    AUTO = "auto"

@dataclass
class GPUStatus:
    """GPU system status information."""
    cuda_available: bool = False
    cupy_available: bool = False
    gpu_memory_gb: float = 0.0
    gpu_name: str = "Unknown"
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    fallback_reason: str = ""

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation_name: str
    backend_used: ComputeBackend
    execution_time_ms: float
    memory_used_mb: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class GPUFallbackManager:
    """
    Intelligent GPU fallback manager for Schwabot trading system.
    
    This class provides seamless GPU acceleration with automatic CPU fallback,
    ensuring trading operations continue even when CUDA is unavailable.
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize the GPU fallback manager.
        
        Args:
            force_cpu: Force CPU-only mode (useful for testing/debugging)
        """
        self.force_cpu = force_cpu
        self.gpu_status = self._detect_gpu_status()
        self.performance_history: List[PerformanceMetrics] = []
        self.current_backend = self._select_backend()
        self.xp = self._get_array_library()
        
        logger.info(f"GPU Fallback Manager initialized with backend: {self.current_backend.value}")
        logger.info(f"GPU Status: {self.gpu_status}")
    
    def _detect_gpu_status(self) -> GPUStatus:
        """Detect GPU and CUDA availability."""
        status = GPUStatus()
        
        # Check if CUDA environment variables are set
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home:
            logger.info(f"CUDA environment detected: {cuda_home}")
        
        # Try to import CuPy
        try:
            import cupy as cp
            status.cupy_available = True
            
            # Get GPU information
            try:
                mempool = cp.get_default_memory_pool()
                status.gpu_memory_gb = mempool.get_limit() / (1024**3)
                
                # Get device info
                device = cp.cuda.Device()
                status.gpu_name = device.attributes['Name'].decode('utf-8')
                status.compute_capability = f"{device.compute_capability[0]}.{device.compute_capability[1]}"
                
                logger.info(f"CuPy GPU detected: {status.gpu_name}")
                logger.info(f"GPU Memory: {status.gpu_memory_gb:.2f} GB")
                logger.info(f"Compute Capability: {status.compute_capability}")
                
            except Exception as e:
                logger.warning(f"Could not get detailed GPU info: {e}")
                
        except ImportError as e:
            status.fallback_reason = f"CuPy not available: {e}"
            logger.info(f"CuPy not available, falling back to NumPy: {e}")
        except Exception as e:
            status.fallback_reason = f"CuPy import error: {e}"
            logger.warning(f"CuPy import failed: {e}")
        
        # Check CUDA availability
        try:
            import cupy.cuda.runtime as runtime
            status.cuda_available = True
            status.cuda_version = runtime.runtimeGetVersion()
            logger.info(f"CUDA Runtime version: {status.cuda_version}")
        except:
            status.cuda_available = False
            logger.info("CUDA runtime not available")
        
        return status
    
    def _select_backend(self) -> ComputeBackend:
        """Select the appropriate computation backend."""
        if self.force_cpu:
            logger.info("Forcing CPU-only mode")
            return ComputeBackend.NUMPY
        
        if self.gpu_status.cupy_available and self.gpu_status.cuda_available:
            logger.info("GPU acceleration available")
            return ComputeBackend.CUPY
        else:
            logger.info("GPU not available, using CPU fallback")
            return ComputeBackend.NUMPY
    
    def _get_array_library(self):
        """Get the appropriate array library (CuPy or NumPy)."""
        if self.current_backend == ComputeBackend.CUPY:
            try:
                import cupy as cp
                logger.info("Using CuPy for GPU acceleration")
                return cp
            except ImportError:
                logger.warning("CuPy import failed, falling back to NumPy")
                return np
        else:
            logger.info("Using NumPy for CPU computation")
            return np
    
    def safe_operation(self, operation_name: str, operation_func: Callable, 
                      fallback_func: Optional[Callable] = None, *args, **kwargs) -> Any:
        """
        Execute an operation with automatic fallback.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Primary operation function
            fallback_func: Fallback operation function (optional)
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Result of the operation
        """
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            # Try primary operation
            result = operation_func(*args, **kwargs)
            
            # Record success
            execution_time = (time.time() - start_time) * 1000
            memory_used = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)
            
            self._record_performance(operation_name, self.current_backend, 
                                   execution_time, memory_used, True)
            
            return result
            
        except Exception as e:
            # Primary operation failed, try fallback
            logger.warning(f"Primary operation '{operation_name}' failed: {e}")
            
            if fallback_func is not None:
                try:
                    logger.info(f"Attempting fallback for '{operation_name}'")
                    result = fallback_func(*args, **kwargs)
                    
                    execution_time = (time.time() - start_time) * 1000
                    memory_used = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)
                    
                    self._record_performance(operation_name, ComputeBackend.NUMPY, 
                                           execution_time, memory_used, True)
                    
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback operation also failed: {fallback_error}")
                    self._record_performance(operation_name, self.current_backend, 
                                           (time.time() - start_time) * 1000, 0, False, str(fallback_error))
                    raise fallback_error
            else:
                execution_time = (time.time() - start_time) * 1000
                self._record_performance(operation_name, self.current_backend, 
                                       execution_time, 0, False, str(e))
                raise e
    
    def _record_performance(self, operation_name: str, backend: ComputeBackend, 
                          execution_time: float, memory_used: float, 
                          success: bool, error_message: Optional[str] = None):
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            backend_used=backend,
            execution_time_ms=execution_time,
            memory_used_mb=memory_used,
            success=success,
            error_message=error_message
        )
        self.performance_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        successful_ops = [m for m in self.performance_history if m.success]
        failed_ops = [m for m in self.performance_history if not m.success]
        
        stats = {
            "total_operations": len(self.performance_history),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.performance_history) if self.performance_history else 0,
            "average_execution_time_ms": np.mean([m.execution_time_ms for m in successful_ops]) if successful_ops else 0,
            "average_memory_usage_mb": np.mean([m.memory_used_mb for m in successful_ops]) if successful_ops else 0,
            "backend_usage": {
                backend.value: len([m for m in self.performance_history if m.backend_used == backend])
                for backend in ComputeBackend
            }
        }
        
        return stats
    
    def switch_backend(self, backend: ComputeBackend) -> bool:
        """Switch computation backend."""
        if backend == self.current_backend:
            return True
        
        try:
            if backend == ComputeBackend.CUPY:
                import cupy as cp
                self.xp = cp
                self.current_backend = ComputeBackend.CUPY
                logger.info("Switched to CuPy backend")
            else:
                self.xp = np
                self.current_backend = ComputeBackend.NUMPY
                logger.info("Switched to NumPy backend")
            
            return True
        except ImportError as e:
            logger.warning(f"Could not switch to {backend.value}: {e}")
            return False

# ============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS
# ============================================================================

# Global GPU fallback manager instance
_gpu_manager: Optional[GPUFallbackManager] = None

def get_gpu_manager() -> GPUFallbackManager:
    """Get the global GPU fallback manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUFallbackManager()
    return _gpu_manager

def safe_array_operation(operation_name: str, operation_func: Callable, 
                        fallback_func: Optional[Callable] = None, *args, **kwargs) -> Any:
    """Convenience function for safe array operations."""
    manager = get_gpu_manager()
    return manager.safe_operation(operation_name, operation_func, fallback_func, *args, **kwargs)

def get_array_library():
    """Get the current array library (CuPy or NumPy)."""
    manager = get_gpu_manager()
    return manager.xp

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    manager = get_gpu_manager()
    return manager.current_backend == ComputeBackend.CUPY

def get_gpu_status() -> GPUStatus:
    """Get current GPU status."""
    manager = get_gpu_manager()
    return manager.gpu_status

def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics."""
    manager = get_gpu_manager()
    return manager.get_performance_stats()

# ============================================================================
# COMMON MATHEMATICAL OPERATIONS WITH FALLBACK
# ============================================================================

def safe_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Safe matrix multiplication with GPU fallback."""
    xp = get_array_library()
    
    def gpu_multiply(a, b):
        return xp.dot(a, b)
    
    def cpu_multiply(a, b):
        return np.dot(a, b)
    
    return safe_array_operation("matrix_multiply", gpu_multiply, cpu_multiply, A, B)

def safe_fft(data: np.ndarray) -> np.ndarray:
    """Safe FFT with GPU fallback."""
    xp = get_array_library()
    
    def gpu_fft(d):
        return xp.fft.fft(d)
    
    def cpu_fft(d):
        return np.fft.fft(d)
    
    return safe_array_operation("fft", gpu_fft, cpu_fft, data)

def safe_eigenvalues(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Safe eigenvalue decomposition with GPU fallback."""
    xp = get_array_library()
    
    def gpu_eig(a):
        return xp.linalg.eig(a)
    
    def cpu_eig(a):
        return np.linalg.eig(a)
    
    return safe_array_operation("eigenvalues", gpu_eig, cpu_eig, A)

def safe_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Safe SVD decomposition with GPU fallback."""
    xp = get_array_library()
    
    def gpu_svd(a):
        return xp.linalg.svd(a)
    
    def cpu_svd(a):
        return np.linalg.svd(a)
    
    return safe_array_operation("svd", gpu_svd, cpu_svd, A)

# ============================================================================
# INITIALIZATION AND STATUS REPORTING
# ============================================================================

def initialize_gpu_system(force_cpu: bool = False) -> GPUFallbackManager:
    """Initialize the GPU system with optional CPU forcing."""
    global _gpu_manager
    _gpu_manager = GPUFallbackManager(force_cpu=force_cpu)
    return _gpu_manager

def report_system_status():
    """Report comprehensive system status."""
    manager = get_gpu_manager()
    status = manager.gpu_status
    stats = manager.get_performance_stats()
    
    print("=" * 60)
    print("SCHWABOT GPU FALLBACK SYSTEM STATUS")
    print("=" * 60)
    print(f"Current Backend: {manager.current_backend.value}")
    print(f"CUDA Available: {status.cuda_available}")
    print(f"CuPy Available: {status.cupy_available}")
    print(f"GPU Name: {status.gpu_name}")
    print(f"GPU Memory: {status.gpu_memory_gb:.2f} GB")
    print(f"Compute Capability: {status.compute_capability}")
    print(f"Fallback Reason: {status.fallback_reason}")
    print("-" * 60)
    print("PERFORMANCE STATISTICS")
    print("-" * 60)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    # Test the GPU fallback system
    initialize_gpu_system()
    report_system_status() 