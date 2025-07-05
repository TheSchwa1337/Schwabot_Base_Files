#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA Helper Utility for Schwabot Trading System

Provides intelligent CUDA/GPU acceleration with automatic CPU fallback.
This module ensures all mathematical operations work regardless of CUDA availability.

Key Features:
- Automatic CUDA detection (CuPy, PyTorch, Numba)
- Seamless fallback to CPU when GPU operations fail
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
- Mathematical integrity preservation
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, optimize, stats
from scipy.fft import fft, fftfreq, ifft
from scipy.sparse import csr_matrix, lil_matrix

logger = logging.getLogger(__name__)


class ComputeMode(Enum):
    """Available computation modes."""

    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class FallbackMetrics:
    """Metrics for fallback performance tracking."""

    timestamp: float
    operation: str
    original_mode: ComputeMode
    fallback_mode: ComputeMode
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    performance_ratio: float = 1.0


class CUDADetector:
    """Detects CUDA availability and manages fallback logic."""

    def __init__(self):
        self.cuda_available = False
        self.cupy_available = False
        self.torch_available = False
        self.numba_cuda_available = False
        self.detected_devices = []
        self._detect_cuda()

    def _detect_cuda(self):
        """Detect available CUDA implementations."""
        # Try CuPy
        try:
            import cupy as cp

            self.cupy_available = True
            self.cuda_available = True
            logger.info("CuPy CUDA acceleration detected")
        except ImportError:
            logger.info("CuPy not available")

        # Try PyTorch
        try:
            import torch

            if torch.cuda.is_available():
                self.torch_available = True
                self.cuda_available = True
                self.detected_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                logger.info(
                    f"PyTorch CUDA acceleration detected: {
                        self.detected_devices}"
                )
        except ImportError:
            logger.info("PyTorch not available")

        # Try Numba CUDA
        try:
            from numba import cuda

            if cuda.is_available():
                self.numba_cuda_available = True
                self.cuda_available = True
                logger.info("Numba CUDA acceleration detected")
        except ImportError:
            logger.info("Numba CUDA not available")

        if not self.cuda_available:
            logger.warning("No CUDA acceleration available - using CPU fallback")

    def get_available_modes(self) -> List[ComputeMode]:
        """Get list of available computation modes."""
        modes = [ComputeMode.CPU]
        if self.cuda_available:
            modes.extend([ComputeMode.CUDA, ComputeMode.AUTO])
        return modes

    def get_status(self) -> Dict[str, Any]:
        """Get current CUDA detection status."""
        return {
            "cuda_available": self.cuda_available,
            "cupy_available": self.cupy_available,
            "torch_available": self.torch_available,
            "numba_cuda_available": self.numba_cuda_available,
            "detected_devices": self.detected_devices,
            "available_modes": [mode.value for mode in self.get_available_modes()],
        }


# Global CUDA detector instance
_cuda_detector = CUDADetector()

# Set up the primary array library
try:
    import cupy as cp

    xp = cp
    USING_CUDA = True
    logger.info("⚡ CUDA Acceleration Enabled (CuPy)")
except ImportError:
    xp = np
    USING_CUDA = False
    logger.info("🔄 CPU Fallback Mode Active (NumPy)")


def safe_cuda_operation(operation: Callable, fallback_operation: Optional[Callable] = None) -> Any:
    """
    Execute a CUDA operation with automatic fallback to CPU.

    Args:
        operation: The CUDA operation to attempt
        fallback_operation: Optional CPU fallback operation

    Returns:
        Result of the operation (GPU or CPU)
    """
    start_time = time.time()

    try:
        if USING_CUDA:
            result = operation()
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"CUDA operation completed in {execution_time:.2f}ms")
            return result
        else:
            # Force CPU mode
            if fallback_operation:
                result = fallback_operation()
            else:
                result = operation()
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"CPU operation completed in {execution_time:.2f}ms")
            return result

    except Exception as e:
        logger.warning(f"CUDA operation failed, falling back to CPU: {e}")

        # Execute CPU fallback
        try:
            if fallback_operation:
                result = fallback_operation()
            else:
                result = operation()

            execution_time = (time.time() - start_time) * 1000
            logger.info(f"CPU fallback completed in {execution_time:.2f}ms")
            return result

        except Exception as cpu_error:
            logger.error(f"Both CUDA and CPU operations failed: {cpu_error}")
            raise


def get_cuda_status() -> Dict[str, Any]:
    """Get comprehensive CUDA status information."""
    return {
        "using_cuda": USING_CUDA,
        "detector_status": _cuda_detector.get_status(),
        "primary_library": "cupy" if USING_CUDA else "numpy",
        "timestamp": time.time(),
    }


def report_cuda_status():
    """Print CUDA status to console."""
    status = get_cuda_status()
    if status["using_cuda"]:
        print("⚡ CUDA Acceleration Enabled")
        print(f"   - Primary Library: {status['primary_library']}")
        print(f"   - Available Devices: {status['detector_status']['detected_devices']}")
    else:
        print("🔄 CPU Fallback Mode Active")
        print(f"   - Primary Library: {status['primary_library']}")


# Convenience functions for common operations


def safe_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Safe matrix multiplication with CUDA fallback."""
    return safe_cuda_operation(lambda: xp.dot(A, B), lambda: np.dot(A, B))


def safe_tensor_contraction(
    A: np.ndarray, B: np.ndarray, axes: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Safe tensor contraction with CUDA fallback."""
    return safe_cuda_operation(
        lambda: xp.tensordot(A, B, axes=axes), lambda: np.tensordot(A, B, axes=axes)
    )


def safe_fft(data: np.ndarray) -> np.ndarray:
    """Safe FFT with CUDA fallback."""
    return safe_cuda_operation(lambda: xp.fft.fft(data), lambda: fft(data))


def safe_convolution(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Safe convolution with CUDA fallback."""
    return safe_cuda_operation(
        lambda: xp.convolve(data, kernel, mode="same"),
        lambda: np.convolve(data, kernel, mode="same"),
    )


def safe_eigenvalue_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Safe eigenvalue decomposition with CUDA fallback."""
    return safe_cuda_operation(
        lambda: (xp.linalg.eigvals(A), xp.linalg.eigh(A)[1]), lambda: linalg.eigh(A)
    )


def safe_matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Safe matrix inverse with CUDA fallback."""
    return safe_cuda_operation(lambda: xp.linalg.inv(A), lambda: linalg.inv(A))


def safe_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Safe SVD decomposition with CUDA fallback."""
    return safe_cuda_operation(lambda: xp.linalg.svd(A), lambda: linalg.svd(A))


# Export key variables and functions
__all__ = [
    "xp",
    "USING_CUDA",
    "safe_cuda_operation",
    "get_cuda_status",
    "report_cuda_status",
    "safe_matrix_multiply",
    "safe_tensor_contraction",
    "safe_fft",
    "safe_convolution",
    "safe_eigenvalue_decomposition",
    "safe_matrix_inverse",
    "safe_svd",
    "ComputeMode",
    "FallbackMetrics",
    "CUDADetector",
]
