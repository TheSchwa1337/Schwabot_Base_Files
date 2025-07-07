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
- System-aware hardware scaling and fit testing
"""

import logging
import time
import warnings
import hashlib
import json
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


@dataclass
class SystemFitProfile:
    """System-aware hardware profile for GPU scaling and fit testing."""
    
    gpu_tier: str
    device_type: str
    matrix_size: int
    precision: str
    system_hash: str
    gpu_hash: str
    can_run_gpu_logic: bool
    memory_gb: float = 0.0
    compute_capability: str = ""
    max_threads_per_block: int = 0
    max_blocks_per_grid: int = 0


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


def build_system_fit_profile() -> SystemFitProfile:
    """Build system-aware hardware profile for GPU scaling."""
    
    # Default CPU profile
    cpu_profile = {
        "cores": 4,
        "memory_gb": 8.0,
        "architecture": "x86_64"
    }
    
    # Default GPU profile
    gpu_profile = {
        "tier": "TIER_LOW",
        "memory_gb": 2.0,
        "compute_capability": "3.5",
        "matrix_size": 16,
        "use_half_precision": False,
        "max_threads_per_block": 1024,
        "max_blocks_per_grid": 65535
    }
    
    device_type = "DESKTOP"  # Default
    
    # Try to detect actual GPU capabilities
    try:
        if USING_CUDA:
            import cupy as cp
            mem_info = cp.cuda.runtime.memGetInfo()
            gpu_profile["memory_gb"] = mem_info[1] / (1024**3)  # Total memory in GB
            
            # Determine GPU tier based on memory
            if gpu_profile["memory_gb"] >= 8:
                gpu_profile["tier"] = "TIER_ULTRA"
                gpu_profile["matrix_size"] = 64
                gpu_profile["use_half_precision"] = True
            elif gpu_profile["memory_gb"] >= 4:
                gpu_profile["tier"] = "TIER_HIGH"
                gpu_profile["matrix_size"] = 32
                gpu_profile["use_half_precision"] = True
            elif gpu_profile["memory_gb"] >= 2:
                gpu_profile["tier"] = "TIER_MID"
                gpu_profile["matrix_size"] = 24
            else:
                gpu_profile["tier"] = "TIER_LOW"
                gpu_profile["matrix_size"] = 16
                
    except Exception as e:
        logger.warning(f"Could not detect GPU capabilities: {e}")
    
    # Try to detect CPU profile
    try:
        import psutil
        cpu_profile["cores"] = psutil.cpu_count()
        cpu_profile["memory_gb"] = psutil.virtual_memory().total / (1024**3)
        
        # Determine device type
        if cpu_profile["memory_gb"] < 4:
            device_type = "EMBEDDED"
        elif cpu_profile["memory_gb"] < 8:
            device_type = "LAPTOP"
        else:
            device_type = "DESKTOP"
            
    except ImportError:
        logger.warning("psutil not available - using default CPU profile")
    
    # Create combined profile
    combined = {
        "gpu": gpu_profile,
        "cpu": cpu_profile,
        "device_type": device_type
    }
    
    system_hash = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()
    gpu_hash = hashlib.sha256(json.dumps(gpu_profile, sort_keys=True).encode()).hexdigest()
    
    # Determine if GPU logic can run
    can_run_gpu_logic = gpu_profile["tier"] in ["TIER_MID", "TIER_HIGH", "TIER_ULTRA"]
    
    precision = 'half' if gpu_profile.get('use_half_precision') else 'float'
    
    return SystemFitProfile(
        gpu_tier=gpu_profile['tier'],
        device_type=device_type,
        matrix_size=gpu_profile['matrix_size'],
        precision=precision,
        system_hash=system_hash,
        gpu_hash=gpu_hash,
        can_run_gpu_logic=can_run_gpu_logic,
        memory_gb=gpu_profile['memory_gb'],
        compute_capability=gpu_profile['compute_capability'],
        max_threads_per_block=gpu_profile['max_threads_per_block'],
        max_blocks_per_grid=gpu_profile['max_blocks_per_grid']
    )


def test_matrix_fit() -> bool:
    """Test if matrix operations fit the current system profile."""
    try:
        A = xp.random.rand(FIT_PROFILE.matrix_size, FIT_PROFILE.matrix_size)
        B = xp.random.rand(FIT_PROFILE.matrix_size, FIT_PROFILE.matrix_size)
        result = xp.dot(A, B)
        assert result.shape == (FIT_PROFILE.matrix_size, FIT_PROFILE.matrix_size)
        logger.info(f"✅ Matrix fit test passed: {FIT_PROFILE.matrix_size}x{FIT_PROFILE.matrix_size}")
        return True
    except Exception as e:
        logger.warning(f"❌ Matrix fit test failed: {str(e)}")
        return False


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

# Build system fit profile
FIT_PROFILE = build_system_fit_profile()

# Log system profile
logger.info(f"🧠 Detected GPU Tier: {FIT_PROFILE.gpu_tier}")
logger.info(f"🧠 Device Type: {FIT_PROFILE.device_type}")
logger.info(f"🧠 Matrix Ops Size: {FIT_PROFILE.matrix_size} ({FIT_PROFILE.precision}-precision)")
logger.info(f"🧠 System Hash: {FIT_PROFILE.system_hash[:12]}...")
logger.info(f"🧠 GPU Memory: {FIT_PROFILE.memory_gb:.1f}GB")

# Test matrix fit
if USING_CUDA:
    test_matrix_fit()


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
    "SystemFitProfile",
    "FIT_PROFILE",
    "build_system_fit_profile",
    "test_matrix_fit",
]
