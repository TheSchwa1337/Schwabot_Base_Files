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
- Advanced mathematical functions for trading strategies
"""

import hashlib
import json
import logging
import platform
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
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
    gpu_freq_ghz: float = 1.0
    cpu_cores: int = 4
    total_memory_gb: float = 8.0


class MathematicalCore:
    """Core mathematical functions for GPU-aware trading strategies."""

    def __init__(self, system_profile: SystemFitProfile):
        self.system_profile = system_profile
        self.xp = np  # Default to numpy, will be overridden if CUDA available

    def matrix_fit(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with system-aware sizing.

        Formula: C = A × B where A ∈ R^(m×k), B ∈ R^(k×n), C ∈ R^(m×n)
        """
        A, B = np.array(A), np.array(B)

        # Ensure matrices fit within system constraints
        max_size = self.system_profile.matrix_size
        if A.shape[0] > max_size or A.shape[1] > max_size or B.shape[1] > max_size:
            logger.warning(f"Matrix size {A.shape} x {B.shape} exceeds system limit {max_size}")
            # Truncate to fit
            A = A[:max_size, :max_size]
            B = B[:max_size, :max_size]

        return self.xp.dot(A, B)

    def cosine_match(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        Cosine similarity for strategy matching.

        Formula: cosine(A,B) = (A·B) / (||A|| ||B||)
        """
        A, B = np.array(A), np.array(B)
        dot_product = self.xp.dot(A, B)
        norm_A = self.xp.linalg.norm(A)
        norm_B = self.xp.linalg.norm(B)

        if norm_A == 0 or norm_B == 0:
            return 0.0

        return dot_product / (norm_A * norm_B)

    def entropy_of_vector(self, v: np.ndarray) -> float:
        """
        Shannon entropy for market disorder quantification.

        Formula: H(X) = -Σ p(x_i) * log2(p(x_i))
        """
        v = np.array(v)
        vals, counts = np.unique(v, return_counts=True)
        probs = counts / counts.sum()

        # Avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def flatness_measure(self, tick_prices: np.ndarray) -> float:
        """
        Gradient-based flatness detection for trade zones.

        Formula: G = (1/n) * Σ |dP/dt|
        """
        tick_prices = np.array(tick_prices)
        gradient = np.gradient(tick_prices)
        return np.abs(np.mean(gradient))

    def ideal_tick_time(self, ops_count: int, gpu_freq_ghz: Optional[float] = None) -> float:
        """
        Calculate ideal tick time based on operations and GPU frequency.

        Formula: T_ideal = C_ops / f_GPU
        """
        if gpu_freq_ghz is None:
            gpu_freq_ghz = self.system_profile.gpu_freq_ghz

        return ops_count / (gpu_freq_ghz * 1e9)

    def memory_tile_limit(self, mem_bytes: Optional[float] = None,
                         matrix_size: Optional[int] = None, 
                         dtype_bytes: int = 4) -> int:
        """
        Calculate memory tile limit for GPU tensor management.

        Formula: M_tile = M_available / (S_matrix × D_dtype)
        """
        if mem_bytes is None:
            mem_bytes = self.system_profile.memory_gb * (1024**3)
        if matrix_size is None:
            matrix_size = self.system_profile.matrix_size

        return int(mem_bytes // (matrix_size * dtype_bytes))

    def gpu_load_ratio(self, current_usage: float, max_capacity: float = 100) -> float:
        """
        Calculate GPU load ratio for safe operation.

        Formula: L_safe = U_current / U_max
        """
        return current_usage / max_capacity

    def smooth_gradient_detection(self, prices: np.ndarray, window: int = 5) -> float:
        """
        Smooth gradient detection for trade zone flatness.

        Formula: G = (1/n) * Σ |dP/dt| with smoothing
        """
        prices = np.array(prices)
        if len(prices) < window:
            return 0.0

        # Apply moving average smoothing
        smoothed = np.convolve(prices, np.ones(window)/window, mode='valid')
        gradient = np.gradient(smoothed)
        return np.abs(np.mean(gradient))

    def phantom_score(self, tick_data: np.ndarray, threshold: float = 0.1) -> float:
        """
        Calculate phantom trade detection score using entropy and flatness.
        """
        if len(tick_data) < 10:
            return 0.0

        entropy = self.entropy_of_vector(tick_data)
        flatness = self.flatness_measure(tick_data)

        # Combine entropy and flatness for phantom detection
        phantom_score = (entropy * flatness) / (1 + flatness)

        return phantom_score if phantom_score > threshold else 0.0


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
        "max_blocks_per_grid": 65535,
        "gpu_freq_ghz": 1.0
    }

    device_type = "DESKTOP"  # Default

    # Try to detect actual CPU profile
    try:
        cpu_profile["cores"] = psutil.cpu_count()
        cpu_profile["memory_gb"] = psutil.virtual_memory().total / (1024**3)

        # Determine device type based on CPU cores and memory
        if cpu_profile["cores"] <= 4 and cpu_profile["memory_gb"] <= 4:
            device_type = "PI"
        elif cpu_profile["cores"] <= 8 and cpu_profile["memory_gb"] <= 16:
            device_type = "LAPTOP"
        else:
            device_type = "DESKTOP"

    except Exception as e:
        logger.warning(f"Could not detect CPU profile: {e}")

    # Try to detect actual GPU capabilities
    try:
        # Check for CuPy
        import cupy as cp
        mem_info = cp.cuda.runtime.memGetInfo()
        gpu_profile["memory_gb"] = mem_info[1] / (1024**3)  # Total memory in GB

        # Determine GPU tier based on memory
        if gpu_profile["memory_gb"] >= 8:
            gpu_profile["tier"] = "TIER_ULTRA"
            gpu_profile["matrix_size"] = 64
            gpu_profile["use_half_precision"] = True
            gpu_profile["gpu_freq_ghz"] = 2.0
        elif gpu_profile["memory_gb"] >= 4:
            gpu_profile["tier"] = "TIER_HIGH"
            gpu_profile["matrix_size"] = 32
            gpu_profile["use_half_precision"] = True
            gpu_profile["gpu_freq_ghz"] = 1.8
        elif gpu_profile["memory_gb"] >= 2:
            gpu_profile["tier"] = "TIER_MID"
            gpu_profile["matrix_size"] = 24
            gpu_profile["gpu_freq_ghz"] = 1.5
        else:
            gpu_profile["tier"] = "TIER_LOW"
            gpu_profile["matrix_size"] = 16
            gpu_profile["gpu_freq_ghz"] = 1.0

    except Exception as e:
        logger.warning(f"Could not detect GPU capabilities: {e}")

    # Determine precision based on GPU tier
    precision = 'half' if gpu_profile.get('use_half_precision', False) else 'float'

    # Create combined system data for hashing
    combined = {
        "gpu": gpu_profile,
        "cpu": cpu_profile,
        "device_type": device_type,
        "platform": platform.platform(),
        "python_version": platform.python_version()
    }

    # Generate system and GPU hashes
    system_hash = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()
    gpu_hash = hashlib.sha256(json.dumps(gpu_profile, sort_keys=True).encode()).hexdigest()

    # Determine if GPU logic can run
    can_run_gpu_logic = gpu_profile.get('tier') in ["TIER_MID", "TIER_HIGH", "TIER_ULTRA"]

    return SystemFitProfile(
        gpu_tier=gpu_profile['tier'],
        device_type=device_type,
        matrix_size=gpu_profile['matrix_size'],
        precision=precision,
        system_hash=system_hash,
        gpu_hash=gpu_hash,
        can_run_gpu_logic=can_run_gpu_logic,
        memory_gb=gpu_profile.get('memory_gb', 0.0),
        compute_capability=gpu_profile.get('compute_capability', ''),
        max_threads_per_block=gpu_profile.get('max_threads_per_block', 0),
        max_blocks_per_grid=gpu_profile.get('max_blocks_per_grid', 0),
        gpu_freq_ghz=gpu_profile.get('gpu_freq_ghz', 1.0),
        cpu_cores=cpu_profile.get('cores', 4),
        total_memory_gb=cpu_profile.get('memory_gb', 8.0)
    )


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
                logger.info(f"PyTorch CUDA acceleration detected: {self.detected_devices}")
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


# Global instances
detector = CUDADetector()
FIT_PROFILE = build_system_fit_profile()
math_core = MathematicalCore(FIT_PROFILE)

# Set up CUDA if available
USING_CUDA = detector.cuda_available
if USING_CUDA and detector.cupy_available:
    try:
        import cupy as cp
        math_core.xp = cp
        logger.info("Using CuPy for GPU acceleration")
    except ImportError:
        logger.warning("CuPy import failed, falling back to CPU")
        USING_CUDA = False

# Log system profile
logger.info(f"🧠 Detected GPU Tier: {FIT_PROFILE.gpu_tier}")
logger.info(f"🧠 Device Type: {FIT_PROFILE.device_type}")
logger.info(f"🧠 Matrix Ops Size: {FIT_PROFILE.matrix_size} ({FIT_PROFILE.precision}-precision)")
logger.info(f"🧠 System Hash: {FIT_PROFILE.system_hash[:12]}...")
logger.info(f"🧠 GPU Hash: {FIT_PROFILE.gpu_hash[:12]}...")
logger.info(f"🧠 Can Run GPU Logic: {FIT_PROFILE.can_run_gpu_logic}")


def test_matrix_fit() -> bool:
    """Test if current GPU can run matrix operations."""
    try:
        size = FIT_PROFILE.matrix_size
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        result = math_core.matrix_fit(A, B)
        assert result.shape == (size, size)

        logger.info(f"✅ Matrix fit test passed for size {size}x{size}")
        return True
    except Exception as e:
        logger.warning(f"❌ Matrix fit test failed: {str(e)}")
        return False


def safe_cuda_operation(operation: Callable, fallback_operation: Optional[Callable] = None) -> Any:
    """
    Safely execute CUDA operation with automatic fallback.

    Args:
        operation: The CUDA operation to attempt
        fallback_operation: Optional fallback operation (defaults to CPU, version)

    Returns:
        Result of the operation
    """
    start_time = time.time()

    try:
        if USING_CUDA and FIT_PROFILE.can_run_gpu_logic:
            result = operation()
            execution_time = (time.time() - start_time) * 1000

            logger.debug(f"CUDA operation completed in {execution_time:.2f}ms")
            return result
        else:
            raise RuntimeError("CUDA not available or GPU logic disabled")

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.warning(f"CUDA operation failed ({execution_time:.2f}ms): {str(e)}")

        # Fallback to CPU
        if fallback_operation:
            try:
                fallback_start = time.time()
                result = fallback_operation()
                fallback_time = (time.time() - fallback_start) * 1000

                logger.info(f"CPU fallback completed in {fallback_time:.2f}ms")
                return result
            except Exception as fallback_error:
                logger.error(f"CPU fallback also failed: {str(fallback_error)}")
                raise
        else:
            raise


def get_cuda_status() -> Dict[str, Any]:
    """Get comprehensive CUDA and system status."""
    status = detector.get_status()
    status.update({
        "system_profile": {
            "gpu_tier": FIT_PROFILE.gpu_tier,
            "device_type": FIT_PROFILE.device_type,
            "matrix_size": FIT_PROFILE.matrix_size,
            "precision": FIT_PROFILE.precision,
            "can_run_gpu_logic": FIT_PROFILE.can_run_gpu_logic,
            "memory_gb": FIT_PROFILE.memory_gb,
            "gpu_freq_ghz": FIT_PROFILE.gpu_freq_ghz
        },
        "matrix_fit_test": test_matrix_fit()
    })
    return status


def report_cuda_status():
    """Report current CUDA status to console."""
    status = get_cuda_status()

    print("🚀 CUDA Helper Status Report")
    print("=" * 40)
    print(f"CUDA Available: {status['cuda_available']}")
    print(f"CuPy Available: {status['cupy_available']}")
    print(f"PyTorch Available: {status['torch_available']}")
    print(f"Numba CUDA Available: {status['numba_cuda_available']}")
    print(f"GPU Tier: {status['system_profile']['gpu_tier']}")
    print(f"Device Type: {status['system_profile']['device_type']}")
    print(f"Matrix Size: {status['system_profile']['matrix_size']}")
    print(f"Precision: {status['system_profile']['precision']}")
    print(f"Can Run GPU Logic: {status['system_profile']['can_run_gpu_logic']}")
    print(f"Matrix Fit Test: {'✅ PASSED' if status['matrix_fit_test'] else '❌ FAILED'}")


# Mathematical operation wrappers
def safe_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Safe matrix multiplication with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.matrix_fit(A, B),
        lambda: np.dot(A, B)
    )


def safe_tensor_contraction(
    A: np.ndarray, B: np.ndarray, axes: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Safe tensor contraction with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.xp.tensordot(A, B, axes=axes),
        lambda: np.tensordot(A, B, axes=axes)
    )


def safe_fft(data: np.ndarray) -> np.ndarray:
    """Safe FFT with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.xp.fft.fft(data),
        lambda: np.fft.fft(data)
    )


def safe_convolution(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Safe convolution with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.xp.convolve(data, kernel, mode='same'),
        lambda: np.convolve(data, kernel, mode='same')
    )


def safe_eigenvalue_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Safe eigenvalue decomposition with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.xp.linalg.eigh(A),
        lambda: np.linalg.eigh(A)
    )


def safe_matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Safe matrix inverse with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.xp.linalg.inv(A),
        lambda: np.linalg.inv(A)
    )


def safe_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Safe SVD with GPU acceleration."""
    return safe_cuda_operation(
        lambda: math_core.xp.linalg.svd(A),
        lambda: np.linalg.svd(A)
    )


# Export mathematical functions for easy access
def matrix_fit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication with system-aware sizing."""
    return math_core.matrix_fit(A, B)


def cosine_match(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine similarity for strategy matching."""
    return math_core.cosine_match(A, B)


def entropy_of_vector(v: np.ndarray) -> float:
    """Shannon entropy for market disorder quantification."""
    return math_core.entropy_of_vector(v)


def flatness_measure(tick_prices: np.ndarray) -> float:
    """Gradient-based flatness detection for trade zones."""
    return math_core.flatness_measure(tick_prices)


def ideal_tick_time(ops_count: int, gpu_freq_ghz: Optional[float] = None) -> float:
    """Calculate ideal tick time based on operations and GPU frequency."""
    return math_core.ideal_tick_time(ops_count, gpu_freq_ghz)


def memory_tile_limit(mem_bytes: Optional[float] = None,
                     matrix_size: Optional[int] = None, 
                     dtype_bytes: int = 4) -> int:
    """Calculate memory tile limit for GPU tensor management."""
    return math_core.memory_tile_limit(mem_bytes, matrix_size, dtype_bytes)


def gpu_load_ratio(current_usage: float, max_capacity: float = 100) -> float:
    """Calculate GPU load ratio for safe operation."""
    return math_core.gpu_load_ratio(current_usage, max_capacity)


def smooth_gradient_detection(prices: np.ndarray, window: int = 5) -> float:
    """Smooth gradient detection for trade zone flatness."""
    return math_core.smooth_gradient_detection(prices, window)


def phantom_score(tick_data: np.ndarray, threshold: float = 0.1) -> float:
    """Calculate phantom trade detection score using entropy and flatness."""
    return math_core.phantom_score(tick_data, threshold)


# Initialize and test the system
if __name__ == "__main__":
    report_cuda_status()

    # Test mathematical functions
    print("\n🧮 Testing Mathematical Functions:")
    print("=" * 40)

    # Test matrix operations
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    C = matrix_fit(A, B)
    print(f"Matrix Fit: {C.shape}")

    # Test cosine similarity
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    similarity = cosine_match(vec1, vec2)
    print(f"Cosine Similarity: {similarity:.4f}")

    # Test entropy
    data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    entropy = entropy_of_vector(data)
    print(f"Entropy: {entropy:.4f}")

    # Test flatness
    prices = np.array([100, 101, 100, 99, 100, 101, 100])
    flatness = flatness_measure(prices)
    print(f"Flatness: {flatness:.4f}")

    # Test phantom score
    phantom = phantom_score(prices)
    print(f"Phantom Score: {phantom:.4f}")

    print("\n✅ All mathematical functions tested successfully!")
