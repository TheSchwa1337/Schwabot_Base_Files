#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Acceleration Utilities for Schwabot
=======================================

Provides GPU acceleration using CuPy/Torch with fallback to NumPy.
This module automatically detects available GPU hardware and
provides optimized mathematical operations.

Key Features:
• Automatic GPU detection and backend selection
• CuPy and PyTorch support with NumPy fallback
• Performance monitoring and optimization
• Memory management and error handling
• Seamless fallback to CPU when GPU fails
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Import type aliases
try:
    from utils.type_aliases import (
        BackendType,
        DriftCoefficient,
        EntropyValue,
        PhaseValue,
        ProcessingTime,
        SignalField,
        TimeIndex,
    )
    TYPE_ALIASES_AVAILABLE = True
except ImportError:
    TYPE_ALIASES_AVAILABLE = False
    logger.warning("Type aliases not available, using basic types")

# ============================================================================
# BACKEND DETECTION AND INITIALIZATION
# ============================================================================

# Global backend state
USING_GPU = False
BACKEND_NAME = "numpy"
XP = None

def detect_backend() -> Tuple[bool, str, Any]:
    """
    Detect available GPU backend.
    
    Returns:
        Tuple of (using_gpu, backend_name, xp_module)
    """
    global USING_GPU, BACKEND_NAME, XP
    
    # Try CuPy first (preferred for GPU acceleration)
    try:
        import cupy as cp
        if cp.cuda.is_available():
            USING_GPU = True
            BACKEND_NAME = "cupy"
            XP = cp
            logger.info("✅ CuPy GPU backend detected and initialized")
            return True, "cupy", cp
    except ImportError:
        logger.debug("CuPy not available")
    except Exception as e:
        logger.warning(f"CuPy initialization failed: {e}")
    
    # Try PyTorch as fallback
    try:
        import torch
        if torch.cuda.is_available():
            USING_GPU = True
            BACKEND_NAME = "torch"
            XP = torch
            logger.info("✅ PyTorch GPU backend detected and initialized")
            return True, "torch", torch
    except ImportError:
        logger.debug("PyTorch not available")
    except Exception as e:
        logger.warning(f"PyTorch initialization failed: {e}")
    
    # Fallback to NumPy
    try:
        import numpy as np
        USING_GPU = False
        BACKEND_NAME = "numpy"
        XP = np
        logger.info("✅ NumPy CPU backend initialized")
        return False, "numpy", np
    except ImportError:
        logger.error("❌ NumPy not available - no backend found")
        return False, "none", None

# Initialize backend on module import
USING_GPU, BACKEND_NAME, XP = detect_backend()

# ============================================================================
# CORE MATHEMATICAL OPERATIONS
# ============================================================================

def gradient(signal: Any) -> Any:
    """
    Compute gradient using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Gradient array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.gradient(signal)
        elif BACKEND_NAME == "torch":
            # PyTorch doesn't have gradient, use diff
            return XP.diff(signal)
        else:  # numpy
            return XP.gradient(signal)
    except Exception as e:
        logger.error(f"Gradient computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.gradient(signal)

def dot(a: Any, b: Any) -> float:
    """
    Compute dot product using the selected backend.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Dot product result
    """
    try:
        if BACKEND_NAME == "cupy":
            return float(XP.dot(a, b))
        elif BACKEND_NAME == "torch":
            return float(XP.dot(a, b))
        else:  # numpy
            return float(XP.dot(a, b))
    except Exception as e:
        logger.error(f"Dot product computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return float(np.dot(a, b))

def mean(signal: Any) -> float:
    """
    Compute mean using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Mean value
    """
    try:
        if BACKEND_NAME == "cupy":
            return float(XP.mean(signal))
        elif BACKEND_NAME == "torch":
            return float(XP.mean(signal))
        else:  # numpy
            return float(XP.mean(signal))
    except Exception as e:
        logger.error(f"Mean computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return float(np.mean(signal))

def std(signal: Any) -> float:
    """
    Compute standard deviation using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Standard deviation value
    """
    try:
        if BACKEND_NAME == "cupy":
            return float(XP.std(signal))
        elif BACKEND_NAME == "torch":
            return float(XP.std(signal))
        else:  # numpy
            return float(XP.std(signal))
    except Exception as e:
        logger.error(f"Standard deviation computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return float(np.std(signal))

def max(signal: Any) -> float:
    """
    Compute maximum using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Maximum value
    """
    try:
        if BACKEND_NAME == "cupy":
            return float(XP.max(signal))
        elif BACKEND_NAME == "torch":
            return float(XP.max(signal))
        else:  # numpy
            return float(XP.max(signal))
    except Exception as e:
        logger.error(f"Maximum computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return float(np.max(signal))

def min(signal: Any) -> float:
    """
    Compute minimum using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Minimum value
    """
    try:
        if BACKEND_NAME == "cupy":
            return float(XP.min(signal))
        elif BACKEND_NAME == "torch":
            return float(XP.min(signal))
        else:  # numpy
            return float(XP.min(signal))
    except Exception as e:
        logger.error(f"Minimum computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return float(np.min(signal))

def sum(signal: Any) -> float:
    """
    Compute sum using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Sum value
    """
    try:
        if BACKEND_NAME == "cupy":
            return float(XP.sum(signal))
        elif BACKEND_NAME == "torch":
            return float(XP.sum(signal))
        else:  # numpy
            return float(XP.sum(signal))
    except Exception as e:
        logger.error(f"Sum computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return float(np.sum(signal))

# ============================================================================
# ADVANCED MATHEMATICAL OPERATIONS
# ============================================================================

def fft(signal: Any) -> Any:
    """
    Compute FFT using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        FFT result array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.fft.fft(signal)
        elif BACKEND_NAME == "torch":
            return XP.fft.fft(XP.tensor(signal))
        else:  # numpy
            return XP.fft.fft(signal)
    except Exception as e:
        logger.error(f"FFT computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.fft.fft(signal)

def ifft(signal: Any) -> Any:
    """
    Compute inverse FFT using the selected backend.
    
    Args:
        signal: Input signal array
        
    Returns:
        Inverse FFT result array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.fft.ifft(signal)
        elif BACKEND_NAME == "torch":
            return XP.fft.ifft(XP.tensor(signal))
        else:  # numpy
            return XP.fft.ifft(signal)
    except Exception as e:
        logger.error(f"Inverse FFT computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.fft.ifft(signal)

def convolve(a: Any, b: Any, mode: str = 'full') -> Any:
    """
    Compute convolution using the selected backend.
    
    Args:
        a: First array
        b: Second array
        mode: Convolution mode
        
    Returns:
        Convolution result array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.convolve(a, b, mode=mode)
        elif BACKEND_NAME == "torch":
            # PyTorch uses different convolution API
            return XP.conv1d(XP.tensor(a), XP.tensor(b))
        else:  # numpy
            return XP.convolve(a, b, mode=mode)
    except Exception as e:
        logger.error(f"Convolution computation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.convolve(a, b, mode=mode)

# ============================================================================
# ARRAY OPERATIONS
# ============================================================================

def array(data: Any) -> Any:
    """
    Create array using the selected backend.
    
    Args:
        data: Input data
        
    Returns:
        Array object
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.array(data)
        elif BACKEND_NAME == "torch":
            return XP.tensor(data)
        else:  # numpy
            return XP.array(data)
    except Exception as e:
        logger.error(f"Array creation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.array(data)

def zeros(shape: Tuple[int, ...]) -> Any:
    """
    Create zeros array using the selected backend.
    
    Args:
        shape: Array shape
        
    Returns:
        Zeros array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.zeros(shape)
        elif BACKEND_NAME == "torch":
            return XP.zeros(shape)
        else:  # numpy
            return XP.zeros(shape)
    except Exception as e:
        logger.error(f"Zeros array creation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.zeros(shape)

def ones(shape: Tuple[int, ...]) -> Any:
    """
    Create ones array using the selected backend.
    
    Args:
        shape: Array shape
        
    Returns:
        Ones array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.ones(shape)
        elif BACKEND_NAME == "torch":
            return XP.ones(shape)
        else:  # numpy
            return XP.ones(shape)
    except Exception as e:
        logger.error(f"Ones array creation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.ones(shape)

def linspace(start: float, stop: float, num: int) -> Any:
    """
    Create linearly spaced array using the selected backend.
    
    Args:
        start: Start value
        stop: Stop value
        num: Number of points
        
    Returns:
        Linearly spaced array
    """
    try:
        if BACKEND_NAME == "cupy":
            return XP.linspace(start, stop, num)
        elif BACKEND_NAME == "torch":
            return XP.linspace(start, stop, num)
        else:  # numpy
            return XP.linspace(start, stop, num)
    except Exception as e:
        logger.error(f"Linspace creation error: {e}")
        # Fallback to CPU
        import numpy as np
        return np.linspace(start, stop, num)

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def monitor_gpu_performance(func):
    """
    Decorator to monitor GPU operation performance.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            logger.debug(f"GPU operation {func.__name__} completed in {processing_time:.6f}s")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"GPU operation {func.__name__} failed after {processing_time:.6f}s: {e}")
            raise
    return wrapper

# ============================================================================
# GPU MANAGEMENT
# ============================================================================

class GPUManager:
    """
    Manager for GPU operations and memory.
    
    Provides centralized management of GPU operations,
    memory allocation, and performance monitoring.
    """
    
    def __init__(self):
        """Initialize GPU manager."""
        self.logger = logging.getLogger(__name__)
        self.operation_count = 0
        self.total_processing_time = 0.0
        self.memory_allocated = 0
        self.fallback_count = 0
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information.
        
        Returns:
            Dictionary with GPU information
        """
        gpu_info = {
            'using_gpu': USING_GPU,
            'backend': BACKEND_NAME,
            'available': USING_GPU,
        }
        
        if BACKEND_NAME == "cupy":
            try:
                gpu_info.update({
                    'device_count': XP.cuda.runtime.getDeviceCount(),
                    'current_device': XP.cuda.runtime.getDevice(),
                    'memory_allocated': XP.cuda.runtime.memGetInfo()[1],
                    'memory_total': XP.cuda.runtime.memGetInfo()[0],
                })
            except Exception as e:
                self.logger.warning(f"Could not get CuPy GPU info: {e}")
        
        elif BACKEND_NAME == "torch":
            try:
                gpu_info.update({
                    'device_count': XP.cuda.device_count(),
                    'current_device': XP.cuda.current_device(),
                    'memory_allocated': XP.cuda.memory_allocated(),
                    'memory_total': XP.cuda.get_device_properties(0).total_memory,
                })
            except Exception as e:
                self.logger.warning(f"Could not get PyTorch GPU info: {e}")
        
        return gpu_info
    
    def clear_memory(self):
        """Clear GPU memory."""
        if BACKEND_NAME == "cupy":
            try:
                XP.cuda.runtime.deviceReset()
                self.logger.info("CuPy GPU memory cleared")
            except Exception as e:
                self.logger.warning(f"Could not clear CuPy GPU memory: {e}")
        
        elif BACKEND_NAME == "torch":
            try:
                XP.cuda.empty_cache()
                self.logger.info("PyTorch GPU memory cleared")
            except Exception as e:
                self.logger.warning(f"Could not clear PyTorch GPU memory: {e}")
    
    def record_operation(self, operation_name: str, processing_time: float, success: bool = True):
        """
        Record operation statistics.
        
        Args:
            operation_name: Name of the operation
            processing_time: Processing time in seconds
            success: Whether operation was successful
        """
        self.operation_count += 1
        self.total_processing_time += processing_time
        
        if not success:
            self.fallback_count += 1
        
        self.logger.debug(f"GPU operation recorded: {operation_name}, "
                         f"time: {processing_time:.6f}s, success: {success}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get GPU operation statistics.
        
        Returns:
            Dictionary with statistics
        """
        avg_time = (self.total_processing_time / max(self.operation_count, 1))
        
        return {
            'operation_count': self.operation_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'fallback_count': self.fallback_count,
            'success_rate': (self.operation_count - self.fallback_count) / max(self.operation_count, 1),
            'gpu_info': self.get_gpu_info()
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.operation_count = 0
        self.total_processing_time = 0.0
        self.fallback_count = 0
        self.logger.info("GPU operation statistics reset")

# Global GPU manager instance
gpu_manager = GPUManager()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_backend() -> str:
    """
    Get current backend name.
    
    Returns:
        Backend name
    """
    return BACKEND_NAME

def is_gpu_available() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        True if GPU is available
    """
    return USING_GPU

def switch_to_cpu():
    """
    Force switch to CPU backend.
    """
    global USING_GPU, BACKEND_NAME, XP
    
    try:
        import numpy as np
        USING_GPU = False
        BACKEND_NAME = "numpy"
        XP = np
        logger.info("Switched to CPU backend")
    except ImportError:
        logger.error("Could not switch to CPU backend - NumPy not available")

def switch_to_gpu():
    """
    Attempt to switch to GPU backend.
    """
    global USING_GPU, BACKEND_NAME, XP
    
    # Re-detect backend
    USING_GPU, BACKEND_NAME, XP = detect_backend()

# ============================================================================
# EXPORTED FUNCTIONS
# ============================================================================

__all__ = [
    'gradient',
    'dot',
    'mean',
    'std',
    'max',
    'min',
    'sum',
    'fft',
    'ifft',
    'convolve',
    'array',
    'zeros',
    'ones',
    'linspace',
    'monitor_gpu_performance',
    'GPUManager',
    'gpu_manager',
    'get_backend',
    'is_gpu_available',
    'switch_to_cpu',
    'switch_to_gpu',
    'USING_GPU',
    'BACKEND_NAME',
    'XP'
] 