#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU Fallback Utilities for Schwabot
===================================

Provides graceful degradation layer when GPU operations fail.
This module ensures that mathematical operations can continue
even when GPU acceleration is unavailable or fails.

Key Features:
• Safe mathematical operations using NumPy
• Graceful fallback from GPU to CPU
• Error handling and recovery
• Performance monitoring for CPU operations
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Import type aliases
try:
    from utils.type_aliases import DriftCoefficient, EntropyValue, PhaseValue, ProcessingTime, SignalField, TimeIndex
    TYPE_ALIASES_AVAILABLE = True
except ImportError:
    TYPE_ALIASES_AVAILABLE = False
    logger.warning("Type aliases not available, using basic types")

# ============================================================================
# CORE MATHEMATICAL OPERATIONS
# ============================================================================

def safe_gradient(signal: np.ndarray) -> np.ndarray:
    """
    Safe gradient computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Gradient array
    """
    try:
        if len(signal) < 2:
            return np.array([0.0])
        return np.gradient(signal)
    except Exception as e:
        logger.error(f"Gradient computation error: {e}")
        return np.array([0.0])

def safe_dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Safe dot product computation using NumPy.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Dot product result
    """
    try:
        return float(np.dot(a, b))
    except Exception as e:
        logger.error(f"Dot product computation error: {e}")
        return 0.0

def safe_entropy(signal: np.ndarray) -> float:
    """
    Safe Shannon entropy computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Shannon entropy value
    """
    try:
        if len(signal) == 0:
            return 0.0
        
        # Normalize signal
        signal_abs = np.abs(signal)
        total = np.sum(signal_abs)
        
        if total == 0:
            return 0.0
        
        # Calculate probabilities
        p = signal_abs / total
        
        # Calculate entropy (avoid log(0))
        entropy = -np.sum(p * np.log2(p + 1e-12))
        
        return float(entropy)
    except Exception as e:
        logger.error(f"Entropy computation error: {e}")
        return 0.0

def safe_mean(signal: np.ndarray) -> float:
    """
    Safe mean computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Mean value
    """
    try:
        if len(signal) == 0:
            return 0.0
        return float(np.mean(signal))
    except Exception as e:
        logger.error(f"Mean computation error: {e}")
        return 0.0

def safe_std(signal: np.ndarray) -> float:
    """
    Safe standard deviation computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Standard deviation value
    """
    try:
        if len(signal) < 2:
            return 0.0
        return float(np.std(signal))
    except Exception as e:
        logger.error(f"Standard deviation computation error: {e}")
        return 0.0

def safe_max(signal: np.ndarray) -> float:
    """
    Safe maximum computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Maximum value
    """
    try:
        if len(signal) == 0:
            return 0.0
        return float(np.max(signal))
    except Exception as e:
        logger.error(f"Maximum computation error: {e}")
        return 0.0

def safe_min(signal: np.ndarray) -> float:
    """
    Safe minimum computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Minimum value
    """
    try:
        if len(signal) == 0:
            return 0.0
        return float(np.min(signal))
    except Exception as e:
        logger.error(f"Minimum computation error: {e}")
        return 0.0

# ============================================================================
# ADVANCED MATHEMATICAL OPERATIONS
# ============================================================================

def safe_fft(signal: np.ndarray) -> np.ndarray:
    """
    Safe FFT computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        FFT result array
    """
    try:
        if len(signal) == 0:
            return np.array([])
        return np.fft.fft(signal)
    except Exception as e:
        logger.error(f"FFT computation error: {e}")
        return np.array([])

def safe_ifft(signal: np.ndarray) -> np.ndarray:
    """
    Safe inverse FFT computation using NumPy.
    
    Args:
        signal: Input signal array
        
    Returns:
        Inverse FFT result array
    """
    try:
        if len(signal) == 0:
            return np.array([])
        return np.fft.ifft(signal)
    except Exception as e:
        logger.error(f"Inverse FFT computation error: {e}")
        return np.array([])

def safe_convolve(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    Safe convolution computation using NumPy.
    
    Args:
        a: First array
        b: Second array
        mode: Convolution mode ('full', 'same', 'valid')
        
    Returns:
        Convolution result array
    """
    try:
        if len(a) == 0 or len(b) == 0:
            return np.array([])
        return np.convolve(a, b, mode=mode)
    except Exception as e:
        logger.error(f"Convolution computation error: {e}")
        return np.array([])

def safe_correlate(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    Safe correlation computation using NumPy.
    
    Args:
        a: First array
        b: Second array
        mode: Correlation mode ('full', 'same', 'valid')
        
    Returns:
        Correlation result array
    """
    try:
        if len(a) == 0 or len(b) == 0:
            return np.array([])
        return np.correlate(a, b, mode=mode)
    except Exception as e:
        logger.error(f"Correlation computation error: {e}")
        return np.array([])

# ============================================================================
# SIGNAL PROCESSING OPERATIONS
# ============================================================================

def safe_smooth(signal: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Safe signal smoothing using moving average.
    
    Args:
        signal: Input signal array
        window_size: Size of smoothing window
        
    Returns:
        Smoothed signal array
    """
    try:
        if len(signal) == 0:
            return np.array([])
        
        if window_size < 1:
            window_size = 1
        
        if window_size > len(signal):
            window_size = len(signal)
        
        # Apply moving average smoothing
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        
        return smoothed
    except Exception as e:
        logger.error(f"Signal smoothing error: {e}")
        return signal

def safe_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Safe signal normalization.
    
    Args:
        signal: Input signal array
        
    Returns:
        Normalized signal array
    """
    try:
        if len(signal) == 0:
            return np.array([])
        
        signal_std = safe_std(signal)
        if signal_std == 0:
            return np.zeros_like(signal)
        
        normalized = (signal - safe_mean(signal)) / signal_std
        return normalized
    except Exception as e:
        logger.error(f"Signal normalization error: {e}")
        return signal

def safe_noise_reduction(signal: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Safe noise reduction using thresholding.
    
    Args:
        signal: Input signal array
        threshold: Noise threshold
        
    Returns:
        Noise-reduced signal array
    """
    try:
        if len(signal) == 0:
            return np.array([])
        
        # Apply threshold-based noise reduction
        signal_abs = np.abs(signal)
        max_val = safe_max(signal_abs)
        
        if max_val == 0:
            return signal
        
        threshold_val = threshold * max_val
        reduced = np.where(signal_abs < threshold_val, 0, signal)
        
        return reduced
    except Exception as e:
        logger.error(f"Noise reduction error: {e}")
        return signal

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def monitor_performance(func):
    """
    Decorator to monitor CPU operation performance.
    
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
            logger.debug(f"CPU operation {func.__name__} completed in {processing_time:.6f}s")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"CPU operation {func.__name__} failed after {processing_time:.6f}s: {e}")
            raise
    return wrapper

# ============================================================================
# FALLBACK MANAGEMENT
# ============================================================================

class CPUFallbackManager:
    """
    Manager for CPU fallback operations.
    
    Provides centralized management of CPU fallback operations
    and performance monitoring.
    """
    
    def __init__(self):
        """Initialize CPU fallback manager."""
        self.logger = logging.getLogger(__name__)
        self.operation_count = 0
        self.total_processing_time = 0.0
        self.fallback_count = 0
        
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
        
        self.logger.debug(f"CPU operation recorded: {operation_name}, "
                         f"time: {processing_time:.6f}s, success: {success}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get CPU fallback statistics.
        
        Returns:
            Dictionary with statistics
        """
        avg_time = (self.total_processing_time / max(self.operation_count, 1))
        
        return {
            'operation_count': self.operation_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'fallback_count': self.fallback_count,
            'success_rate': (self.operation_count - self.fallback_count) / max(self.operation_count, 1)
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.operation_count = 0
        self.total_processing_time = 0.0
        self.fallback_count = 0
        self.logger.info("CPU fallback statistics reset")

# Global CPU fallback manager instance
cpu_fallback_manager = CPUFallbackManager()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_cpu_available() -> bool:
    """
    Check if CPU operations are available.
    
    Returns:
        True if CPU operations are available
    """
    try:
        # Test basic NumPy operation
        test_array = np.array([1.0, 2.0, 3.0])
        result = safe_mean(test_array)
        return result == 2.0
    except Exception:
        return False

def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU information.
    
    Returns:
        Dictionary with CPU information
    """
    try:
        import psutil
        
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available': True
        }
    except ImportError:
        cpu_info = {
            'cpu_count': 1,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'available': True
        }
    
    return cpu_info

# ============================================================================
# EXPORTED FUNCTIONS
# ============================================================================

__all__ = [
    'safe_gradient',
    'safe_dot',
    'safe_entropy',
    'safe_mean',
    'safe_std',
    'safe_max',
    'safe_min',
    'safe_fft',
    'safe_ifft',
    'safe_convolve',
    'safe_correlate',
    'safe_smooth',
    'safe_normalize',
    'safe_noise_reduction',
    'monitor_performance',
    'CPUFallbackManager',
    'cpu_fallback_manager',
    'is_cpu_available',
    'get_cpu_info'
] 