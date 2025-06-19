#!/usr/bin/env python3
"""
Mathematical Constants - Schwabot Unified Framework
==================================================

Core mathematical constants for the Schwabot trading system including:
- Golden ratio and Fibonacci sequences for profit routing
- Quantum mechanical constants for drift analysis 
- Thermal dynamics constants for volume processing
- Advanced signal processing constants

Based on SP 1.27-AE framework and Nexus mathematical integration.
"""

import os
import platform
import numpy as np
from pathlib import Path
from typing import Any


# =====================================
# ADVANCED MATHEMATICAL CONSTANTS
# =====================================

# Golden Ratio and Fibonacci Constants (Profit Routing)
PSI_INFINITY = 1.618033988749895  # Golden ratio for allocation
FIBONACCI_SCALING = 1.272019649514069  # φ^(1/2) for fractal scaling
INVERSE_PSI = 0.618033988749895  # 1/φ for counter-rotation

# Quantum Mechanical Constants (Drift Analysis)
PLANCK_CONSTANT = 6.62607015e-34  # Quantum energy scaling
REDUCED_PLANCK = 1.0545718176461565e-34  # ℏ for angular momentum
QUANTUM_ENTROPY_SCALE = 1.3806485e-23  # Boltzmann constant adaptation

# Thermal Dynamics Constants (Volume Processing)
THERMAL_CONDUCTIVITY_BTC = 0.024  # W/(m·K) for BTC thermal modeling
STEFAN_BOLTZMANN = 5.670374419e-8  # σ for thermal radiation
AVOGADRO_TRADING = 6.02214076e23 / 1e15  # Scaled for tick processing

# Signal Processing Constants (Advanced Drift)
NYQUIST_SCALING = 2.0  # Sampling rate factor
BUTTERWORTH_ORDER = 4  # Default filter order
HAMMING_ALPHA = 0.54  # Window function parameter
BLACKMAN_ALPHA = 0.16  # Advanced window parameter

# Trading Mathematics Constants
KELLY_SAFETY_FACTOR = 0.25  # Fractional Kelly criterion
SHARPE_TARGET = 1.5  # Target Sharpe ratio
MAX_DRAWDOWN_LIMIT = 0.15  # Maximum allowed drawdown
VOLATILITY_NORMALIZATION = 252  # Trading days per year

# Ferris Wheel Logic Constants (Temporal Cycles)
FERRIS_PRIMARY_CYCLE = 16  # Primary rotation period
FERRIS_HARMONIC_RATIOS = [1, 2, 4, 8, 16, 32]  # Harmonic subdivisions
FERRIS_PHASE_OFFSET = np.pi / 4  # 45-degree phase shift

# Hash Density Constants (Pattern Recognition)
SHA256_ENTROPY_BITS = 256  # Full entropy for hash operations
PATTERN_SIMILARITY_THRESHOLD = 0.95  # Pattern matching threshold
RECURSIVE_DEPTH_LIMIT = 100  # Maximum recursion depth
BLOOM_FILTER_BITS = 1024  # Bloom filter size

# Drift Shell Constants (Ring Allocation)
DEFAULT_SHELL_RADIUS = 144.44  # Base shell radius
RING_ALLOCATION_FACTOR = 2 * np.pi  # 2πr/n allocation
SUBSURFACE_DECAY_RATE = 0.1  # Exponential decay rate
GRAYSCALE_ENTROPY_SCALE = 255.0  # 8-bit grayscale normalization

# Zero Point Energy Constants (Quantum Trading)
ZPE_CAVITY_LENGTH = 1e-6  # Micrometers for ZPE calculation
ZPE_FREQUENCY_CUTOFF = 1e12  # Hz cutoff frequency
VACUUM_IMPEDANCE = 376.730313668  # Ohms, Z₀

# Matrix Operation Constants
MATRIX_CONDITION_LIMIT = 1e12  # Condition number limit
EIGENVALUE_TOLERANCE = 1e-10  # Eigenvalue computation tolerance
SVD_TOLERANCE = 1e-12  # Singular value decomposition tolerance

# API and System Constants
MAX_API_RETRIES = 3  # Maximum API retry attempts
API_TIMEOUT_SECONDS = 30.0  # API request timeout
LATENCY_THRESHOLD_MS = 100.0  # Latency warning threshold

# Error Handling Constants
MAX_ERROR_STACK_SIZE = 1000  # Maximum error history
ERROR_DECAY_FACTOR = 0.95  # Error importance decay
FALLBACK_TIMEOUT = 5.0  # Fallback system timeout


# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling"""
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and 
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))
    
    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """Print message safely with Windows CLI compatibility"""
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            emoji_mapping = {
                '🚨': '[ALERT]', '⚠️': '[WARNING]', '✅': '[SUCCESS]',
                '❌': '[ERROR]', '🔄': '[PROCESSING]', '🎯': '[TARGET]',
                '📊': '[DATA]', '🔍': '[SEARCH]', '⚡': '[FAST]',
                '🎉': '[COMPLETE]', '🔧': '[TOOLS]', '📈': '[PROFIT]'
            }
            for emoji, marker in emoji_mapping.items():
                message = message.replace(emoji, marker)
        return message
    
    @staticmethod
    def log_safe(logger: Any, level: str, message: str) -> None:
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            ascii_message = safe_message.encode('ascii', 
                                             errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)


# =====================================
# PATH AND CONFIGURATION CONSTANTS
# =====================================

# Shared constants across the Schwabot code-base
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_FRACTAL_PATH = CONFIG_DIR / "fractal_core.yaml"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


# =====================================
# DERIVED MATHEMATICAL CONSTANTS
# =====================================

# Composite constants for advanced calculations
QUANTUM_THERMAL_COUPLING = (PLANCK_CONSTANT * THERMAL_CONDUCTIVITY_BTC) / (
    STEFAN_BOLTZMANN * QUANTUM_ENTROPY_SCALE)

FERRIS_GOLDEN_HARMONIC = PSI_INFINITY * FERRIS_PRIMARY_CYCLE / (2 * np.pi)

KELLY_SHARPE_COMPOSITE = KELLY_SAFETY_FACTOR * SHARPE_TARGET / np.sqrt(
    VOLATILITY_NORMALIZATION)

# Numerical stability constants
EPSILON_FLOAT32 = np.finfo(np.float32).eps
EPSILON_FLOAT64 = np.finfo(np.float64).eps
MAX_SAFE_INTEGER = 2**53 - 1

# Performance optimization constants
VECTORIZATION_THRESHOLD = 1000  # Use vectorized ops above this size
PARALLEL_PROCESSING_THRESHOLD = 10000  # Use parallel processing above this
MEMORY_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for large operations
