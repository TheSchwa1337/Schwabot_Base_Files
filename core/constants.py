#!/usr/bin/env python3
"""
Constants - Core System Constants and Configuration
==================================================

Defines all mathematical constants, thresholds, and configuration values
used throughout the Schwabot trading system. Includes Windows CLI compatibility
handlers for cross-platform operation.
"""

import os
import platform
import numpy as np
from pathlib import Path
from typing import Any

# Mathematical constants for advanced calculations
PSI_INFINITY = 1.618033988749895  # Golden ratio for allocation
FIBONACCI_SCALING = 1.272019649514069  # φ^(1/2) for fractal scaling
INVERSE_PSI = 0.618033988749895  # 1/φ for counter-rotation

# Configuration directories
CONFIG_DIR = Path(__file__).parent / "config"
DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"

# Mathematical thresholds and limits
KELLY_SAFETY_FACTOR = 0.25  # Kelly criterion safety factor
SHARPE_TARGET = 2.0  # Target Sharpe ratio
MAX_POSITION_SIZE = 0.1  # Maximum position size (10%)
MIN_POSITION_SIZE = 0.001  # Minimum position size (0.1%)

# Signal processing constants
SAMPLE_RATE = 1000  # Hz - Signal sampling rate
NYQUIST_FREQUENCY = SAMPLE_RATE / 2  # Nyquist frequency
BUTTERWORTH_ORDER = 4  # Default filter order

# Fractal and pattern analysis
FRACTAL_DIMENSION_LIMIT = 2.5  # Maximum fractal dimension
PATTERN_SIMILARITY_THRESHOLD = 0.95  # Pattern matching threshold
RECURSIVE_DEPTH_LIMIT = 100  # Maximum recursion depth

# Thermal and entropy constants
THERMAL_DECAY_RATE = 0.95  # Thermal state decay rate
ENTROPY_THRESHOLD = 0.5  # Entropy threshold for state changes
VOID_WELL_DEPTH = 0.1  # Void-well analysis depth

# Performance and latency thresholds
LATENCY_THRESHOLD_MS = 100.0  # Latency warning threshold
MAX_ERROR_STACK_SIZE = 1000  # Maximum error history
ERROR_DECAY_FACTOR = 0.95  # Error importance decay

# Ferris wheel and temporal analysis
FERRIS_HARMONIC_RATIOS = [1, 2, 4, 8, 16, 32]  # Harmonic subdivisions
TEMPORAL_COMPRESSION_FACTOR = 0.8  # Time compression factor

# Advanced mathematical constants
SVD_TOLERANCE = 1e-12  # Singular value decomposition tolerance
EIGENVALUE_THRESHOLD = 1e-10  # Eigenvalue significance threshold


# Windows CLI compatibility handler
class WindowsCliCompatibilityHandler:
    """Handles Windows CLI compatibility for cross-platform operation"""

    @staticmethod
    def is_windows_cli() -> bool:
        """Check if running in Windows CLI environment"""
        return platform.system() == "Windows" and (
            "cmd" in os.environ.get("COMSPEC", "").lower()
            or "powershell" in os.environ.get("PSModulePath", "").lower()
        )

    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """Safely print messages with optional emoji support"""
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            # Strip emojis for Windows CLI compatibility
            import re

            message = re.sub(r"[^\w\s\-_.,!?]", "", message)
        return message

    @staticmethod
    def log_safe(logger: Any, level: str, message: str) -> None:
        """Safely log messages with CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        if hasattr(logger, level.lower()):
            getattr(logger, level.lower())(safe_message)

    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Safely format error messages for CLI compatibility"""
        error_msg = str(error)
        if context:
            error_msg = f"{context}: {error_msg}"
        return WindowsCliCompatibilityHandler.safe_print(
            error_msg, use_emoji=False
        )


# Shared constants across the Schwabot code-base
DEFAULT_TIMEOUT = 30.0  # Default timeout in seconds
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts
DEFAULT_BATCH_SIZE = 1000  # Default batch processing size

# Composite constants for advanced calculations
KELLY_SHARPE_COMPOSITE = KELLY_SAFETY_FACTOR * SHARPE_TARGET / np.sqrt(2)
FRACTAL_THERMAL_RATIO = FRACTAL_DIMENSION_LIMIT * THERMAL_DECAY_RATE

# Performance optimization constants
VECTORIZATION_THRESHOLD = 1000  # Use vectorized ops above this size
PARALLEL_PROCESSING_THRESHOLD = 10000  # Use parallel processing above this
