# !/usr/bin/env python3
"""
Fractal Core - Fractal mathematics for Schwabot trading system.

Provides fractal quantization, pattern recognition, and mathematical
fractal operations for the trading system.

CUDA Integration:
- GPU-accelerated fractal operations with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = 'cupy (GPU)'
    xp = cp
except ImportError:
    import numpy as np
    USING_CUDA = False
    _backend = 'numpy (CPU)'
    xp = np

# CUDA Helper Integration (for additional utilities)
try:
    from ..utils.cuda_helper import (
        get_cuda_status,
        report_cuda_status,
        safe_convolution,
        safe_cuda_operation,
        safe_eigenvalue_decomposition,
        safe_fft,
        safe_matrix_inverse,
        safe_matrix_multiply,
        safe_svd,
        safe_tensor_contraction,
        xp as helper_xp,
    )
    CUDA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info(f"⚡ CUDA acceleration enabled in Fractal Core: {_backend}")
except ImportError:
    CUDA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("🔄 CUDA helper not available - using CPU-only mode in Fractal Core")

# Dual State Router Integration
try:
    from ..system.dual_state_router import (
        ComputeMode,
        StrategyTier,
        get_dual_state_router,
        route_task,
    )
    DUAL_STATE_AVAILABLE = True
    logger.info("🔄 Dual State Router integration enabled in Fractal Core")
except ImportError:
    DUAL_STATE_AVAILABLE = False
    logger.warning("⚠️ Dual State Router not available in Fractal Core")

logger = logging.getLogger(__name__)
logger.info(f"Fractal Core initialized with backend: {_backend}")


@dataclass
class FractalQuantizationResult:
    """Result of fractal quantization operation."""

    quantized_vector: np.ndarray
    fractal_dimension: float
    self_similarity_score: float
    compression_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def fractal_quantize_vector(
    vector: Union[List[float], np.ndarray], precision: int = 8, method: str = "mandelbrot"
) -> FractalQuantizationResult:
    """
    Quantize a vector using fractal mathematics.

    Args:
        vector: Input vector to quantize
        precision: Bit precision for quantization (4, 8, 16, 42)
        method: Quantization method ('mandelbrot', 'julia', 'sierpinski')

    Returns:
        FractalQuantizationResult with quantized vector and metadata
    """
    try:
        # Use dual state router if available for profit-tiered orchestration
        if DUAL_STATE_AVAILABLE:
            dual_state_router = get_dual_state_router()
            task_data = {
                "vector": vector.tolist() if hasattr(vector, "tolist") else list(vector),
                "precision": precision,
                "method": method,
                "operation": "fractal_quantization",
            }

            result = dual_state_router.route(task_id="fractal_quantization", data=task_data)

            if result.get("success", False) and "quantized_vector" in result:
                # Extract result from dual state router
                quantized_vector = np.array(result["quantized_vector"])
                fractal_dim = result.get("fractal_dimension", 1.0)
                similarity = result.get("self_similarity_score", 0.5)
                compression = result.get("compression_ratio", 1.0)

                return FractalQuantizationResult(
                    quantized_vector=quantized_vector,
                    fractal_dimension=fractal_dim,
                    self_similarity_score=similarity,
                    compression_ratio=compression,
                    metadata={
                        "method": method,
                        "precision": precision,
                        "original_length": len(vector),
                        "quantized_length": len(quantized_vector),
                        "dual_state_routed": True,
                    },
                )
            else:
                # Fallback to direct computation
                logger.debug("Dual state router returned no result, using direct computation")

        # Direct computation (fallback or when dual state router not available)
        # Convert to numpy array if needed
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float64)

        # Normalize vector to [0, 1] range with CUDA acceleration
        v_min, v_max = safe_cuda_operation(
            lambda: (xp.min(vector), xp.max(vector)), lambda: (np.min(vector), np.max(vector))
        )

        if v_max > v_min:
            normalized = safe_cuda_operation(
                lambda: (vector - v_min) / (v_max - v_min),
                lambda: (vector - v_min) / (v_max - v_min),
            )
        else:
            normalized = vector * 0.5  # Handle constant vectors

        # Apply fractal quantization based on method
        if method == "mandelbrot":
            quantized = _mandelbrot_quantize(normalized, precision)
        elif method == "julia":
            quantized = _julia_quantize(normalized, precision)
        elif method == "sierpinski":
            quantized = _sierpinski_quantize(normalized, precision)
        else:
            quantized = _mandelbrot_quantize(normalized, precision)  # Default

        # Calculate fractal dimension with CUDA acceleration
        fractal_dim = _calculate_fractal_dimension(quantized)

        # Calculate self-similarity score with CUDA acceleration
        similarity = _calculate_self_similarity(quantized)

        # Calculate compression ratio
        compression = len(vector) / len(quantized) if len(quantized) > 0 else 1.0

        return FractalQuantizationResult(
            quantized_vector=quantized,
            fractal_dimension=fractal_dim,
            self_similarity_score=similarity,
            compression_ratio=compression,
            metadata={
                "method": method,
                "precision": precision,
                "original_length": len(vector),
                "quantized_length": len(quantized),
                "dual_state_routed": False,
            },
        )

    except Exception as e:
        logger.error(f"Fractal quantization failed: {e}")
        # Return fallback quantization
        return FractalQuantizationResult(
            quantized_vector=np.array(vector, dtype=np.float64),
            fractal_dimension=1.0,
            self_similarity_score=0.5,
            compression_ratio=1.0,
            metadata={"error": str(e), "method": "fallback", "dual_state_routed": False},
        )


def quantize_vector(vector: Union[List[float], np.ndarray], precision: int = 8) -> np.ndarray:
    """
    Simple vector quantization function.

    Args:
        vector: Input vector to quantize
        precision: Bit precision for quantization

    Returns:
        Quantized vector
    """
    try:
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float64)

        # Simple quantization to discrete levels with CUDA acceleration
        max_val = 2**precision - 1
        quantized = safe_cuda_operation(
            lambda: xp.round(vector * max_val) / max_val,
            lambda: np.round(vector * max_val) / max_val,
        )

        return quantized

    except Exception as e:
        logger.error(f"Vector quantization failed: {e}")
        return np.array(vector, dtype=np.float64)


def _mandelbrot_quantize(vector: np.ndarray, precision: int) -> np.ndarray:
    """Quantize using Mandelbrot set mathematics."""
    # Mandelbrot iteration count for each point
    max_iter = 2**precision
    quantized = safe_cuda_operation(lambda: xp.zeros_like(vector), lambda: np.zeros_like(vector))

    for i, val in enumerate(vector):
        # Use value as complex number parameter
        c = complex(val * 2 - 1, 0.5)  # Map to Mandelbrot parameter space
        z = 0

        # Mandelbrot iteration
        for iter_count in range(max_iter):
            z = z * z + c
            if abs(z) > 2:
                break

        # Quantize based on iteration count
        quantized[i] = iter_count / max_iter

    return quantized


def _julia_quantize(vector: np.ndarray, precision: int) -> np.ndarray:
    """Quantize using Julia set mathematics."""
    max_iter = 2**precision
    quantized = safe_cuda_operation(lambda: xp.zeros_like(vector), lambda: np.zeros_like(vector))

    # Fixed Julia parameter
    c = complex(-0.7, 0.27)

    for i, val in enumerate(vector):
        # Use value as starting point
        z = complex(val * 2 - 1, 0.5)

        # Julia iteration
        for iter_count in range(max_iter):
            z = z * z + c
            if abs(z) > 2:
                break

        quantized[i] = iter_count / max_iter

    return quantized


def _sierpinski_quantize(vector: np.ndarray, precision: int) -> np.ndarray:
    """Quantize using Sierpinski triangle mathematics."""
    quantized = safe_cuda_operation(lambda: xp.zeros_like(vector), lambda: np.zeros_like(vector))

    for i, val in enumerate(vector):
        # Convert to binary representation
        binary = format(int(val * (2**precision - 1)), f"0{precision}b")

        # Count 1s in binary (Sierpinski pattern)
        ones_count = binary.count("1")
        quantized[i] = ones_count / precision

    return quantized


def _calculate_fractal_dimension(vector: np.ndarray) -> float:
    """Calculate fractal dimension using box-counting method."""
    try:
        # Simplified box-counting for 1D vector
        if len(vector) < 2:
            return 1.0

        # Use different box sizes
        box_sizes = [1, 2, 4, 8, 16]
        box_counts = []

        for size in box_sizes:
            if size >= len(vector):
                break

            # Count boxes needed to cover the vector
            boxes_needed = safe_cuda_operation(
                lambda: xp.ceil(len(vector) / size), lambda: np.ceil(len(vector) / size)
            )
            box_counts.append(boxes_needed)

        if len(box_counts) < 2:
            return 1.0

        # Calculate fractal dimension using log-log plot slope
        log_sizes = [math.log(1 / size) for size in box_sizes[: len(box_counts)]]
        log_counts = [math.log(count) for count in box_counts]

        # Linear regression to find slope
        n = len(log_sizes)
        sum_x = sum(log_sizes)
        sum_y = sum(log_counts)
        sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
        sum_x2 = sum(x * x for x in log_sizes)

        if sum_x2 * n - sum_x * sum_x == 0:
            return 1.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return abs(slope)

    except Exception as e:
        logger.error(f"Fractal dimension calculation failed: {e}")
        return 1.0


def _calculate_self_similarity(vector: np.ndarray) -> float:
    """Calculate self-similarity score of the vector."""
    try:
        if len(vector) < 4:
            return 0.5

        # Compare different scales of the vector
        scales = [1, 2, 4]
        similarities = []

        for scale in scales:
            if scale >= len(vector):
                break

            # Create scaled version
            scaled = vector[::scale]
            if len(scaled) < 2:
                continue

            # Calculate correlation with original
            correlation = np.corrcoef(vector[: len(scaled)], scaled)[0, 1]
            if not np.isnan(correlation):
                similarities.append(abs(correlation))

        if similarities:
            return float(np.mean(similarities))
        else:
            return 0.5

    except Exception:
        return 0.5


def generate_fractal_hash(vector: np.ndarray, length: int = 64) -> str:
    """
    Generate fractal hash from vector.

    Args:
        vector: Input vector
        length: Hash length in bits

    Returns:
        Fractal hash string
    """
    try:
        # Quantize vector
        quantized = fractal_quantize_vector(vector, precision=8)

        # Convert to binary string
        binary = ""
        for val in quantized.quantized_vector:
            # Convert to binary representation
            binary_val = format(int(val * 255), "08b")
            binary += binary_val

        # Truncate or pad to desired length
        if len(binary) > length:
            binary = binary[:length]
        else:
            binary = binary.ljust(length, "0")

        # Convert to hex
        hex_hash = ""
        for i in range(0, len(binary), 4):
            chunk = binary[i : i + 4]
            hex_hash += format(int(chunk, 2), "x")

        return hex_hash

    except Exception as e:
        logger.error(f"Fractal hash generation failed: {e}")
        return "0" * (length // 4)


def fractal_pattern_match(
    pattern: np.ndarray, target: np.ndarray, threshold: float = 0.8
) -> Tuple[bool, float]:
    """
    Match fractal pattern in target vector.

    Args:
        pattern: Pattern to match
        target: Target vector
        threshold: Similarity threshold

    Returns:
        Tuple of (match_found, similarity_score)
    """
    try:
        if len(pattern) > len(target):
            return False, 0.0

        best_score = 0.0

        # Slide pattern over target
        for i in range(len(target) - len(pattern) + 1):
            segment = target[i : i + len(pattern)]

            # Calculate similarity
            correlation = np.corrcoef(pattern, segment)[0, 1]
            if not np.isnan(correlation):
                score = abs(correlation)
                if score > best_score:
                    best_score = score

        match_found = best_score >= threshold
        return match_found, float(best_score)

    except Exception as e:
        logger.error(f"Fractal pattern matching failed: {e}")
        return False, 0.0
