# !/usr/bin/env python3
"""
Fractal Core - Fractal mathematics for Schwabot trading system.

Provides fractal quantization, pattern recognition, and mathematical
fractal operations for the trading system.
"""

import math
import logging
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FractalQuantizationResult:
    """Result of fractal quantization operation."""
    quantized_vector: np.ndarray
    fractal_dimension: float
    self_similarity_score: float
    compression_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def fractal_quantize_vector(vector: Union[List[float], np.ndarray],
                           precision: int = 8,
                           method: str = "mandelbrot") -> FractalQuantizationResult:
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
        # Convert to numpy array if needed
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float64)

        # Normalize vector to [0, 1] range
        v_min, v_max = vector.min(), vector.max()
        if v_max > v_min:
            normalized = (vector - v_min) / (v_max - v_min)
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

        # Calculate fractal dimension
        fractal_dim = _calculate_fractal_dimension(quantized)

        # Calculate self-similarity score
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
                "quantized_length": len(quantized)
            }
        )

    except Exception as e:
        logger.error(f"Fractal quantization failed: {e}")
        # Return fallback quantization
        return FractalQuantizationResult(
            quantized_vector=np.array(vector, dtype=np.float64),
            fractal_dimension=1.0,
            self_similarity_score=0.5,
            compression_ratio=1.0,
            metadata={"error": str(e), "method": "fallback"}
        )


def quantize_vector(vector: Union[List[float], np.ndarray],
                   precision: int = 8) -> np.ndarray:
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
        
        # Simple quantization to discrete levels
        max_val = 2 ** precision - 1
        quantized = np.round(vector * max_val) / max_val
        
        return quantized
        
    except Exception as e:
        logger.error(f"Vector quantization failed: {e}")
        return np.array(vector, dtype=np.float64)


def _mandelbrot_quantize(vector: np.ndarray, precision: int) -> np.ndarray:
    """Quantize using Mandelbrot set mathematics."""
    # Mandelbrot iteration count for each point
    max_iter = 2 ** precision
    quantized = np.zeros_like(vector)

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
    max_iter = 2 ** precision
    quantized = np.zeros_like(vector)

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
    quantized = np.zeros_like(vector)

    for i, val in enumerate(vector):
        # Convert to binary representation
        binary = format(int(val * (2**precision - 1)), f'0{precision}b')

        # Count 1s in binary (Sierpinski pattern)
        ones_count = binary.count('1')
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

            # Count boxes needed
            count = 0
            for i in range(0, len(vector), size):
                segment = vector[i:i+size]
                if len(segment) > 0 and np.std(segment) > 0.01:
                    count += 1

            box_counts.append(count)

        if len(box_counts) < 2:
            return 1.0

        # Calculate fractal dimension from slope
        log_sizes = [math.log(size) for size in box_sizes[:len(box_counts)]]
        log_counts = [math.log(count) if count > 0 else 0 for count in box_counts]

        # Linear regression for slope
        slope = np.polyfit(log_sizes, log_counts, 1)[0]
        return abs(slope)

    except Exception:
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
            correlation = np.corrcoef(vector[:len(scaled)], scaled)[0, 1]
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
            binary_val = format(int(val * 255), '08b')
            binary += binary_val
        
        # Truncate or pad to desired length
        if len(binary) > length:
            binary = binary[:length]
        else:
            binary = binary.ljust(length, '0')
        
        # Convert to hex
        hex_hash = ""
        for i in range(0, len(binary), 4):
            chunk = binary[i:i+4]
            hex_hash += format(int(chunk, 2), 'x')
        
        return hex_hash
        
    except Exception as e:
        logger.error(f"Fractal hash generation failed: {e}")
        return "0" * (length // 4)


def fractal_pattern_match(pattern: np.ndarray, target: np.ndarray,
                         threshold: float = 0.8) -> Tuple[bool, float]:
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
            segment = target[i:i+len(pattern)]
            
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