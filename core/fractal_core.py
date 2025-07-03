#!/usr/bin/env python3
"""
Fractal Core - Fractal mathematics for Schwabot trading system.

Provides fractal quantization, pattern recognition, and mathematical
fractal operations for the trading system.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import logging

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
        scales = [2, 4, 8]
        similarities = []
        
        for scale in scales:
            if scale >= len(vector):
                break
            
            # Create downsampled version
            downsampled = vector[::scale]
            
            # Calculate correlation with original
            if len(downsampled) > 1:
                correlation = np.corrcoef(vector[:len(downsampled)*scale], 
                                        np.repeat(downsampled, scale))[0, 1]
                if not np.isnan(correlation):
                    similarities.append(abs(correlation))
        
        return np.mean(similarities) if similarities else 0.5
        
    except Exception:
        return 0.5


def generate_fractal_hash(vector: np.ndarray, length: int = 64) -> str:
    """Generate a fractal hash from a vector."""
    try:
        # Quantize the vector
        result = fractal_quantize_vector(vector)
        
        # Use quantized values to generate hash
        hash_values = []
        for val in result.quantized_vector:
            # Convert to integer and take modulo
            int_val = int(val * 255)
            hash_values.append(int_val)
        
        # Pad or truncate to desired length
        while len(hash_values) < length:
            hash_values.extend(hash_values[:length - len(hash_values)])
        
        hash_values = hash_values[:length]
        
        # Convert to hex string
        hex_hash = ''.join(f'{val:02x}' for val in hash_values)
        return hex_hash
        
    except Exception as e:
        logger.error(f"Fractal hash generation failed: {e}")
        return "0" * length


def fractal_pattern_match(pattern: np.ndarray, target: np.ndarray, 
                         threshold: float = 0.8) -> Tuple[bool, float]:
    """
    Match fractal patterns between two vectors.
    
    Returns:
        Tuple of (match_found, similarity_score)
    """
    try:
        # Quantize both vectors
        pattern_result = fractal_quantize_vector(pattern)
        target_result = fractal_quantize_vector(target)
        
        # Calculate similarity using multiple metrics
        similarities = []
        
        # Vector similarity
        if len(pattern_result.quantized_vector) == len(target_result.quantized_vector):
            vec_sim = np.corrcoef(pattern_result.quantized_vector, 
                                target_result.quantized_vector)[0, 1]
            if not np.isnan(vec_sim):
                similarities.append(abs(vec_sim))
        
        # Fractal dimension similarity
        dim_sim = 1.0 - abs(pattern_result.fractal_dimension - target_result.fractal_dimension)
        similarities.append(dim_sim)
        
        # Self-similarity score similarity
        sim_sim = 1.0 - abs(pattern_result.self_similarity_score - target_result.self_similarity_score)
        similarities.append(sim_sim)
        
        # Average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        match_found = avg_similarity >= threshold
        
        return match_found, avg_similarity
        
    except Exception as e:
        logger.error(f"Fractal pattern matching failed: {e}")
        return False, 0.0


# Export main functions
__all__ = [
    "fractal_quantize_vector",
    "generate_fractal_hash", 
    "fractal_pattern_match",
    "FractalQuantizationResult"
] 