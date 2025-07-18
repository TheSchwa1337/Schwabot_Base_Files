import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# CUDA Integration with Fallback
try:
    import cupy as cp

    USING_CUDA = True
    _backend = 'cupy (GPU)'
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = 'numpy (CPU)'
    xp = np

# CUDA Helper Integration (for additional, utilities)
try:
    from utils.cuda_helper import (
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
    )

    CUDA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("⚡ CUDA acceleration available for GPU Handlers: {0}".format(_backend))
except ImportError:
    xp = np
    CUDA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("🎯 Profit optimization: Initializing CPU-optimized computational pipeline for GPU Handlers")

# Import CPU handlers for fallback
try:
    from .cpu_handlers import run_cpu_strategy
except ImportError:

    def run_cpu_strategy(task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback CPU strategy when GPU is not available."""
        return {"profit_delta": 0.0, "success": False, "fallback": True}


logger = logging.getLogger(__name__)
logger.info("GPU Handlers initialized with backend: {0}".format(_backend))


def run_gpu_strategy(task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute GPU-based strategy using ZBE (Zero Bottleneck, Entropy).

    Args:
        task_id: Strategy identifier
        data: Input data for strategy execution

    Returns:
        Strategy result with performance metrics
    """
    start_time = time.time()

    try:
        # Check if CUDA is available
        if not CUDA_AVAILABLE:
            logger.info(
                "🎯 Profit optimization: Adaptive switching to CPU-based strategy execution for {0}".format(task_id)
            )
            return run_cpu_strategy(task_id, data)

        # Route to appropriate GPU handler based on task_id
        if "matrix_match" in task_id:
            result = _gpu_matrix_match(data)
        elif "ghost_tick" in task_id:
            result = _gpu_ghost_tick_detector(data)
        elif "altitude" in task_id:
            result = _gpu_altitude_rebalance(data)
        elif "fractal" in task_id:
            result = _gpu_fractal_analysis(data)
        elif "tensor" in task_id:
            result = _gpu_tensor_operations(data)
        elif "spectral" in task_id:
            result = _gpu_spectral_analysis(data)
        elif "entropy" in task_id:
            result = _gpu_entropy_calculation(data)
        else:
            # Default GPU handler
            result = _gpu_generic_strategy(data)

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Add performance metrics
        result.update(
            {
                "task_id": task_id,
                "execution_time_ms": execution_time_ms,
                "compute_mode": "zbe",
                "execution_engine": "gpu",
                "success": True,
                "cuda_available": CUDA_AVAILABLE,
            }
        )

        logger.debug("GPU strategy {0} completed in {1}ms".format(task_id, execution_time_ms))
        return result

    except Exception as e:
        logger.error("GPU strategy {0} failed: {1}".format(task_id, e))
        # Fallback to CPU
        logger.info("Falling back to CPU for {0}".format(task_id))
        return run_cpu_strategy(task_id, data)


def _gpu_matrix_match(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based matrix matching for hash-to-matrix operations."""
    try:
        hash_vector = safe_cuda_operation(
            lambda: xp.array(data.get("hash_vector", [])),
            lambda: np.array(data.get("hash_vector", [])),
        )
        matrices = data.get("matrices", [])
        threshold = data.get("threshold", 0.8)

        if hash_vector.size == 0 or len(matrices) == 0:
            return {"profit_delta": 0.0, "match_found": False}

        best_match = None
        best_score = 0.0

        # Process matrices in parallel batches
        batch_size = 10
        for i in range(0, len(matrices), batch_size):
            batch = matrices[i : i + batch_size]

            # Convert batch to GPU arrays
            batch_arrays = []
            for matrix_data in batch:
                matrix = safe_cuda_operation(
                    lambda: xp.array(matrix_data.get("matrix", [])),
                    lambda: np.array(matrix_data.get("matrix", [])),
                )
                if matrix.size > 0:
                    batch_arrays.append((matrix_data, matrix))

            # Calculate similarities in parallel
            for matrix_data, matrix in batch_arrays:
                # Flatten matrix for comparison
                flat_matrix = safe_cuda_operation(lambda: matrix.flatten(), lambda: matrix.flatten())

                # Calculate cosine similarity with GPU acceleration
                similarity = _gpu_cosine_similarity(hash_vector, flat_matrix)

                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = matrix_data

        profit_delta = best_score * 0.15 if best_match else 0.0  # Higher profit for GPU

        return {
            "profit_delta": profit_delta,
            "match_found": best_match is not None,
            "similarity_score": best_score,
            "matched_matrix": best_match,
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU matrix match failed: {0}".format(e))
        return {"profit_delta": 0.0, "match_found": False}


def _gpu_ghost_tick_detector(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based ghost tick detection for short-term signals."""
    try:
        price_data = safe_cuda_operation(
            lambda: xp.array(data.get("price_data", [])),
            lambda: np.array(data.get("price_data", [])),
        )
        volume_data = safe_cuda_operation(
            lambda: xp.array(data.get("volume_data", [])),
            lambda: np.array(data.get("volume_data", [])),
        )

        if len(price_data) < 10:
            return {"profit_delta": 0.0, "ghost_detected": False}

        # Calculate price momentum with GPU acceleration
        price_momentum = safe_cuda_operation(lambda: xp.diff(price_data), lambda: np.diff(price_data))

        # Calculate volume statistics with GPU acceleration
        volume_mean = safe_cuda_operation(lambda: xp.mean(volume_data), lambda: np.mean(volume_data))
        volume_std = safe_cuda_operation(lambda: xp.std(volume_data), lambda: np.std(volume_data))

        # Calculate volume anomaly
        volume_anomaly = safe_cuda_operation(
            lambda: (volume_data[-1] - volume_mean) / (volume_std + 1e-6),
            lambda: (volume_data[-1] - volume_mean) / (volume_std + 1e-6),
        )

        # Detect ghost tick with GPU acceleration
        price_change = safe_cuda_operation(
            lambda: xp.abs(price_momentum[-1]) if len(price_momentum) > 0 else 0,
            lambda: abs(price_momentum[-1]) if len(price_momentum) > 0 else 0,
        )

        ghost_score = safe_cuda_operation(
            lambda: price_change * (1.0 - xp.minimum(xp.abs(volume_anomaly), 1.0)),
            lambda: price_change * (1.0 - min(abs(volume_anomaly), 1.0)),
        )

        # Calculate profit potential (higher for, GPU)
        profit_delta = float(ghost_score * 0.8)  # 8% of ghost score

        return {
            "profit_delta": profit_delta,
            "ghost_detected": float(ghost_score) > 0.1,
            "ghost_score": float(ghost_score),
            "price_momentum": float(price_momentum[-1]) if len(price_momentum) > 0 else 0.0,
            "volume_anomaly": float(volume_anomaly),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU ghost tick detection failed: {0}".format(e))
        return {"profit_delta": 0.0, "ghost_detected": False}


def _gpu_altitude_rebalance(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based altitude rebalancing for portfolio optimization."""
    try:
        positions = data.get("positions", [])
        target_weights = safe_cuda_operation(
            lambda: xp.array(data.get("target_weights", [])),
            lambda: np.array(data.get("target_weights", [])),
        )

        if len(positions) == 0:
            return {"profit_delta": 0.0, "rebalanced": False}

        # Calculate current weights with GPU acceleration
        position_values = safe_cuda_operation(
            lambda: xp.array([pos.get("value", 0) for pos in positions]),
            lambda: np.array([pos.get("value", 0) for pos in positions]),
        )

        total_value = safe_cuda_operation(lambda: xp.sum(position_values), lambda: np.sum(position_values))

        current_weights = safe_cuda_operation(
            lambda: (position_values / total_value if total_value > 0 else xp.zeros_like(position_values)),
            lambda: (position_values / total_value if total_value > 0 else np.zeros_like(position_values)),
        )

        # Calculate rebalancing needs with GPU acceleration
        rebalance_diffs = safe_cuda_operation(
            lambda: target_weights - current_weights, lambda: target_weights - current_weights
        )

        # Calculate rebalancing magnitude
        rebalance_magnitude = safe_cuda_operation(
            lambda: xp.sum(xp.abs(rebalance_diffs)), lambda: np.sum(np.abs(rebalance_diffs))
        )

        # Calculate profit potential (higher for, GPU)
        profit_delta = float(rebalance_magnitude * 0.3)  # 3% of rebalancing magnitude

        return {
            "profit_delta": profit_delta,
            "rebalanced": float(rebalance_magnitude) > 0.5,
            "rebalance_magnitude": float(rebalance_magnitude),
            "current_weights": current_weights.tolist(),
            "target_weights": target_weights.tolist(),
            "rebalance_diffs": rebalance_diffs.tolist(),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU altitude rebalance failed: {0}".format(e))
        return {"profit_delta": 0.0, "rebalanced": False}


def _gpu_fractal_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based fractal analysis for pattern recognition."""
    try:
        time_series = safe_cuda_operation(
            lambda: xp.array(data.get("time_series", [])),
            lambda: np.array(data.get("time_series", [])),
        )

        if len(time_series) < 20:
            return {"profit_delta": 0.0, "fractal_detected": False}

        # Calculate fractal dimension using GPU-accelerated box-counting
        fractal_dim = _gpu_calculate_fractal_dimension(time_series)

        # Calculate self-similarity with GPU acceleration
        similarity = _gpu_calculate_self_similarity(time_series)

        # Detect fractal patterns
        fractal_score = safe_cuda_operation(lambda: fractal_dim * similarity, lambda: fractal_dim * similarity)

        # Calculate profit potential (higher for, GPU)
        profit_delta = float(fractal_score * 0.5)  # 5% of fractal score

        return {
            "profit_delta": profit_delta,
            "fractal_detected": float(fractal_score) > 0.5,
            "fractal_dimension": float(fractal_dim),
            "self_similarity": float(similarity),
            "fractal_score": float(fractal_score),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU fractal analysis failed: {0}".format(e))
        return {"profit_delta": 0.0, "fractal_detected": False}


def _gpu_tensor_operations(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based tensor operations for advanced mathematics."""
    try:
        tensor_a = safe_cuda_operation(
            lambda: xp.array(data.get("tensor_a", [])), lambda: np.array(data.get("tensor_a", []))
        )
        tensor_b = safe_cuda_operation(
            lambda: xp.array(data.get("tensor_b", [])), lambda: np.array(data.get("tensor_b", []))
        )
        operation = data.get("operation", "multiply")

        if tensor_a.size == 0 or tensor_b.size == 0:
            return {"profit_delta": 0.0, "operation_completed": False}

        # Perform tensor operations with GPU acceleration
        if operation == "multiply":
            result = safe_matrix_multiply(tensor_a, tensor_b)
        elif operation == "tensordot":
            result = safe_tensor_contraction(tensor_a, tensor_b, axes=0)
        elif operation == "outer":
            result = safe_cuda_operation(lambda: xp.outer(tensor_a, tensor_b), lambda: np.outer(tensor_a, tensor_b))
        elif operation == "eigenvalue":
            result = safe_eigenvalue_decomposition(tensor_a)
        elif operation == "svd":
            result = safe_svd(tensor_a)
        else:
            result = safe_cuda_operation(lambda: tensor_a * tensor_b, lambda: tensor_a * tensor_b)

        # Calculate operation complexity score
        complexity = safe_cuda_operation(
            lambda: tensor_a.size * tensor_b.size / 1000.0,
            lambda: tensor_a.size * tensor_b.size / 1000.0,
        )

        # Calculate profit potential (higher for, GPU)
        profit_delta = float(complexity * 0.15)  # 1.5% of complexity

        return {
            "profit_delta": profit_delta,
            "operation_completed": True,
            "result_shape": result.shape if hasattr(result, "shape") else None,
            "result_sum": float(safe_cuda_operation(lambda: xp.sum(result), lambda: np.sum(result))),
            "complexity_score": float(complexity),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU tensor operations failed: {0}".format(e))
        return {"profit_delta": 0.0, "operation_completed": False}


def _gpu_spectral_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based spectral analysis for frequency domain processing."""
    try:
        signal_data = safe_cuda_operation(
            lambda: xp.array(data.get("signal_data", [])),
            lambda: np.array(data.get("signal_data", [])),
        )

        if len(signal_data) < 16:
            return {"profit_delta": 0.0, "spectral_analyzed": False}

        # Apply window function with GPU acceleration
        window = safe_cuda_operation(lambda: xp.hanning(len(signal_data)), lambda: np.hanning(len(signal_data)))
        windowed_signal = safe_cuda_operation(lambda: signal_data * window, lambda: signal_data * window)

        # Compute FFT with GPU acceleration
        fft_result = safe_fft(windowed_signal)
        frequencies = safe_cuda_operation(
            lambda: xp.fft.fftfreq(len(signal_data)), lambda: np.fft.fftfreq(len(signal_data))
        )

        # Calculate power spectrum with GPU acceleration
        power_spectrum = safe_cuda_operation(lambda: xp.abs(fft_result) ** 2, lambda: np.abs(fft_result) ** 2)

        # Find dominant frequencies
        positive_freq_mask = safe_cuda_operation(lambda: frequencies > 0, lambda: frequencies > 0)
        positive_power = safe_cuda_operation(
            lambda: power_spectrum[positive_freq_mask], lambda: power_spectrum[positive_freq_mask]
        )

        dominant_freq_idx = safe_cuda_operation(lambda: xp.argmax(positive_power), lambda: np.argmax(positive_power))
        dominant_frequency = safe_cuda_operation(
            lambda: frequencies[positive_freq_mask][dominant_freq_idx],
            lambda: frequencies[positive_freq_mask][dominant_freq_idx],
        )

        # Calculate spectral entropy with GPU acceleration
        normalized_power = safe_cuda_operation(
            lambda: power_spectrum / xp.sum(power_spectrum),
            lambda: power_spectrum / np.sum(power_spectrum),
        )
        spectral_entropy = safe_cuda_operation(
            lambda: -xp.sum(normalized_power * xp.log(normalized_power + 1e-10)),
            lambda: -np.sum(normalized_power * np.log(normalized_power + 1e-10)),
        )

        # Calculate profit potential (higher for, GPU)
        profit_delta = float(spectral_entropy * 0.3)  # 3% of spectral entropy

        return {
            "profit_delta": profit_delta,
            "spectral_analyzed": True,
            "dominant_frequency": float(dominant_frequency),
            "spectral_entropy": float(spectral_entropy),
            "power_spectrum_max": float(
                safe_cuda_operation(lambda: xp.max(power_spectrum), lambda: np.max(power_spectrum))
            ),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU spectral analysis failed: {0}".format(e))
        return {"profit_delta": 0.0, "spectral_analyzed": False}


def _gpu_entropy_calculation(data: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-based entropy calculation for information theory metrics."""
    try:
        probability_dist = safe_cuda_operation(
            lambda: xp.array(data.get("probability_dist", [])),
            lambda: np.array(data.get("probability_dist", [])),
        )

        if probability_dist.size == 0:
            return {"profit_delta": 0.0, "entropy_calculated": False}

        # Normalize to probability distribution with GPU acceleration
        total = safe_cuda_operation(lambda: xp.sum(probability_dist), lambda: np.sum(probability_dist))

        if total > 0:
            normalized_dist = safe_cuda_operation(lambda: probability_dist / total, lambda: probability_dist / total)
        else:
            normalized_dist = probability_dist

        # Calculate Shannon entropy with GPU acceleration
        entropy = safe_cuda_operation(
            lambda: -xp.sum(normalized_dist * xp.log(normalized_dist + 1e-10)),
            lambda: -np.sum(normalized_dist * np.log(normalized_dist + 1e-10)),
        )

        # Calculate maximum possible entropy
        max_entropy = safe_cuda_operation(lambda: xp.log(len(normalized_dist)), lambda: np.log(len(normalized_dist)))

        # Normalize entropy to [0, 1]
        normalized_entropy = safe_cuda_operation(
            lambda: entropy / max_entropy if max_entropy > 0 else 0,
            lambda: entropy / max_entropy if max_entropy > 0 else 0,
        )

        # Calculate profit potential (higher for, GPU)
        profit_delta = float(normalized_entropy * 0.6)  # 6% of normalized entropy

        return {
            "profit_delta": profit_delta,
            "entropy_calculated": True,
            "shannon_entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "max_entropy": float(max_entropy),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU entropy calculation failed: {0}".format(e))
        return {"profit_delta": 0.0, "entropy_calculated": False}


def _gpu_generic_strategy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generic GPU strategy for fallback operations."""
    try:
        # Simple generic computation with GPU acceleration
        input_data = data.get("input_data", [])
        if isinstance(input_data, list):
            input_array = safe_cuda_operation(lambda: xp.array(input_data), lambda: np.array(input_data))
        else:
            input_array = safe_cuda_operation(lambda: xp.array([input_data]), lambda: np.array([input_data]))

        # Basic statistical analysis with GPU acceleration
        if input_array.size > 0:
            mean_val = safe_cuda_operation(lambda: xp.mean(input_array), lambda: np.mean(input_array))
            std_val = safe_cuda_operation(lambda: xp.std(input_array), lambda: np.std(input_array))
            max_val = safe_cuda_operation(lambda: xp.max(input_array), lambda: np.max(input_array))
            min_val = safe_cuda_operation(lambda: xp.min(input_array), lambda: np.min(input_array))

            # Calculate simple profit metric (higher for, GPU)
            profit_delta = float((max_val - min_val) * 0.15)  # 1.5% of range
        else:
            profit_delta = 0.0
            mean_val = std_val = max_val = min_val = 0.0

        return {
            "profit_delta": profit_delta,
            "generic_completed": True,
            "mean": float(mean_val),
            "std": float(std_val),
            "max": float(max_val),
            "min": float(min_val),
            "gpu_accelerated": True,
        }

    except Exception as e:
        logger.error("GPU generic strategy failed: {0}".format(e))
        return {"profit_delta": 0.0, "generic_completed": False}


# Utility functions for GPU operations
def _gpu_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors with GPU acceleration."""
    try:
        norm_a = safe_cuda_operation(lambda: xp.linalg.norm(a), lambda: np.linalg.norm(a))
        norm_b = safe_cuda_operation(lambda: xp.linalg.norm(b), lambda: np.linalg.norm(b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot_product = safe_cuda_operation(lambda: xp.dot(a, b), lambda: np.dot(a, b))

        return float(dot_product / (norm_a * norm_b))
    except Exception:
        return 0.0


def _gpu_calculate_fractal_dimension(time_series: np.ndarray) -> float:
    """Calculate fractal dimension using GPU-accelerated box-counting method."""
    try:
        if len(time_series) < 4:
            return 1.0

        # GPU-accelerated box-counting
        box_sizes = safe_cuda_operation(lambda: xp.array([1, 2, 4, 8]), lambda: np.array([1, 2, 4, 8]))
        box_counts = []

        for size in box_sizes:
            if size >= len(time_series):
                break

            boxes_needed = safe_cuda_operation(
                lambda: xp.ceil(len(time_series) / size), lambda: np.ceil(len(time_series) / size)
            )
            box_counts.append(boxes_needed)

        if len(box_counts) < 2:
            return 1.0

        # Calculate fractal dimension using log-log plot with GPU acceleration
        log_sizes = safe_cuda_operation(
            lambda: xp.log(1 / box_sizes[: len(box_counts)]),
            lambda: np.log(1 / box_sizes[: len(box_counts)]),
        )
        log_counts = safe_cuda_operation(lambda: xp.log(xp.array(box_counts)), lambda: np.log(np.array(box_counts)))

        # Linear regression with GPU acceleration
        slope = safe_cuda_operation(
            lambda: xp.polyfit(log_sizes, log_counts, 1)[0],
            lambda: np.polyfit(log_sizes, log_counts, 1)[0],
        )
        return float(abs(slope))

    except Exception:
        return 1.0


def _gpu_calculate_self_similarity(time_series: np.ndarray) -> float:
    """Calculate self-similarity score of time series with GPU acceleration."""
    try:
        if len(time_series) < 4:
            return 0.5

        # Compare different scales with GPU acceleration
        scales = safe_cuda_operation(lambda: xp.array([1, 2, 4]), lambda: np.array([1, 2, 4]))
        similarities = []

        for scale in scales:
            if scale >= len(time_series):
                break

            # Create scaled version
            scaled = safe_cuda_operation(lambda: time_series[:: int(scale)], lambda: time_series[:: int(scale)])
            if len(scaled) < 2:
                continue

            # Calculate correlation with GPU acceleration
            correlation = safe_cuda_operation(
                lambda: xp.corrcoef(time_series[: len(scaled)], scaled)[0, 1],
                lambda: np.corrcoef(time_series[: len(scaled)], scaled)[0, 1],
            )
            if not safe_cuda_operation(lambda: xp.isnan(correlation), lambda: np.isnan(correlation)):
                similarities.append(abs(correlation))

        return float(
            safe_cuda_operation(
                lambda: xp.mean(similarities) if similarities else 0.5,
                lambda: np.mean(similarities) if similarities else 0.5,
            )
        )

    except Exception:
        return 0.5


# Export key functions
__all__ = [
    "run_gpu_strategy",
    "_gpu_matrix_match",
    "_gpu_ghost_tick_detector",
    "_gpu_altitude_rebalance",
    "_gpu_fractal_analysis",
    "_gpu_tensor_operations",
    "_gpu_spectral_analysis",
    "_gpu_entropy_calculation",
    "_gpu_generic_strategy",
]


class GPUHandlers:
    """GPU Handlers class for ZBE (Zero Bottleneck, Entropy) operations."""

    def __init__(self):
        """Initialize GPU handlers."""
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0.0,
            "cuda_fallbacks": 0,
        }
        self.cuda_available = CUDA_AVAILABLE

    def run_strategy(self, task_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a GPU-based strategy."""
        self.stats["total_operations"] += 1
        start_time = time.time()

        try:
            result = run_gpu_strategy(task_id, data)

            # Check if this was a CUDA fallback
            if not self.cuda_available and result.get("execution_engine") == "cpu":
                self.stats["cuda_fallbacks"] += 1

            if result.get("success", False):
                self.stats["successful_operations"] += 1
            else:
                self.stats["failed_operations"] += 1

            execution_time = time.time() - start_time
            self.stats["total_execution_time"] += execution_time

            return result

        except Exception as e:
            self.stats["failed_operations"] += 1
            self.logger.error("GPU strategy {0} failed: {1}".format(task_id, e))
            return {
                "task_id": task_id,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000,
                "compute_mode": "zbe",
                "execution_engine": "gpu",
                "success": False,
                "profit_delta": 0.0,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get GPU handler statistics."""
        return {
            "handler_type": "gpu",
            "compute_mode": "zbe",
            "cuda_available": self.cuda_available,
            "stats": self.stats.copy(),
            "average_execution_time": (
                self.stats["total_execution_time"] / self.stats["total_operations"]
                if self.stats["total_operations"] > 0
                else 0.0
            ),
        }

    def reset(self):
        """Reset GPU handler statistics."""
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0.0,
            "cuda_fallbacks": 0,
        }
        self.logger.info("GPU handlers statistics reset")

    def get_cuda_status(self) -> Dict[str, Any]:
        """Get CUDA status information."""
        if hasattr(self, "cuda_available") and self.cuda_available:
            try:
                return get_cuda_status()
            except BaseException:
                return {"cuda_available": False, "error": "Status check failed"}
        else:
            return {"cuda_available": False, "reason": "CUDA not available"}

    async def process_market_data(self, market_data: dict) -> dict:
        """Async wrapper to process market data using GPU strategy."""
        start_time = time.time()
        await asyncio.sleep(0)
        # Use a generic or specific strategy for demonstration
        result = self.run_strategy("market_analysis", market_data)
        processing_time = time.time() - start_time
        return {
            "status": "success" if result.get("success", False) else "error",
            "processing_time": processing_time,
            "signal_strength": result.get("profit_delta", 0.0),
            "confidence": 0.9 if result.get("success", False) else 0.0,
            "result": result,
        }
