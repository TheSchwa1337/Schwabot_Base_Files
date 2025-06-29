# -*- coding: utf-8 -*-
""""""
GPU Offload Manager - Schwabot UROS v1.0
=======================================

Manages GPU acceleration for mathematical calculations including:
- Bit phase resolution
- Tensor score calculations
- Wave entropy computations
- Matrix operations
- Enhanced thermal state management
- Improved error recovery
""""""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.unified_math_system import unified_math
from dual_unicore_handler import DualUnicoreHandler
from utils.safe_print import debug, error, info, safe_print, success, warn

# Initialize Unicode handler
unicore = DualUnicoreHandler()

logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp

    GPU_AVAILABLE = True
    logger.info("CuPy GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available, using CPU fallback")

try:
    import numba
    from numba import cuda

    NUMBA_AVAILABLE = True
    logger.info("Numba GPU acceleration available")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using CPU fallback")


class ThermalState(Enum):
    """Thermal state enumeration."""

    COOL = "cool"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"


@dataclass
class GPUOperation:
    """GPU operation result."""

    operation_name: str
    input_size: int
    execution_time_ms: float
    gpu_memory_used: int
    success: bool
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUPerformance:
    """GPU performance metrics."""

    total_operations: int
    successful_operations: int
    total_execution_time_ms: float
    average_execution_time_ms: float
    total_gpu_memory_used: int
    gpu_utilization: float
    timestamp: datetime


class GPUOffloadManager:
    """"""
    Manages GPU acceleration for mathematical calculations.

    Features:
    - Bit phase resolution on GPU
    - Tensor score calculations
    - Wave entropy computations
    - Matrix operations
    - Performance monitoring
    - Enhanced thermal state management
    - Improved error recovery
    """"""

    def __init__(self):
        self.gpu_available = GPU_AVAILABLE or NUMBA_AVAILABLE
        self.operation_history: List[GPUOperation] = []
        self.performance_metrics: List[GPUPerformance] = []

        # GPU configuration
        self.max_gpu_memory = 1024 * 1024 * 1024  # 1GB default
        self.batch_size = 1000
        self.enable_async = True

        # Enhanced thermal management
        self.thermal_state = ThermalState.COOL
        self.thermal_thresholds = {}
            ThermalState.COOL: 50.0,
                ThermalState.WARM: 70.0,
                    ThermalState.HOT: 80.0,
                    ThermalState.CRITICAL: 85.0,
}
        self.thermal_history: List[Tuple[datetime, float]] = []

        # Initialize GPU if available
        if self.gpu_available:
            self._initialize_gpu()

        logger.info(f"GPU Offload Manager initialized (GPU: {self.gpu_available})")

    def _initialize_gpu(self) -> None:
        """Initialize GPU resources."""
        try:
            if GPU_AVAILABLE:
                # Initialize CuPy
                cp.cuda.Device(0).use()
                logger.info("CuPy GPU initialized")

            if NUMBA_AVAILABLE:
                # Initialize Numba CUDA
                logger.info("Numba CUDA available")

        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            self.gpu_available = False

    def _update_thermal_state(self, temperature: float) -> ThermalState:
        """Update thermal state based on current temperature."""
        timestamp = datetime.now()
        self.thermal_history.append((timestamp, temperature))

        # Keep only recent history
        if len(self.thermal_history) > 1000:
            self.thermal_history = self.thermal_history[-500:]

        # Determine new thermal state
        if temperature < self.thermal_thresholds[ThermalState.COOL]:
            new_state = ThermalState.COOL
        elif temperature < self.thermal_thresholds[ThermalState.WARM]:
            new_state = ThermalState.WARM
        elif temperature < self.thermal_thresholds[ThermalState.HOT]:
            new_state = ThermalState.HOT
        else:
            new_state = ThermalState.CRITICAL

        if new_state != self.thermal_state:
            logger.info(f"Thermal state changed from {self.thermal_state.value} to {new_state.value}")
            self.thermal_state = new_state

        return self.thermal_state

    def _get_current_temperature(self) -> float:
        """Get current temperature (placeholder for actual implementation)."""
        # This would integrate with actual temperature monitoring
        # For now, return a simulated temperature
        if self.thermal_history:
            return self.thermal_history[-1][1]
        return 45.0  # Default cool temperature

    def _should_use_gpu(self, operation_type: str, data_size: int) -> bool:
        """Determine if GPU should be used based on thermal state and data size."""
        current_temp = self._get_current_temperature()
        self._update_thermal_state(current_temp)

        if self.thermal_state == ThermalState.CRITICAL:
            return False
        elif self.thermal_state == ThermalState.HOT and data_size > 10000:
            return False
        elif not self.gpu_available:
            return False
        else:
            return True

    def _cpu_fallback(self, operation: str, *args, **kwargs) -> Any:
        """Enhanced CPU fallback for GPU operations."""
        logger.warning(f"GPU not available, using CPU fallback for {operation}")

        try:
            if operation == "resolve_bit_phase":
                return self._resolve_bit_phase_cpu(*args, **kwargs)
            elif operation == "tensor_score":
                return self._tensor_score_cpu(*args, **kwargs)
            elif operation == "wave_entropy":
                return self._wave_entropy_cpu(*args, **kwargs)
            elif operation == "matrix_operation":
                return self._matrix_operation_cpu(*args, **kwargs)
            else:
                logger.error(f"Unknown operation for CPU fallback: {operation}")
                return None
        except Exception as e:
            logger.error(f"CPU fallback failed for {operation}: {e}")
            return None

    def resolve_bit_phase_gpu(self, hash_strings: List[str], mode: str = "8bit") -> List[int]:
        """"""
        Resolve bit phases from hash strings using GPU acceleration.

        Args:
            hash_strings: List of hash strings to process
            mode: Bit resolution mode ("4bit", "8bit", "42bit")

        Returns:
            List[int]: Resolved bit phases
        """"""
        start_time = time.time()

        try:
            if not self._should_use_gpu("resolve_bit_phase", len(hash_strings)):
                # Fallback to CPU
                return self._resolve_bit_phase_cpu(hash_strings, mode)

            # GPU processing
            if GPU_AVAILABLE:
                return self._resolve_bit_phase_cupy(hash_strings, mode)
            elif NUMBA_AVAILABLE:
                return self._resolve_bit_phase_numba(hash_strings, mode)
            else:
                return self._resolve_bit_phase_cpu(hash_strings, mode)

        except Exception as e:
            logger.error(f"Error in GPU bit phase resolution: {e}")
            return self._resolve_bit_phase_cpu(hash_strings, mode)
        finally:
            execution_time = (time.time() - start_time) * 1000
            self._record_operation("resolve_bit_phase_gpu", len(hash_strings), execution_time, True)

    def _resolve_bit_phase_cupy(self, hash_strings: List[str], mode: str) -> List[int]:
        """Resolve bit phases using CuPy GPU acceleration."""
        try:
            # Convert hash strings to GPU arrays
            hash_array = cp.array([hash_str.encode() for hash_str in hash_strings])

            # Extract relevant segments based on mode
            if mode == "4bit":
                segments = hash_array[:, 0:1]
                max_val = 16
            elif mode == "8bit":
                segments = hash_array[:, 0:2]
                max_val = 256
            elif mode == "42bit":
                segments = hash_array[:, 0:11]
                max_val = 4398046511104
            else:
                segments = hash_array[:, 0:2]
                max_val = 256

            # Convert hex strings to integers on GPU
            hex_strings = cp.char.decode(segments)
            phase_values = cp.array([int(h.decode(), 16) % max_val for h in hex_strings])

            # Transfer result back to CPU
            return cp.asnumpy(phase_values).tolist()

        except Exception as e:
            logger.error(f"Error in CuPy bit phase resolution: {e}")
            return self._resolve_bit_phase_cpu(hash_strings, mode)

    def _resolve_bit_phase_numba(self, hash_strings: List[str], mode: str) -> List[int]:
        """Resolve bit phases using Numba GPU acceleration."""
        try:
            # For Numba, we'll use a simpler approach'
            # Convert to numpy arrays and process in batches
            results = []

            for i in range(0, len(hash_strings), self.batch_size):
                batch = hash_strings[i : i + self.batch_size]
                batch_results = self._resolve_bit_phase_cpu(batch, mode)
                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"Error in Numba bit phase resolution: {e}")
            return self._resolve_bit_phase_cpu(hash_strings, mode)

    def _resolve_bit_phase_cpu(self, hash_strings: List[str], mode: str) -> List[int]:
        """Resolve bit phases using CPU (fallback)."""
        try:
            results = []

            for hash_str in hash_strings:
                if mode == "4bit":
                    phase = int(hash_str[0:1], 16) % 16
                elif mode == "8bit":
                    phase = int(hash_str[0:2], 16) % 256
                elif mode == "42bit":
                    phase = int(hash_str[0:11], 16) % 4398046511104
                else:
                    phase = int(hash_str[0:2], 16) % 256

                results.append(phase)

            return results

        except Exception as e:
            logger.error(f"Error in CPU bit phase resolution: {e}")
            return [0] * len(hash_strings)

    def tensor_score_gpu()
        self, entry_prices: List[float], current_prices: List[float], phases: List[int]
    ) -> List[float]:
        """"""
        Calculate tensor scores using GPU acceleration.

        Args:
            entry_prices: List of entry prices
            current_prices: List of current prices
            phases: List of bit phases

        Returns:
            List[float]: Tensor scores
        """"""
        start_time = time.time()

        try:
            if not self._should_use_gpu("tensor_score", len(entry_prices)):
                # Fallback to CPU
                return self._tensor_score_cpu(entry_prices, current_prices, phases)

            # GPU processing
            if GPU_AVAILABLE:
                return self._tensor_score_cupy(entry_prices, current_prices, phases)
            elif NUMBA_AVAILABLE:
                return self._tensor_score_numba(entry_prices, current_prices, phases)
            else:
                return self._tensor_score_cpu(entry_prices, current_prices, phases)

        except Exception as e:
            logger.error(f"Error in GPU tensor score calculation: {e}")
            return self._tensor_score_cpu(entry_prices, current_prices, phases)
        finally:
            execution_time = (time.time() - start_time) * 1000
            self._record_operation("tensor_score_gpu", len(entry_prices), execution_time, True)

    def _tensor_score_cupy()
        self, entry_prices: List[float], current_prices: List[float], phases: List[int]
    ) -> List[float]:
        """Calculate tensor scores using CuPy GPU acceleration."""
        try:
            # Convert to GPU arrays
            entry_gpu = cp.array(entry_prices, dtype=cp.float32)
            current_gpu = cp.array(current_prices, dtype=cp.float32)
            phases_gpu = cp.array(phases, dtype=cp.float32)

            # Calculate deltas
            deltas = (current_gpu - entry_gpu) / entry_gpu

            # Apply phase multiplier
            tensor_scores = deltas * (phases_gpu + 1)

            # Round to 4 decimal places
            tensor_scores = cp.round(tensor_scores, 4)

            # Transfer result back to CPU
            return cp.asnumpy(tensor_scores).tolist()

        except Exception as e:
            logger.error(f"Error in CuPy tensor score calculation: {e}")
            return self._tensor_score_cpu(entry_prices, current_prices, phases)

    def _tensor_score_numba()
        self, entry_prices: List[float], current_prices: List[float], phases: List[int]
    ) -> List[float]:
        """Calculate tensor scores using Numba GPU acceleration."""
        try:
            # For Numba, we'll use a simpler approach'
            return self._tensor_score_cpu(entry_prices, current_prices, phases)

        except Exception as e:
            logger.error(f"Error in Numba tensor score calculation: {e}")
            return self._tensor_score_cpu(entry_prices, current_prices, phases)

    def _tensor_score_cpu()
        self, entry_prices: List[float], current_prices: List[float], phases: List[int]
    ) -> List[float]:
        """Calculate tensor scores using CPU (fallback)."""
        try:
            results = []

            for entry, current, phase in zip(entry_prices, current_prices, phases):
                if entry <= 0:
                    results.append(0.0)
                    continue

                delta = (current - entry) / entry
                tensor_score = delta * (phase + 1)
                results.append(round(tensor_score, 4))

            return results

        except Exception as e:
            logger.error(f"Error in CPU tensor score calculation: {e}")
            return [0.0] * len(entry_prices)

    def wave_entropy_gpu(self, sequences: List[List[float]]) -> List[float]:
        """"""
        Calculate wave entropy using GPU acceleration.

        Args:
            sequences: List of wave sequences

        Returns:
            List[float]: Entropy values
        """"""
        start_time = time.time()

        try:
            if not self._should_use_gpu("wave_entropy", len(sequences)):
                # Fallback to CPU
                return self._wave_entropy_cpu(sequences)

            # GPU processing
            if GPU_AVAILABLE:
                return self._wave_entropy_cupy(sequences)
            elif NUMBA_AVAILABLE:
                return self._wave_entropy_numba(sequences)
            else:
                return self._wave_entropy_cpu(sequences)

        except Exception as e:
            logger.error(f"Error in GPU wave entropy calculation: {e}")
            return self._wave_entropy_cpu(sequences)
        finally:
            execution_time = (time.time() - start_time) * 1000
            self._record_operation("wave_entropy_gpu", len(sequences), execution_time, True)

    def _wave_entropy_cupy(self, sequences: List[List[float]]) -> List[float]:
        """Calculate wave entropy using CuPy GPU acceleration."""
        try:
            results = []

            for seq in sequences:
                # Convert sequence to GPU array
                seq_gpu = cp.array(seq, dtype=cp.float32)

                # Calculate FFT
                fft_gpu = cp.fft.fft(seq_gpu)

                # Calculate power spectrum
                power_gpu = cp.abs(fft_gpu) ** 2

                # Normalize
                total_power = cp.sum(power_gpu)
                if total_power > 0:
                    normalized_gpu = power_gpu / total_power
                else:
                    normalized_gpu = cp.zeros_like(power_gpu)

                # Calculate entropy
                # Add small epsilon to avoid log(0)
                epsilon = 1e-9
                entropy_gpu = -cp.sum(normalized_gpu * cp.log2(normalized_gpu + epsilon))

                # Transfer result back to CPU
                results.append(float(cp.asnumpy(entropy_gpu)))

            return results

        except Exception as e:
            logger.error(f"Error in CuPy wave entropy calculation: {e}")
            return self._wave_entropy_cpu(sequences)

    def _wave_entropy_numba(self, sequences: List[List[float]]) -> List[float]:
        """Calculate wave entropy using Numba GPU acceleration."""
        try:
            # For Numba, we'll use a simpler approach'
            return self._wave_entropy_cpu(sequences)

        except Exception as e:
            logger.error(f"Error in Numba wave entropy calculation: {e}")
            return self._wave_entropy_cpu(sequences)

    def _wave_entropy_cpu(self, sequences: List[List[float]]) -> List[float]:
        """Calculate wave entropy using CPU (fallback)."""
        try:
            results = []

            for seq in sequences:
                # Calculate FFT
                fft = np.fft.fft(seq)

                # Calculate power spectrum
                power = np.abs(fft) ** 2

                # Normalize
                total_power = np.sum(power)
                if total_power > 0:
                    normalized = power / total_power
                else:
                    normalized = np.zeros_like(power)

                # Calculate entropy
                epsilon = 1e-9
                entropy = -np.sum(normalized * np.log2(normalized + epsilon))

                results.append(float(entropy))

            return results

        except Exception as e:
            logger.error(f"Error in CPU wave entropy calculation: {e}")
            return [0.0] * len(sequences)

    def matrix_operation_gpu(self, matrices: List[np.ndarray], operation: str) -> List[np.ndarray]:
        """"""
        Perform matrix operations using GPU acceleration.

        Args:
            matrices: List of matrices to process
            operation: Operation type ("multiply", "inverse", "eigenvalues", etc.)

        Returns:
            List[np.ndarray]: Operation results
        """"""
        start_time = time.time()

        try:
            if not self._should_use_gpu("matrix_operation", len(matrices)):
                # Fallback to CPU
                return self._matrix_operation_cpu(matrices, operation)

            # GPU processing
            if GPU_AVAILABLE:
                return self._matrix_operation_cupy(matrices, operation)
            elif NUMBA_AVAILABLE:
                return self._matrix_operation_numba(matrices, operation)
            else:
                return self._matrix_operation_cpu(matrices, operation)

        except Exception as e:
            logger.error(f"Error in GPU matrix operation: {e}")
            return self._matrix_operation_cpu(matrices, operation)
        finally:
            execution_time = (time.time() - start_time) * 1000
            self._record_operation(f"matrix_operation_gpu_{operation}", len(matrices), execution_time, True)

    def _matrix_operation_cupy(self, matrices: List[np.ndarray], operation: str) -> List[np.ndarray]:
        """Perform matrix operations using CuPy GPU acceleration."""
        try:
            results = []

            for matrix in matrices:
                # Convert to GPU array
                matrix_gpu = cp.array(matrix, dtype=cp.float32)

                # Perform operation
                if operation == "multiply":
                    result_gpu = cp.dot(matrix_gpu, matrix_gpu)
                elif operation == "inverse":
                    result_gpu = cp.linalg.inv(matrix_gpu)
                elif operation == "eigenvalues":
                    eigenvalues_gpu = cp.linalg.eigvals(matrix_gpu)
                    result_gpu = eigenvalues_gpu
                elif operation == "transpose":
                    result_gpu = cp.transpose(matrix_gpu)
                else:
                    result_gpu = matrix_gpu  # Default to identity operation

                # Transfer result back to CPU
                results.append(cp.asnumpy(result_gpu))

            return results

        except Exception as e:
            logger.error(f"Error in CuPy matrix operation: {e}")
            return self._matrix_operation_cpu(matrices, operation)

    def _matrix_operation_numba(self, matrices: List[np.ndarray], operation: str) -> List[np.ndarray]:
        """Perform matrix operations using Numba GPU acceleration."""
        try:
            # For Numba, we'll use a simpler approach'
            return self._matrix_operation_cpu(matrices, operation)

        except Exception as e:
            logger.error(f"Error in Numba matrix operation: {e}")
            return self._matrix_operation_cpu(matrices, operation)

    def _matrix_operation_cpu(self, matrices: List[np.ndarray], operation: str) -> List[np.ndarray]:
        """Perform matrix operations using CPU (fallback)."""
        try:
            results = []

            for matrix in matrices:
                # Perform operation
                if operation == "multiply":
                    result = unified_math.matrix_multiply(matrix, matrix)
                elif operation == "inverse":
                    result = unified_math.inverse(matrix)
                elif operation == "eigenvalues":
                    result = unified_math.eigenvalues(matrix)
                elif operation == "transpose":
                    result = np.transpose(matrix)
                else:
                    result = matrix  # Default to identity operation

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in CPU matrix operation: {e}")
            return matrices

    def _record_operation(self, operation_name: str, input_size: int, execution_time_ms: float, success: bool) -> None:
        """Record GPU operation for performance tracking."""
        try:
            # Implement real GPU memory tracking
            gpu_memory_used = self._get_gpu_memory_usage()

            operation = GPUOperation()
                operation_name=operation_name,
                    input_size=input_size,
                        execution_time_ms=execution_time_ms,
                        gpu_memory_used=gpu_memory_used,
                        success=success,
                        result=None,
                        )

            self.operation_history.append(operation)

            # Keep only recent operations
            if len(self.operation_history) > 1000:
                self.operation_history = self.operation_history[-500:]

        except Exception as e:
            logger.error(f"Error recording operation: {e}")

    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        try:
            if not self.gpu_available:
                return 0

            if GPU_AVAILABLE:
                # Use CuPy to get GPU memory info
                import cupy as cp

                mem_info = cp.cuda.runtime.memGetInfo()
                return int(mem_info[1] - mem_info[0])  # Total - Free = Used
            elif NUMBA_AVAILABLE:
                # For Numba, estimate based on recent operations
                if hasattr(self, "_estimated_gpu_memory"):
                    return self._estimated_gpu_memory
                else:
                    self._estimated_gpu_memory = 0
                    return 0
            else:
                return 0

        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {e}")
            return 0

    def get_performance_metrics(self) -> GPUPerformance:
        """Get GPU performance metrics."""
        try:
            if not self.operation_history:
                return GPUPerformance()
                    total_operations=0,
                        successful_operations=0,
                            total_execution_time_ms=0.0,
                            average_execution_time_ms=0.0,
                            total_gpu_memory_used=0,
                            gpu_utilization=0.0,
                            timestamp=datetime.now(),
                            )

            total_operations = len(self.operation_history)
            successful_operations = sum(1 for op in self.operation_history if op.success)
            total_execution_time = sum(op.execution_time_ms for op in self.operation_history)
            average_execution_time = total_execution_time / total_operations if total_operations > 0 else 0.0
            total_gpu_memory = sum(op.gpu_memory_used for op in self.operation_history)

            # Calculate GPU utilization (simplified)
            gpu_utilization = unified_math.min()
                1.0, total_execution_time / (total_operations * 100)
            )  # Assume 100ms is full utilization

            performance = GPUPerformance()
                total_operations=total_operations,
                    successful_operations=successful_operations,
                        total_execution_time_ms=total_execution_time,
                        average_execution_time_ms=average_execution_time,
                        total_gpu_memory_used=total_gpu_memory,
                        gpu_utilization=gpu_utilization,
                        timestamp=datetime.now(),
                        )

            self.performance_metrics.append(performance)

            # Keep only recent metrics
            if len(self.performance_metrics) > 100:
                self.performance_metrics = self.performance_metrics[-50:]

            return performance

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return GPUPerformance()
                total_operations=0,
                    successful_operations=0,
                        total_execution_time_ms=0.0,
                        average_execution_time_ms=0.0,
                        total_gpu_memory_used=0,
                        gpu_utilization=0.0,
                        timestamp=datetime.now(),
                        )

    def clear_history(self) -> None:
        """Clear operation and performance history."""
        self.operation_history.clear()
        self.performance_metrics.clear()
        logger.info("GPU operation history cleared")

    def export_performance_data(self, output_path: str = "gpu_performance_data.json") -> None:
        """Export GPU performance data to JSON."""
        try:
            import json

            performance = self.get_performance_metrics()

            export_data = {
                "timestamp": datetime.now().isoformat(),
                "gpu_available": self.gpu_available,
                "cupy_available": GPU_AVAILABLE,
                "numba_available": NUMBA_AVAILABLE,
                "performance_metrics": {}
                "total_operations": performance.total_operations,
                "successful_operations": performance.successful_operations,
                "total_execution_time_ms": performance.total_execution_time_ms,
                "average_execution_time_ms": performance.average_execution_time_ms,
                "total_gpu_memory_used": performance.total_gpu_memory_used,
                "gpu_utilization": performance.gpu_utilization,
}
                            },
                            "recent_operations": []
                    {}
                        "operation_name": op.operation_name,
                            "input_size": op.input_size,
                                "execution_time_ms": op.execution_time_ms,
                                "success": op.success,
}
                    for op in self.operation_history[-50:]  # Last 50 operations
                ],
}
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"GPU performance data exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting GPU performance data: {e}")


def main():
    """Test function for GPU Offload Manager."""
    safe_print("🔗 Testing Enhanced GPU Offload Manager...")

    manager = GPUOffloadManager()

    # Test bit phase resolution
    test_hashes = ["a1b2c3d4", "e5f6g7h8", "i9j0k1l2"]
    phases = manager.resolve_bit_phase_gpu(test_hashes, "8bit")
    safe_print(f"✅ Bit phase resolution: {phases}")

    # Test tensor score calculation
    entry_prices = [100.0, 200.0, 300.0]
    current_prices = [110.0, 190.0, 320.0]
    phases = [1, 2, 3]
    scores = manager.tensor_score_gpu(entry_prices, current_prices, phases)
    safe_print(f"✅ Tensor scores: {scores}")

    # Test wave entropy calculation
    sequences = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]
    entropies = manager.wave_entropy_gpu(sequences)
    safe_print(f"✅ Wave entropies: {entropies}")

    # Test matrix operation
    matrices = [np.random.rand(3, 3)]
    results = manager.matrix_operation_gpu(matrices, "eigenvalues")
    safe_print(f"✅ Matrix eigenvalues: {results}")

    # Get performance metrics
    metrics = manager.get_performance_metrics()
    safe_print(f"📈 Performance metrics: {metrics}")


if __name__ == "__main__":
    main()
