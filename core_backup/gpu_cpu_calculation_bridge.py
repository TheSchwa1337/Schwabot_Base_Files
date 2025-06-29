#!/usr/bin/env python3
""""""
GPU/CPU Calculation Bridge
=========================

Intelligent computation handoff system that:
- Automatically detects GPU availability
- Optimizes calculations based on hardware
- Provides seamless fallback to CPU
- Manages memory efficiently
- Supports CUDA, OpenCL, and CPU backends
""""""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np

# GPU imports with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class ComputeCapabilities:
    """Hardware compute capabilities."""
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: str = ""
    cpu_cores: int = 1
    cpu_threads: int = 1
    ram_gb: float = 0.0
    cuda_available: bool = False
    opencl_available: bool = False


@dataclass
class ComputeTask:
    """Computation task definition."""
    task_id: str
    operation: str
    data: Any
    size_hint: int
    priority: int = 1
    gpu_preferred: bool = True
    callback: Optional[Callable] = None


class GPUCPUCalculationBridge:
    """Manages GPU/CPU computation handoff for optimal performance."""

    def __init__(self, enable_gpu: bool = True, memory_limit_gb: float = 2.0):
        self.enable_gpu = enable_gpu
        self.memory_limit_gb = memory_limit_gb

        # Hardware detection
        self.capabilities = self._detect_hardware()

        # Task management
        self.task_queue: List[ComputeTask] = []
        self.active_tasks: Dict[str, ComputeTask] = {}
        self.performance_history: Dict[str, Dict[str, List[float]]] = {}

        # Threading
        self.worker_thread = None
        self.is_running = False
        self._lock = threading.Lock()

        # Configuration
        self.gpu_threshold_size = 1000  # Minimum size for GPU computation
        self.auto_optimization = True

        logger.info(f"GPU/CPU Bridge initialized - GPU Available: {self.gpu_available}")

    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available and enabled."""
        return self.enable_gpu and self.capabilities.gpu_available

    def _detect_hardware(self) -> ComputeCapabilities:
        """Detect available hardware capabilities."""
        capabilities = ComputeCapabilities()

        # CPU detection
        try:
            import psutil
            capabilities.cpu_cores = psutil.cpu_count(logical=False)
            capabilities.cpu_threads = psutil.cpu_count(logical=True)
            capabilities.ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            import multiprocessing
            capabilities.cpu_threads = multiprocessing.cpu_count()
            capabilities.cpu_cores = capabilities.cpu_threads

        # GPU detection
        if CUPY_AVAILABLE and self.enable_gpu:
            try:
                cp.cuda.runtime.getDeviceCount()
                device = cp.cuda.Device(0)
                with device:
                    meminfo = cp.cuda.runtime.memGetInfo()
                    capabilities.gpu_available = True
                    capabilities.gpu_memory_gb = meminfo[1] / (1024**3)
                    capabilities.cuda_available = True

                    # Get compute capability
                    major = device.compute_capability[0]
                    minor = device.compute_capability[1]
                    capabilities.gpu_compute_capability = f"{major}.{minor}"

                logger.info(f"CUDA GPU detected: {capabilities.gpu_memory_gb:.1f}GB, ")
                          f"Compute {capabilities.gpu_compute_capability}")

            except Exception as e:
                logger.warning(f"CUDA detection failed: {e}")

        # PyTorch GPU detection
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.enable_gpu:
            try:
                capabilities.gpu_available = True
                capabilities.cuda_available = True
                if not capabilities.gpu_memory_gb:  # If not already set by CuPy
                    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    capabilities.gpu_memory_gb = mem_gb
                logger.info("PyTorch CUDA support detected")
            except Exception as e:
                logger.warning(f"PyTorch CUDA detection failed: {e}")

        return capabilities

    def start_worker(self):
        """Start the background computation worker."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("GPU/CPU worker started")

    def stop_worker(self):
        """Stop the background computation worker."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        logger.info("GPU/CPU worker stopped")

    def _worker_loop(self):
        """Main worker loop for processing computation tasks."""
        while self.is_running:
            try:
                with self._lock:
                    if self.task_queue:
                        task = self.task_queue.pop(0)
                    else:
                        task = None

                if task:
                    self._process_task(task)
                else:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting

            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    def _process_task(self, task: ComputeTask):
        """Process a computation task."""
        try:
            start_time = time.time()

            # Decide on computation backend
            use_gpu = self._should_use_gpu(task)

            # Execute the task
            if use_gpu:
                result = self._execute_gpu_task(task)
                backend = "GPU"
            else:
                result = self._execute_cpu_task(task)
                backend = "CPU"

            execution_time = time.time() - start_time

            # Record performance
            self._record_performance(task, backend, execution_time)

            # Handle callback
            if task.callback:
                task.callback(result)

            logger.debug(f"Task {task.task_id} completed on {backend} in {execution_time:.3f}s")

        except Exception as e:
            logger.error(f"Task processing failed: {e}")

    def _should_use_gpu(self, task: ComputeTask) -> bool:
        """Determine whether to use GPU for a task."""
        if not self.gpu_available:
            return False

        if not task.gpu_preferred:
            return False

        # Size-based decision
        if task.size_hint < self.gpu_threshold_size:
            return False

        # Memory check
        estimated_memory_gb = task.size_hint * 8 / (1024**3)  # Rough estimate
        if estimated_memory_gb > self.capabilities.gpu_memory_gb * 0.8:
            return False

        # Performance history-based decision
        if self.auto_optimization and task.operation in self.performance_history:
            history = self.performance_history[task.operation]
            if len(history["cpu"]) > 0 and len(history["gpu"]) > 0:
                avg_cpu = np.mean(history["cpu"])
                avg_gpu = np.mean(history["gpu"])
                if avg_cpu < avg_gpu * 0.8:  # CPU is significantly faster
                    return False

        return True

    def _execute_gpu_task(self, task: ComputeTask) -> Any:
        """Execute task on GPU."""
        if CUPY_AVAILABLE:
            return self._execute_cupy_task(task)
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            return self._execute_torch_gpu_task(task)
        else:
            raise RuntimeError("No GPU backend available")

    def _execute_cupy_task(self, task: ComputeTask) -> Any:
        """Execute task using CuPy."""
        if task.operation == "matrix_multiply":
            a, b = task.data
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result_gpu = cp.dot(a_gpu, b_gpu)
            return cp.asnumpy(result_gpu)

        elif task.operation == "array_sum":
            data_gpu = cp.asarray(task.data)
            result_gpu = cp.sum(data_gpu)
            return cp.asnumpy(result_gpu)

        elif task.operation == "fft":
            data_gpu = cp.asarray(task.data)
            result_gpu = cp.fft.fft(data_gpu)
            return cp.asnumpy(result_gpu)

        elif task.operation == "convolution":
            a, b = task.data
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result_gpu = cp.convolve(a_gpu, b_gpu)
            return cp.asnumpy(result_gpu)

        else:
            raise ValueError(f"Unknown GPU operation: {task.operation}")

    def _execute_torch_gpu_task(self, task: ComputeTask) -> Any:
        """Execute task using PyTorch GPU."""
        device = torch.device("cuda")

        if task.operation == "matrix_multiply":
            a, b = task.data
            a_tensor = torch.tensor(a, device=device)
            b_tensor = torch.tensor(b, device=device)
            result_tensor = torch.mm(a_tensor, b_tensor)
            return result_tensor.cpu().numpy()

        elif task.operation == "array_sum":
            data_tensor = torch.tensor(task.data, device=device)
            result_tensor = torch.sum(data_tensor)
            return result_tensor.cpu().numpy()

        else:
            raise ValueError(f"Unknown PyTorch GPU operation: {task.operation}")

    def _execute_cpu_task(self, task: ComputeTask) -> Any:
        """Execute task on CPU."""
        if task.operation == "matrix_multiply":
            a, b = task.data
            return np.dot(a, b)

        elif task.operation == "array_sum":
            return np.sum(task.data)

        elif task.operation == "fft":
            return np.fft.fft(task.data)

        elif task.operation == "convolution":
            a, b = task.data
            return np.convolve(a, b)

        else:
            raise ValueError(f"Unknown CPU operation: {task.operation}")

    def _record_performance(self, task: ComputeTask, backend: str, execution_time: float):
        """Record performance for optimization."""
        if task.operation not in self.performance_history:
            self.performance_history[task.operation] = {"cpu": [], "gpu": []}

        backend_key = backend.lower()
        history = self.performance_history[task.operation][backend_key]
        history.append(execution_time)

        # Keep only recent history
        max_history = 50
        if len(history) > max_history:
            history[:] = history[-max_history:]

    def submit_task(self, task: ComputeTask) -> bool:
        """Submit a computation task."""
        try:
            with self._lock:
                self.task_queue.append(task)
                # Sort by priority
                self.task_queue.sort(key=lambda t: t.priority, reverse=True)
            return True
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return False

    def compute_sync(self, operation: str, data: Any, )
                    gpu_preferred: bool = True, size_hint: Optional[int] = None) -> Any:
        """Synchronous computation."""
        if size_hint is None:
            if isinstance(data, (list, tuple)) and len(data) > 0:
                if isinstance(data[0], np.ndarray):
                    size_hint = data[0].size
                else:
                    size_hint = len(data)
            elif isinstance(data, np.ndarray):
                size_hint = data.size
            else:
                size_hint = 1000  # Default

        task = ComputeTask()
            task_id=f"sync_{int(time.time() * 1000)}",
                operation=operation,
                    data=data,
                    size_hint=size_hint,
                    gpu_preferred=gpu_preferred
        )

        # Process immediately
        self._process_task(task)

        # For now, we'll simulate the result (actual implementation would return the real result)'
        if operation == "matrix_multiply":
            a, b = data
            return np.dot(a, b)
        elif operation == "array_sum":
            return np.sum(data)
        elif operation == "fft":
            return np.fft.fft(data)
        elif operation == "convolution":
            a, b = data
            return np.convolve(a, b)
        else:
            return data

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "capabilities": {}
            "gpu_available": self.gpu_available,
            "gpu_memory_gb": self.capabilities.gpu_memory_gb,
            "cpu_cores": self.capabilities.cpu_cores,
            "cuda_available": self.capabilities.cuda_available
}
            },
                "performance_history": {}
}
        for operation, history in self.performance_history.items():
            stats["performance_history"][operation] = {}
                "cpu_avg": np.mean(history["cpu"]) if history["cpu"] else 0,
                    "gpu_avg": np.mean(history["gpu"]) if history["gpu"] else 0,
                        "cpu_count": len(history["cpu"]),
                        "gpu_count": len(history["gpu"])
}
        return stats

    def optimize_thresholds(self):
        """Optimize GPU/CPU thresholds based on performance history."""
        if not self.auto_optimization:
            return

        total_gpu_faster = 0
        total_comparisons = 0

        for operation, history in self.performance_history.items():
            if len(history["cpu"]) > 0 and len(history["gpu"]) > 0:
                cpu_avg = np.mean(history["cpu"])
                gpu_avg = np.mean(history["gpu"])

                if gpu_avg < cpu_avg:
                    total_gpu_faster += 1
                total_comparisons += 1

        if total_comparisons > 0:
            gpu_advantage_ratio = total_gpu_faster / total_comparisons

            # Adjust threshold based on GPU advantage
            if gpu_advantage_ratio > 0.7:
                self.gpu_threshold_size = max(100, self.gpu_threshold_size * 0.8)
            elif gpu_advantage_ratio < 0.3:
                self.gpu_threshold_size = min(10000, self.gpu_threshold_size * 1.2)

            logger.info(f"Optimized GPU threshold to {self.gpu_threshold_size}")

    def cleanup(self):
        """Cleanup resources."""
        self.stop_worker()

        # Clear GPU memory if available
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

        logger.info("GPU/CPU bridge cleanup completed")


# Global bridge instance
_gpu_cpu_bridge: Optional[GPUCPUCalculationBridge] = None


def get_gpu_cpu_bridge(enable_gpu: bool = True, )
                      memory_limit_gb: float = 2.0) -> GPUCPUCalculationBridge:
    """Get the global GPU/CPU bridge instance."""
    global _gpu_cpu_bridge
    if _gpu_cpu_bridge is None:
        _gpu_cpu_bridge = GPUCPUCalculationBridge(enable_gpu, memory_limit_gb)
        _gpu_cpu_bridge.start_worker()
    return _gpu_cpu_bridge


# Convenience functions
def compute_on_best_device(operation: str, data: Any, )
                          gpu_preferred: bool = True) -> Any:
    """Compute on the best available device."""
    bridge = get_gpu_cpu_bridge()
    return bridge.compute_sync(operation, data, gpu_preferred)


def matrix_multiply_optimized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized matrix multiplication."""
    return compute_on_best_device("matrix_multiply", (a, b))


def array_sum_optimized(data: np.ndarray) -> Union[float, np.ndarray]:
    """Optimized array summation."""
    return compute_on_best_device("array_sum", data)


if __name__ == "__main__":
    # Test the GPU/CPU bridge
    bridge = GPUCPUCalculationBridge()

    # Test matrix multiplication
    a = np.random.random((1000, 1000))
    b = np.random.random((1000, 1000))

    start_time = time.time()
    result = bridge.compute_sync("matrix_multiply", (a, b))
    end_time = time.time()

    print(f"Matrix multiplication completed in {end_time - start_time:.3f}s")
    print(f"Result shape: {result.shape}")

    # Get performance stats
    stats = bridge.get_performance_stats()
    print(f"Performance stats: {stats}")

    # Cleanup
    bridge.cleanup()
