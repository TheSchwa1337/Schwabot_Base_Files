"""
Profit-Driven Backend Dispatcher
Dynamically selects CPU vs GPU based on profit metrics and performance learning.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import numpy as np

# Import both backends
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# Registry to track performance/profit per operation
class ProfitBackendRegistry:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.operation_stats = defaultdict(
            lambda: {
                'cpu': {
                    'profit': 0.0,
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'success_rate': 0.0,
                },
                'gpu': {
                    'profit': 0.0,
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'success_rate': 0.0,
                },
            }
        )
        self.recent_performance = defaultdict(
            lambda: {'cpu': deque(maxlen=max_history), 'gpu': deque(maxlen=max_history)}
        )
        self.lock = threading.RLock()

    def update_stats(
        self,
        op_name: str,
        backend: str,
        execution_time: float,
        profit: float = 0.0,
        success: bool = True,
    ):
        """Update performance statistics for an operation."""
        with self.lock:
            stats = self.operation_stats[op_name][backend]
            recent = self.recent_performance[op_name][backend]

            # Update basic stats
            stats['count'] += 1
            stats['total_time'] += execution_time
            stats['avg_time'] = stats['total_time'] / stats['count']

            # Update profit (cumulative)
            stats['profit'] += profit

            # Update success rate
            if success:
                stats['success_rate'] = (stats['success_rate'] * (stats['count'] - 1) + 1.0) / stats['count']
            else:
                stats['success_rate'] = (stats['success_rate'] * (stats['count'] - 1)) / stats['count']

            # Store recent performance for trend analysis
            recent.append(
                {
                    'time': execution_time,
                    'profit': profit,
                    'success': success,
                    'timestamp': time.time(),
                }
            )

    def get_backend_recommendation(self, op_name: str, data_size: Optional[int] = None) -> str:
        """Get the recommended backend based on profit and performance metrics."""
        with self.lock:
            if op_name not in self.operation_stats:
                # No history, use GPU if available for large operations, CPU
                # for small
                if HAS_CUPY and data_size and data_size > 1000:
                    return 'gpu'
                return 'cpu'

            cpu_stats = self.operation_stats[op_name]['cpu']
            gpu_stats = self.operation_stats[op_name]['gpu']

            # Calculate profit per second for each backend
            cpu_profit_rate = cpu_stats['profit'] / max(cpu_stats['total_time'], 0.001)
            gpu_profit_rate = gpu_stats['profit'] / max(gpu_stats['total_time'], 0.001)

            # Consider recent performance trends (last 10 operations)
            cpu_recent = list(self.recent_performance[op_name]['cpu'])[-10:]
            gpu_recent = list(self.recent_performance[op_name]['gpu'])[-10:]

            cpu_recent_avg = np.mean([p['time'] for p in cpu_recent]) if cpu_recent else float('inf')
            gpu_recent_avg = np.mean([p['time'] for p in gpu_recent]) if gpu_recent else float('inf')

            # Decision logic: prioritize profit rate, then speed, then success
            # rate
            if gpu_profit_rate > cpu_profit_rate * 1.1:  # GPU 10% more profitable
                return 'gpu'
            elif cpu_profit_rate > gpu_profit_rate * 1.1:  # CPU 10% more profitable
                return 'cpu'
            elif gpu_recent_avg < cpu_recent_avg * 0.8:  # GPU 20% faster
                return 'gpu'
            elif cpu_recent_avg < gpu_recent_avg * 0.8:  # CPU 20% faster
                return 'cpu'
            elif gpu_stats['success_rate'] > cpu_stats['success_rate']:
                return 'gpu'
            else:
                return 'cpu'

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of all operation statistics."""
        with self.lock:
            summary = {
                'total_operations': sum(
                    stats['cpu']['count'] + stats['gpu']['count'] for stats in self.operation_stats.values()
                ),
                'operations': {},
            }

            for op_name, stats in self.operation_stats.items():
                summary['operations'][op_name] = {
                    'cpu': dict(stats['cpu']),
                    'gpu': dict(stats['gpu']),
                    'recommended': self.get_backend_recommendation(op_name),
                }

            return summary


# Global registry instance
registry = ProfitBackendRegistry()


# Operation implementations
class BackendOperations:
    """CPU and GPU implementations of common mathematical operations."""

    @staticmethod
    def matrix_multiply(a, b, **kwargs):
        """Matrix multiplication with backend selection."""
        return np.dot(a, b)

    @staticmethod
    def matrix_multiply_gpu(a, b, **kwargs):
        """GPU matrix multiplication."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        return cp.dot(a, b)

    @staticmethod
    def elementwise_multiply(a, b, **kwargs):
        """Element-wise multiplication."""
        return a * b

    @staticmethod
    def elementwise_multiply_gpu(a, b, **kwargs):
        """GPU element-wise multiplication."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        return a * b

    @staticmethod
    def convolution(data, kernel, **kwargs):
        """Convolution operation."""
        # Simplified convolution - you might want to use scipy.signal.convolve
        return np.convolve(data.flatten(), kernel.flatten(), mode='same').reshape(data.shape)

    @staticmethod
    def convolution_gpu(data, kernel, **kwargs):
        """GPU convolution operation."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        return cp.convolve(data.flatten(), kernel.flatten(), mode='same').reshape(data.shape)

    @staticmethod
    def fft(data, **kwargs):
        """Fast Fourier Transform."""
        return np.fft.fft(data)

    @staticmethod
    def fft_gpu(data, **kwargs):
        """GPU Fast Fourier Transform."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        return cp.fft.fft(data)


# Operation mapping
OPERATIONS = {
    'matrix_multiply': (BackendOperations.matrix_multiply, BackendOperations.matrix_multiply_gpu),
    'elementwise_multiply': (
        BackendOperations.elementwise_multiply,
        BackendOperations.elementwise_multiply_gpu,
    ),
    'convolution': (BackendOperations.convolution, BackendOperations.convolution_gpu),
    'fft': (BackendOperations.fft, BackendOperations.fft_gpu),
}


def dispatch_op(op_name: str, *args, profit: float = 0.0, data_size: Optional[int] = None, **kwargs) -> Any:
    """
    Dispatch an operation to the most profitable backend.

    Args:
        op_name: Name of the operation to perform
        *args: Arguments for the operation
        profit: Expected or actual profit from this operation
        data_size: Size of the data (for initial backend selection)
        **kwargs: Additional keyword arguments

    Returns:
        Result of the operation
    """
    if op_name not in OPERATIONS:
        raise ValueError(f"Unknown operation: {op_name}")

    cpu_op, gpu_op = OPERATIONS[op_name]

    # Get backend recommendation
    recommended_backend = registry.get_backend_recommendation(op_name, data_size)

    # Execute operation with timing
    start_time = time.time()
    success = True

    try:
        if recommended_backend == 'gpu' and HAS_CUPY:
            result = gpu_op(*args, **kwargs)
        else:
            result = cpu_op(*args, **kwargs)
    except Exception as e:
        # Fallback to CPU if GPU fails
        if recommended_backend == 'gpu':
            start_time = time.time()
            result = cpu_op(*args, **kwargs)
            recommended_backend = 'cpu'
        else:
            raise e

    execution_time = time.time() - start_time

    # Update registry with performance data
    registry.update_stats(op_name, recommended_backend, execution_time, profit, success)

    return result


def get_profit_stats() -> Dict[str, Any]:
    """Get current profit and performance statistics."""
    return registry.get_stats_summary()


def reset_stats():
    """Reset all performance statistics."""
    global registry
    registry = ProfitBackendRegistry()


# Convenience functions for common operations
def matrix_multiply(a, b, profit=0.0, **kwargs):
    """Profit-driven matrix multiplication."""
    data_size = a.size if hasattr(a, 'size') else len(a) * len(a[0])
    return dispatch_op('matrix_multiply', a, b, profit=profit, data_size=data_size, **kwargs)


def elementwise_multiply(a, b, profit=0.0, **kwargs):
    """Profit-driven element-wise multiplication."""
    data_size = a.size if hasattr(a, 'size') else len(a)
    return dispatch_op('elementwise_multiply', a, b, profit=profit, data_size=data_size, **kwargs)


def convolution(data, kernel, profit=0.0, **kwargs):
    """Profit-driven convolution."""
    data_size = data.size if hasattr(data, 'size') else len(data)
    return dispatch_op('convolution', data, kernel, profit=profit, data_size=data_size, **kwargs)


def fft(data, profit=0.0, **kwargs):
    """Profit-driven FFT."""
    data_size = data.size if hasattr(data, 'size') else len(data)
    return dispatch_op('fft', data, profit=profit, data_size=data_size, **kwargs)
