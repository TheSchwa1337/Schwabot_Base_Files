import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import psutil


class PerformanceMonitor:
    """
    Performance monitor for Schwabot system.

    Tracks various performance metrics including:
    - CPU usage
    - Memory usage
    - GPU utilization (if available)
    - Network I/O
    - Disk I/O
    - Process-specific metrics
    """

    def __init__(
        self,
        update_interval: float = 1.0,
        max_history: int = 1000,
        enable_gpu_monitoring: bool = True,
    ):
        """
        Initialize the performance monitor.

        Args:
            update_interval: Update interval in seconds
            max_history: Maximum number of data points to keep
            enable_gpu_monitoring: Enable GPU monitoring if available
        """
        self.logger = logging.getLogger("PerformanceMonitor")

        # Configuration
        self.update_interval = update_interval
        self.max_history = max_history
        self.enable_gpu_monitoring = enable_gpu_monitoring

        # Data storage
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.network_history = deque(maxlen=max_history)
        self.disk_history = deque(maxlen=max_history)
        self.gpu_history = deque(maxlen=max_history)

        # Current metrics
        self.current_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_available": 0,
            "memory_used": 0,
            "network_sent": 0,
            "network_recv": 0,
            "disk_read": 0,
            "disk_write": 0,
            "gpu_utilization": 0.0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
        }

        # Process-specific metrics
        self.process_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_rss": 0,
            "memory_vms": 0,
            "num_threads": 0,
            "num_fds": 0,
        }

        # Callbacks
        self.metric_callbacks: List[Callable] = []

        # Async state
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Initialize GPU monitoring if available
        self.gpu_available = False
        if self.enable_gpu_monitoring:
            self._init_gpu_monitoring()

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu_available = True
            self.logger.info("GPU monitoring initialized")
        except ImportError:
            self.logger.warning("pynvml not available, GPU monitoring disabled")
        except Exception as e:
            self.logger.warning(f"GPU monitoring initialization failed: {e}")

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics if available"""
        if not self.gpu_available:
            return {"utilization": 0.0, "memory_used": 0, "memory_total": 0}

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu

            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = memory_info.used
            memory_total = memory_info.total

            return {
                "utilization": gpu_util,
                "memory_used": memory_used,
                "memory_total": memory_total,
            }
        except Exception as e:
            self.logger.error(f"GPU metrics collection failed: {e}")
            return {"utilization": 0.0, "memory_used": 0, "memory_total": 0}

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()

            # Network metrics
            network = psutil.net_io_counters()

            # Disk metrics
            disk = psutil.disk_io_counters()

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "memory_used": memory.used,
                "network_sent": network.bytes_sent,
                "network_recv": network.bytes_recv,
                "disk_read": disk.read_bytes if disk else 0,
                "disk_write": disk.write_bytes if disk else 0,
            }
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return {}

    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get process-specific metrics"""
        try:
            process = psutil.Process()

            return {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_rss": process.memory_info().rss,
                "memory_vms": process.memory_info().vms,
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0,
            }
        except Exception as e:
            self.logger.error(f"Process metrics collection failed: {e}")
            return {}

    def update_metrics(self):
        """Update all performance metrics"""
        timestamp = time.time()

        # Get system metrics
        system_metrics = self._get_system_metrics()
        self.current_metrics.update(system_metrics)

        # Get process metrics
        process_metrics = self._get_process_metrics()
        self.process_metrics.update(process_metrics)

        # Get GPU metrics
        if self.enable_gpu_monitoring:
            gpu_metrics = self._get_gpu_metrics()
            self.current_metrics.update(
                {
                    "gpu_utilization": gpu_metrics["utilization"],
                    "gpu_memory_used": gpu_metrics["memory_used"],
                    "gpu_memory_total": gpu_metrics["memory_total"],
                }
            )

        # Store historical data
        self.cpu_history.append(
            {"timestamp": timestamp, "value": self.current_metrics["cpu_percent"]}
        )

        self.memory_history.append(
            {"timestamp": timestamp, "value": self.current_metrics["memory_percent"]}
        )

        self.network_history.append(
            {
                "timestamp": timestamp,
                "sent": self.current_metrics["network_sent"],
                "recv": self.current_metrics["network_recv"],
            }
        )

        self.disk_history.append(
            {
                "timestamp": timestamp,
                "read": self.current_metrics["disk_read"],
                "write": self.current_metrics["disk_write"],
            }
        )

        if self.enable_gpu_monitoring:
            self.gpu_history.append(
                {
                    "timestamp": timestamp,
                    "utilization": self.current_metrics["gpu_utilization"],
                    "memory_used": self.current_metrics["gpu_memory_used"],
                }
            )

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "system": self.current_metrics.copy(),
            "process": self.process_metrics.copy(),
            "timestamp": time.time(),
        }

    def get_historical_metrics(self, metric_type: str = "all") -> Dict[str, Any]:
        """Get historical performance metrics"""
        if metric_type == "all":
            return {
                "cpu": list(self.cpu_history),
                "memory": list(self.memory_history),
                "network": list(self.network_history),
                "disk": list(self.disk_history),
                "gpu": list(self.gpu_history) if self.enable_gpu_monitoring else [],
            }
        elif metric_type == "cpu":
            return list(self.cpu_history)
        elif metric_type == "memory":
            return list(self.memory_history)
        elif metric_type == "network":
            return list(self.network_history)
        elif metric_type == "disk":
            return list(self.disk_history)
        elif metric_type == "gpu":
            return list(self.gpu_history) if self.enable_gpu_monitoring else []
        else:
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with statistics"""

        def calculate_stats(data_list, key="value"):
            if not data_list:
                return {"avg": 0, "min": 0, "max": 0, "count": 0}

            values = [item[key] for item in data_list if key in item]
            if not values:
                return {"avg": 0, "min": 0, "max": 0, "count": 0}

            return {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        return {
            "cpu": calculate_stats(self.cpu_history),
            "memory": calculate_stats(self.memory_history),
            "network": {
                "sent": calculate_stats(self.network_history, "sent"),
                "recv": calculate_stats(self.network_history, "recv"),
            },
            "disk": {
                "read": calculate_stats(self.disk_history, "read"),
                "write": calculate_stats(self.disk_history, "write"),
            },
            "gpu": (
                calculate_stats(self.gpu_history, "utilization")
                if self.enable_gpu_monitoring
                else {"avg": 0, "min": 0, "max": 0, "count": 0}
            ),
            "current": self.get_current_metrics(),
        }

    def register_metric_callback(self, callback: Callable):
        """Register a callback for metric updates"""
        self.metric_callbacks.append(callback)

    async def start(self):
        """Start the performance monitor"""
        if self.is_running:
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Performance Monitor started")

    async def stop(self):
        """Stop the performance monitor"""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance Monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Update metrics
                self.update_metrics()

                # Trigger callbacks
                current_metrics = self.get_current_metrics()
                for callback in self.metric_callbacks:
                    try:
                        callback(current_metrics)
                    except Exception as e:
                        self.logger.error(f"Metric callback error: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitor loop error: {e}")
                await asyncio.sleep(1)

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds"""
        alerts = []
        current_time = time.time()

        # CPU alert
        if self.current_metrics["cpu_percent"] > 80:
            alerts.append(
                {
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"High CPU usage: {self.current_metrics['cpu_percent']:.1f}%",
                    "timestamp": current_time,
                }
            )

        # Memory alert
        if self.current_metrics["memory_percent"] > 85:
            alerts.append(
                {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"High memory usage: {self.current_metrics['memory_percent']:.1f}%",
                    "timestamp": current_time,
                }
            )

        # GPU alert
        if self.enable_gpu_monitoring and self.current_metrics["gpu_utilization"] > 90:
            alerts.append(
                {
                    "type": "high_gpu",
                    "severity": "warning",
                    "message": f"High GPU utilization: {self.current_metrics['gpu_utilization']:.1f}%",
                    "timestamp": current_time,
                }
            )

        return alerts
