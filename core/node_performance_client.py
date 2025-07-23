#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ NODE PERFORMANCE CLIENT - HARDWARE MONITORING
================================================

Client component for monitoring node performance and reporting
to the Upstream Timing Protocol.

Features:
- Real-time hardware monitoring
- Performance metric collection
- Automatic reporting to Flask server
- Hardware-specific optimization
"""

import json
import logging
import time
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import psutil
import requests
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class HardwareMetrics:
    """Hardware performance metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    network_latency: float
    fractal_sync_time: float
    timestamp: float

class NodePerformanceClient:
    """Client for monitoring and reporting node performance."""
    
    def __init__(self, node_id: str, flask_server_url: str = "http://localhost:5000"):
        """Initialize the node performance client."""
        self.node_id = node_id
        self.flask_server_url = flask_server_url
        
        # Performance tracking
        self.metrics_history: List[HardwareMetrics] = []
        self.last_report_time = 0
        self.report_interval = 5  # seconds
        
        # Hardware detection
        self.gpu_available = self._detect_gpu()
        self.gpu_name = self._get_gpu_name()
        
        # Threading
        self.running = False
        self.monitoring_thread = None
        
        logger.info(f"Node Performance Client initialized for {node_id}")
        logger.info(f"GPU Available: {self.gpu_available} ({self.gpu_name})")
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            # Try NVIDIA GPU detection
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_gpu_name(self) -> str:
        """Get GPU name."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "Unknown"
    
    def collect_metrics(self) -> HardwareMetrics:
        """Collect current hardware metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU metrics
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            
            if self.gpu_available:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            gpu_info = lines[0].split(', ')
                            gpu_usage = float(gpu_info[0])
                            gpu_memory_used = float(gpu_info[1])
                            gpu_memory_total = float(gpu_info[2])
                            gpu_memory_usage = (gpu_memory_used / gpu_memory_total) * 100
                except:
                    pass
            
            # Network latency (ping localhost)
            network_latency = self._measure_network_latency()
            
            # Fractal sync time (simulated for now)
            fractal_sync_time = self._measure_fractal_sync()
            
            return HardwareMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                network_latency=network_latency,
                fractal_sync_time=fractal_sync_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return HardwareMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=0.0,
                gpu_memory_usage=0.0,
                network_latency=0.0,
                fractal_sync_time=0.0,
                timestamp=time.time()
            )
    
    def _measure_network_latency(self) -> float:
        """Measure network latency to Flask server."""
        try:
            start_time = time.time()
            response = requests.get(f"{self.flask_server_url}/api/health", timeout=5)
            end_time = time.time()
            
            if response.status_code == 200:
                return (end_time - start_time) * 1000  # Convert to milliseconds
        except:
            pass
        return 1000.0  # Default high latency if measurement fails
    
    def _measure_fractal_sync(self) -> float:
        """Measure Forever Fractal synchronization time."""
        # This is a placeholder - implement actual fractal sync measurement
        # For now, return a simulated value based on hardware performance
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Simulate fractal sync time based on system load
            base_sync_time = 5.0  # Base 5ms
            load_factor = (cpu_usage + memory_usage) / 200.0  # Normalize to 0-1
            sync_time = base_sync_time + (load_factor * 15.0)  # Max 20ms
            
            return sync_time
        except:
            return 10.0  # Default sync time
    
    def report_metrics(self, metrics: HardwareMetrics):
        """Report metrics to Flask server."""
        try:
            # Prepare metrics data
            metrics_data = {
                'node_id': self.node_id,
                'latency': metrics.network_latency,
                'tick_sync': 2.0,  # Simulated tick sync time
                'cpu_load': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'gpu_memory': metrics.gpu_memory_usage,
                'network_latency': metrics.network_latency,
                'fractal_sync': metrics.fractal_sync_time,
                'status': 'online'
            }
            
            # Send to Flask server
            response = requests.post(
                f"{self.flask_server_url}/api/upstream/nodes/{self.node_id}/update",
                json=metrics_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"Metrics reported successfully")
            else:
                logger.warning(f"Failed to report metrics: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error reporting metrics: {e}")
    
    def register_node(self):
        """Register this node with the Flask server."""
        try:
            # Collect initial metrics
            metrics = self.collect_metrics()
            
            # Prepare registration data
            registration_data = {
                'node_id': self.node_id,
                'latency': metrics.network_latency,
                'tick_sync': 2.0,
                'cpu_load': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'gpu_memory': metrics.gpu_memory_usage,
                'network_latency': metrics.network_latency,
                'fractal_sync': metrics.fractal_sync_time,
                'status': 'online'
            }
            
            # Register with Flask server
            response = requests.post(
                f"{self.flask_server_url}/api/upstream/nodes/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Node registered successfully: {result}")
                return True
            else:
                logger.error(f"Failed to register node: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        def monitoring_loop():
            while self.running:
                try:
                    # Collect metrics
                    metrics = self.collect_metrics()
                    
                    # Store in history
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 100 metrics
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]
                    
                    # Report metrics
                    self.report_metrics(metrics)
                    
                    time.sleep(self.report_interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(self.report_interval)
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Node performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Node performance monitoring stopped") 