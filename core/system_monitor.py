#!/usr/bin/env python3
"""
System Monitor - Real-time System Health and Performance Monitoring
=================================================================

Comprehensive system monitoring and health checking for the Schwabot
mathematical trading framework. Monitors CPU, memory, network, and
trading system performance in real-time.

Key Features:
- Real-time system resource monitoring
- Trading system health checks
- Performance bottleneck detection
- Alert system for critical issues
- Resource usage optimization
- System stability metrics
- Integration with mathematical frameworks

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import threading
import psutil
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from collections import deque, defaultdict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """System status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: float
    memory_available: float
    disk_usage_percent: float
    network_io_sent: float
    network_io_recv: float
    process_count: int
    thread_count: int
    load_average: Tuple[float, float, float]


@dataclass
class TradingSystemMetrics:
    """Trading system specific metrics"""
    
    timestamp: float
    active_strategies: int
    total_positions: int
    total_pnl: float
    order_queue_size: int
    tick_processing_rate: float
    signal_generation_rate: float
    risk_level: float
    system_latency: float


@dataclass
class SystemAlert:
    """System alert container"""
    
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: float
    component: str
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """Real-time system monitoring and health checking"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize system monitor"""
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=self.config.get('max_history_size', 1000))
        self.trading_metrics_history: deque = deque(maxlen=self.config.get('max_history_size', 1000))
        
        # Alert management
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: deque = deque(maxlen=self.config.get('max_alert_history', 100))
        
        # Thresholds and limits
        self.thresholds = self._initialize_thresholds()
        
        # Callbacks and hooks
        self.alert_callbacks: List[Callable[[SystemAlert], None]] = []
        self.metrics_callbacks: List[Callable[[SystemMetrics], None]] = []
        
        # Component health tracking
        self.component_health: Dict[str, SystemStatus] = {}
        self.component_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.monitoring_start_time = time.time()
        self.total_checks = 0
        self.total_alerts = 0
        
        logger.info(f"SystemMonitor v{self.version} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'monitoring_interval': 1.0,  # seconds
            'max_history_size': 1000,
            'max_alert_history': 100,
            'enable_cpu_monitoring': True,
            'enable_memory_monitoring': True,
            'enable_disk_monitoring': True,
            'enable_network_monitoring': True,
            'enable_trading_monitoring': True,
            'enable_alerting': True,
            'enable_performance_tracking': True,
            'alert_cooldown': 60.0,  # seconds between repeated alerts
            'health_check_timeout': 5.0  # seconds
        }
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize monitoring thresholds"""
        return {
            'cpu': {
                'warning': 70.0,
                'critical': 90.0
            },
            'memory': {
                'warning': 80.0,
                'critical': 95.0
            },
            'disk': {
                'warning': 85.0,
                'critical': 95.0
            },
            'network': {
                'warning': 1000000.0,  # 1MB/s
                'critical': 5000000.0  # 5MB/s
            },
            'trading': {
                'latency_warning': 0.1,  # 100ms
                'latency_critical': 0.5,  # 500ms
                'risk_warning': 0.7,
                'risk_critical': 0.9
            }
        }
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]) -> None:
        """Add callback for system alerts"""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Add callback for system metrics"""
        self.metrics_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start system monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect trading metrics if enabled
                if self.config.get('enable_trading_monitoring', True):
                    trading_metrics = self._collect_trading_metrics()
                    self.trading_metrics_history.append(trading_metrics)
                
                # Check thresholds and generate alerts
                if self.config.get('enable_alerting', True):
                    self._check_thresholds(system_metrics)
                
                # Execute callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(system_metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")
                
                # Update performance tracking
                self.total_checks += 1
                
                # Sleep for monitoring interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.get('monitoring_interval', 1.0) - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024 ** 3)  # GB
            memory_available = memory.available / (1024 ** 3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io_sent = network.bytes_sent / (1024 ** 2)  # MB
            network_io_recv = network.bytes_recv / (1024 ** 2)  # MB
            
            # Process metrics
            process_count = len(psutil.pids())
            thread_count = psutil.cpu_count()
            
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except AttributeError:
                # Windows fallback
                load_avg = (cpu_percent / 100.0, cpu_percent / 100.0, cpu_percent / 100.0)
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used=memory_used,
                memory_available=memory_available,
                disk_usage_percent=disk_usage_percent,
                network_io_sent=network_io_sent,
                network_io_recv=network_io_recv,
                process_count=process_count,
                thread_count=thread_count,
                load_average=load_avg
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used=0.0,
                memory_available=0.0,
                disk_usage_percent=0.0,
                network_io_sent=0.0,
                network_io_recv=0.0,
                process_count=0,
                thread_count=0,
                load_average=(0.0, 0.0, 0.0)
            )
    
    def _collect_trading_metrics(self) -> TradingSystemMetrics:
        """Collect trading system specific metrics"""
        try:
            # This would integrate with your trading system components
            # For now, return placeholder metrics
            
            metrics = TradingSystemMetrics(
                timestamp=time.time(),
                active_strategies=0,
                total_positions=0,
                total_pnl=0.0,
                order_queue_size=0,
                tick_processing_rate=0.0,
                signal_generation_rate=0.0,
                risk_level=0.0,
                system_latency=0.0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return TradingSystemMetrics(
                timestamp=time.time(),
                active_strategies=0,
                total_positions=0,
                total_pnl=0.0,
                order_queue_size=0,
                tick_processing_rate=0.0,
                signal_generation_rate=0.0,
                risk_level=0.0,
                system_latency=0.0
            )
    
    def _check_thresholds(self, metrics: SystemMetrics) -> None:
        """Check metrics against thresholds and generate alerts"""
        try:
            # Check CPU usage
            if self.config.get('enable_cpu_monitoring', True):
                self._check_cpu_thresholds(metrics)
            
            # Check memory usage
            if self.config.get('enable_memory_monitoring', True):
                self._check_memory_thresholds(metrics)
            
            # Check disk usage
            if self.config.get('enable_disk_monitoring', True):
                self._check_disk_thresholds(metrics)
            
            # Check network usage
            if self.config.get('enable_network_monitoring', True):
                self._check_network_thresholds(metrics)
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    def _check_cpu_thresholds(self, metrics: SystemMetrics) -> None:
        """Check CPU usage thresholds"""
        try:
            cpu_warning = self.thresholds['cpu']['warning']
            cpu_critical = self.thresholds['cpu']['critical']
            
            if metrics.cpu_percent >= cpu_critical:
                self._create_alert(
                    'cpu_critical',
                    AlertLevel.CRITICAL,
                    f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                    'system',
                    'cpu_percent',
                    metrics.cpu_percent,
                    cpu_critical
                )
            elif metrics.cpu_percent >= cpu_warning:
                self._create_alert(
                    'cpu_warning',
                    AlertLevel.WARNING,
                    f"CPU usage high: {metrics.cpu_percent:.1f}%",
                    'system',
                    'cpu_percent',
                    metrics.cpu_percent,
                    cpu_warning
                )
            
        except Exception as e:
            logger.error(f"Error checking CPU thresholds: {e}")
    
    def _check_memory_thresholds(self, metrics: SystemMetrics) -> None:
        """Check memory usage thresholds"""
        try:
            memory_warning = self.thresholds['memory']['warning']
            memory_critical = self.thresholds['memory']['critical']
            
            if metrics.memory_percent >= memory_critical:
                self._create_alert(
                    'memory_critical',
                    AlertLevel.CRITICAL,
                    f"Memory usage critical: {metrics.memory_percent:.1f}%",
                    'system',
                    'memory_percent',
                    metrics.memory_percent,
                    memory_critical
                )
            elif metrics.memory_percent >= memory_warning:
                self._create_alert(
                    'memory_warning',
                    AlertLevel.WARNING,
                    f"Memory usage high: {metrics.memory_percent:.1f}%",
                    'system',
                    'memory_percent',
                    metrics.memory_percent,
                    memory_warning
                )
            
        except Exception as e:
            logger.error(f"Error checking memory thresholds: {e}")
    
    def _check_disk_thresholds(self, metrics: SystemMetrics) -> None:
        """Check disk usage thresholds"""
        try:
            disk_warning = self.thresholds['disk']['warning']
            disk_critical = self.thresholds['disk']['critical']
            
            if metrics.disk_usage_percent >= disk_critical:
                self._create_alert(
                    'disk_critical',
                    AlertLevel.CRITICAL,
                    f"Disk usage critical: {metrics.disk_usage_percent:.1f}%",
                    'system',
                    'disk_usage_percent',
                    metrics.disk_usage_percent,
                    disk_critical
                )
            elif metrics.disk_usage_percent >= disk_warning:
                self._create_alert(
                    'disk_warning',
                    AlertLevel.WARNING,
                    f"Disk usage high: {metrics.disk_usage_percent:.1f}%",
                    'system',
                    'disk_usage_percent',
                    metrics.disk_usage_percent,
                    disk_warning
                )
            
        except Exception as e:
            logger.error(f"Error checking disk thresholds: {e}")
    
    def _check_network_thresholds(self, metrics: SystemMetrics) -> None:
        """Check network usage thresholds"""
        try:
            network_warning = self.thresholds['network']['warning']
            network_critical = self.thresholds['network']['critical']
            
            total_network_io = metrics.network_io_sent + metrics.network_io_recv
            
            if total_network_io >= network_critical:
                self._create_alert(
                    'network_critical',
                    AlertLevel.CRITICAL,
                    f"Network I/O critical: {total_network_io:.2f} MB/s",
                    'system',
                    'network_io_total',
                    total_network_io,
                    network_critical
                )
            elif total_network_io >= network_warning:
                self._create_alert(
                    'network_warning',
                    AlertLevel.WARNING,
                    f"Network I/O high: {total_network_io:.2f} MB/s",
                    'system',
                    'network_io_total',
                    total_network_io,
                    network_warning
                )
            
        except Exception as e:
            logger.error(f"Error checking network thresholds: {e}")
    
    def _create_alert(self, alert_id: str, level: AlertLevel, message: str,
                     component: str, metric_name: str, metric_value: float,
                     threshold: float) -> None:
        """Create and dispatch system alert"""
        try:
            # Check if alert already exists and is within cooldown
            if alert_id in self.active_alerts:
                existing_alert = self.active_alerts[alert_id]
                cooldown = self.config.get('alert_cooldown', 60.0)
                if time.time() - existing_alert.timestamp < cooldown:
                    return
            
            alert = SystemAlert(
                alert_id=alert_id,
                level=level,
                message=message,
                timestamp=time.time(),
                component=component,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.total_alerts += 1
            
            # Execute callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"System alert: {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                del self.active_alerts[alert_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_system_status(self) -> SystemStatus:
        """Get overall system status"""
        try:
            if not self.system_metrics_history:
                return SystemStatus.OFFLINE
            
            # Check for critical alerts
            critical_alerts = [a for a in self.active_alerts.values() 
                             if a.level == AlertLevel.CRITICAL and not a.resolved]
            if critical_alerts:
                return SystemStatus.CRITICAL
            
            # Check for warning alerts
            warning_alerts = [a for a in self.active_alerts.values() 
                            if a.level == AlertLevel.WARNING and not a.resolved]
            if warning_alerts:
                return SystemStatus.WARNING
            
            return SystemStatus.HEALTHY
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return SystemStatus.OFFLINE
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics"""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def get_metrics_history(self, count: int = 100) -> List[SystemMetrics]:
        """Get system metrics history"""
        try:
            metrics = list(self.system_metrics_history)
            return metrics[-count:] if count > 0 else metrics
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            uptime = time.time() - self.monitoring_start_time
            
            return {
                'version': self.version,
                'uptime': uptime,
                'total_checks': self.total_checks,
                'total_alerts': self.total_alerts
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
