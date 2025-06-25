from __future__ import annotations

# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Ops and Observability - Comprehensive Monitoring and Logging System.

This module provides enterprise-grade observability including:
- Structured logging with ELK/Loki integration
- Prometheus metrics for latency, PnL, hit rate, memory, GC
- Health endpoints and monitoring
- Slack alerts and notifications
- Integration with all Schwabot core systems
- Unified mathematics and trading metrics
"""


import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import socket
import psutil
import gc
from pathlib import Path
import aiohttp
import structlog
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, generate_latest,
    CONTENT_TYPE_LATEST, start_http_server
)

# Import unified mathematics
try:
    from core.unified_mathematics_config import get_unified_math
    unified_math = get_unified_math()
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False

# Import all core systems for integration
try:
    from core.capital_controls import get_capital_controls, get_capital_status
    from core.enhanced_risk_manager import get_enhanced_risk_manager, get_risk_summary
    from core.risk_guard import get_risk_guard, get_risk_status
    from core.vecu_core import get_vecu_core
    from core.ferris_rde_core import get_ferris_rde
    from core.secure_api_manager import get_secure_api_manager
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False

# Import centralized CLI handler
try:
#     from core.utils.windows_cli_compatibility import (  # F811: duplicate import
        safe_print, safe_format_error, log_safe
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"
    def log_safe(logger, level: str, message: str) -> None:
        getattr(logger, level.lower())(message)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    component: str
    trace_id: str
    span_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class MetricData:
    """Metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str
    timestamp: datetime
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Alert:
    """Alert data."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrometheusMetrics:
    """Prometheus metrics collection."""

    def __init__(self) -> None:
        """Initialize Prometheus metrics."""
        # Trading metrics
        self.trades_total = Counter('schwabot_trades_total', 'Total number of trades', ['asset', 'side', 'status'])
        self.trade_pnl = Histogram('schwabot_trade_pnl', 'Trade PnL distribution', ['asset', 'side'])
        self.trade_latency = Histogram('schwabot_trade_latency_seconds', 'Trade execution latency', ['asset', 'side'])
        self.hit_rate = Gauge('schwabot_hit_rate', 'Trading hit rate percentage', ['asset'])
        self.portfolio_value = Gauge('schwabot_portfolio_value_usd', 'Current portfolio value in USD')
        self.portfolio_pnl = Gauge('schwabot_portfolio_pnl_usd', 'Current portfolio PnL in USD')

        # Risk metrics
        self.var_95 = Gauge('schwabot_var_95_percent', '95% Value at Risk')
        self.var_99 = Gauge('schwabot_var_99_percent', '99% Value at Risk')
        self.portfolio_volatility = Gauge('schwabot_portfolio_volatility', 'Portfolio volatility')
        self.drawdown = Gauge('schwabot_drawdown_percent', 'Current drawdown percentage')

        # System metrics
        self.memory_usage_bytes = Gauge('schwabot_memory_usage_bytes', 'Memory usage in bytes')
        self.cpu_usage_percent = Gauge('schwabot_cpu_usage_percent', 'CPU usage percentage')
        self.gc_collections = Counter('schwabot_gc_collections_total', 'Total garbage collection events')
        self.gc_time_seconds = Histogram('schwabot_gc_time_seconds', 'Garbage collection time')

        # API metrics
        self.api_requests_total = Counter('schwabot_api_requests_total', 'Total API requests', ['api_type', 'endpoint', 'status'])
        self.api_latency = Histogram('schwabot_api_latency_seconds', 'API request latency', ['api_type', 'endpoint'])
        self.api_errors = Counter('schwabot_api_errors_total', 'Total API errors', ['api_type', 'endpoint', 'error_type'])

        # VECU and Ferris metrics
        self.vecu_timing_accuracy = Gauge('schwabot_vecu_timing_accuracy', 'VECU timing accuracy')
        self.ferris_wheel_phase = Gauge('schwabot_ferris_wheel_phase', 'Current Ferris wheel phase')
        self.ferris_wheel_confidence = Gauge('schwabot_ferris_wheel_confidence', 'Ferris wheel confidence score')

        # Capital controls metrics
        self.position_size_requests = Counter('schwabot_position_size_requests_total', 'Position size calculation requests', ['method'])
        self.rebalancing_events = Counter('schwabot_rebalancing_events_total', 'Portfolio rebalancing events')
        self.risk_violations = Counter('schwabot_risk_violations_total', 'Risk limit violations', ['violation_type'])

        # Unified mathematics metrics
        self.math_operations = Counter('schwabot_math_operations_total', 'Mathematical operations', ['operation_type'])
        self.math_latency = Histogram('schwabot_math_latency_seconds', 'Mathematical operation latency', ['operation_type'])

        # Circuit breaker metrics
        self.circuit_breaker_trips = Counter('schwabot_circuit_breaker_trips_total', 'Circuit breaker trips', ['trigger'])
        self.circuit_breaker_state = Gauge('schwabot_circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open)')

        # Health check metrics
        self.health_check_duration = Histogram('schwabot_health_check_duration_seconds', 'Health check duration', ['component'])
        self.health_check_status = Gauge('schwabot_health_check_status', 'Health check status (0=unhealthy, 1=healthy)', ['component'])


class StructuredLogger:
    """Structured logging with ELK/Loki integration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize structured logger."""
        self.config = config
        self.log_queue = queue.Queue()
        self.log_worker = None
        self.running = False

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger()

        # Start log worker
        self.start_log_worker()

    def start_log_worker(self) -> None:
        """Start log worker thread."""
        self.running = True
        self.log_worker = threading.Thread(target=self._log_worker, daemon=True)
        self.log_worker.start()

    def _log_worker(self) -> None:
        """Log worker thread."""
        while self.running:
            try:
                log_entry = self.log_queue.get(timeout=1)
                self._send_log(log_entry)
            except queue.Empty:
                continue
            except Exception as e:
                safe_print(f"Log worker error: {e}")

    def _send_log(self, log_entry: LogEntry) -> None:
        """Send log to ELK/Loki."""
        try:
            # Format log entry for ELK/Loki
            log_data = {
                'timestamp': log_entry.timestamp.isoformat(),
                'level': log_entry.level.value,
                'message': log_entry.message,
                'component': log_entry.component,
                'trace_id': log_entry.trace_id,
                'span_id': log_entry.span_id,
                'user_id': log_entry.user_id,
                'session_id': log_entry.session_id,
                'tags': log_entry.tags,
                **log_entry.metadata
            }

            # Send to configured endpoints
            if self.config.get('elk_enabled'):
                self._send_to_elk(log_data)

            if self.config.get('loki_enabled'):
                self._send_to_loki(log_data)

            # Also log locally
            log_method = getattr(self.logger, log_entry.level.value)
            log_method(
                log_entry.message,
                component=log_entry.component,
                trace_id=log_entry.trace_id,
                **log_entry.metadata
            )

        except Exception as e:
            safe_print(f"Log sending error: {e}")

    def _send_to_elk(self, log_data: Dict[str, Any]) -> None:
        """Send log to ELK stack."""
        # Implementation for ELK stack
        pass

    def _send_to_loki(self, log_data: Dict[str, Any]) -> None:
        """Send log to Loki."""
        # Implementation for Loki
        pass

    def unified_math.log(self, level: LogLevel, message: str, component: str, **kwargs) -> None:
        """Log a message."""
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            component=component,
            trace_id=kwargs.get('trace_id', str(uuid.uuid4())),
            span_id=kwargs.get('span_id', str(uuid.uuid4())),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            metadata={k: v for k, v in kwargs.items() if k not in ['trace_id', 'span_id', 'user_id', 'session_id']},
            tags=kwargs.get('tags', [])
        )

        self.log_queue.put(log_entry)


class HealthMonitor:
    """Health monitoring system."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize health monitor."""
        self.config = config
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        self.monitoring_interval = config.get('health_check_interval', 30)
        self.monitoring_thread = None
        self.running = False

        # Register default health checks
        self._register_default_health_checks()

    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        self.register_health_check("system", self._check_system_health)
        self.register_health_check("memory", self._check_memory_health)
        self.register_health_check("cpu", self._check_cpu_health)
        self.register_health_check("disk", self._check_disk_health)
        self.register_health_check("network", self._check_network_health)

        if CORE_SYSTEMS_AVAILABLE:
            self.register_health_check("capital_controls", self._check_capital_controls_health)
            self.register_health_check("risk_manager", self._check_risk_manager_health)
            self.register_health_check("risk_guard", self._check_risk_guard_health)
            self.register_health_check("vecu", self._check_vecu_health)
            self.register_health_check("ferris_rde", self._check_ferris_rde_health)
            self.register_health_check("api_manager", self._check_api_manager_health)

    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func

    def start_monitoring(self) -> None:
        """Start health monitoring."""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()

    def _monitoring_worker(self):
        """Health monitoring worker thread."""
        while self.running:
            try:
                for name, check_func in self.health_checks.items():
                    start_time = time.time()
                    try:
                        result = check_func()
                        response_time = time.time() - start_time

                        health_check = HealthCheck(
                            component=name,
                            status="healthy" if result else "unhealthy",
                            timestamp=datetime.now(),
                            response_time=response_time,
                            details=result if isinstance(result, dict) else {}
                        )
                    except Exception as e:
                        response_time = time.time() - start_time
                        health_check = HealthCheck(
                            component=name,
                            status="unhealthy",
                            timestamp=datetime.now(),
                            response_time=response_time,
                            error=str(e)
                        )

                    self.health_status[name] = health_check

                time.sleep(self.monitoring_interval)

            except Exception as e:
                safe_print(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _check_system_health(self) -> Dict[str, Any]:
        """Check system health."""
        return {
            'uptime': time.time(),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'platform': psutil.sys.platform
        }

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }

    def _check_cpu_health(self) -> Dict[str, Any]:
        """Check CPU health."""
        return {
            'usage_percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count()
        }

    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health."""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }

    def _check_network_health(self) -> Dict[str, Any]:
        """Check network health."""
        return {
            'connections': len(psutil.net_connections())
        }

    def _check_capital_controls_health(self) -> Dict[str, Any]:
        """Check capital controls health."""
        try:
            capital_controls = get_capital_controls()
            status = get_capital_status()
            return {
                'total_capital': status.get('total_capital', 0),
                'current_capital': status.get('current_capital', 0),
                'drawdown': status.get('current_drawdown', 0)
            }
        except Exception as e:
            raise Exception(f"Capital controls health check failed: {e}")

    def _check_risk_manager_health(self) -> Dict[str, Any]:
        """Check risk manager health."""
        try:
            risk_manager = get_enhanced_risk_manager()
            summary = get_risk_summary()
            return {
                'total_risk_checks': summary.get('total_risk_checks', 0),
                'risk_violations': summary.get('risk_violations', 0),
                'monitoring_active': summary.get('monitoring_active', False)
            }
        except Exception as e:
            raise Exception(f"Risk manager health check failed: {e}")

    def _check_risk_guard_health(self) -> Dict[str, Any]:
        """Check risk guard health."""
        try:
            risk_guard = get_risk_guard()
            status = get_risk_status()
            return {
                'circuit_breaker_state': status.get('circuit_breaker_state', 'unknown'),
                'trading_allowed': status.get('trading_allowed', False)
            }
        except Exception as e:
            raise Exception(f"Risk guard health check failed: {e}")

    def _check_vecu_health(self) -> Dict[str, Any]:
        """Check VECU health."""
        try:
            vecu = get_vecu_core()
            return {
                'status': 'operational',
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            raise Exception(f"VECU health check failed: {e}")

    def _check_ferris_rde_health(self) -> Dict[str, Any]:
        """Check Ferris RDE health."""
        try:
            ferris = get_ferris_rde()
            return {
                'status': 'operational',
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            raise Exception(f"Ferris RDE health check failed: {e}")

    def _check_api_manager_health(self) -> Dict[str, Any]:
        """Check API manager health."""
        try:
            api_manager = get_secure_api_manager()
            stats = api_manager.get_api_statistics()
            return {
                'total_requests': stats.get('total_requests', 0),
                'successful_requests': stats.get('successful_requests', 0),
                'error_rate': stats.get('error_rate', 0)
            }
        except Exception as e:
            raise Exception(f"API manager health check failed: {e}")

    def get_health_status(self) -> Dict[str, HealthCheck]:
        """Get current health status."""
        return self.health_status

    def get_overall_health(self) -> str:
        """Get overall health status."""
        if not self.health_status:
            return "unknown"

        unhealthy_count = sum(1 for check in self.health_status.values() if check.status == "unhealthy")
        total_count = len(self.health_status)

        if unhealthy_count == 0:
            return "healthy"
        elif unhealthy_count < total_count / 2:
            return "degraded"
        else:
            return "unhealthy"


class AlertManager:
    """Alert management system."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize alert manager."""
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[str, Callable] = {}
        self.slack_webhook_url = config.get('slack_webhook_url')

        # Register alert handlers
        self._register_alert_handlers()

    def _register_alert_handlers(self) -> None:
        """Register alert handlers."""
        if self.slack_webhook_url:
            self.register_alert_handler("slack", self._send_slack_alert)

    def register_alert_handler(self, name: str, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers[name] = handler

    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.alerts.append(alert)

        # Send alert through handlers
        for handler_name, handler in self.alert_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                safe_print(f"Alert handler {handler_name} failed: {e}")

        return alert

    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack."""
        if not self.slack_webhook_url:
            return

        try:
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "#36a64",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }

            slack_message = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Component",
                            "value": alert.component,
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        }
                    ],
                    "footer": "Schwabot Alert System"
                }]
            }

            # Add metadata if present
            if alert.metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in alert.metadata.items()])
                slack_message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": metadata_text,
                    "short": False
                })

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=slack_message,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 200:
                        safe_print(f"Slack alert failed: {response.status}")

        except Exception as e:
            safe_print(f"Slack alert error: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get active (unacknowledged) alerts."""
        return [alert for alert in self.alerts if not alert.acknowledged]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                break


class OpsObservability:
    """
    Ops and Observability - Comprehensive monitoring and logging system.

    Provides enterprise-grade observability including:
    - Structured logging with ELK/Loki integration
    - Prometheus metrics for latency, PnL, hit rate, memory, GC
    - Health endpoints and monitoring
    - Slack alerts and notifications
    - Integration with all Schwabot core systems
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Ops and Observability system."""
        self.config = config or {}

        # Initialize components
        self.metrics = PrometheusMetrics()
        self.logger = StructuredLogger(self.config)
        self.health_monitor = HealthMonitor(self.config)
        self.alert_manager = AlertManager(self.config)

        # Start services
        self._start_services()

        # Performance tracking
        self.start_time = time.time()
        self.total_operations = 0

        safe_safe_print("🔍 Ops and Observability initialized")

    def _start_services(self) -> None:
        """Start observability services."""
        # Start Prometheus metrics server
        metrics_port = self.config.get('prometheus_port', 8000)
        try:
            start_http_server(metrics_port)
            safe_safe_print(f"✅ Prometheus metrics server started on port {metrics_port}")
        except Exception as e:
            safe_safe_print(f"❌ Prometheus server failed: {safe_format_error(e, 'prometheus_start')}")

        # Start health monitoring
        self.health_monitor.start_monitoring()
        safe_safe_print("✅ Health monitoring started")

    def log_operation(
        self,
        operation: str,
        component: str,
        level: LogLevel = LogLevel.INFO,
        duration: Optional[float] = None,
        success: Optional[bool] = None,
        **kwargs
    ) -> None:
        """Log an operation with metrics."""
        try:
            # Update metrics
            if duration is not None:
                if operation == "trade":
                    self.metrics.trade_latency.observe(duration)
                elif operation == "api_request":
                    self.metrics.api_latency.observe(duration)
                elif operation == "math_operation":
                    self.metrics.math_latency.observe(duration)
                elif operation == "health_check":
                    self.metrics.health_check_duration.observe(duration)

            # Log operation
            self.logger.log(
                level=level,
                message=f"Operation: {operation}",
                component=component,
                duration=duration,
                success=success,
                **kwargs
            )

            self.total_operations += 1

        except Exception as e:
            safe_safe_print(f"❌ Operation logging failed: {safe_format_error(e, 'operation_logging')}")

    def record_trade(
        self,
        asset: str,
        side: str,
        pnl: float,
        latency: float,
        success: bool
    ) -> None:
        """Record trade metrics."""
        try:
            # Update trade metrics
            self.metrics.trades_total.labels(asset=asset, side=side, status="success" if success else "failed").inc()
            self.metrics.trade_pnl.labels(asset=asset, side=side).observe(pnl)
            self.metrics.trade_latency.labels(asset=asset, side=side).observe(latency)

            # Log trade
            self.log_operation(
                operation="trade",
                component="trading_engine",
                level=LogLevel.INFO if success else LogLevel.ERROR,
                duration=latency,
                success=success,
                asset=asset,
                side=side,
                pnl=pnl
            )

        except Exception as e:
            safe_safe_print(f"❌ Trade recording failed: {safe_format_error(e, 'trade_recording')}")

    def record_api_request(
        self,
        api_type: str,
        endpoint: str,
        status_code: int,
        latency: float,
        error_type: Optional[str] = None
    ) -> None:
        """Record API request metrics."""
        try:
            # Update API metrics
            status = "success" if status_code < 400 else "error"
            self.metrics.api_requests_total.labels(api_type=api_type, endpoint=endpoint, status=status).inc()
            self.metrics.api_latency.labels(api_type=api_type, endpoint=endpoint).observe(latency)

            if error_type:
                self.metrics.api_errors.labels(api_type=api_type, endpoint=endpoint, error_type=error_type).inc()

            # Log API request
            self.log_operation(
                operation="api_request",
                component="api_manager",
                level=LogLevel.INFO if status_code < 400 else LogLevel.ERROR,
                duration=latency,
                success=status_code < 400,
                api_type=api_type,
                endpoint=endpoint,
                status_code=status_code,
                error_type=error_type
            )

        except Exception as e:
            safe_safe_print(f"❌ API recording failed: {safe_format_error(e, 'api_recording')}")

    def record_risk_violation(
        self,
        violation_type: str,
        component: str,
        details: Dict[str, Any]
    ) -> None:
        """Record risk violation."""
        try:
            # Update risk metrics
            self.metrics.risk_violations.labels(violation_type=violation_type).inc()

            # Create alert
            self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Risk Violation: {violation_type}",
                message=f"Risk violation detected in {component}",
                component=component,
                metadata=details
            )

            # Log violation
            self.log_operation(
                operation="risk_violation",
                component=component,
                level=LogLevel.WARNING,
                violation_type=violation_type,
                **details
            )

        except Exception as e:
            safe_safe_print(f"❌ Risk violation recording failed: {safe_format_error(e, 'risk_violation_recording')}")

    def record_math_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool,
        **kwargs
    ) -> None:
        """Record mathematical operation."""
        try:
            # Update math metrics
            self.metrics.math_operations.labels(operation_type=operation_type).inc()
            self.metrics.math_latency.labels(operation_type=operation_type).observe(duration)

            # Log operation
            self.log_operation(
                operation="math_operation",
                component="unified_mathematics",
                level=LogLevel.INFO if success else LogLevel.ERROR,
                duration=duration,
                success=success,
                operation_type=operation_type,
                **kwargs
            )

        except Exception as e:
            safe_safe_print(f"❌ Math operation recording failed: {safe_format_error(e, 'math_recording')}")

    def update_system_metrics(self) -> None:
        """Update system metrics."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.memory_usage_bytes.set(memory.used)

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.cpu_usage_percent.set(cpu_percent)

            # GC metrics
            gc_stats = gc.get_stats()
            for stat in gc_stats:
                self.metrics.gc_collections.labels(generation=stat['generation']).inc()
                self.metrics.gc_time_seconds.labels(generation=stat['generation']).observe(stat['collections_duration'])

            # Update core system metrics if available
            if CORE_SYSTEMS_AVAILABLE:
                self._update_core_system_metrics()

        except Exception as e:
            safe_safe_print(f"❌ System metrics update failed: {safe_format_error(e, 'system_metrics')}")

    def _update_core_system_metrics(self) -> None:
        """Update core system metrics."""
        try:
            # Capital controls metrics
            capital_status = get_capital_status()
            self.metrics.portfolio_value.set(capital_status.get('current_capital', 0))
            self.metrics.portfolio_pnl.set(capital_status.get('total_pnl', 0))
            self.metrics.drawdown.set(capital_status.get('current_drawdown', 0) * 100)

            # Risk manager metrics
            risk_summary = get_risk_summary()
            if risk_summary.get('latest_metrics'):
                metrics = risk_summary['latest_metrics']
                self.metrics.var_95.set(metrics.get('var_95', 0) * 100)
                self.metrics.var_99.set(metrics.get('var_99', 0) * 100)
                self.metrics.portfolio_volatility.set(metrics.get('volatility', 0))

            # Risk guard metrics
            risk_status = get_risk_status()
            circuit_breaker_state = 1 if risk_status.get('circuit_breaker_state') == 'open' else 0
            self.metrics.circuit_breaker_state.set(circuit_breaker_state)

            # API manager metrics
            api_manager = get_secure_api_manager()
            api_stats = api_manager.get_api_statistics()
            # API metrics are updated in record_api_request

        except Exception as e:
            safe_safe_print(f"❌ Core system metrics update failed: {safe_format_error(e, 'core_metrics')}")

    def get_health_endpoint(self) -> Dict[str, Any]:
        """Get health endpoint data."""
        try:
            overall_health = self.health_monitor.get_overall_health()
            health_status = self.health_monitor.get_health_status()

            return {
                'status': overall_health,
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - self.start_time,
                'version': '1.0.0',
                'components': {
                    name: {
                        'status': check.status,
                        'response_time': check.response_time,
                        'last_check': check.timestamp.isoformat(),
                        'details': check.details,
                        'error': check.error
                    }
                    for name, check in health_status.items()
                }
            }

        except Exception as e:
            safe_safe_print(f"❌ Health endpoint failed: {safe_format_error(e, 'health_endpoint')}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_metrics_endpoint(self) -> str:
        """Get Prometheus metrics endpoint."""
        try:
            return generate_latest()
        except Exception as e:
            safe_safe_print(f"❌ Metrics endpoint failed: {safe_format_error(e, 'metrics_endpoint')}")
            return ""

    def get_observability_summary(self) -> Dict[str, Any]:
        """Get observability system summary."""
        return {
            'uptime': time.time() - self.start_time,
            'total_operations': self.total_operations,
            'health_status': self.health_monitor.get_overall_health(),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'components': {
                'metrics': 'active',
                'logging': 'active',
                'health_monitoring': 'active',
                'alerting': 'active'
            }
        }


# Global Ops and Observability instance
ops_observability = OpsObservability()


# Convenience functions for external access
def get_ops_observability() -> OpsObservability:
    """Get global Ops and Observability instance."""
    return ops_observability


def log_operation(
    operation: str,
    component: str,
    level: LogLevel = LogLevel.INFO,
    duration: Optional[float] = None,
    success: Optional[bool] = None,
    **kwargs
) -> None:
    """Log an operation."""
    ops_observability.log_operation(operation, component, level, duration, success, **kwargs)


def record_trade(asset: str, side: str, pnl: float, latency: float, success: bool) -> None:
    """Record trade metrics."""
    ops_observability.record_trade(asset, side, pnl, latency, success)


def record_api_request(api_type: str, endpoint: str, status_code: int, latency: float, error_type: Optional[str] = None) -> None:
    """Record API request metrics."""
    ops_observability.record_api_request(api_type, endpoint, status_code, latency, error_type)


def record_risk_violation(violation_type: str, component: str, details: Dict[str, Any]) -> None:
    """Record risk violation."""
    ops_observability.record_risk_violation(violation_type, component, details)


def record_math_operation(operation_type: str, duration: float, success: bool, **kwargs) -> None:
    """Record mathematical operation."""
    ops_observability.record_math_operation(operation_type, duration, success, **kwargs)


def get_health_endpoint() -> Dict[str, Any]:
    """Get health endpoint data."""
    return ops_observability.get_health_endpoint()


def get_metrics_endpoint() -> str:
    """Get Prometheus metrics endpoint."""
    return ops_observability.get_metrics_endpoint()


def get_observability_summary() -> Dict[str, Any]:
    """Get observability system summary."""
    return ops_observability.get_observability_summary()


# Example usage
if __name__ == "__main__":
    # Test Ops and Observability
    safe_print("🔍 Testing Ops and Observability...")

    ops = get_ops_observability()

    # Test operation logging
    log_operation(
        operation="test_operation",
        component="test_component",
        level=LogLevel.INFO,
        duration=0.1,
        success=True,
        test_data="example"
    )

    # Test trade recording
    record_trade("BTC", "buy", 150.0, 0.05, True)

    # Test API recording
    record_api_request("coinmarketcap", "/v1/cryptocurrency/quotes/latest", 200, 0.1)

    # Test risk violation recording
    record_risk_violation(
        "drawdown_limit",
        "capital_controls",
        {"current_drawdown": 0.25, "limit": 0.20}
    )

    # Test math operation recording
    record_math_operation(
        "eigenvector_calculation",
        0.02,
        True,
        matrix_size=100
    )

    # Update system metrics
    ops.update_system_metrics()

    # Get health endpoint
    health = get_health_endpoint()
    safe_print(f"✅ Health status: {health['status']}")

    # Get observability summary
    summary = get_observability_summary()
    safe_print(f"✅ Observability summary: {summary}")
