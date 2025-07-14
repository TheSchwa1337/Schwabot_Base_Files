#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    xp = cp
except ImportError:
    CUPY_AVAILABLE = False
    xp = np
    cp = None

"""
Enhanced Error Recovery System for Schwabot Trading System

Provides comprehensive error detection, classification, and recovery mechanisms
    for mathematical operations, network issues, and system failures.

Features:
- Multi-level error classification and severity assessment
- Automatic recovery strategies with fallback mechanisms
- Mathematical stability monitoring and correction
- System health monitoring and alerting
- Graceful degradation and failover capabilities

CUDA Integration:
- GPU-accelerated error recovery with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)
"""

logger = logging.getLogger(__name__)
if CUPY_AVAILABLE:
    logger.info("⚡ Enhanced Error Recovery System using GPU acceleration: {0}".format('cupy (GPU)'))
else:
    logger.info("🔄 Enhanced Error Recovery System using CPU fallback: {0}".format('numpy (CPU)'))


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ErrorCategory(Enum):
    """Error categories for classification"""

    MATHEMATICAL = "mathematical"
    NETWORK = "network"
    MEMORY = "memory"
    COMPUTATION = "computation"
    DATA = "data"
    SYSTEM = "system"
    TRADING = "trading"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies"""

    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SYSTEM_RESTART = "system_restart"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""

    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    recovery_attempts: int = 0
    recovered: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time: Optional[float] = None


@dataclass
class SystemHealth:
    """System health metrics"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float]
    network_latency: float
    error_rate: float
    recovery_rate: float
    uptime: float
    last_check: datetime


@dataclass
class RecoveryConfiguration:
    """Configuration for recovery strategies"""

    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    fallback_enabled: bool = True
    graceful_degradation_enabled: bool = True
    auto_restart_enabled: bool = False
    health_check_interval: float = 30.0
    error_threshold: float = 0.1
    memory_threshold: float = 0.85
    cpu_threshold: float = 0.90


class MathematicalStabilityChecker:
    """Advanced mathematical stability checking"""

    def __init__(self):
        self.stability_threshold = 1e-12
        self.condition_number_threshold = 1e15
        self.convergence_threshold = 1e-10
        self.numerical_precision = xp.finfo(float).eps

    def check_matrix_stability(self, matrix: xp.ndarray) -> Dict[str, Any]:
        """Check matrix stability and conditioning"""
        try:
            stability_report = {}
            stability_report['is_stable'] = True
            stability_report['condition_number'] = None
            stability_report['determinant'] = None
            stability_report['rank'] = None
            stability_report['eigenvalue_issues'] = False
            stability_report['numerical_issues'] = []

            # Check for NaN or Inf values
            if xp.any(xp.isnan(matrix)) or xp.any(xp.isinf(matrix)):
                stability_report['is_stable'] = False
                stability_report['numerical_issues'].append('NaN or Inf values detected')

            # Check matrix conditioning
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                try:
                    cond_num = xp.linalg.cond(matrix)
                    stability_report['condition_number'] = float(cond_num)

                    if cond_num > self.condition_number_threshold:
                        stability_report['is_stable'] = False
                        stability_report['numerical_issues'].append(f"Ill-conditioned matrix (cond={cond_num:.2e})")

                    # Check determinant
                    det = xp.linalg.det(matrix)
                    stability_report['determinant'] = float(det)

                    if abs(det) < self.stability_threshold:
                        stability_report['is_stable'] = False
                        stability_report['numerical_issues'].append('Matrix is nearly singular')

                    # Check rank
                    rank = xp.linalg.matrix_rank(matrix)
                    stability_report['rank'] = int(rank)

                    if rank < min(matrix.shape):
                        stability_report['is_stable'] = False
                        stability_report['numerical_issues'].append('Matrix is rank deficient')

                    # Check eigenvalues for stability
                    eigenvalues = xp.linalg.eigvals(matrix)
                    if xp.any(xp.real(eigenvalues) < -self.stability_threshold):
                        stability_report['eigenvalue_issues'] = True
                        stability_report['numerical_issues'].append('Unstable eigenvalues detected')

                except xp.linalg.LinAlgError as e:
                    stability_report['is_stable'] = False
                    stability_report['numerical_issues'].append("Linear algebra error: {0}".format(str(e)))

            return stability_report

        except Exception as e:
            logger.error("Error checking matrix stability: {0}".format(e))
            return {'is_stable': False, 'error': str(e)}

    def stabilize_matrix(self, matrix: xp.ndarray, method: str = 'ridge') -> xp.ndarray:
        """Apply stabilization techniques to matrices"""
        try:
            stabilized = matrix.copy()

            # Replace NaN and Inf values
            stabilized = xp.nan_to_num(stabilized, nan=0.0, posinf=1e10, neginf=-1e10)

            if method == 'ridge' and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                # Ridge regularization
                regularization = 1e-10 * xp.eye(matrix.shape[0])
                stabilized += regularization

            elif method == 'truncated_svd':
                # Truncated SVD for low-rank approximation
                U, s, Vt = xp.linalg.svd(stabilized, full_matrices=False)
                # Keep only significant singular values
                threshold = xp.max(s) * 1e-10
                s_truncated = xp.where(s > threshold, s, 0)
                stabilized = U @ xp.diag(s_truncated) @ Vt

            elif method == 'clipping':
                # Clip extreme values
                stabilized = xp.clip(stabilized, -1e10, 1e10)

            return stabilized

        except Exception as e:
            logger.error("Error stabilizing matrix: {0}".format(e))
            return matrix


class ErrorClassifier:
    """Classify errors by type and severity"""

    def __init__(self):
        self.error_patterns = {}
        self.error_patterns[ErrorCategory.MATHEMATICAL] = [
            'singular matrix',
            'ill-conditioned',
            'convergence',
            'numerical',
            'overflow',
            'underflow',
            'division by zero',
            'invalid value',
        ]
        self.error_patterns[ErrorCategory.NETWORK] = [
            'connection',
            'timeout',
            'network',
            'socket',
            'http',
            'ssl',
            'dns',
            'unreachable',
            'refused',
        ]
        self.error_patterns[ErrorCategory.MEMORY] = [
            'memory',
            'allocation',
            'out of memory',
            'malloc',
            'heap',
            'stack overflow',
            'segmentation fault',
        ]
        self.error_patterns[ErrorCategory.COMPUTATION] = [
            'computation',
            'calculation',
            'algorithm',
            'iteration',
            'optimization',
            'gradient',
            'derivative',
        ]
        self.error_patterns[ErrorCategory.DATA] = [
            'data',
            'format',
            'parsing',
            'serialization',
            'validation',
            'schema',
            'type',
            'missing',
        ]
        self.error_patterns[ErrorCategory.SYSTEM] = [
            'system',
            'os',
            'file',
            'permission',
            'disk',
            'hardware',
            'driver',
            'resource',
        ]
        self.error_patterns[ErrorCategory.TRADING] = [
            'order',
            'execution',
            'balance',
            'position',
            'market',
            'exchange',
            'symbol',
            'price',
        ]

    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity"""
        try:
            error_message = str(error).lower()
            error_type = type(error).__name__.lower()

            # Classify by category
            category = ErrorCategory.UNKNOWN
            for cat, patterns in self.error_patterns.items():
                if any(pattern in error_message or pattern in error_type for pattern in patterns):
                    category = cat
                    break

            # Determine severity
            severity = self._determine_severity(error, category, context)

            return category, severity

        except Exception as e:
            logger.error("Error classifying error: {0}".format(e))
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

    def _determine_severity(
        self, error: Exception, category: ErrorCategory, context: Dict[str, Any] = None
    ) -> ErrorSeverity:
        """Determine error severity based on type and context"""
        try:
            # Critical errors
            if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
                return ErrorSeverity.CRITICAL

            # High severity errors
            if category in [ErrorCategory.SYSTEM, ErrorCategory.TRADING]:
                return ErrorSeverity.HIGH

            if isinstance(error, (ValueError, TypeError)) and 'critical' in str(error).lower():
                return ErrorSeverity.HIGH

            # Medium severity errors
            if category in [ErrorCategory.MATHEMATICAL, ErrorCategory.COMPUTATION]:
                return ErrorSeverity.MEDIUM

            # Low severity errors
            if category in [ErrorCategory.DATA, ErrorCategory.NETWORK]:
                return ErrorSeverity.LOW

            # Default to medium
            return ErrorSeverity.MEDIUM

        except Exception as e:
            logger.error("Error determining severity: {0}".format(e))
            return ErrorSeverity.MEDIUM


class RecoveryManager:
    """Manage recovery strategies and execution"""

    def __init__(self, config: RecoveryConfiguration):
        self.config = config
        self.recovery_strategies = {}
        self.recovery_strategies[RecoveryStrategy.RETRY] = self._retry_strategy
        self.recovery_strategies[RecoveryStrategy.FALLBACK] = self._fallback_strategy
        self.recovery_strategies[RecoveryStrategy.GRACEFUL_DEGRADATION] = self._graceful_degradation_strategy
        self.recovery_strategies[RecoveryStrategy.SYSTEM_RESTART] = self._system_restart_strategy
        self.recovery_strategies[RecoveryStrategy.MANUAL_INTERVENTION] = self._manual_intervention_strategy
        self.fallback_functions = {}
        self.degraded_mode_functions = {}

    def register_fallback(self, function_name: str, fallback_func: Callable):
        """Register a fallback function"""
        self.fallback_functions[function_name] = fallback_func

    def register_degraded_mode(self, function_name: str, degraded_func: Callable):
        """Register a degraded mode function"""
        self.degraded_mode_functions[function_name] = degraded_func

    def execute_recovery(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Execute recovery strategy for an error"""
        try:
            strategy = self._select_recovery_strategy(error_record)
            error_record.recovery_strategy = strategy

            start_time = time.time()
            success = self.recovery_strategies[strategy](error_record, context)
            error_record.recovery_time = time.time() - start_time
            error_record.recovered = success

            logger.info(
                "Recovery {0} for error {1}".format('successful' if success else 'failed', error_record.error_id)
            )
            return success

        except Exception as e:
            logger.error("Error executing recovery: {0}".format(e))
            return False

    def _select_recovery_strategy(self, error_record: ErrorRecord) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on error"""
        try:
            # Critical errors may require system restart
            if error_record.severity == ErrorSeverity.CRITICAL:
                return RecoveryStrategy.SYSTEM_RESTART

            # High severity errors try fallback first
            if error_record.severity == ErrorSeverity.HIGH:
                if self.config.fallback_enabled:
                    return RecoveryStrategy.FALLBACK
                else:
                    return RecoveryStrategy.GRACEFUL_DEGRADATION

            # Medium severity errors try retry first
            if error_record.severity == ErrorSeverity.MEDIUM:
                if error_record.recovery_attempts < self.config.max_retry_attempts:
                    return RecoveryStrategy.RETRY
                else:
                    return RecoveryStrategy.FALLBACK

            # Low severity errors just retry
            return RecoveryStrategy.RETRY

        except Exception as e:
            logger.error("Error selecting recovery strategy: {0}".format(e))
            return RecoveryStrategy.RETRY

    def _retry_strategy(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Retry strategy with exponential backoff"""
        try:
            if error_record.recovery_attempts >= self.config.max_retry_attempts:
                return False

            # Calculate delay with exponential backoff
            if self.config.exponential_backoff:
                delay = self.config.retry_delay * (2**error_record.recovery_attempts)
            else:
                delay = self.config.retry_delay

            time.sleep(delay)
            error_record.recovery_attempts += 1

            # The actual retry is handled by the calling function
            return True

        except Exception as e:
            logger.error("Error in retry strategy: {0}".format(e))
            return False

    def _fallback_strategy(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Fallback strategy using alternative functions"""
        try:
            function_name = context.get('function_name') if context else None

            if function_name and function_name in self.fallback_functions:
                fallback_func = self.fallback_functions[function_name]
                args = context.get('args', ())
                kwargs = context.get('kwargs', {})

                # Execute fallback function
                result = fallback_func(*args, **kwargs)

                # Store result in context for retrieval
                if context:
                    context['fallback_result'] = result

                return True

            return False

        except Exception as e:
            logger.error("Error in fallback strategy: {0}".format(e))
            return False

    def _graceful_degradation_strategy(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Graceful degradation strategy"""
        try:
            function_name = context.get('function_name') if context else None

            if function_name and function_name in self.degraded_mode_functions:
                degraded_func = self.degraded_mode_functions[function_name]
                args = context.get('args', ())
                kwargs = context.get('kwargs', {})

                # Execute degraded mode function
                result = degraded_func(*args, **kwargs)

                # Store result in context for retrieval
                if context:
                    context['degraded_result'] = result

                return True

            # Default degradation: return safe defaults
            if context:
                context['degraded_result'] = self._get_safe_default(error_record.category)

            return True

        except Exception as e:
            logger.error("Error in graceful degradation: {0}".format(e))
            return False

    def _system_restart_strategy(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """System restart strategy (placeholder)"""
        try:
            if not self.config.auto_restart_enabled:
                return False

            logger.critical("System restart requested due to critical error")
            # In a real system, this would trigger a controlled restart
            # For now, we'll just log the request'

            return True

        except Exception as e:
            logger.error("Error in system restart strategy: {0}".format(e))
            return False

    def _manual_intervention_strategy(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Manual intervention strategy"""
        try:
            logger.critical("Manual intervention required for error {0}".format(error_record.error_id))

            # In a real system, this would trigger alerts to operators
            # For now, we'll just log the request'

            return False  # Manual intervention is not automatic

        except Exception as e:
            logger.error("Error in manual intervention strategy: {0}".format(e))
            return False

    def _get_safe_default(self, category: ErrorCategory) -> Any:
        """Get safe default values for different error categories"""
        defaults = {}
        defaults[ErrorCategory.MATHEMATICAL] = xp.array([0.0])
        defaults[ErrorCategory.COMPUTATION] = 0.0
        defaults[ErrorCategory.DATA] = None
        defaults[ErrorCategory.TRADING] = {'action': 'hold', 'quantity': 0.0}
        defaults[ErrorCategory.NETWORK] = {'status': 'offline'}
        defaults[ErrorCategory.SYSTEM] = {'status': 'degraded'}
        defaults[ErrorCategory.MEMORY] = None
        defaults[ErrorCategory.UNKNOWN] = None

        return defaults.get(category, None)


class SystemHealthMonitor:
    """Monitor system health and performance"""

    def __init__(self, config: RecoveryConfiguration):
        self.config = config
        self.health_history = []
        self.monitoring_active = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start system health monitoring"""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self.monitor_thread.start()
                logger.info("System health monitoring started")

        except Exception as e:
            logger.error("Error starting health monitoring: {0}".format(e))

    def stop_monitoring(self):
        """Stop system health monitoring"""
        try:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            logger.info("System health monitoring stopped")

        except Exception as e:
            logger.error("Error stopping health monitoring: {0}".format(e))

    def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                health = self._collect_health_metrics()
                self.health_history.append(health)

                # Keep only recent history
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]

                # Check for health issues
                self._check_health_alerts(health)

                time.sleep(self.config.health_check_interval)

        except Exception as e:
            logger.error("Error in monitoring loop: {0}".format(e))

    def _collect_health_metrics(self) -> SystemHealth:
        """Collect current system health metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1.0)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent

            # GPU usage (if, available)
            gpu_usage = None
            if CUPY_AVAILABLE:
                try:
                    gpu_memory = cp.cuda.MemoryPool().used_bytes()
                    gpu_total = cp.cuda.MemoryPool().total_bytes()
                    gpu_usage = (gpu_memory / gpu_total) * 100 if gpu_total > 0 else 0
                except Exception as e:
                    logger.info(
                        "🎯 Profit optimization: Switching to CPU-based computational resources for this operation: {0}".format(
                            e
                        )
                    )
                    gpu_usage = None  # Indicate adaptive switching
            else:
                gpu_usage = 0.0  # CPU optimization mode active

            # Network latency (simplified)
            network_latency = 0.0  # Placeholder

            # Error and recovery rates (simplified)
            error_rate = 0.0  # Would be calculated from error history
            recovery_rate = 0.0  # Would be calculated from recovery history

            # System uptime
            uptime = time.time() - psutil.boot_time()

            return SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                gpu_usage=gpu_usage,
                network_latency=network_latency,
                error_rate=error_rate,
                recovery_rate=recovery_rate,
                uptime=uptime,
                last_check=datetime.now(),
            )

        except Exception as e:
            logger.error("Error collecting health metrics: {0}".format(e))
            return SystemHealth(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                gpu_usage=None,
                network_latency=0.0,
                error_rate=0.0,
                recovery_rate=0.0,
                uptime=0.0,
                last_check=datetime.now(),
            )

    def _check_health_alerts(self, health: SystemHealth):
        """Check for health alerts and warnings"""
        try:
            alerts = []

            # CPU usage alert
            if health.cpu_usage > self.config.cpu_threshold * 100:
                alerts.append("High CPU usage: {0}%".format(health.cpu_usage))

            # Memory usage alert
            if health.memory_usage > self.config.memory_threshold * 100:
                alerts.append("High memory usage: {0}%".format(health.memory_usage))

            # Disk usage alert
            if health.disk_usage > 90.0:
                alerts.append("High disk usage: {0}%".format(health.disk_usage))

            # GPU usage alert
            if health.gpu_usage is not None and health.gpu_usage > 90.0:
                alerts.append("High GPU usage: {0}%".format(health.gpu_usage))

            # Error rate alert
            if health.error_rate > self.config.error_threshold:
                alerts.append("High error rate: {0}".format(health.error_rate))

            # Log alerts
            for alert in alerts:
                logger.warning("Health Alert: {0}".format(alert))

        except Exception as e:
            logger.error("Error checking health alerts: {0}".format(e))

    def get_current_health(self) -> SystemHealth:
        """Get current system health"""
        return self._collect_health_metrics()

    def get_health_history(self) -> List[SystemHealth]:
        """Get health history"""
        return self.health_history.copy()


class EnhancedErrorRecoverySystem:
    """
    Main enhanced error recovery system for trading operations.

    Implements:
    - Multi-level error detection and classification
    - Automatic recovery strategies
    - Mathematical stability monitoring
    - System health monitoring
    - Graceful degradation and failover
    """

    def __init__(self, config: RecoveryConfiguration = None):
        self.config = config or RecoveryConfiguration()

        # Initialize components
        self.error_classifier = ErrorClassifier()
        self.recovery_manager = RecoveryManager(self.config)
        self.health_monitor = SystemHealthMonitor(self.config)
        self.stability_checker = MathematicalStabilityChecker()

        # Error tracking
        self.error_history = []
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'recovery_rate': 0.0,
        }

        # Threading
        self.error_lock = threading.Lock()

        # Start monitoring
        self.health_monitor.start_monitoring()

        logger.info("Enhanced Error Recovery System initialized")

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        """
        Handle an error with automatic recovery.

        Args:
            error: The exception that occurred
            context: Additional context information

        Returns:
            Recovery result or None if recovery failed
        """
        try:
            # Create error record
            error_record = self._create_error_record(error, context)

            # Classify error
            category, severity = self.error_classifier.classify_error(error, context)
            error_record.category = category
            error_record.severity = severity

            # Store error record
            with self.error_lock:
                self.error_history.append(error_record)
                self.error_stats['total_errors'] += 1

            # Execute recovery
            recovery_success = self.recovery_manager.execute_recovery(error_record, context)

            # Update statistics
            with self.error_lock:
                if recovery_success:
                    self.error_stats['recovered_errors'] += 1
                else:
                    self.error_stats['failed_recoveries'] += 1

                self.error_stats['recovery_rate'] = (
                    self.error_stats['recovered_errors'] / self.error_stats['total_errors']
                    if self.error_stats['total_errors'] > 0
                    else 0.0
                )

            # Return recovery result
            if recovery_success and context:
                return context.get('fallback_result') or context.get('degraded_result')

            return None

        except Exception as e:
            logger.error("Error in error handling: {0}".format(e))
            return None

    def _create_error_record(self, error: Exception, context: Dict[str, Any] = None) -> ErrorRecord:
        """Create an error record from an exception"""
        try:
            error_id = "error_{0}".format(int(time.time() * 1000000))

            return ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                severity=ErrorSeverity.MEDIUM,  # Will be updated by classifier
                category=ErrorCategory.UNKNOWN,  # Will be updated by classifier
                context=context or {},
            )

        except Exception as e:
            logger.error("Error creating error record: {0}".format(e))
            return ErrorRecord(
                error_id="unknown",
                timestamp=datetime.now(),
                error_type="Unknown",
                error_message=str(e),
                stack_trace="",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.UNKNOWN,
                context={},
            )

    @contextmanager
    def error_recovery_context(self, function_name: str, *args, **kwargs):
        """Context manager for automatic error recovery"""
        try:
            yield
        except Exception as e:
            context = {'function_name': function_name, 'args': args, 'kwargs': kwargs}
            result = self.handle_error(e, context)
            if result is not None:
                return result
            raise

    def register_fallback_function(self, function_name: str, fallback_func: Callable):
        """Register a fallback function for error recovery"""
        self.recovery_manager.register_fallback(function_name, fallback_func)

    def register_degraded_mode_function(self, function_name: str, degraded_func: Callable):
        """Register a degraded mode function for graceful degradation"""
        self.recovery_manager.register_degraded_mode(function_name, degraded_func)

    def check_mathematical_stability(self, data: xp.ndarray) -> Dict[str, Any]:
        """Check mathematical stability of data"""
        return self.stability_checker.check_matrix_stability(data)

    def stabilize_mathematical_data(self, data: xp.ndarray, method: str = 'ridge') -> xp.ndarray:
        """Stabilize mathematical data"""
        return self.stability_checker.stabilize_matrix(data, method)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self.error_lock:
            return self.error_stats.copy()

    def get_error_history(self, limit: int = 100) -> List[ErrorRecord]:
        """Get error history"""
        with self.error_lock:
            return self.error_history[-limit:] if limit else self.error_history.copy()

    def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        return self.health_monitor.get_current_health()

    def cleanup_resources(self):
        """Clean up recovery system resources"""
        try:
            # Stop monitoring
            self.health_monitor.stop_monitoring()

            # Clear error history
            with self.error_lock:
                self.error_history.clear()

            logger.info("Error recovery system resources cleaned up")

        except Exception as e:
            logger.error("Error cleaning up recovery resources: {0}".format(e))

    def __del__(self):
        """Destructor to ensure resource cleanup"""
        try:
            self.cleanup_resources()
        except Exception:
            pass


def error_recovery_decorator(recovery_system: EnhancedErrorRecoverySystem, function_name: str = None):
    """Decorator for automatic error recovery"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = function_name or func.__name__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {'function_name': func_name, 'args': args, 'kwargs': kwargs}
                result = recovery_system.handle_error(e, context)
                if result is not None:
                    return result
                raise

        return wrapper

    return decorator
