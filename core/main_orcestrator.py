"""
Main Orchestrator - System-Wide Coordination Engine
==================================================

Comprehensive system orchestrator for the Schwabot mathematical trading framework.
Provides centralized component management, health monitoring, event processing,
and system-wide coordination.

Key Features:
- Component lifecycle management and dependency resolution
- Real-time health monitoring and heartbeat tracking
- Event processing and prioritization system
- Graceful shutdown and error recovery
- System state management and coordination
- Integration with all core components
- Windows CLI compatibility with emoji fallbacks

Component Management:
- Registration and lifecycle control
- Dependency resolution and startup ordering
- Health monitoring and error detection
- Automatic restart and recovery procedures
- Performance metrics and resource tracking

Event System:
- Prioritized event processing (CRITICAL, HIGH, NORMAL, LOW, BACKGROUND)
- Event history and audit trail
- Real-time event routing and handling
- System-wide event coordination

Integration Points:
- All core components for system coordination
- enhanced_windows_cli_compatibility.py: CLI compatibility
- thermal_boundary_manager.py: Thermal-aware operations
- error_handling_pipeline.py: Error management
- profit_routing_engine.py: Profit optimization coordination

Windows CLI compatible with flake8 compliance.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional

import psutil

# Import core components
try:
    from core.config import get_config_manager
    from core.thermal_boundary_manager import create_thermal_boundary_manager
    from core.error_handling_pipeline import ErrorHandlingPipeline
    from core.enhanced_windows_cli_compatibility import safe_print, safe_format_error
    CLI_HANDLER_AVAILABLE = True
except ImportError as e:
    CLI_HANDLER_AVAILABLE = False
    
    def safe_print(message: str, use_emoji: bool = True) -> str:
        return message
        
    def safe_format_error(error: Exception, context: str = "") -> str:
        return f"Error: {str(error)} | Context: {context}"

# Configure logging
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"


class SystemState(Enum):
    """System state enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


class Priority(Enum):
    """Priority levels for system events."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ComponentInfo:
    """Component information container."""
    name: str
    component_type: str
    status: ComponentStatus
    priority: Priority
    dependencies: List[str]
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemEvent:
    """System event container."""
    event_id: str
    event_type: str
    timestamp: datetime
    component: str
    message: str
    priority: Priority
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result container."""
    component: str
    timestamp: datetime
    status: ComponentStatus
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_count: int
    warnings: List[str] = field(default_factory=list)


class MainOrchestrator:
    """Main system orchestrator for component management and coordination."""

    def __init__(self, config_path: str = "./config/orchestrator_config.json"):
        """Initialize the main orchestrator."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # System state
        self._running = False
        self.system_state = SystemState.SHUTDOWN
        self.start_time: Optional[datetime] = None
        
        # Component management
        self.components: Dict[str, ComponentInfo] = {}
        self.component_instances: Dict[str, Any] = {}
        
        # Event system
        self.event_queue: PriorityQueue = PriorityQueue()
        self.event_history: List[SystemEvent] = []
        self.max_event_history = 1000
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.heartbeat_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        
        # Background threads
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._event_processor_thread: Optional[threading.Thread] = None
        
        # Shutdown handlers
        self.shutdown_handlers: List[Callable[[], None]] = []
        
        # Initialize system
        self._load_configuration()
        self._setup_signal_handlers()
        self._initialize_core_components()
        
        safe_print("🎯 Main Orchestrator initialized")

    def _load_configuration(self) -> None:
        """Load orchestrator configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Use default configuration
                self.config = {
                    'heartbeat_interval': 30,
                    'health_check_interval': 60,
                    'max_event_history': 1000,
                    'auto_restart_components': True,
                    'max_restart_attempts': 3
                }
                logger.info("Using default configuration")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use minimal default configuration
            self.config = {'heartbeat_interval': 30, 'health_check_interval': 60}

    def _setup_signal_handlers(self) -> None:
        """Setup system signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers configured")
        except Exception as e:
            logger.warning(f"Signal handler setup failed: {e}")

    def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        try:
            # Initialize configuration manager
            try:
                config_manager = get_config_manager()
                self.component_instances['config_manager'] = config_manager
                logger.info("Configuration manager initialized")
            except Exception as e:
                logger.warning(f"Config manager initialization failed: {e}")
                
        except Exception as e:
            logger.error(f"Core component initialization failed: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.shutdown()

    def start_system(self) -> None:
        """Start the main orchestrator system."""
        try:
            if self._running:
                logger.warning("System already running")
                return
            
            self._running = True
            self.system_state = SystemState.STARTING
            self.start_time = datetime.now()
            
            # Start background workers
            self._start_background_workers()
            
            # Start all registered components
            self._start_all_components()
            
            self.system_state = SystemState.RUNNING
            logger.info("Main orchestrator system started")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.system_state = SystemState.ERROR

    def _start_background_workers(self) -> None:
        """Start background worker threads."""
        try:
            # Heartbeat processor
            self._heartbeat_thread = threading.Thread(
                target=self._process_heartbeats,
                daemon=True,
                name="HeartbeatProcessor"
            )
            self._heartbeat_thread.start()
            
            # Health check processor
            self._health_check_thread = threading.Thread(
                target=self._perform_health_checks,
                daemon=True,
                name="HealthCheckProcessor"
            )
            self._health_check_thread.start()
            
            # Event processor
            self._event_processor_thread = threading.Thread(
                target=self._process_events,
                daemon=True,
                name="EventProcessor"
            )
            self._event_processor_thread.start()

            logger.info("Background workers started")
            
        except Exception as e:
            logger.error(f"Failed to start background workers: {e}")
    
    def _start_all_components(self) -> None:
        """Start all registered components."""
        try:
            for component_name, component_info in self.components.items():
                if component_info.status == ComponentStatus.INITIALIZING:
                    self.start_component(component_name)
                    
        except Exception as e:
            logger.error(f"Failed to start all components: {e}")

    def register_component(self, name: str, component_type: str,
                          priority: Priority = Priority.NORMAL,
                          dependencies: Optional[List[str]] = None) -> bool:
        """Register a new component with the orchestrator."""
        try:
            if name in self.components:
                logger.warning(f"Component {name} already registered")
                return False

            component_info = ComponentInfo(
                name=name,
                component_type=component_type,
                status=ComponentStatus.INITIALIZING,
                priority=priority,
                dependencies=dependencies or []
            )

            self.components[name] = component_info
            logger.info(f"Component {name} registered")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {name}: {e}")
            return False
    
    def start_component(self, name: str) -> bool:
        """Start a specific component."""
        try:
            if name not in self.components:
                logger.error(f"Component {name} not registered")
                return False

            component_info = self.components[name]

            # Check dependencies
            for dep in component_info.dependencies:
                if dep not in self.components:
                    logger.error(f"Component {name} depends on {dep} which is not registered")
                    return False
                if self.components[dep].status != ComponentStatus.RUNNING:
                    logger.error(f"Component {name} depends on {dep} which is not running")
                    return False

            # Update component status
            component_info.status = ComponentStatus.RUNNING
            component_info.start_time = datetime.now()
            component_info.last_heartbeat = datetime.now()

            logger.info(f"Component {name} started")
            return True

        except Exception as e:
            logger.error(f"Failed to start component {name}: {e}")
            return False

    def stop_component(self, name: str) -> bool:
        """Stop a specific component."""
        try:
            if name not in self.components:
                logger.error(f"Component {name} not registered")
                return False
            
            component_info = self.components[name]
            component_info.status = ComponentStatus.SHUTDOWN
            component_info.last_heartbeat = datetime.now()

            logger.info(f"Component {name} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop component {name}: {e}")
            return False

    def get_component(self, name: str) -> Optional[Any]:
        """Get a component instance by name."""
        return self.component_instances.get(name)

    def get_component_status(self, name: str) -> Optional[ComponentStatus]:
        """Get the status of a component."""
        if name in self.components:
            return self.components[name].status
        return None

    def update_component_heartbeat(self, name: str) -> None:
        """Update component heartbeat."""
        if name in self.components:
            self.components[name].last_heartbeat = datetime.now()

    def _process_heartbeats(self) -> None:
        """Process component heartbeats."""
        while self._running:
            try:
                time.sleep(self.heartbeat_interval)

                current_time = datetime.now()
                for name, component_info in self.components.items():
                    if (component_info.last_heartbeat and 
                        (current_time - component_info.last_heartbeat).total_seconds() > 
                        self.heartbeat_interval * 2):
                        
                        logger.warning(f"Component {name} heartbeat timeout")
                        component_info.status = ComponentStatus.ERROR
                        component_info.error_count += 1

                        # Log event
                        self._log_event(
                            "heartbeat_timeout",
                            name,
                            f"Heartbeat timeout for component {name}",
                            Priority.HIGH
                        )

            except Exception as e:
                logger.error(f"Heartbeat processing error: {e}")

    def _perform_health_checks(self) -> None:
        """Perform periodic health checks."""
        while self._running:
            try:
                time.sleep(self.health_check_interval)
                
                for name, component_info in self.components.items():
                    if component_info.status == ComponentStatus.RUNNING:
                        health_check = self._check_component_health(name)
                        self.health_checks[name] = health_check
                        
            except Exception as e:
                logger.error(f"Health check processing error: {e}")

    def _check_component_health(self, name: str) -> HealthCheck:
        """Check health of a specific component."""
        try:
            start_time = time.time()
            
            # Get component info
            component_info = self.components.get(name)
            if not component_info:
                return HealthCheck(
                    component=name,
                    timestamp=datetime.now(),
                    status=ComponentStatus.UNKNOWN,
                    response_time_ms=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    error_count=0,
                    warnings=["Component not found"]
                )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Get system metrics
            memory_usage_mb = self._get_memory_usage()
            cpu_usage_percent = self._get_cpu_usage()
            
            return HealthCheck(
                component=name,
                timestamp=datetime.now(),
                status=component_info.status,
                response_time_ms=response_time_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                error_count=component_info.error_count,
                warnings=[]
            )
            
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return HealthCheck(
                component=name,
                timestamp=datetime.now(),
                status=ComponentStatus.ERROR,
                response_time_ms=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                error_count=0,
                warnings=[f"Health check error: {e}"]
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.debug(f"Memory usage check failed: {e}")
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.debug(f"CPU usage check failed: {e}")
            return 0.0

    def _log_event(self, event_type: str, component: str, message: str,
                   priority: Priority, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a system event."""
        try:
            event = SystemEvent(
                event_id=f"{event_type}_{int(time.time())}",
                event_type=event_type,
                timestamp=datetime.now(),
                component=component,
                message=message,
                priority=priority,
                data=data or {}
            )
            
            self.event_queue.put((priority.value, event))

            # Maintain event history
            self.event_history.append(event)
            if len(self.event_history) > self.max_event_history:
                self.event_history.pop(0)

        except Exception as e:
            logger.error(f"Event logging failed: {e}")

    def _process_events(self) -> None:
        """Process system events."""
        while self._running:
            try:
                # Get next event from queue
                priority, event = self.event_queue.get(timeout=1)
                
                # Handle event based on priority
                if event.priority == Priority.CRITICAL:
                    self._handle_critical_event(event)
                elif event.priority == Priority.HIGH:
                    logger.warning(f"High priority event: {event.message}")
                else:
                    logger.info(f"Event: {event.message}")
                    
            except Exception as e:
                # Timeout is expected, other exceptions are logged
                if "timeout" not in str(e).lower():
                    logger.error(f"Event processing error: {e}")

    def _handle_critical_event(self, event: SystemEvent) -> None:
        """Handle critical system events."""
        try:
            logger.critical(f"Critical event: {event.message}")
            
            # Attempt component restart if configured
            if self.config.get('auto_restart_components', True):
                max_attempts = self.config.get('max_restart_attempts', 3)
                component_info = self.components.get(event.component)
                
                if component_info and component_info.error_count < max_attempts:
                    self._attempt_component_restart(event.component)
                    
        except Exception as e:
            logger.error(f"Critical event handling failed: {e}")

    def _attempt_component_restart(self, component_name: str) -> None:
        """Attempt to restart a failed component."""
        try:
            logger.info(f"Attempting to restart component: {component_name}")
            
            # Stop component
            self.stop_component(component_name)
            
            # Wait briefly
            time.sleep(2)
            
            # Start component
            if self.start_component(component_name):
                logger.info(f"Component {component_name} restarted successfully")
                self._log_event(
                    "component_restarted",
                    component_name,
                    f"Component {component_name} restarted successfully",
                    Priority.HIGH
                )
            else:
                logger.error(f"Failed to restart component {component_name}")
                
        except Exception as e:
            logger.error(f"Component restart failed for {component_name}: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            uptime = None
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
            
            component_statuses = {}
            for name, info in self.components.items():
                component_statuses[name] = {
                    'status': info.status.value,
                    'type': info.component_type,
                    'priority': info.priority.value,
                    'error_count': info.error_count,
                    'uptime': (datetime.now() - info.start_time).total_seconds() if info.start_time else None
                }
            
            return {
                'system_state': self.system_state.value,
                'running': self._running,
                'uptime_seconds': uptime,
                'components': component_statuses,
                'health_checks': len(self.health_checks),
                'event_queue_size': self.event_queue.qsize(),
                'event_history_size': len(self.event_history),
                'memory_usage_mb': self._get_memory_usage(),
                'cpu_usage_percent': self._get_cpu_usage()
            }
            
        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {'error': str(e)}

    def add_shutdown_handler(self, handler: Callable[[], None]) -> None:
        """Add a shutdown handler."""
        self.shutdown_handlers.append(handler)

    def shutdown(self) -> None:
        """Shutdown the main orchestrator system."""
        try:
            logger.info("Initiating system shutdown")
            self.system_state = SystemState.SHUTTING_DOWN
            self._running = False
            
            # Stop all components
            for name in list(self.components.keys()):
                self.stop_component(name)

            # Call shutdown handlers
            for handler in self.shutdown_handlers:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Shutdown handler failed: {e}")
            
            self.system_state = SystemState.SHUTDOWN
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            self.system_state = SystemState.EMERGENCY_STOP


# Global orchestrator instance
_orchestrator: Optional[MainOrchestrator] = None


def get_orchestrator() -> MainOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MainOrchestrator()
    return _orchestrator


def main():
    """Main function for testing the orchestrator."""
    try:
        # Create orchestrator
        orchestrator = get_orchestrator()
        
        # Start system
        orchestrator.start_system()
        
        # Get system status
        status = orchestrator.get_system_status()
        safe_print(f"📊 System Status: {status}")
        
        # Keep running
        try:
            while orchestrator._running:
                time.sleep(1)
        except KeyboardInterrupt:
            safe_print("🛑 Shutdown requested")
            orchestrator.shutdown()
        
    except Exception as e:
        safe_print(f"❌ Orchestrator test failed: {safe_format_error(e, 'main_test')}")


if __name__ == "__main__":
    main()