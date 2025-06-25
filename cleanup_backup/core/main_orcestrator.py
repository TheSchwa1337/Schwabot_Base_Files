from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Main Orchestrator - Central Coordination and System Management for Schwabot
===========================================================================

This module implements the main orchestrator system for Schwabot, providing
central coordination, component management, and system lifecycle control.
It manages the initialization, coordination, and shutdown of all trading
components and ensures proper system operation.

Core Functionality:
- Component lifecycle management
- System coordination and messaging
- Configuration management
- Health monitoring and diagnostics
- Error handling and recovery
- Performance optimization
"""

import logging
import json
import signal
import sys
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import os
import queue
import traceback
from queue import PriorityQueue

logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"

class SystemState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    EMERGENCY_STOP = "emergency_stop"

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ComponentInfo:
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
    event_id: str
    event_type: str
    timestamp: datetime
    component: str
    message: str
    priority: Priority
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    component: str
    timestamp: datetime
    status: ComponentStatus
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_count: int
    warnings: List[str] = field(default_factory=list)

class MainOrchestrator:
    """Main orchestrator for the Schwabot system."""
    
    def __init__(self, config_path: str = "./config/orchestrator_config.json"):
        """Initialize the main orchestrator."""
        self.config_path = config_path
        self.system_state = SystemState.STARTING
        self.components: Dict[str, ComponentInfo] = {}
        self.component_instances: Dict[str, Any] = {}
        self.event_queue: PriorityQueue[Any] = PriorityQueue()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.event_history: List[SystemEvent] = []
        self.shutdown_handlers: List[Callable[[], None]] = []
        self.heartbeat_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.max_event_history = 1000
        self._load_configuration()
        self._setup_signal_handlers()
        self._start_background_workers()
        logger.info("MainOrchestrator initialized")

    def _load_configuration(self) -> None:
        """Load orchestrator configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.heartbeat_interval = config.get("heartbeat_interval", 30)
                self.health_check_interval = config.get("health_check_interval", 60)
                self.max_event_history = config.get("max_event_history", 1000)
                
                # Load component configurations
                for comp_config in config.get("components", []):
                    component_info = ComponentInfo(
                        name=comp_config["name"],
                        component_type=comp_config["type"],
                        status=ComponentStatus.UNKNOWN,
                        priority=Priority(comp_config.get("priority", 3)),
                        dependencies=comp_config.get("dependencies", [])
                    )
                    self.components[component_info.name] = component_info
                
                logger.info(f"Loaded configuration for {len(self.components)} components")
            else:
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()

    def _create_default_configuration(self) -> None:
        """Create default component configuration."""
        default_components = [
            {
                "name": "market_data_engine",
                "type": "data_engine",
                "priority": 1,
                "dependencies": []
            },
            {
                "name": "trading_engine",
                "type": "trading",
                "priority": 1,
                "dependencies": ["market_data_engine"]
            },
            {
                "name": "risk_manager",
                "type": "risk",
                "priority": 1,
                "dependencies": ["trading_engine"]
            },
            {
                "name": "portfolio_manager",
                "type": "portfolio",
                "priority": 2,
                "dependencies": ["trading_engine", "risk_manager"]
            },
            {
                "name": "analytics_engine",
                "type": "analytics",
                "priority": 3,
                "dependencies": ["market_data_engine"]
            },
            {
                "name": "reporting_engine",
                "type": "reporting",
                "priority": 4,
                "dependencies": ["analytics_engine", "portfolio_manager"]
            }
        ]
        
        for comp_config in default_components:
            component_info = ComponentInfo(
                name=comp_config["name"],
                component_type=comp_config["type"],
                status=ComponentStatus.UNKNOWN,
                priority=Priority(comp_config["priority"]),
                dependencies=comp_config["dependencies"]
            )
            self.components[component_info.name] = component_info
        
        self._save_configuration()
        logger.info("Default configuration created")

    def _save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            config = {
                "heartbeat_interval": self.heartbeat_interval,
                "health_check_interval": self.health_check_interval,
                "max_event_history": self.max_event_history,
                "components": [asdict(comp) for comp in self.components.values()]
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating shutdown")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _start_background_workers(self) -> None:
        """Start background worker threads."""
        # Heartbeat worker
        def heartbeat_worker() -> None:
            while self.system_state != SystemState.SHUTTING_DOWN:
                try:
                    self._process_heartbeats()
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Error in heartbeat worker: {e}")
        
        # Health check worker
        def health_check_worker() -> None:
            while self.system_state != SystemState.SHUTTING_DOWN:
                try:
                    self._perform_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check worker: {e}")
        
        # Event processor worker
        def event_processor_worker() -> None:
            while self.system_state != SystemState.SHUTTING_DOWN:
                try:
                    self._process_events()
                    time.sleep(1)  # Process events every second
                except Exception as e:
                    logger.error(f"Error in event processor worker: {e}")
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.health_check_thread = threading.Thread(target=health_check_worker, daemon=True)
        self.event_processor_thread = threading.Thread(target=event_processor_worker, daemon=True)
        
        self.heartbeat_thread.start()
        self.health_check_thread.start()
        self.event_processor_thread.start()
        
        logger.info("Background workers started")

    def register_component(self, name: str, component_type: str, 
                          priority: Priority = Priority.NORMAL,
                          dependencies: Optional[List[str]] = None) -> None:
        """Register a new component with the orchestrator."""
        if name in self.components:
            logger.warning(f"Component {name} already registered")
            return
        
        component_info = ComponentInfo(
            name=name,
            component_type=component_type,
            status=ComponentStatus.INITIALIZING,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.components[name] = component_info
        logger.info(f"Component registered: {name} ({component_type})")

    def start_component(self, name: str, component_instance: Any) -> bool:
        """Start a component and register its instance."""
        if name not in self.components:
            logger.error(f"Component {name} not registered")
            return False
        
        component_info = self.components[name]
        
        # Check dependencies
        for dep in component_info.dependencies:
            if dep not in self.component_instances:
                logger.error(f"Component {name} depends on {dep} which is not started")
                return False
        
        try:
            # Store component instance
            self.component_instances[name] = component_instance
            
            # Update component status
            component_info.status = ComponentStatus.RUNNING
            component_info.start_time = datetime.now()
            component_info.last_heartbeat = datetime.now()
            
            # Log component start
            self._log_event(
                f"component_started",
                name,
                f"Component {name} started successfully",
                Priority.NORMAL
            )
            
            logger.info(f"Component started: {name}")
            return True
            
        except Exception as e:
            component_info.status = ComponentStatus.ERROR
            component_info.error_count += 1
            
            self._log_event(
                f"component_start_failed",
                name,
                f"Failed to start component {name}: {e}",
                Priority.HIGH
            )
            
            logger.error(f"Failed to start component {name}: {e}")
            return False

    def stop_component(self, name: str) -> bool:
        """Stop a component."""
        if name not in self.components:
            logger.error(f"Component {name} not found")
            return False
        
        component_info = self.components[name]
        
        try:
            # Update component status
            component_info.status = ComponentStatus.SHUTDOWN
            component_info.last_heartbeat = datetime.now()
            
            # Remove component instance
            if name in self.component_instances:
                del self.component_instances[name]
            
            # Log component stop
            self._log_event(
                f"component_stopped",
                name,
                f"Component {name} stopped",
                Priority.NORMAL
            )
            
            logger.info(f"Component stopped: {name}")
            return True
            
        except Exception as e:
            self._log_event(
                f"component_stop_failed",
                name,
                f"Failed to stop component {name}: {e}",
                Priority.HIGH
            )
            
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
        """Update the heartbeat for a component."""
        if name in self.components:
            self.components[name].last_heartbeat = datetime.now()

    def _process_heartbeats(self) -> None:
        """Process component heartbeats and detect failures."""
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.heartbeat_interval * 2)
        
        for name, component_info in self.components.items():
            if component_info.status == ComponentStatus.RUNNING:
                if (component_info.last_heartbeat and 
                    current_time - component_info.last_heartbeat > timeout_threshold):
                    
                    # Component heartbeat timeout
                    component_info.status = ComponentStatus.ERROR
                    component_info.error_count += 1
                    
                    self._log_event(
                        f"component_heartbeat_timeout",
                        name,
                        f"Component {name} heartbeat timeout",
                        Priority.HIGH
                    )
                    
                    logger.warning(f"Component {name} heartbeat timeout")

    def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        for name, component_info in self.components.items():
            try:
                start_time = time.time()
                
                # Basic health check
                is_healthy = self._check_component_health(name)
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Get system metrics
                memory_usage = self._get_memory_usage()
                cpu_usage = self._get_cpu_usage()
                
                health_check = HealthCheck(
                    component=name,
                    timestamp=datetime.now(),
                    status=component_info.status,
                    response_time_ms=response_time,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    error_count=component_info.error_count
                )
                
                self.health_checks[name] = health_check
                
                # Update component status if needed
                if not is_healthy and component_info.status == ComponentStatus.RUNNING:
                    component_info.status = ComponentStatus.ERROR
                    component_info.error_count += 1
                    
                    self._log_event(
                        f"component_health_check_failed",
                        name,
                        f"Health check failed for component {name}",
                        Priority.HIGH
                    )
                
            except Exception as e:
                logger.error(f"Error performing health check for {name}: {e}")

    def _check_component_health(self, name: str) -> bool:
        """Check the health of a specific component."""
        # This is a simplified health check
        # In a real system, you would implement component-specific health checks
        component_info = self.components.get(name)
        if not component_info:
            return False
        
        # Check if component is running and has recent heartbeat
        if component_info.status != ComponentStatus.RUNNING:
            return False
        
        if not component_info.last_heartbeat:
            return False
        
        # Check if heartbeat is recent
        time_since_heartbeat = datetime.now() - component_info.last_heartbeat
        if time_since_heartbeat > timedelta(seconds=self.heartbeat_interval * 2):
            return False
        
        return True

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0

    def _log_event(self, event_type: str, component: str, message: str, 
                  priority: Priority, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a system event."""
        event_id = f"event_{int(datetime.now().timestamp())}_{hash(message) % 10000}"
        
        event = SystemEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            component=component,
            message=message,
            priority=priority,
            data=data or {}
        )
        
        # Add to event queue
        self.event_queue.put((priority.value, event))
        
        # Add to history
        self.event_history.append(event)
        
        # Trim history if too long
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]

    def _process_events(self) -> None:
        """Process events from the event queue."""
        try:
            while not self.event_queue.empty():
                priority, event = self.event_queue.get_nowait()
                
                # Log the event
                log_level = logging.INFO
                if priority <= Priority.CRITICAL.value:
                    log_level = logging.CRITICAL
                elif priority <= Priority.HIGH.value:
                    log_level = logging.ERROR
                elif priority <= Priority.NORMAL.value:
                    log_level = logging.INFO
                else:
                    log_level = logging.DEBUG
                
                logger.unified_math.log(log_level, f"[{event.component}] {event.message}")
                
                # Handle critical events
                if priority <= Priority.CRITICAL.value:
                    self._handle_critical_event(event)
                
                self.event_queue.task_done()
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing events: {e}")

    def _handle_critical_event(self, event: SystemEvent) -> None:
        """Handle critical system events."""
        if event.event_type == "component_heartbeat_timeout":
            # Attempt to restart component
            self._attempt_component_restart(event.component)
        elif event.event_type == "component_health_check_failed":
            # Log and potentially take corrective action
            logger.critical(f"Critical health check failure: {event.message}")

    def _attempt_component_restart(self, component_name: str) -> None:
        """Attempt to restart a failed component."""
        logger.info(f"Attempting to restart component: {component_name}")
        
        # This is a simplified restart mechanism
        # In a real system, you would implement proper restart logic
        component_info = self.components.get(component_name)
        if component_info:
            component_info.status = ComponentStatus.INITIALIZING
            component_info.error_count += 1
            
            self._log_event(
                f"component_restart_attempted",
                component_name,
                f"Restart attempted for component {component_name}",
                Priority.HIGH
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        component_statuses = {}
        for name, component_info in self.components.items():
            component_statuses[name] = {
                "status": component_info.status.value,
                "type": component_info.component_type,
                "priority": component_info.priority.value,
                "start_time": component_info.start_time.isoformat() if component_info.start_time else None,
                "last_heartbeat": component_info.last_heartbeat.isoformat() if component_info.last_heartbeat else None,
                "error_count": component_info.error_count
            }
        
        health_summary = {}
        for name, health_check in self.health_checks.items():
            health_summary[name] = {
                "status": health_check.status.value,
                "response_time_ms": health_check.response_time_ms,
                "memory_usage_mb": health_check.memory_usage_mb,
                "cpu_usage_percent": health_check.cpu_usage_percent,
                "error_count": health_check.error_count
            }
        
        return {
            "system_state": self.system_state.value,
            "total_components": len(self.components),
            "running_components": len([c for c in self.components.values() if c.status == ComponentStatus.RUNNING]),
            "error_components": len([c for c in self.components.values() if c.status == ComponentStatus.ERROR]),
            "component_statuses": component_statuses,
            "health_summary": health_summary,
            "recent_events": len(self.event_history),
            "timestamp": datetime.now().isoformat()
        }

    def add_shutdown_handler(self, handler: Callable[[], None]) -> None:
        """Add a shutdown handler function."""
        self.shutdown_handlers.append(handler)

    def shutdown(self) -> None:
        """Shutdown the orchestrator and all components."""
        logger.info("Initiating system shutdown")
        self.system_state = SystemState.SHUTTING_DOWN
        
        # Call shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        # Stop all components
        for name in list(self.components.keys()):
            self.stop_component(name)
        
        # Wait for background workers to finish
        time.sleep(2)
        
        logger.info("System shutdown completed")
        sys.exit(0)

    def start_system(self) -> None:
        """Start the orchestrator system."""
        logger.info("Starting MainOrchestrator system")
        self.system_state = SystemState.RUNNING
        
        # Initialize all components
        for name, component_info in self.components.items():
            if not component_info.dependencies:  # Start components with no dependencies first
                component_info.status = ComponentStatus.INITIALIZING
        
        logger.info("MainOrchestrator system started")

def main() -> None:
    """Main function for testing and demonstration."""
    orchestrator = MainOrchestrator("./test_orchestrator_config.json")
    
    # Start the system
    orchestrator.start_system()
    
    # Register and start some test components
    orchestrator.register_component("test_engine", "test", Priority.NORMAL)
    
    # Simulate component instance
    class TestComponent:
        """Test component for demonstration."""
        
        def __init__(self, name: str) -> None:
            self.name = name
        
        def heartbeat(self) -> None:
            """Send heartbeat."""
            pass
    
    test_component = TestComponent("test_engine")
    orchestrator.start_component("test_engine", test_component)
    
    # Simulate some heartbeats
    for _ in range(5):
        test_component.heartbeat()
        time.sleep(1)
    
    # Get system status
    status = orchestrator.get_system_status()
    safe_print(f"System status: {json.dumps(status, indent=2, default=str)}")
    
    # Shutdown
    orchestrator.shutdown()

if __name__ == "__main__":
    main() 