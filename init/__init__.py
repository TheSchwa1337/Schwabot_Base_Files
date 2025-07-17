#!/usr/bin/env python3
"""
System Initialization Module
============================

Handles system-wide initialization and component management.
Provides dependency resolution and startup sequence generation.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InitStatus(Enum):
    """System initialization status enumeration."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    CONFIGURING = "configuring"
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class ComponentType(Enum):
    """Component type enumeration."""
    CORE = "core"
    API = "api"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    MONITORING = "monitoring"
    TRADING = "trading"


@dataclass
class ComponentConfig:
    """Component configuration data."""
    component_id: str
    component_type: ComponentType
    dependencies: List[str]
    config_data: Dict[str, Any]
    is_required: bool
    startup_timeout: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitResult:
    """Component initialization result."""
    component_id: str
    success: bool
    start_time: float
    end_time: float
    duration: float
    error_message: Optional[str] = None


class SystemInitializer:
    """
    System initialization manager.
    
    Handles component initialization, dependency resolution,
    and startup sequence generation.
    """

    def __init__(self, config_path: str = "./config/system_init_config.json"):
        """Initialize the system initializer."""
        self.config_path = config_path
        self.init_status: InitStatus = InitStatus.PENDING
        self.component_configs: Dict[str, ComponentConfig] = {}
        self.init_results: Dict[str, InitResult] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.startup_sequence: List[str] = []
        
        self._load_configuration()
        self._build_dependency_graph()
        self._generate_startup_sequence()
        
        logger.info("SystemInitializer initialized")

    def _load_configuration(self) -> None:
        """Load system initialization configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Load component configurations
                for comp_config in config.get("components", []):
                    component_id = comp_config["component_id"]
                    self.component_configs[component_id] = ComponentConfig(
                        component_id=component_id,
                        component_type=ComponentType(comp_config["component_type"]),
                        dependencies=comp_config.get("dependencies", []),
                        config_data=comp_config.get("config_data", {}),
                        is_required=comp_config.get("is_required", True),
                        startup_timeout=comp_config.get("startup_timeout", 30),
                        metadata=comp_config.get("metadata", {})
                    )

                logger.info(f"Loaded configuration for {len(self.component_configs)} components")
            else:
                self._create_default_configuration()

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()

    def _create_default_configuration(self) -> None:
        """Create default system initialization configuration."""
        default_components = [
            {
                "component_id": "core_system",
                "component_type": "core",
                "dependencies": [],
                "config_data": {"version": "1.0"},
                "is_required": True,
                "startup_timeout": 30
            },
            {
                "component_id": "api_gateway",
                "component_type": "api",
                "dependencies": ["core_system"],
                "config_data": {"port": 8080, "host": "localhost"},
                "is_required": True,
                "startup_timeout": 15
            },
            {
                "component_id": "database",
                "component_type": "database",
                "dependencies": ["core_system"],
                "config_data": {"connection_string": "sqlite:///schwabot.db"},
                "is_required": True,
                "startup_timeout": 20
            },
            {
                "component_id": "trading_engine",
                "component_type": "trading",
                "dependencies": ["core_system", "database"],
                "config_data": {"max_positions": 100},
                "is_required": True,
                "startup_timeout": 45
            }
        ]
        
        for comp_config in default_components:
            component_id = comp_config["component_id"]
            self.component_configs[component_id] = ComponentConfig(
                component_id=component_id,
                component_type=ComponentType(comp_config["component_type"]),
                dependencies=comp_config.get("dependencies", []),
                config_data=comp_config.get("config_data", {}),
                is_required=comp_config.get("is_required", True),
                startup_timeout=comp_config.get("startup_timeout", 30),
                metadata=comp_config.get("metadata", {})
            )

        self._save_configuration()
        logger.info("Default system initialization configuration created")

    def _save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            config = {
                "components": [
                    {
                        "component_id": comp.component_id,
                        "component_type": comp.component_type.value,
                        "dependencies": comp.dependencies,
                        "config_data": comp.config_data,
                        "is_required": comp.is_required,
                        "startup_timeout": comp.startup_timeout,
                        "metadata": comp.metadata
                    }
                    for comp in self.component_configs.values()
                ]
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _build_dependency_graph(self) -> None:
        """Build dependency graph for components."""
        try:
            for component_id, config in self.component_configs.items():
                self.dependency_graph[component_id] = config.dependencies
        except Exception as e:
            logger.error(f"Error building dependency graph: {e}")

    def _generate_startup_sequence(self) -> None:
        """Generate startup sequence based on dependencies."""
        try:
            # Topological sort for dependency resolution
            visited = set()
            temp_visited = set()
            sequence = []

            def visit(component_id: str) -> None:
                if component_id in temp_visited:
                    raise ValueError(f"Circular dependency detected: {component_id}")
                if component_id in visited:
                    return

                temp_visited.add(component_id)

                for dependency in self.dependency_graph.get(component_id, []):
                    visit(dependency)

                temp_visited.remove(component_id)
                visited.add(component_id)
                sequence.append(component_id)

            # Visit all components
            for component_id in self.component_configs.keys():
                if component_id not in visited:
                    visit(component_id)

            self.startup_sequence = sequence
            logger.info(f"Generated startup sequence: {sequence}")

        except Exception as e:
            logger.error(f"Error generating startup sequence: {e}")
            # Fallback to alphabetical order
            self.startup_sequence = sorted(self.component_configs.keys())

    def initialize_system(self) -> bool:
        """Initialize the entire system."""
        try:
            self.init_status = InitStatus.INITIALIZING
            logger.info("Starting system initialization")

            # Initialize components in dependency order
            for component_id in self.startup_sequence:
                if not self._initialize_component(component_id):
                    if self.component_configs[component_id].is_required:
                        logger.error(f"Required component {component_id} failed to initialize")
                        self.init_status = InitStatus.ERROR
                        return False
                    else:
                        logger.warning(f"Optional component {component_id} failed to initialize")
                else:
                    logger.info(f"Component {component_id} initialized successfully")

            self.init_status = InitStatus.READY
            logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during system initialization: {e}")
            self.init_status = InitStatus.ERROR
            return False

    def _initialize_component(self, component_id: str) -> bool:
        """Initialize a specific component."""
        try:
            config = self.component_configs[component_id]
            logger.info(f"Initializing component: {component_id}")

            start_time = time.time()
            success = self._initialize_component_by_type(component_id, config)
            end_time = time.time()

            init_result = InitResult(
                component_id=component_id,
                success=success,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                error_message=None if success else "Component initialization failed"
            )

            self.init_results[component_id] = init_result

            if success:
                logger.info(f"Component {component_id} initialized successfully")
            else:
                logger.error(f"Component {component_id} failed to initialize")

            return success

        except Exception as e:
            logger.error(f"Error initializing component {component_id}: {e}")
            return False

    def _initialize_component_by_type(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize component based on its type."""
        try:
            if config.component_type == ComponentType.CORE:
                return self._initialize_core_component(component_id, config)
            elif config.component_type == ComponentType.API:
                return self._initialize_api_component(component_id, config)
            elif config.component_type == ComponentType.DATABASE:
                return self._initialize_database_component(component_id, config)
            elif config.component_type == ComponentType.NETWORK:
                return self._initialize_network_component(component_id, config)
            elif config.component_type == ComponentType.SECURITY:
                return self._initialize_security_component(component_id, config)
            elif config.component_type == ComponentType.MONITORING:
                return self._initialize_monitoring_component(component_id, config)
            elif config.component_type == ComponentType.TRADING:
                return self._initialize_trading_component(component_id, config)
            else:
                logger.warning(f"Unknown component type: {config.component_type}")
                return False

        except Exception as e:
            logger.error(f"Error in component type initialization: {e}")
            return False

    def _initialize_core_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize core system component."""
        try:
            # Core system initialization logic
            logger.info(f"Initializing core component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing core component {component_id}: {e}")
            return False

    def _initialize_api_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize API component."""
        try:
            # API component initialization logic
            logger.info(f"Initializing API component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing API component {component_id}: {e}")
            return False

    def _initialize_database_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize database component."""
        try:
            # Database component initialization logic
            logger.info(f"Initializing database component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing database component {component_id}: {e}")
            return False

    def _initialize_network_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize network component."""
        try:
            # Network component initialization logic
            logger.info(f"Initializing network component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing network component {component_id}: {e}")
            return False

    def _initialize_security_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize security component."""
        try:
            # Security component initialization logic
            logger.info(f"Initializing security component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing security component {component_id}: {e}")
            return False

    def _initialize_monitoring_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize monitoring component."""
        try:
            # Monitoring component initialization logic
            logger.info(f"Initializing monitoring component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing monitoring component {component_id}: {e}")
            return False

    def _initialize_trading_component(self, component_id: str, config: ComponentConfig) -> bool:
        """Initialize trading component."""
        try:
            # Trading component initialization logic
            logger.info(f"Initializing trading component: {component_id}")
            time.sleep(0.1)  # Simulate initialization time
            return True
        except Exception as e:
            logger.error(f"Error initializing trading component {component_id}: {e}")
            return False

    def get_init_status(self) -> InitStatus:
        """Get current initialization status."""
        return self.init_status

    def get_component_status(self, component_id: str) -> Optional[InitResult]:
        """Get initialization status for a specific component."""
        return self.init_results.get(component_id)

    def get_all_component_statuses(self) -> Dict[str, InitResult]:
        """Get initialization status for all components."""
        return self.init_results.copy()

    def shutdown_system(self) -> bool:
        """Shutdown the entire system."""
        try:
            logger.info("Starting system shutdown")
            self.init_status = InitStatus.SHUTTING_DOWN

            # Shutdown components in reverse dependency order
            for component_id in reversed(self.startup_sequence):
                self._shutdown_component(component_id)

            self.init_status = InitStatus.SHUTDOWN
            logger.info("System shutdown completed")
            return True

        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            self.init_status = InitStatus.ERROR
            return False

    def _shutdown_component(self, component_id: str) -> None:
        """Shutdown a specific component."""
        try:
            logger.info(f"Shutting down component: {component_id}")
            # Component shutdown logic would go here
            time.sleep(0.05)  # Simulate shutdown time
        except Exception as e:
            logger.error(f"Error shutting down component {component_id}: {e}")

    def get_init_statistics(self) -> Dict[str, Any]:
        """Get initialization statistics."""
        total_components = len(self.component_configs)
        successful_components = sum(1 for result in self.init_results.values() if result.success)
        failed_components = total_components - successful_components

        total_duration = sum(result.duration for result in self.init_results.values())
        avg_duration = total_duration / len(self.init_results) if self.init_results else 0

        return {
            "total_components": total_components,
            "successful_components": successful_components,
            "failed_components": failed_components,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "init_status": self.init_status.value,
            "startup_sequence": self.startup_sequence
        }


def main() -> None:
    """Main function for testing the system initializer."""
    try:
        initializer = SystemInitializer()
        
        print("System Initializer Test")
        print("=" * 30)
        
        # Test initialization
        success = initializer.initialize_system()
        
        if success:
            print("✅ System initialization successful")
            
            # Show statistics
            stats = initializer.get_init_statistics()
            print(f"Total components: {stats['total_components']}")
            print(f"Successful: {stats['successful_components']}")
            print(f"Failed: {stats['failed_components']}")
            print(f"Average duration: {stats['average_duration']:.3f}s")
            
            # Test shutdown
            shutdown_success = initializer.shutdown_system()
            if shutdown_success:
                print("✅ System shutdown successful")
            else:
                print("❌ System shutdown failed")
        else:
            print("❌ System initialization failed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    main()
