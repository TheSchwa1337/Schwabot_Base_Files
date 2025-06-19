#!/usr/bin/env python3
"""
Schwabot Integration Orchestrator
=================================

Comprehensive integration layer that connects all system components with the
centralized configuration management system. This orchestrator ensures all
components work together seamlessly and can be configured through the
centralized configuration system.

Key Integration Points:
- Configuration-driven component initialization
- Cross-component communication and data flow
- Real-time configuration updates and hot-reloading
- Component health monitoring and status reporting
- Graceful error handling and fallback mechanisms
- Performance optimization and resource management

Integrated Components:
- Mathematical libraries (mathlib, mathlib_v2, mathlib_v3)
- GAN filtering system (entropy generation and discrimination)
- Trading system (BTC integration, strategy execution)
- Risk management (constraints, monitoring, position sizing)
- Real-time processing (tick processing, market data)
- High-performance computing (GEMM operations, optimization)
- Configuration management (hot-reloading, validation)

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union, Set
from enum import Enum
import warnings
from datetime import datetime, timedelta

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler,
        safe_print, safe_log
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback CLI handler
    class CLIHandler:
        @staticmethod
        def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
            emoji_mapping = {
                '✅': '[SUCCESS]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🚨': '[ALERT]',
                '🎉': '[COMPLETE]', '🔄': '[PROCESSING]', '⏳': '[WAITING]', '⭐': '[STAR]',
                '🚀': '[LAUNCH]', '🔧': '[TOOLS]', '🛠️': '[REPAIR]', '⚡': '[FAST]',
                '🔍': '[SEARCH]', '🎯': '[TARGET]', '🔥': '[HOT]', '❄️': '[COOL]',
                '📊': '[DATA]', '📈': '[PROFIT]', '📉': '[LOSS]', '💰': '[MONEY]',
                '🧪': '[TEST]', '⚖️': '[BALANCE]', '🌡️': '[TEMP]', '🔬': '[ANALYZE]',
                '🎛️': '[CONTROL]', '🔗': '[CONNECT]', '🌐': '[NETWORK]', '⚙️': '[CONFIG]'
            }
            if force_ascii:
                for emoji, replacement in emoji_mapping.items():
                    message = message.replace(emoji, replacement)
            return message

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class IntegrationMode(Enum):
    """Integration mode enumeration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


@dataclass
class ComponentInfo:
    """Component information container"""
    name: str
    status: ComponentStatus = ComponentStatus.UNINITIALIZED
    instance: Optional[Any] = None
    config_section: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[Callable[[], bool]] = None
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    restart_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationMetrics:
    """Integration system metrics"""
    total_components: int = 0
    running_components: int = 0
    failed_components: int = 0
    avg_response_time: float = 0.0
    total_requests: int = 0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class IntegrationOrchestrator:
    """
    Comprehensive integration orchestrator for Schwabot system
    
    This class manages the integration of all system components with the
    centralized configuration system, providing unified control and monitoring.
    """
    
    def __init__(self, config_manager: Optional[Any] = None) -> None:
        """
        Initialize integration orchestrator
        
        Args:
            config_manager: Configuration manager instance
        """
        self.cli_handler = CLIHandler()
        
        # Configuration management
        if config_manager is None:
            from core.config import get_config_manager
            self.config_manager = get_config_manager()
        else:
            self.config_manager = config_manager
        
        # Component registry
        self.components: Dict[str, ComponentInfo] = {}
        self.component_lock = threading.RLock()
        
        # Integration state
        self.mode = IntegrationMode.DEVELOPMENT
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Monitoring and metrics
        self.metrics = IntegrationMetrics()
        self.health_check_interval = 30  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Initialize component registry
        self._initialize_component_registry()
        
        # Add configuration watcher
        self.config_manager.add_watcher(self._on_configuration_changed)
        
        logger.info("Integration Orchestrator initialized")
    
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """
        Safe print function with CLI compatibility
        
        Args:
            message: Message to print
            force_ascii: Force ASCII conversion
        """
        config = self.config_manager.get_config()
        if force_ascii is None:
            force_ascii = config.system.force_ascii_output
        
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """
        Safe logging function with CLI compatibility
        
        Args:
            level: Log level
            message: Message to log
            context: Additional context
            
        Returns:
            True if logging was successful
        """
        if CLI_COMPATIBILITY_AVAILABLE:
            return safe_log(logger, level, message, context)
        else:
            try:
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(message)
                return True
            except Exception:
                return False
    
    def _initialize_component_registry(self) -> None:
        """Initialize the component registry with all available components"""
        try:
            # Mathematical libraries
            self.register_component(ComponentInfo(
                name="mathlib_v1",
                config_section="mathlib",
                dependencies=[],
                health_check=self._check_mathlib_v1_health
            ))
            
            self.register_component(ComponentInfo(
                name="mathlib_v2",
                config_section="mathlib",
                dependencies=["mathlib_v1"],
                health_check=self._check_mathlib_v2_health
            ))
            
            self.register_component(ComponentInfo(
                name="mathlib_v3",
                config_section="mathlib",
                dependencies=["mathlib_v1", "mathlib_v2"],
                health_check=self._check_mathlib_v3_health
            ))
            
            # GAN filtering system
            self.register_component(ComponentInfo(
                name="gan_filter",
                config_section="advanced",
                dependencies=["mathlib_v3"],
                health_check=self._check_gan_filter_health
            ))
            
            # Trading system components
            self.register_component(ComponentInfo(
                name="btc_integration",
                config_section="trading",
                dependencies=["mathlib_v2", "risk_monitor"],
                health_check=self._check_btc_integration_health
            ))
            
            self.register_component(ComponentInfo(
                name="strategy_logic",
                config_section="trading",
                dependencies=["mathlib_v1", "mathlib_v2"],
                health_check=self._check_strategy_logic_health
            ))
            
            # Risk management
            self.register_component(ComponentInfo(
                name="risk_monitor",
                config_section="trading",
                dependencies=["mathlib_v1"],
                health_check=self._check_risk_monitor_health
            ))
            
            # Real-time processing
            self.register_component(ComponentInfo(
                name="tick_processor",
                config_section="realtime",
                dependencies=["mathlib_v1"],
                health_check=self._check_tick_processor_health
            ))
            
            # High-performance computing
            self.register_component(ComponentInfo(
                name="rittle_gemm",
                config_section="mathlib",
                dependencies=[],
                health_check=self._check_rittle_gemm_health
            ))
            
            self.register_component(ComponentInfo(
                name="math_optimization_bridge",
                config_section="mathlib",
                dependencies=["rittle_gemm"],
                health_check=self._check_math_optimization_bridge_health
            ))
            
            self.safe_log('info', f'Registered {len(self.components)} components')
            
        except Exception as e:
            error_msg = f"Error initializing component registry: {e}"
            self.safe_log('error', error_msg)
    
    def register_component(self, component_info: ComponentInfo) -> bool:
        """
        Register a component with the orchestrator
        
        Args:
            component_info: Component information
            
        Returns:
            True if registration was successful
        """
        try:
            with self.component_lock:
                self.components[component_info.name] = component_info
                self.safe_log('info', f'Registered component: {component_info.name}')
                return True
                
        except Exception as e:
            error_msg = f"Error registering component {component_info.name}: {e}"
            self.safe_log('error', error_msg)
            return False
    
    def start_integration(self) -> bool:
        """
        Start the integration orchestrator
        
        Returns:
            True if startup was successful
        """
        try:
            if self.is_running:
                self.safe_log('warning', 'Integration orchestrator already running')
                return True
            
            self.safe_print("🚀 Starting Schwabot Integration Orchestrator")
            self.start_time = datetime.now()
            
            # Get configuration
            config = self.config_manager.get_config()
            self.mode = IntegrationMode(config.system.environment.value)
            
            self.safe_print(f"⚙️ Mode: {self.mode.value}")
            self.safe_print(f"🔧 Components to initialize: {len(self.components)}")
            
            # Initialize components in dependency order
            initialization_order = self._get_initialization_order()
            self.safe_print(f"📋 Initialization order: {', '.join(initialization_order)}")
            
            success_count = 0
            for component_name in initialization_order:
                if self._initialize_component(component_name):
                    success_count += 1
                    self.safe_print(f"✅ {component_name} initialized")
                else:
                    self.safe_print(f"❌ {component_name} failed to initialize")
            
            # Start monitoring
            self._start_monitoring()
            
            self.is_running = True
            
            self.safe_print(f"🎉 Integration orchestrator started")
            self.safe_print(f"   Successfully initialized: {success_count}/{len(self.components)} components")
            
            # Update metrics
            self._update_metrics()
            
            return True
            
        except Exception as e:
            error_msg = f"Error starting integration orchestrator: {e}"
            self.safe_log('error', error_msg)
            self.safe_print(f"❌ {error_msg}")
            return False
    
    def _get_initialization_order(self) -> List[str]:
        """Get component initialization order based on dependencies"""
        try:
            order = []
            remaining = set(self.components.keys())
            
            while remaining:
                # Find components with no unresolved dependencies
                ready = []
                for name in remaining:
                    component = self.components[name]
                    if all(dep in order for dep in component.dependencies):
                        ready.append(name)
                
                if not ready:
                    # Circular dependency or missing dependency
                    self.safe_log('warning', f'Circular/missing dependencies for: {remaining}')
                    # Add remaining components anyway
                    ready = list(remaining)
                
                # Sort ready components by name for consistent ordering
                ready.sort()
                order.extend(ready)
                remaining -= set(ready)
            
            return order
            
        except Exception as e:
            self.safe_log('error', f'Error determining initialization order: {e}')
            return list(self.components.keys())
    
    def _initialize_component(self, component_name: str) -> bool:
        """Initialize a specific component"""
        try:
            with self.component_lock:
                if component_name not in self.components:
                    self.safe_log('error', f'Component not found: {component_name}')
                    return False
                
                component = self.components[component_name]
                component.status = ComponentStatus.INITIALIZING
                
                # Get configuration for this component
                config = self.config_manager.get_config()
                
                # Initialize based on component type
                if component_name == "mathlib_v1":
                    component.instance = self._initialize_mathlib_v1(config)
                elif component_name == "mathlib_v2":
                    component.instance = self._initialize_mathlib_v2(config)
                elif component_name == "mathlib_v3":
                    component.instance = self._initialize_mathlib_v3(config)
                elif component_name == "gan_filter":
                    component.instance = self._initialize_gan_filter(config)
                elif component_name == "btc_integration":
                    component.instance = self._initialize_btc_integration(config)
                elif component_name == "strategy_logic":
                    component.instance = self._initialize_strategy_logic(config)
                elif component_name == "risk_monitor":
                    component.instance = self._initialize_risk_monitor(config)
                elif component_name == "tick_processor":
                    component.instance = self._initialize_tick_processor(config)
                elif component_name == "rittle_gemm":
                    component.instance = self._initialize_rittle_gemm(config)
                elif component_name == "math_optimization_bridge":
                    component.instance = self._initialize_math_optimization_bridge(config)
                
                if component.instance is not None:
                    component.status = ComponentStatus.RUNNING
                    self.safe_log('info', f'Component {component_name} initialized successfully')
                    return True
                else:
                    component.status = ComponentStatus.ERROR
                    component.error_count += 1
                    self.safe_log('error', f'Component {component_name} initialization failed')
                    return False
                    
        except Exception as e:
            self.safe_log('error', f'Error initializing component {component_name}: {e}')
            if component_name in self.components:
                self.components[component_name].status = ComponentStatus.ERROR
                self.components[component_name].error_count += 1
            return False
    
    def _initialize_mathlib_v1(self, config: Any) -> Optional[Any]:
        """Initialize MathLib V1"""
        try:
            from mathlib import MathLib
            return MathLib()
        except ImportError:
            self.safe_log('warning', 'MathLib V1 not available')
            return None
    
    def _initialize_mathlib_v2(self, config: Any) -> Optional[Any]:
        """Initialize MathLib V2"""
        try:
            from mathlib.mathlib_v2 import MathLibV2
            return MathLibV2()
        except ImportError:
            self.safe_log('warning', 'MathLib V2 not available')
            return None
    
    def _initialize_mathlib_v3(self, config: Any) -> Optional[Any]:
        """Initialize MathLib V3"""
        try:
            from core.mathlib_v3 import MathLibV3
            return MathLibV3()
        except ImportError:
            self.safe_log('warning', 'MathLib V3 not available')
            return None
    
    def _initialize_gan_filter(self, config: Any) -> Optional[Any]:
        """Initialize GAN filter system"""
        try:
            if not config.advanced.gan_enabled:
                self.safe_log('info', 'GAN filter disabled in configuration')
                return None
            
            from core.gan_filter import EntropyGAN, GANConfig, GANMode
            
            gan_config = GANConfig(
                noise_dim=100,
                signal_dim=64,
                batch_size=config.advanced.gan_batch_size,
                epochs=1000,
                mode=GANMode.VANILLA
            )
            
            return EntropyGAN(gan_config)
            
        except ImportError:
            self.safe_log('warning', 'GAN filter not available (PyTorch required)')
            return None
        except Exception as e:
            self.safe_log('error', f'Error initializing GAN filter: {e}')
            return None
    
    def _initialize_btc_integration(self, config: Any) -> Optional[Any]:
        """Initialize BTC integration"""
        try:
            from core.simplified_btc_integration import SimplifiedBTCIntegration
            return SimplifiedBTCIntegration()
        except ImportError:
            self.safe_log('warning', 'BTC integration not available')
            return None
    
    def _initialize_strategy_logic(self, config: Any) -> Optional[Any]:
        """Initialize strategy logic"""
        try:
            from core.strategy_logic import StrategyLogic
            return StrategyLogic()
        except ImportError:
            self.safe_log('warning', 'Strategy logic not available')
            return None
    
    def _initialize_risk_monitor(self, config: Any) -> Optional[Any]:
        """Initialize risk monitor"""
        try:
            from core.risk_monitor import RiskMonitor
            return RiskMonitor()
        except ImportError:
            self.safe_log('warning', 'Risk monitor not available')
            return None
    
    def _initialize_tick_processor(self, config: Any) -> Optional[Any]:
        """Initialize tick processor"""
        try:
            from core.tick_processor import TickProcessor
            return TickProcessor()
        except ImportError:
            self.safe_log('warning', 'Tick processor not available')
            return None
    
    def _initialize_rittle_gemm(self, config: Any) -> Optional[Any]:
        """Initialize Rittle GEMM"""
        try:
            from core.rittle_gemm import RittleGEMM
            return RittleGEMM()
        except ImportError:
            self.safe_log('warning', 'Rittle GEMM not available')
            return None
    
    def _initialize_math_optimization_bridge(self, config: Any) -> Optional[Any]:
        """Initialize mathematical optimization bridge"""
        try:
            from core.mathematical_optimization_bridge import MathematicalOptimizationBridge
            return MathematicalOptimizationBridge()
        except ImportError:
            self.safe_log('warning', 'Mathematical optimization bridge not available')
            return None
    
    def _start_monitoring(self) -> None:
        """Start the monitoring thread"""
        try:
            if self.monitoring_thread is not None:
                return
            
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.safe_log('info', 'Monitoring thread started')
            
        except Exception as e:
            self.safe_log('error', f'Error starting monitoring: {e}')
    
    def _monitoring_worker(self) -> None:
        """Monitoring worker thread"""
        while self.is_running:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Update metrics
                self._update_metrics()
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.safe_log('error', f'Error in monitoring worker: {e}')
                time.sleep(self.health_check_interval)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all components"""
        try:
            with self.component_lock:
                for component_name, component in self.components.items():
                    if component.status == ComponentStatus.RUNNING and component.health_check:
                        try:
                            is_healthy = component.health_check()
                            component.last_health_check = datetime.now()
                            
                            if not is_healthy:
                                component.status = ComponentStatus.ERROR
                                component.error_count += 1
                                self.safe_log('warning', f'Health check failed for {component_name}')
                                
                        except Exception as e:
                            component.status = ComponentStatus.ERROR
                            component.error_count += 1
                            self.safe_log('error', f'Health check error for {component_name}: {e}')
                            
        except Exception as e:
            self.safe_log('error', f'Error performing health checks: {e}')
    
    def _update_metrics(self) -> None:
        """Update system metrics"""
        try:
            with self.component_lock:
                self.metrics.total_components = len(self.components)
                self.metrics.running_components = sum(
                    1 for c in self.components.values() 
                    if c.status == ComponentStatus.RUNNING
                )
                self.metrics.failed_components = sum(
                    1 for c in self.components.values() 
                    if c.status == ComponentStatus.ERROR
                )
                
                if self.start_time:
                    self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
                # Calculate error rate
                total_errors = sum(c.error_count for c in self.components.values())
                if self.metrics.total_requests > 0:
                    self.metrics.error_rate = total_errors / self.metrics.total_requests
                
        except Exception as e:
            self.safe_log('error', f'Error updating metrics: {e}')
    
    def _on_configuration_changed(self, config: Any) -> None:
        """Handle configuration changes"""
        try:
            self.safe_log('info', 'Configuration changed, updating components...')
            
            # Check if GAN system needs to be enabled/disabled
            if hasattr(config.advanced, 'gan_enabled'):
                gan_component = self.components.get('gan_filter')
                if gan_component:
                    if config.advanced.gan_enabled and gan_component.status != ComponentStatus.RUNNING:
                        self._initialize_component('gan_filter')
                    elif not config.advanced.gan_enabled and gan_component.status == ComponentStatus.RUNNING:
                        gan_component.status = ComponentStatus.PAUSED
                        self.safe_log('info', 'GAN filter paused due to configuration change')
            
            # Update other component configurations as needed
            self._trigger_event('configuration_changed', config)
            
        except Exception as e:
            self.safe_log('error', f'Error handling configuration change: {e}')
    
    def _trigger_event(self, event_name: str, data: Any = None) -> None:
        """Trigger an event to all registered handlers"""
        try:
            if event_name in self.event_handlers:
                for handler in self.event_handlers[event_name]:
                    try:
                        handler(data)
                    except Exception as e:
                        self.safe_log('error', f'Error in event handler for {event_name}: {e}')
                        
        except Exception as e:
            self.safe_log('error', f'Error triggering event {event_name}: {e}')
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component instance by name
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found/available
        """
        try:
            with self.component_lock:
                if name in self.components:
                    component = self.components[name]
                    if component.status == ComponentStatus.RUNNING:
                        return component.instance
                    else:
                        self.safe_log('warning', f'Component {name} not running (status: {component.status.value})')
                        return None
                else:
                    self.safe_log('error', f'Component {name} not found')
                    return None
                    
        except Exception as e:
            self.safe_log('error', f'Error getting component {name}: {e}')
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            System status dictionary
        """
        try:
            with self.component_lock:
                component_status = {}
                for name, component in self.components.items():
                    component_status[name] = {
                        'status': component.status.value,
                        'error_count': component.error_count,
                        'restart_count': component.restart_count,
                        'last_health_check': component.last_health_check.isoformat() if component.last_health_check else None,
                        'dependencies': component.dependencies
                    }
                
                return {
                    'orchestrator': {
                        'mode': self.mode.value,
                        'running': self.is_running,
                        'start_time': self.start_time.isoformat() if self.start_time else None,
                        'uptime_seconds': self.metrics.uptime_seconds
                    },
                    'components': component_status,
                    'metrics': {
                        'total_components': self.metrics.total_components,
                        'running_components': self.metrics.running_components,
                        'failed_components': self.metrics.failed_components,
                        'error_rate': self.metrics.error_rate,
                        'avg_response_time': self.metrics.avg_response_time
                    }
                }
                
        except Exception as e:
            self.safe_log('error', f'Error getting system status: {e}')
            return {'error': str(e)}
    
    def shutdown(self) -> bool:
        """
        Shutdown the integration orchestrator
        
        Returns:
            True if shutdown was successful
        """
        try:
            if not self.is_running:
                return True
            
            self.safe_print("🛑 Shutting down Integration Orchestrator")
            
            self.is_running = False
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            # Shutdown components
            with self.component_lock:
                for component in self.components.values():
                    component.status = ComponentStatus.SHUTDOWN
            
            self.safe_print("✅ Integration Orchestrator shutdown complete")
            return True
            
        except Exception as e:
            error_msg = f"Error shutting down integration orchestrator: {e}"
            self.safe_log('error', error_msg)
            return False
    
    # Health check methods for components
    def _check_mathlib_v1_health(self) -> bool:
        """Health check for MathLib V1"""
        try:
            component = self.components.get('mathlib_v1')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_mathlib_v2_health(self) -> bool:
        """Health check for MathLib V2"""
        try:
            component = self.components.get('mathlib_v2')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_mathlib_v3_health(self) -> bool:
        """Health check for MathLib V3"""
        try:
            component = self.components.get('mathlib_v3')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_gan_filter_health(self) -> bool:
        """Health check for GAN filter"""
        try:
            component = self.components.get('gan_filter')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_btc_integration_health(self) -> bool:
        """Health check for BTC integration"""
        try:
            component = self.components.get('btc_integration')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_strategy_logic_health(self) -> bool:
        """Health check for strategy logic"""
        try:
            component = self.components.get('strategy_logic')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_risk_monitor_health(self) -> bool:
        """Health check for risk monitor"""
        try:
            component = self.components.get('risk_monitor')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_tick_processor_health(self) -> bool:
        """Health check for tick processor"""
        try:
            component = self.components.get('tick_processor')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_rittle_gemm_health(self) -> bool:
        """Health check for Rittle GEMM"""
        try:
            component = self.components.get('rittle_gemm')
            return component and component.instance is not None
        except Exception:
            return False
    
    def _check_math_optimization_bridge_health(self) -> bool:
        """Health check for mathematical optimization bridge"""
        try:
            component = self.components.get('math_optimization_bridge')
            return component and component.instance is not None
        except Exception:
            return False


# Global orchestrator instance
_orchestrator_instance: Optional[IntegrationOrchestrator] = None


def get_integration_orchestrator(config_manager: Optional[Any] = None) -> IntegrationOrchestrator:
    """
    Get or create the global integration orchestrator
    
    Args:
        config_manager: Optional configuration manager
        
    Returns:
        IntegrationOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = IntegrationOrchestrator(config_manager)
    return _orchestrator_instance


def main() -> None:
    """
    Main function for testing integration orchestrator
    
    Demonstrates the complete integration of all system components with
    centralized configuration management.
    """
    try:
        print("🚀 Integration Orchestrator Test")
        print("=" * 50)
        
        # Initialize orchestrator
        print("🔧 Initializing Integration Orchestrator...")
        orchestrator = get_integration_orchestrator()
        
        # Start integration
        print("\n🎯 Starting system integration...")
        success = orchestrator.start_integration()
        
        if success:
            print("✅ Integration started successfully")
            
            # Get system status
            print("\n📊 System Status:")
            status = orchestrator.get_system_status()
            
            print(f"   Mode: {status['orchestrator']['mode']}")
            print(f"   Running: {status['orchestrator']['running']}")
            print(f"   Components: {status['metrics']['running_components']}/{status['metrics']['total_components']}")
            
            # Show component details
            print("\n🔍 Component Status:")
            for name, info in status['components'].items():
                status_emoji = "✅" if info['status'] == 'running' else "❌" if info['status'] == 'error' else "⏳"
                print(f"   {status_emoji} {name}: {info['status']}")
            
            # Test component access
            print("\n🧪 Testing Component Access:")
            mathlib_v1 = orchestrator.get_component('mathlib_v1')
            if mathlib_v1:
                print("   ✅ MathLib V1 accessible")
            else:
                print("   ❌ MathLib V1 not accessible")
            
            gan_filter = orchestrator.get_component('gan_filter')
            if gan_filter:
                print("   ✅ GAN Filter accessible")
            else:
                print("   ⚠️ GAN Filter not accessible (may be disabled or PyTorch unavailable)")
            
            # Test configuration integration
            print("\n⚙️ Testing Configuration Integration:")
            config_manager = orchestrator.config_manager
            config = config_manager.get_config()
            print(f"   GAN enabled: {config.advanced.gan_enabled}")
            print(f"   GAN batch size: {config.advanced.gan_batch_size}")
            
            # Simulate configuration change
            print("\n🔄 Testing Configuration Hot-Reload:")
            config_manager.update_config('advanced', 'gan_batch_size', 128)
            updated_config = config_manager.get_config()
            print(f"   Updated GAN batch size: {updated_config.advanced.gan_batch_size}")
            
            print("\n🎉 Integration Orchestrator test completed successfully!")
            
            # Shutdown
            print("\n🛑 Shutting down...")
            orchestrator.shutdown()
            
        else:
            print("❌ Integration failed to start")
            
    except Exception as e:
        print(f"❌ Integration Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 