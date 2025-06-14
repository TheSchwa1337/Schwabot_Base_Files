"""
Enhanced Hook System
===================

Provides dynamic hook routing with thermal-aware, profit-synchronized decision making.
Integrates with the complete trading system architecture including memory agents,
thermal management, and profit trajectory analysis.
"""

import os
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from pathlib import Path

# Import system components
try:
    from .memory_agent import MemoryAgent
    from .memory_map import get_memory_map
    from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor, ProfitZoneState
    from .thermal_zone_manager import ThermalZoneManager, ThermalZone
except ImportError:
    # Fallback imports for standalone testing
    MemoryAgent = None
    get_memory_map = None
    ProfitTrajectoryCoprocessor = None
    ThermalZoneManager = None

# Setup logging
logger = logging.getLogger(__name__)

class HookState(Enum):
    """Hook execution states"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    THROTTLED = "throttled"
    FAILED = "failed"
    COOLDOWN = "cooldown"

@dataclass
class HookPerformance:
    """Hook performance tracking"""
    hook_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_profit: float
    average_execution_time: float
    last_executed: Optional[datetime]
    confidence_score: float
    thermal_efficiency: float

@dataclass
class HookContext:
    """Context for hook execution decisions"""
    thermal_zone: str
    profit_zone: str
    thermal_temp: float
    profit_vector_strength: float
    memory_confidence: float
    timestamp: datetime

class DynamicHookRouter:
    """
    Dynamic hook router that activates/deactivates hooks based on
    thermal state, profit trajectory, and memory agent feedback.
    """
    
    def __init__(self, config_path: str = "config/hook_config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Component instances
        self.profit_coprocessor: Optional[Any] = None
        self.thermal_manager: Optional[Any] = None
        self.memory_agent: Optional[Any] = None
        self.memory_map = None
        
        # Hook registry
        self.hook_registry: Dict[str, Any] = {}
        self.hook_performance: Dict[str, HookPerformance] = {}
        self.hook_states: Dict[str, HookState] = {}
        self.hook_cooldowns: Dict[str, datetime] = {}
        
        # Load configuration
        self._load_configuration()
        
        # Initialize components if available
        self._initialize_components()
        
    def _load_configuration(self) -> None:
        """Load hook configuration from YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._default_config()
                self._save_configuration()
        except Exception as e:
            logger.error(f"Error loading hook configuration: {e}")
            self.config = self._default_config()
            
    def _default_config(self) -> Dict[str, Any]:
        """Default hook configuration"""
        return {
            "debug": {
                "clusters": os.getenv("DEBUG_CLUSTERS", "0") == "1",
                "drifts": os.getenv("DEBUG_DRIFTS", "0") == "1",
                "simulate_strategy": os.getenv("SIMULATE_STRATEGY", "0") == "1"
            },
            "hooks": {
                "ncco_manager": {
                    "enabled": True,
                    "thermal_zones": ["cool", "normal", "warm"],
                    "profit_zones": ["surging", "stable"],
                    "confidence_threshold": 0.6,
                    "cooldown_seconds": 30,
                    "max_thermal_temp": 80.0
                },
                "sfsss_router": {
                    "enabled": True,
                    "thermal_zones": ["cool", "normal"],
                    "profit_zones": ["surging"],
                    "confidence_threshold": 0.7,
                    "cooldown_seconds": 45,
                    "max_thermal_temp": 75.0
                },
                "cluster_mapper": {
                    "enabled": True,
                    "thermal_zones": ["cool", "normal", "warm"],
                    "profit_zones": ["surging", "stable", "volatile"],
                    "confidence_threshold": 0.5,
                    "cooldown_seconds": 20,
                    "max_thermal_temp": 85.0
                },
                "drift_engine": {
                    "enabled": True,
                    "thermal_zones": ["cool", "normal"],
                    "profit_zones": ["surging", "volatile"],
                    "confidence_threshold": 0.8,
                    "cooldown_seconds": 60,
                    "max_thermal_temp": 70.0
                },
                "echo_logger": {
                    "enabled": True,
                    "thermal_zones": ["cool", "normal", "warm", "hot"],
                    "profit_zones": ["surging", "stable", "drawdown", "volatile"],
                    "confidence_threshold": 0.3,
                    "cooldown_seconds": 5,
                    "max_thermal_temp": 90.0
                },
                "vault_router": {
                    "enabled": True,
                    "thermal_zones": ["cool", "normal", "warm"],
                    "profit_zones": ["surging", "stable"],
                    "confidence_threshold": 0.9,
                    "cooldown_seconds": 120,
                    "max_thermal_temp": 75.0
                }
            },
            "thresholds": {
                "thermal_critical_temp": 85.0,
                "profit_vector_minimum": 0.3,
                "memory_confidence_minimum": 0.4,
                "hook_failure_threshold": 0.2
            },
            "echo_feedback": {
                "enabled": True,
                "success_weight": 0.7,
                "failure_weight": 0.3,
                "decay_factor": 0.95
            }
        }
        
    def _save_configuration(self) -> None:
        """Save current configuration to YAML"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving hook configuration: {e}")
            
    def _initialize_components(self) -> None:
        """Initialize system components if available"""
        try:
            # Initialize profit coprocessor if available
            if ProfitTrajectoryCoprocessor:
                self.profit_coprocessor = ProfitTrajectoryCoprocessor()
                
            # Initialize thermal manager if available
            if ThermalZoneManager and self.profit_coprocessor:
                self.thermal_manager = ThermalZoneManager(self.profit_coprocessor)
                self.thermal_manager.start_monitoring(interval=30.0)
                
            # Initialize memory components if available
            if get_memory_map:
                self.memory_map = get_memory_map()
                
            if MemoryAgent and self.memory_map:
                self.memory_agent = MemoryAgent(
                    agent_id="hook_system_agent",
                    memory_map=self.memory_map,
                    profit_coprocessor=self.profit_coprocessor,
                    thermal_manager=self.thermal_manager
                )
                
            logger.info("Hook system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            
    def register_hook(self, hook_id: str, hook_instance: Any) -> None:
        """Register a hook instance"""
        with self._lock:
            self.hook_registry[hook_id] = hook_instance
            
            # Initialize performance tracking
            self.hook_performance[hook_id] = HookPerformance(
                hook_id=hook_id,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_profit=0.0,
                average_execution_time=0.0,
                last_executed=None,
                confidence_score=0.5,
                thermal_efficiency=1.0
            )
            self.hook_states[hook_id] = HookState.ACTIVE
            
            logger.info(f"Registered hook: {hook_id}")
            
    def get_current_context(self) -> HookContext:
        """Get current system context for hook decisions"""
        thermal_zone = "unknown"
        thermal_temp = 65.0
        
        if self.thermal_manager and hasattr(self.thermal_manager, 'current_state') and self.thermal_manager.current_state:
            thermal_zone = self.thermal_manager.current_state.zone.value
            thermal_temp = max(
                self.thermal_manager.current_state.cpu_temp,
                self.thermal_manager.current_state.gpu_temp
            )
            
        profit_zone = "unknown"
        profit_vector_strength = 0.5
        
        if self.profit_coprocessor and hasattr(self.profit_coprocessor, 'last_vector') and self.profit_coprocessor.last_vector:
            profit_zone = self.profit_coprocessor.last_vector.zone_state.value
            profit_vector_strength = self.profit_coprocessor.last_vector.vector_strength
            
        memory_confidence = 0.5
        if self.memory_agent and hasattr(self.memory_agent, 'get_agent_statistics'):
            try:
                stats = self.memory_agent.get_agent_statistics()
                memory_confidence = stats.get("success_rate", 0.5)
            except Exception:
                pass
                
        return HookContext(
            thermal_zone=thermal_zone,
            profit_zone=profit_zone,
            thermal_temp=thermal_temp,
            profit_vector_strength=profit_vector_strength,
            memory_confidence=memory_confidence,
            timestamp=datetime.now(timezone.utc)
        )
        
    def should_execute_hook(self, hook_id: str, context: HookContext) -> tuple[bool, str]:
        """Determine if a hook should be executed based on current context"""
        if hook_id not in self.config["hooks"]:
            return False, f"Hook {hook_id} not configured"
            
        hook_config = self.config["hooks"][hook_id]
        
        # Check if hook is enabled
        if not hook_config.get("enabled", True):
            return False, "Hook disabled in configuration"
            
        # Check hook state
        current_state = self.hook_states.get(hook_id, HookState.INACTIVE)
        if current_state != HookState.ACTIVE:
            return False, f"Hook state is {current_state.value}"
            
        # Check cooldown period
        if hook_id in self.hook_cooldowns:
            cooldown_seconds = hook_config.get("cooldown_seconds", 30)
            time_since_cooldown = (context.timestamp - self.hook_cooldowns[hook_id]).total_seconds()
            if time_since_cooldown < cooldown_seconds:
                return False, f"Hook in cooldown for {cooldown_seconds - time_since_cooldown:.1f}s"
                
        # Check thermal zone compatibility
        allowed_thermal_zones = hook_config.get("thermal_zones", [])
        if allowed_thermal_zones and context.thermal_zone not in allowed_thermal_zones:
            return False, f"Thermal zone {context.thermal_zone} not in allowed zones {allowed_thermal_zones}"
            
        # Check profit zone compatibility
        allowed_profit_zones = hook_config.get("profit_zones", [])
        if allowed_profit_zones and context.profit_zone not in allowed_profit_zones:
            return False, f"Profit zone {context.profit_zone} not in allowed zones {allowed_profit_zones}"
            
        # Check confidence threshold
        confidence_threshold = hook_config.get("confidence_threshold", 0.5)
        if context.memory_confidence < confidence_threshold:
            return False, f"Memory confidence {context.memory_confidence:.3f} below threshold {confidence_threshold}"
            
        # Check thermal temperature limits
        max_thermal_temp = hook_config.get("max_thermal_temp", 85.0)
        if context.thermal_temp > max_thermal_temp:
            return False, f"Thermal temperature {context.thermal_temp:.1f}°C exceeds limit {max_thermal_temp}°C"
            
        # Check profit vector strength
        min_profit_vector = self.config["thresholds"]["profit_vector_minimum"]
        if context.profit_vector_strength < min_profit_vector:
            return False, f"Profit vector strength {context.profit_vector_strength:.3f} below minimum {min_profit_vector}"
            
        return True, "All conditions met"
        
    def execute_hook(self, hook_id: str, method_name: str, *args, **kwargs) -> Any:
        """Execute a hook method with performance tracking and echo feedback"""
        with self._lock:
            context = self.get_current_context()
            
            # Check if hook should execute
            should_execute, reason = self.should_execute_hook(hook_id, context)
            if not should_execute:
                logger.debug(f"Hook {hook_id} execution denied: {reason}")
                return None
                
            if hook_id not in self.hook_registry:
                logger.error(f"Hook {hook_id} not found in registry")
                return None
                
            hook_instance = self.hook_registry[hook_id]
            if not hasattr(hook_instance, method_name):
                logger.error(f"Method {method_name} not found on hook {hook_id}")
                return None
                
            # Execute hook
            start_time = datetime.now(timezone.utc)
            execution_success = False
            result = None
            
            try:
                result = getattr(hook_instance, method_name)(*args, **kwargs)
                execution_success = True
                
                # Update hook performance
                self._update_hook_performance(hook_id, True, 0.0, start_time, context)
                
                # Echo success to memory if available
                self._echo_hook_result(hook_id, method_name, True, context, result)
                
                logger.debug(f"Hook {hook_id}.{method_name} executed successfully")
                
            except Exception as e:
                execution_success = False
                
                # Update hook performance
                self._update_hook_performance(hook_id, False, 0.0, start_time, context)
                
                # Echo failure to memory
                self._echo_hook_result(hook_id, method_name, False, context, str(e))
                
                logger.error(f"Error executing hook {hook_id}.{method_name}: {e}")
                
            # Set cooldown
            hook_config = self.config["hooks"].get(hook_id, {})
            cooldown_seconds = hook_config.get("cooldown_seconds", 30)
            self.hook_cooldowns[hook_id] = datetime.now(timezone.utc) + timedelta(seconds=cooldown_seconds)
                
            return result
            
    def _update_hook_performance(self, hook_id: str, success: bool, 
                               profit: float, start_time: datetime, context: HookContext) -> None:
        """Update hook performance metrics"""
        if hook_id not in self.hook_performance:
            return
            
        performance = self.hook_performance[hook_id]
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Update counters
        performance.total_executions += 1
        if success:
            performance.successful_executions += 1
        else:
            performance.failed_executions += 1
            
        # Update averages using exponential moving average
        alpha = 0.1
        performance.average_execution_time = (
            (1 - alpha) * performance.average_execution_time + 
            alpha * execution_time
        )
        performance.average_profit = (
            (1 - alpha) * performance.average_profit + 
            alpha * profit
        )
        
        # Update confidence score
        success_rate = performance.successful_executions / performance.total_executions
        performance.confidence_score = success_rate
        
        # Update thermal efficiency
        thermal_factor = 1.0 - (context.thermal_temp / 100.0)  # Higher temp = lower efficiency
        performance.thermal_efficiency = success_rate * thermal_factor
        
        performance.last_executed = datetime.now(timezone.utc)
        
        # Check if hook should be throttled due to poor performance
        failure_threshold = self.config["thresholds"]["hook_failure_threshold"]
        if success_rate < failure_threshold and performance.total_executions > 10:
            self.hook_states[hook_id] = HookState.THROTTLED
            logger.warning(f"Hook {hook_id} throttled due to poor performance (success rate: {success_rate:.2%})")
            
    def _echo_hook_result(self, hook_id: str, method_name: str, success: bool, 
                         context: HookContext, result: Any) -> None:
        """Echo hook execution result to memory system for learning"""
        if not self.config["echo_feedback"]["enabled"]:
            return
            
        try:
            if self.memory_agent and hasattr(self.memory_agent, 'memory_map'):
                # Create echo data
                echo_data = {
                    "hook_id": hook_id,
                    "method_name": method_name,
                    "success": success,
                    "thermal_zone": context.thermal_zone,
                    "profit_zone": context.profit_zone,
                    "thermal_temp": context.thermal_temp,
                    "profit_vector_strength": context.profit_vector_strength,
                    "memory_confidence": context.memory_confidence,
                    "timestamp": context.timestamp.isoformat(),
                    "result": str(result)[:200] if result else ""
                }
                
                # Store in memory map
                self.memory_agent.memory_map.add_strategy_success(f"hook_{hook_id}", echo_data)
                
        except Exception as e:
            logger.debug(f"Error echoing hook result: {e}")
            
    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hook system statistics"""
        context = self.get_current_context()
        
        return {
            "current_context": asdict(context),
            "hook_states": {k: v.value for k, v in self.hook_states.items()},
            "hook_performance": {
                hook_id: {
                    "total_executions": perf.total_executions,
                    "success_rate": (perf.successful_executions / max(1, perf.total_executions)),
                    "average_profit": perf.average_profit,
                    "average_execution_time": perf.average_execution_time,
                    "confidence_score": perf.confidence_score,
                    "thermal_efficiency": perf.thermal_efficiency,
                    "last_executed": perf.last_executed.isoformat() if perf.last_executed else None
                }
                for hook_id, perf in self.hook_performance.items()
            },
            "system_health": {
                "total_hooks": len(self.hook_registry),
                "active_hooks": sum(1 for state in self.hook_states.values() if state == HookState.ACTIVE),
                "throttled_hooks": sum(1 for state in self.hook_states.values() if state == HookState.THROTTLED),
                "thermal_zone": context.thermal_zone,
                "profit_zone": context.profit_zone,
                "components": {
                    "profit_coprocessor": self.profit_coprocessor is not None,
                    "thermal_manager": self.thermal_manager is not None,
                    "memory_agent": self.memory_agent is not None,
                    "memory_map": self.memory_map is not None
                }
            }
        }
        
    def reset_hook_performance(self, hook_id: Optional[str] = None) -> None:
        """Reset performance metrics for a hook or all hooks"""
        if hook_id:
            if hook_id in self.hook_performance:
                self.hook_performance[hook_id] = HookPerformance(
                    hook_id=hook_id,
                    total_executions=0,
                    successful_executions=0,
                    failed_executions=0,
                    average_profit=0.0,
                    average_execution_time=0.0,
                    last_executed=None,
                    confidence_score=0.5,
                    thermal_efficiency=1.0
                )
                self.hook_states[hook_id] = HookState.ACTIVE
                if hook_id in self.hook_cooldowns:
                    del self.hook_cooldowns[hook_id]
        else:
            # Reset all hooks
            for hook_id in self.hook_performance.keys():
                self.reset_hook_performance(hook_id)
                
    def enable_hook(self, hook_id: str) -> bool:
        """Enable a specific hook"""
        if hook_id in self.config["hooks"]:
            self.config["hooks"][hook_id]["enabled"] = True
            self.hook_states[hook_id] = HookState.ACTIVE
            self._save_configuration()
            logger.info(f"Hook {hook_id} enabled")
            return True
        return False
        
    def disable_hook(self, hook_id: str) -> bool:
        """Disable a specific hook"""
        if hook_id in self.config["hooks"]:
            self.config["hooks"][hook_id]["enabled"] = False
            self.hook_states[hook_id] = HookState.INACTIVE
            self._save_configuration()
            logger.info(f"Hook {hook_id} disabled")
            return True
        return False
        
    def shutdown(self) -> None:
        """Gracefully shutdown the hook system"""
        logger.info("Shutting down hook system...")
        
        if self.thermal_manager and hasattr(self.thermal_manager, 'stop_monitoring'):
            self.thermal_manager.stop_monitoring()
            
        # Clean up memory agents
        if self.memory_agent and hasattr(self.memory_agent, 'cleanup_old_data'):
            self.memory_agent.cleanup_old_data(days_to_keep=30)
            
        logger.info("Hook system shutdown complete") 