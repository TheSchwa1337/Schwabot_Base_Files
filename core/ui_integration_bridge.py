"""
UI Integration Bridge for Schwabot Visual Core
=============================================

Bridges core Schwabot systems with the visual interface, providing:
- Real-time data synchronization
- Command routing from UI to core systems
- State management and persistence
- Error handling and fallback mechanisms
"""

import threading
import time
import logging
import json
import yaml
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
from pathlib import Path

# Try to import core components
try:
    from core.quantum_antipole_engine import QuantumAntiPoleEngine, AntiPoleState
    from core.hash_affinity_vault import HashAffinityVault
    from core.master_orchestrator import MasterOrchestrator
    from core.system_monitor import SystemMonitor
    from core.profit_navigator import ProfitNavigator
    from core.strategy_execution_mapper import StrategyExecutionMapper
    from core.thermal_zone_manager import ThermalZoneManager
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    # Create mock classes for development
    class QuantumAntiPoleEngine:
        def process_tick(self, **kwargs): return None
    class HashAffinityVault:
        def log_tick(self, **kwargs): return None
    class MasterOrchestrator:
        def get_system_health(self): return 1.0
    class SystemMonitor:
        def get_metrics(self): return {}
    class ProfitNavigator:
        def calculate_profit_zones(self): return []
    class StrategyExecutionMapper:
        def get_active_strategy(self): return "Mock"
    class ThermalZoneManager:
        def get_thermal_state(self): return {"temperature": 50.0}

logger = logging.getLogger(__name__)

@dataclass
class UIState:
    """Current state of the UI system"""
    connected: bool = False
    last_update: float = 0.0
    active_panels: List[str] = None
    error_count: int = 0
    
    def __post_init__(self):
        if self.active_panels is None:
            self.active_panels = ["system", "profit", "brain", "flow", "settings"]

@dataclass
class SystemMetrics:
    """Aggregated system metrics for UI display"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_temperature: float = 0.0
    thermal_state: str = "normal"
    system_health: float = 1.0
    active_strategies: List[str] = None
    
    def __post_init__(self):
        if self.active_strategies is None:
            self.active_strategies = []

@dataclass 
class TradingMetrics:
    """Trading-specific metrics"""
    total_profit: float = 0.0
    daily_profit: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    active_positions: int = 0
    last_trade_time: Optional[str] = None
    profit_zones: List[Dict] = None
    
    def __post_init__(self):
        if self.profit_zones is None:
            self.profit_zones = []

@dataclass
class DecisionTrace:
    """Trace of a trading decision"""
    timestamp: str
    decision_type: str
    confidence: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning: str
    execution_time_ms: float

class UIIntegrationBridge:
    """Main bridge between core systems and UI"""
    
    def __init__(self):
        self.ui_state = UIState()
        self.running = False
        
        # Core system instances
        self.core_systems = {}
        self.callbacks: Dict[str, List[Callable]] = {
            'system_update': [],
            'trading_update': [],
            'decision_update': [],
            'error_update': []
        }
        
        # Data buffers
        self.system_metrics_history = deque(maxlen=1000)
        self.trading_metrics_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=100)
        self.error_log = deque(maxlen=500)
        
        # Thread safety
        self._lock = threading.Lock()
        self._thread = None
        
        # Initialize core systems
        self._initialize_core_systems()
        
    def _initialize_core_systems(self):
        """Initialize and configure core Schwabot systems"""
        try:
            if CORE_AVAILABLE:
                # Initialize core components
                self.core_systems['antipole_engine'] = QuantumAntiPoleEngine()
                self.core_systems['hash_vault'] = HashAffinityVault()
                self.core_systems['orchestrator'] = MasterOrchestrator()
                self.core_systems['system_monitor'] = SystemMonitor()
                self.core_systems['profit_navigator'] = ProfitNavigator()
                self.core_systems['strategy_mapper'] = StrategyExecutionMapper()
                self.core_systems['thermal_manager'] = ThermalZoneManager()
                
                logger.info("Core systems initialized successfully")
            else:
                # Initialize mock systems for development
                self.core_systems['antipole_engine'] = QuantumAntiPoleEngine()
                self.core_systems['hash_vault'] = HashAffinityVault()
                self.core_systems['orchestrator'] = MasterOrchestrator()
                self.core_systems['system_monitor'] = SystemMonitor()
                self.core_systems['profit_navigator'] = ProfitNavigator()
                self.core_systems['strategy_mapper'] = StrategyExecutionMapper()
                self.core_systems['thermal_manager'] = ThermalZoneManager()
                
                logger.warning("Using mock core systems for development")
                
        except Exception as e:
            logger.error(f"Failed to initialize core systems: {e}")
            self.core_systems = {}
    
    def start(self):
        """Start the integration bridge"""
        if not self.running:
            self.running = True
            self.ui_state.connected = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            logger.info("UI Integration Bridge started")
    
    def stop(self):
        """Stop the integration bridge"""
        self.running = False
        self.ui_state.connected = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("UI Integration Bridge stopped")
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown callback type: {event_type}")
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """Notify all callbacks for an event type"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
    
    def _update_loop(self):
        """Main update loop for gathering system data"""
        while self.running:
            try:
                # Gather system metrics
                system_metrics = self._gather_system_metrics()
                self.system_metrics_history.append(system_metrics)
                self._notify_callbacks('system_update', system_metrics)
                
                # Gather trading metrics
                trading_metrics = self._gather_trading_metrics()
                self.trading_metrics_history.append(trading_metrics)
                self._notify_callbacks('trading_update', trading_metrics)
                
                # Check for new decisions
                self._check_for_decisions()
                
                # Update UI state
                self.ui_state.last_update = time.time()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                self.ui_state.error_count += 1
                self._log_error("update_loop", str(e))
                time.sleep(5)  # Longer delay on error
    
    def _gather_system_metrics(self) -> SystemMetrics:
        """Gather current system metrics from core components"""
        metrics = SystemMetrics()
        
        try:
            # Get system health from orchestrator
            if 'orchestrator' in self.core_systems:
                metrics.system_health = self.core_systems['orchestrator'].get_system_health()
            
            # Get thermal state
            if 'thermal_manager' in self.core_systems:
                thermal_state = self.core_systems['thermal_manager'].get_thermal_state()
                metrics.thermal_state = thermal_state.get('state', 'normal')
                metrics.gpu_temperature = thermal_state.get('temperature', 0.0)
            
            # Get active strategies
            if 'strategy_mapper' in self.core_systems:
                active_strategy = self.core_systems['strategy_mapper'].get_active_strategy()
                metrics.active_strategies = [active_strategy] if active_strategy else []
            
            # CPU/Memory/GPU will be updated by hardware monitor
            
        except Exception as e:
            logger.error(f"Error gathering system metrics: {e}")
        
        return metrics
    
    def _gather_trading_metrics(self) -> TradingMetrics:
        """Gather current trading metrics from core components"""
        metrics = TradingMetrics()
        
        try:
            # Get profit information from navigator
            if 'profit_navigator' in self.core_systems:
                profit_zones = self.core_systems['profit_navigator'].calculate_profit_zones()
                metrics.profit_zones = profit_zones or []
                
                # Calculate total profit (simplified)
                if profit_zones:
                    metrics.total_profit = sum(zone.get('profit', 0) for zone in profit_zones)
            
            # Get hash vault statistics
            if 'hash_vault' in self.core_systems:
                vault_stats = getattr(self.core_systems['hash_vault'], 'get_statistics', lambda: {})()
                metrics.total_trades = vault_stats.get('total_ticks', 0)
                metrics.win_rate = vault_stats.get('success_rate', 0.0)
            
        except Exception as e:
            logger.error(f"Error gathering trading metrics: {e}")
        
        return metrics
    
    def _check_for_decisions(self):
        """Check for new trading decisions and log them"""
        try:
            # This would integrate with the actual decision logging system
            # For now, we'll simulate decision detection
            
            if 'antipole_engine' in self.core_systems:
                # Check if engine has new decisions (this is conceptual)
                # In reality, decisions would be logged through a different mechanism
                pass
                
        except Exception as e:
            logger.error(f"Error checking decisions: {e}")
    
    def log_decision(self, decision_type: str, confidence: float, 
                    inputs: Dict, outputs: Dict, reasoning: str = ""):
        """Log a trading decision for UI display"""
        try:
            decision = DecisionTrace(
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                decision_type=decision_type,
                confidence=confidence,
                inputs=inputs,
                outputs=outputs,
                reasoning=reasoning,
                execution_time_ms=0.0  # Would be calculated in real implementation
            )
            
            with self._lock:
                self.decision_history.append(decision)
            
            self._notify_callbacks('decision_update', decision)
            logger.info(f"Decision logged: {decision_type} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
    
    def _log_error(self, source: str, message: str):
        """Log an error with timestamp"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'message': message
        }
        
        with self._lock:
            self.error_log.append(error_entry)
        
        self._notify_callbacks('error_update', error_entry)
    
    # Public interface methods for UI
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics"""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def get_current_trading_metrics(self) -> Optional[TradingMetrics]:
        """Get the most recent trading metrics"""
        return self.trading_metrics_history[-1] if self.trading_metrics_history else None
    
    def get_recent_decisions(self, count: int = 10) -> List[DecisionTrace]:
        """Get recent trading decisions"""
        with self._lock:
            return list(self.decision_history)[-count:]
    
    def get_system_metrics_history(self, count: int = 100) -> List[SystemMetrics]:
        """Get historical system metrics"""
        return list(self.system_metrics_history)[-count:]
    
    def get_trading_metrics_history(self, count: int = 100) -> List[TradingMetrics]:
        """Get historical trading metrics"""
        return list(self.trading_metrics_history)[-count:]
    
    def get_error_log(self, count: int = 50) -> List[Dict]:
        """Get recent error entries"""
        with self._lock:
            return list(self.error_log)[-count:]
    
    def execute_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command from the UI"""
        try:
            logger.info(f"Executing command: {command} with parameters: {parameters}")
            
            if command == "start_trading":
                return self._start_trading(parameters)
            elif command == "stop_trading":
                return self._stop_trading(parameters)
            elif command == "update_settings":
                return self._update_settings(parameters)
            elif command == "force_decision":
                return self._force_decision(parameters)
            elif command == "reset_system":
                return self._reset_system(parameters)
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def _start_trading(self, parameters: Dict) -> Dict[str, Any]:
        """Start automated trading"""
        try:
            # This would integrate with the strategy execution system
            logger.info("Starting automated trading")
            return {"success": True, "message": "Trading started"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _stop_trading(self, parameters: Dict) -> Dict[str, Any]:
        """Stop automated trading"""
        try:
            logger.info("Stopping automated trading")
            return {"success": True, "message": "Trading stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_settings(self, parameters: Dict) -> Dict[str, Any]:
        """Update system settings"""
        try:
            logger.info(f"Updating settings: {parameters}")
            # This would update the core system configurations
            return {"success": True, "message": "Settings updated"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _force_decision(self, parameters: Dict) -> Dict[str, Any]:
        """Force a trading decision for testing"""
        try:
            decision_type = parameters.get('type', 'MANUAL')
            confidence = parameters.get('confidence', 0.8)
            
            self.log_decision(
                decision_type=decision_type,
                confidence=confidence,
                inputs=parameters,
                outputs={"action": "simulated"},
                reasoning="Manual trigger from UI"
            )
            
            return {"success": True, "message": f"Decision {decision_type} logged"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _reset_system(self, parameters: Dict) -> Dict[str, Any]:
        """Reset system components"""
        try:
            reset_type = parameters.get('type', 'soft')
            logger.info(f"Performing {reset_type} system reset")
            
            if reset_type == 'hard':
                # Clear all data buffers
                self.system_metrics_history.clear()
                self.trading_metrics_history.clear()
                self.decision_history.clear()
                self.error_log.clear()
            
            return {"success": True, "message": f"System {reset_type} reset completed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Singleton instance for global access
ui_bridge = UIIntegrationBridge()

def get_ui_bridge() -> UIIntegrationBridge:
    """Get the global UI bridge instance"""
    return ui_bridge 