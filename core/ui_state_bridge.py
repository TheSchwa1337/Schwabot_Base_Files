# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
#!/usr/bin/env python3
"""UI State Bridge - State Management and Synchronization for Schwabot UI Components.

This module provides a bridge between the core mathematical systems and UI components,
ensuring proper state synchronization, real-time updates, and state persistence.

Key Features:
- Real-time state synchronization with mathematical engines
- State persistence and recovery
- UI component state management
- Dashboard state coordination
- Safe state transitions and validation

This is a low-risk implementation focused on state management without complex mathematics.
"""

import logging
import threading
import time
import json
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

# Import CLI handler for safe output
try:
    from core.type_binding_system import cli_handler
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    # Fallback for CLI safety
    def safe_print(msg: str) -> None:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', errors='replace').decode('ascii'))

logger = logging.getLogger(__name__)


class StateType(Enum):
    """Types of UI states."""
    DASHBOARD = "dashboard"
    TRADING = "trading"
    MATHEMATICAL = "mathematical"
    SYSTEM = "system"
    CONFIGURATION = "configuration"


class StateStatus(Enum):
    """State status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOADING = "loading"
    ERROR = "error"
    SYNCHRONIZING = "synchronizing"


@dataclass
class UIState:
    """Represents a UI component state."""
    state_id: str
    state_type: StateType
    status: StateStatus
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    parent_state_id: Optional[str] = None
    child_states: List[str] = field(default_factory=list)


@dataclass
class StateTransition:
    """Represents a state transition."""
    from_state_id: str
    to_state_id: str
    transition_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSubscription:
    """Represents a state subscription."""
    subscriber_id: str
    state_ids: Set[str] = field(default_factory=set)
    callback: Callable[[Dict[str, Any]], None]
    last_update: datetime = field(default_factory=datetime.now)


class UIStateBridge:
    """UI State Bridge for managing state synchronization and persistence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the UI State Bridge."""
        self.config = config or self._default_config()
        self.version = "1.0.0"
        
        # State storage
        self.states: Dict[str, UIState] = {}
        self.state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.transitions: List[StateTransition] = []
        
        # Subscriptions and callbacks
        self.subscriptions: Dict[str, StateSubscription] = {}
        self.state_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Synchronization
        self.sync_lock = threading.Lock()
        self.sync_thread: Optional[threading.Thread] = None
        self.sync_active = False
        
        # Performance tracking
        self.metrics = {
            "total_states": 0,
            "total_transitions": 0,
            "total_subscriptions": 0,
            "sync_operations": 0,
            "last_sync": datetime.now()
        }
        
        # Initialize default states
        self._initialize_default_states()
        
        # Start synchronization if enabled
        if self.config.get("enable_auto_sync", True):
            self._start_synchronization()
        
        if CLI_HANDLER_AVAILABLE:
            cli_handler.log_safe(logger, "info", f"UI State Bridge v{self.version} initialized")
        else:
            logger.info(f"UI State Bridge v{self.version} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enable_auto_sync": True,
            "sync_interval_seconds": 1.0,
            "max_state_history": 1000,
            "enable_persistence": True,
            "persistence_interval_seconds": 30.0,
            "state_validation": True,
            "transition_logging": True
        }
    
    def _initialize_default_states(self) -> None:
        """Initialize default UI states."""
        default_states = [
            UIState(
                state_id="dashboard_main",
                state_type=StateType.DASHBOARD,
                status=StateStatus.ACTIVE,
                data={"panels": [], "layout": "default"},
                metadata={"description": "Main dashboard state"}
            ),
            UIState(
                state_id="trading_overview",
                state_type=StateType.TRADING,
                status=StateStatus.ACTIVE,
                data={"active_trades": [], "portfolio_value": 0.0},
                metadata={"description": "Trading overview state"}
            ),
            UIState(
                state_id="mathematical_engine",
                state_type=StateType.MATHEMATICAL,
                status=StateStatus.ACTIVE,
                data={"active_calculations": [], "performance_metrics": {}},
                metadata={"description": "Mathematical engine state"}
            ),
            UIState(
                state_id="system_health",
                state_type=StateType.SYSTEM,
                status=StateStatus.ACTIVE,
                data={"system_status": "healthy", "alerts": []},
                metadata={"description": "System health state"}
            )
        ]
        
        for state in default_states:
            self.states[state.state_id] = state
            self.metrics["total_states"] += 1
    
    def create_state(self, state_id: str, state_type: StateType, 
                    initial_data: Optional[Dict[str, Any]] = None,
                    parent_state_id: Optional[str] = None) -> bool:
        """Create a new UI state."""
        try:
            with self.sync_lock:
                if state_id in self.states:
                    if CLI_HANDLER_AVAILABLE:
                        cli_handler.log_safe(logger, "warning", f"State {state_id} already exists")
                    else:
                        logger.warning(f"State {state_id} already exists")
                    return False
                
                state = UIState(
                    state_id=state_id,
                    state_type=state_type,
                    status=StateStatus.ACTIVE,
                    data=initial_data or {},
                    parent_state_id=parent_state_id
                )
                
                self.states[state_id] = state
                self.metrics["total_states"] += 1
                
                # Update parent state if specified
                if parent_state_id and parent_state_id in self.states:
                    self.states[parent_state_id].child_states.append(state_id)
                
                if CLI_HANDLER_AVAILABLE:
                    cli_handler.log_safe(logger, "info", f"Created state: {state_id}")
                else:
                    logger.info(f"Created state: {state_id}")
                
                return True
                
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error creating state {state_id}: {e}")
            else:
                logger.error(f"Error creating state {state_id}: {e}")
            return False
    
    def update_state(self, state_id: str, data: Dict[str, Any], 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing UI state."""
        try:
            with self.sync_lock:
                if state_id not in self.states:
                    if CLI_HANDLER_AVAILABLE:
                        cli_handler.log_safe(logger, "warning", f"State {state_id} not found")
                    else:
                        logger.warning(f"State {state_id} not found")
                    return False
                
                state = self.states[state_id]
                
                # Store previous state in history
                self.state_history[state_id].append(UIState(
                    state_id=state.state_id,
                    state_type=state.state_type,
                    status=state.status,
                    data=state.data.copy(),
                    metadata=state.metadata.copy(),
                    timestamp=state.timestamp,
                    version=state.version
                ))
                
                # Update state
                state.data.update(data)
                if metadata:
                    state.metadata.update(metadata)
                state.timestamp = datetime.now()
                state.version += 1
                
                # Notify subscribers
                self._notify_state_subscribers(state_id)
                
                return True
                
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error updating state {state_id}: {e}")
            else:
                logger.error(f"Error updating state {state_id}: {e}")
            return False
    
    def get_state(self, state_id: str) -> Optional[UIState]:
        """Get a UI state by ID."""
        return self.states.get(state_id)
    
    def get_states_by_type(self, state_type: StateType) -> List[UIState]:
        """Get all states of a specific type."""
        return [state for state in self.states.values() if state.state_type == state_type]
    
    def delete_state(self, state_id: str) -> bool:
        """Delete a UI state."""
        try:
            with self.sync_lock:
                if state_id not in self.states:
                    return False
                
                state = self.states[state_id]
                
                # Remove from parent state
                if state.parent_state_id and state.parent_state_id in self.states:
                    parent = self.states[state.parent_state_id]
                    if state_id in parent.child_states:
                        parent.child_states.remove(state_id)
                
                # Remove child states
                for child_id in state.child_states:
                    if child_id in self.states:
                        del self.states[child_id]
                
                # Remove state
                del self.states[state_id]
                self.metrics["total_states"] -= 1
                
                if CLI_HANDLER_AVAILABLE:
                    cli_handler.log_safe(logger, "info", f"Deleted state: {state_id}")
                else:
                    logger.info(f"Deleted state: {state_id}")
                
                return True
                
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error deleting state {state_id}: {e}")
            else:
                logger.error(f"Error deleting state {state_id}: {e}")
            return False
    
    def transition_state(self, from_state_id: str, to_state_id: str, 
                        transition_type: str = "manual",
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a state transition."""
        try:
            if from_state_id not in self.states or to_state_id not in self.states:
                return False
            
            transition = StateTransition(
                from_state_id=from_state_id,
                to_state_id=to_state_id,
                transition_type=transition_type,
                metadata=metadata or {}
            )
            
            self.transitions.append(transition)
            self.metrics["total_transitions"] += 1
            
            if self.config.get("transition_logging", True):
                if CLI_HANDLER_AVAILABLE:
                    cli_handler.log_safe(logger, "info", f"State transition: {from_state_id} -> {to_state_id}")
                else:
                    logger.info(f"State transition: {from_state_id} -> {to_state_id}")
            
            return True
            
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error creating transition: {e}")
            else:
                logger.error(f"Error creating transition: {e}")
            return False
    
    def subscribe_to_state(self, subscriber_id: str, state_ids: List[str],
                          callback: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to state updates."""
        try:
            subscription = StateSubscription(
                subscriber_id=subscriber_id,
                state_ids=set(state_ids),
                callback=callback
            )
            
            self.subscriptions[subscriber_id] = subscription
            self.metrics["total_subscriptions"] += 1
            
            # Register callbacks for each state
            for state_id in state_ids:
                self.state_callbacks[state_id].append(callback)
            
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "info", f"Subscription created: {subscriber_id}")
            else:
                logger.info(f"Subscription created: {subscriber_id}")
            
            return True
            
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error creating subscription: {e}")
            else:
                logger.error(f"Error creating subscription: {e}")
            return False
    
    def unsubscribe_from_state(self, subscriber_id: str) -> bool:
        """Unsubscribe from state updates."""
        try:
            if subscriber_id not in self.subscriptions:
                return False
            
            subscription = self.subscriptions[subscriber_id]
            
            # Remove callbacks for each state
            for state_id in subscription.state_ids:
                if state_id in self.state_callbacks:
                    if subscription.callback in self.state_callbacks[state_id]:
                        self.state_callbacks[state_id].remove(subscription.callback)
            
            del self.subscriptions[subscriber_id]
            self.metrics["total_subscriptions"] -= 1
            
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "info", f"Subscription removed: {subscriber_id}")
            else:
                logger.info(f"Subscription removed: {subscriber_id}")
            
            return True
            
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error removing subscription: {e}")
            else:
                logger.error(f"Error removing subscription: {e}")
            return False
    
    def _notify_state_subscribers(self, state_id: str) -> None:
        """Notify subscribers of state changes."""
        if state_id not in self.state_callbacks:
            return
        
        state = self.states.get(state_id)
        if not state:
            return
        
        state_data = {
            "state_id": state_id,
            "state_type": state.state_type.value,
            "status": state.status.value,
            "data": state.data,
            "metadata": state.metadata,
            "timestamp": state.timestamp.isoformat(),
            "version": state.version
        }
        
        for callback in self.state_callbacks[state_id]:
            try:
                callback(state_data)
            except Exception as e:
                if CLI_HANDLER_AVAILABLE:
                    cli_handler.log_safe(logger, "error", f"Error in state callback: {e}")
                else:
                    logger.error(f"Error in state callback: {e}")
    
    def _start_synchronization(self) -> None:
        """Start the synchronization thread."""
        if self.sync_active:
            return
        
        self.sync_active = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        if CLI_HANDLER_AVAILABLE:
            cli_handler.log_safe(logger, "info", "State synchronization started")
        else:
            logger.info("State synchronization started")
    
    def _sync_loop(self) -> None:
        """Synchronization loop."""
        while self.sync_active:
            try:
                self._perform_synchronization()
                time.sleep(self.config.get("sync_interval_seconds", 1.0))
            except Exception as e:
                if CLI_HANDLER_AVAILABLE:
                    cli_handler.log_safe(logger, "error", f"Error in sync loop: {e}")
                else:
                    logger.error(f"Error in sync loop: {e}")
                time.sleep(5.0)  # Longer delay on error
    
    def _perform_synchronization(self) -> None:
        """Perform state synchronization."""
        self.metrics["sync_operations"] += 1
        self.metrics["last_sync"] = datetime.now()
        
        # Update subscription timestamps
        current_time = datetime.now()
        for subscription in self.subscriptions.values():
            subscription.last_update = current_time
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status and metrics."""
        return {
            "version": self.version,
            "total_states": self.metrics["total_states"],
            "total_transitions": self.metrics["total_transitions"],
            "total_subscriptions": self.metrics["total_subscriptions"],
            "sync_operations": self.metrics["sync_operations"],
            "last_sync": self.metrics["last_sync"].isoformat(),
            "sync_active": self.sync_active,
            "config": self.config
        }
    
    def export_state_data(self) -> Dict[str, Any]:
        """Export all state data for persistence."""
        return {
            "states": {k: asdict(v) for k, v in self.states.items()},
            "transitions": [asdict(t) for t in self.transitions[-100:]],  # Last 100 transitions
            "metrics": self.metrics,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_state_data(self, data: Dict[str, Any]) -> bool:
        """Import state data from persistence."""
        try:
            with self.sync_lock:
                # Clear existing states
                self.states.clear()
                
                # Import states
                for state_id, state_data in data.get("states", {}).items():
                    state = UIState(
                        state_id=state_data["state_id"],
                        state_type=StateType(state_data["state_type"]),
                        status=StateStatus(state_data["status"]),
                        data=state_data["data"],
                        metadata=state_data["metadata"],
                        timestamp=datetime.fromisoformat(state_data["timestamp"]),
                        version=state_data["version"],
                        parent_state_id=state_data.get("parent_state_id"),
                        child_states=state_data.get("child_states", [])
                    )
                    self.states[state_id] = state
                
                # Import transitions
                self.transitions = []
                for transition_data in data.get("transitions", []):
                    transition = StateTransition(
                        from_state_id=transition_data["from_state_id"],
                        to_state_id=transition_data["to_state_id"],
                        transition_type=transition_data["transition_type"],
                        timestamp=datetime.fromisoformat(transition_data["timestamp"]),
                        metadata=transition_data["metadata"]
                    )
                    self.transitions.append(transition)
                
                # Update metrics
                self.metrics.update(data.get("metrics", {}))
                
                if CLI_HANDLER_AVAILABLE:
                    cli_handler.log_safe(logger, "info", "State data imported successfully")
                else:
                    logger.info("State data imported successfully")
                
                return True
                
        except Exception as e:
            if CLI_HANDLER_AVAILABLE:
                cli_handler.log_safe(logger, "error", f"Error importing state data: {e}")
            else:
                logger.error(f"Error importing state data: {e}")
            return False


# Global bridge instance
_ui_state_bridge: Optional[UIStateBridge] = None


def get_ui_state_bridge() -> UIStateBridge:
    """Get the global UI state bridge instance."""
    global _ui_state_bridge
    if _ui_state_bridge is None:
        _ui_state_bridge = UIStateBridge()
    return _ui_state_bridge


def main() -> None:
    """Demo of UI State Bridge functionality."""
    try:
        bridge = get_ui_state_bridge()
        safe_print(f"✅ UI State Bridge v{bridge.version} initialized")
        
        # Create a test state
        bridge.create_state("test_panel", StateType.DASHBOARD, {"test_data": "value"})
        
        # Update the state
        bridge.update_state("test_panel", {"test_data": "updated_value"})
        
        # Get bridge status
        status = bridge.get_bridge_status()
        safe_print(f"📊 Bridge Status: {status['total_states']} states, {status['total_subscriptions']} subscriptions")
        
        safe_print("🎉 UI State Bridge demo completed successfully!")
        
    except Exception as e:
        safe_print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
