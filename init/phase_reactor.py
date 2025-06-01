"""
Phase Reactor for Schwabot System
Policy engine for phase-based action triggers
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
from pathlib import Path
from .event_bus import EventBus
from .phase_state import PhaseState, PhaseMetrics

@dataclass
class PolicyAction:
    """Container for policy action configuration"""
    name: str
    callback: Callable
    conditions: Dict[str, Any]
    cooldown: float
    last_triggered: float = 0.0

@dataclass
class PolicyState:
    """Container for policy state"""
    active: bool
    last_update: float
    current_phase: str
    actions_triggered: List[str]

class PhaseReactor:
    """Policy engine for phase-based action triggers"""
    
    def __init__(self, event_bus: EventBus, phase_state: PhaseState, log_dir: str = "logs"):
        self.event_bus = event_bus
        self.phase_state = phase_state
        self.policies: Dict[str, PolicyAction] = {}
        self.state = PolicyState(
            active=True,
            last_update=datetime.now().timestamp(),
            current_phase="INITIALIZING",
            actions_triggered=[]
        )
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "phase_reactor.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PhaseReactor')
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default policies
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default policy actions"""
        # Panic zone policy
        self.register_policy(
            "panic_zone",
            self._handle_panic_zone,
            {
                "entropy_score": lambda x: x > 4.5,
                "coherence_score": lambda x: x < 0.4
            },
            cooldown=30.0
        )
        
        # High velocity policy
        self.register_policy(
            "high_velocity",
            self._handle_high_velocity,
            {
                "velocity_class": lambda x: x == "HIGH"
            },
            cooldown=5.0
        )
        
        # Liquidity vacuum policy
        self.register_policy(
            "liquidity_vacuum",
            self._handle_liquidity_vacuum,
            {
                "liquidity_status": lambda x: x == "vacuum"
            },
            cooldown=15.0
        )
        
        # TPF state change policy
        self.register_policy(
            "tpf_state_change",
            self._handle_tpf_state_change,
            {
                "tpf_state": lambda x: x in ["STABILIZING", "DETONATING"]
            },
            cooldown=10.0
        )
    
    def register_policy(self, name: str, callback: Callable, conditions: Dict[str, Callable], cooldown: float) -> None:
        """
        Register a new policy action
        
        Args:
            name: Policy name
            callback: Action callback
            conditions: Condition mapping
            cooldown: Cooldown period in seconds
        """
        with self._lock:
            self.policies[name] = PolicyAction(
                name=name,
                callback=callback,
                conditions=conditions,
                cooldown=cooldown
            )
            self.logger.info(f"Registered policy: {name}")
    
    def _check_policy_conditions(self, policy: PolicyAction) -> bool:
        """
        Check if policy conditions are met
        
        Args:
            policy: Policy action to check
            
        Returns:
            True if conditions are met
        """
        metrics = self.phase_state.get_phase_metrics()
        current_time = datetime.now().timestamp()
        
        # Check cooldown
        if current_time - policy.last_triggered < policy.cooldown:
            return False
        
        # Check conditions
        for key, condition in policy.conditions.items():
            value = getattr(metrics, key).value
            if not condition(value):
                return False
        
        return True
    
    def _handle_panic_zone(self) -> None:
        """Handle panic zone policy"""
        self.logger.warning("Panic zone detected - triggering safety measures")
        self.event_bus.update("trading_enabled", False, "phase_reactor", {
            "reason": "panic_zone",
            "entropy": self.phase_state.get_metric_value("entropy_score"),
            "coherence": self.phase_state.get_metric_value("coherence_score")
        })
    
    def _handle_high_velocity(self) -> None:
        """Handle high velocity policy"""
        self.logger.info("High velocity detected - adjusting strategy")
        self.event_bus.update("current_strategy", "momentum", "phase_reactor", {
            "reason": "high_velocity",
            "velocity": self.phase_state.get_metric_value("velocity_class")
        })
    
    def _handle_liquidity_vacuum(self) -> None:
        """Handle liquidity vacuum policy"""
        self.logger.warning("Liquidity vacuum detected - pausing trading")
        self.event_bus.update("trading_enabled", False, "phase_reactor", {
            "reason": "liquidity_vacuum",
            "status": self.phase_state.get_metric_value("liquidity_status")
        })
    
    def _handle_tpf_state_change(self) -> None:
        """Handle TPF state change policy"""
        tpf_state = self.phase_state.get_metric_value("tpf_state")
        self.logger.info(f"TPF state change detected: {tpf_state}")
        self.event_bus.update("tpf_action", "monitor", "phase_reactor", {
            "reason": "tpf_state_change",
            "state": tpf_state
        })
    
    def process_tick(self) -> None:
        """Process current tick through policy engine"""
        with self._lock:
            current_time = datetime.now().timestamp()
            self.state.last_update = current_time
            
            # Check each policy
            for name, policy in self.policies.items():
                if self._check_policy_conditions(policy):
                    try:
                        policy.callback()
                        policy.last_triggered = current_time
                        self.state.actions_triggered.append(name)
                        self.logger.info(f"Triggered policy: {name}")
                    except Exception as e:
                        self.logger.error(f"Error executing policy {name}: {e}")
            
            # Trim action history
            if len(self.state.actions_triggered) > 100:
                self.state.actions_triggered = self.state.actions_triggered[-100:]
    
    def get_policy_state(self) -> Dict[str, Any]:
        """
        Get current policy engine state
        
        Returns:
            Dictionary of policy engine state
        """
        with self._lock:
            return {
                'active': self.state.active,
                'last_update': self.state.last_update,
                'current_phase': self.state.current_phase,
                'actions_triggered': self.state.actions_triggered,
                'policies': {
                    name: {
                        'cooldown': policy.cooldown,
                        'last_triggered': policy.last_triggered
                    }
                    for name, policy in self.policies.items()
                }
            } 