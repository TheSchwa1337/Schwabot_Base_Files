"""State Tracker - Centralized State Management for Trading Pipeline.
"""State Tracker - Centralized State Management for Trading Pipeline.
"""State Tracker - Centralized State Management for Trading Pipeline.
"""State Tracker - Centralized State Management for Trading Pipeline.


This module provides centralized tracking and routing of critical system state
variables including tick phase, portfolio shifts, and validation states.
"""
"""
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemState:

    """Current system state snapshot."""
"""
"""

    tick_phase: Optional[str] = None
    portfolio_shift: Optional[Dict[str, Any]] = None
    state_valid: Optional[bool] = None
    timestamp: datetime = field(default_factory = datetime.now)

# Additional state tracking
    market_conditions: Dict[str, Any] = field(default_factory = dict)
    risk_metrics: Dict[str, float] = field(default_factory = dict)
    execution_flags: Dict[str, bool] = field(default_factory = dict)


class StateTracker:

    """Centralized state tracking and routing for the trading system."""
"""
"""

    def __init__(self):

        """Initialize the state tracker."""
"""
"""
        self.current_state = SystemState()
        self.state_history = []
        self.max_history = 100

# State change callbacks
        self.callbacks = {
            'tick_phase_change': [],
            'portfolio_shift': [],
            'validation_change': [],
        }

        logger.info("StateTracker initialized")

    def update_tick_phase(self, tick_phase: str) -> None:

        """Update tick phase and trigger callbacks."""
"""
"""
        if tick_phase != self.current_state.tick_phase:
            old_phase = self.current_state.tick_phase
            self.current_state.tick_phase = tick_phase
            self.current_state.timestamp = datetime.now()

            logger.debug(f"Tick phase changed: {old_phase} -> {tick_phase}")
            self._trigger_callbacks('tick_phase_change', tick_phase)

    def update_portfolio_shift(self, portfolio_shift: Dict[str, Any]) -> None:

        """Update portfolio shift and trigger callbacks."""
"""
"""
        self.current_state.portfolio_shift = portfolio_shift
        self.current_state.timestamp = datetime.now()

        logger.debug(f"Portfolio shift updated: {portfolio_shift}")
        self._trigger_callbacks('portfolio_shift', portfolio_shift)

    def update_validation_state(self, state_valid: bool) -> None:

        """Update validation state and trigger callbacks."""
"""
"""
        if state_valid != self.current_state.state_valid:
            self.current_state.state_valid = state_valid
            self.current_state.timestamp = datetime.now()

            logger.debug(f"Validation state changed: {state_valid}")
            self._trigger_callbacks('validation_change', state_valid)

    def update_market_conditions(self, conditions: Dict[str, Any]) -> None:

        """Update market conditions."""
"""
"""
        self.current_state.market_conditions.update(conditions)
        self.current_state.timestamp = datetime.now()

    def update_risk_metrics(self, metrics: Dict[str, float]) -> None:

        """Update risk metrics."""
"""
"""
        self.current_state.risk_metrics.update(metrics)
        self.current_state.timestamp = datetime.now()

    def set_execution_flag(self, flag_name: str, value: bool) -> None:

        """Set execution flag."""
"""
"""
        self.current_state.execution_flags[flag_name] = value
        self.current_state.timestamp = datetime.now()

    def get_current_state(self) -> SystemState:

        """Get current system state."""
"""
"""
        return self.current_state

    def is_ready_for_execution(self) -> bool:

        """Check if system is ready for trade execution."""
"""
"""
        return (
            self.current_state.tick_phase is not None and
            self.current_state.portfolio_shift is not None and
            self.current_state.state_valid is True
        )

    def register_callback(self, event_type: str, callback) -> None:

        """Register a callback for state changes."""
"""
"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown callback event type: {event_type}")

    def _trigger_callbacks(self, event_type: str, value: Any) -> None:

        """Trigger callbacks for a specific event type."""
"""
"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(value)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")

    def store_state_snapshot(self) -> None:

        """Store current state in history."""
"""
"""
        snapshot = SystemState(
            tick_phase = self.current_state.tick_phase,
            portfolio_shift = self.current_state.portfolio_shift.copy() if self.current_state.portfolio_shift else None,
            state_valid = self.current_state.state_valid,
            timestamp = datetime.now(),
            market_conditions = self.current_state.market_conditions.copy(),
            risk_metrics = self.current_state.risk_metrics.copy(),
            execution_flags = self.current_state.execution_flags.copy()
        )

        self.state_history.append(snapshot)

# Maintain history size
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]

    def get_state_summary(self) -> Dict[str, Any]:

        """Get summary of current state."""
"""
"""
        return {
            'tick_phase': self.current_state.tick_phase,
            'portfolio_shift_available': self.current_state.portfolio_shift is not None,
            'state_valid': self.current_state.state_valid,
            'ready_for_execution': self.is_ready_for_execution(),
            'timestamp': self.current_state.timestamp.isoformat(),
            'market_conditions_count': len(self.current_state.market_conditions),
            'risk_metrics_count': len(self.current_state.risk_metrics),
            'execution_flags': self.current_state.execution_flags,
            'history_size': len(self.state_history)
        }


def create_state_tracker() -> StateTracker:

    """Create and return a new StateTracker instance."""
"""
"""
    return StateTracker()
