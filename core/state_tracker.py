# -*- coding: utf-8 -*-
"""
State Tracker Module.

This module provides centralized state tracking and routing for the trading system,
maintaining system state snapshots and managing state transitions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Current system state snapshot."""

    tick_phase: Optional[str] = None
    portfolio_shift: Optional[Dict[str, Any]] = None
    state_valid: Optional[bool] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional state tracking
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    execution_flags: Dict[str, bool] = field(default_factory=dict)


class StateTracker:
    """Centralized state tracking and routing for the trading system."""

    def __init__(self):
        """Initialize the state tracker."""
        self.current_state = SystemState()
        self.state_history: List[SystemState] = []
        self.max_history = 100

        # State change callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "tick_phase_change": [],
            "portfolio_shift": [],
            "validation_change": [],
        }

        logger.info("StateTracker initialized")

    def update_tick_phase(self, tick_phase: str) -> None:
        """Update tick phase and trigger callbacks.

        Args:
            tick_phase: New tick phase value.
        """
        if tick_phase != self.current_state.tick_phase:
            old_phase = self.current_state.tick_phase
            self.current_state.tick_phase = tick_phase
            self.current_state.timestamp = datetime.now()

            logger.debug(f"Tick phase changed: {old_phase} -> {tick_phase}")
            self._trigger_callbacks("tick_phase_change", tick_phase)

    def update_portfolio_shift(self, portfolio_shift: Dict[str, Any]) -> None:
        """Update portfolio shift and trigger callbacks.

        Args:
            portfolio_shift: New portfolio shift data.
        """
        self.current_state.portfolio_shift = portfolio_shift
        self.current_state.timestamp = datetime.now()

        logger.debug(f"Portfolio shift updated: {portfolio_shift}")
        self._trigger_callbacks("portfolio_shift", portfolio_shift)

    def update_validation_state(self, state_valid: bool) -> None:
        """Update validation state and trigger callbacks.

        Args:
            state_valid: New validation state.
        """
        if state_valid != self.current_state.state_valid:
            self.current_state.state_valid = state_valid
            self.current_state.timestamp = datetime.now()

            logger.debug(f"Validation state changed: {state_valid}")
            self._trigger_callbacks("validation_change", state_valid)

    def update_market_conditions(self, conditions: Dict[str, Any]) -> None:
        """Update market conditions.

        Args:
            conditions: New market conditions data.
        """
        self.current_state.market_conditions.update(conditions)
        self.current_state.timestamp = datetime.now()

    def update_risk_metrics(self, metrics: Dict[str, float]) -> None:
        """Update risk metrics.

        Args:
            metrics: New risk metrics data.
        """
        self.current_state.risk_metrics.update(metrics)
        self.current_state.timestamp = datetime.now()

    def set_execution_flag(self, flag_name: str, value: bool) -> None:
        """Set execution flag.

        Args:
            flag_name: Name of the execution flag.
            value: Flag value to set.
        """
        self.current_state.execution_flags[flag_name] = value
        self.current_state.timestamp = datetime.now()

    def get_current_state(self) -> SystemState:
        """Get current system state.

        Returns:
            Current system state.
        """
        return self.current_state

    def is_ready_for_execution(self) -> bool:
        """Check if system is ready for trade execution.

        Returns:
            True if system is ready for execution.
        """
        return (
            self.current_state.tick_phase is not None
            and self.current_state.portfolio_shift is not None
            and self.current_state.state_valid is True
        )

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for state changes.

        Args:
            event_type: Type of event to register for.
            callback: Callback function to register.
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown callback event type: {event_type}")

    def _trigger_callbacks(self, event_type: str, value: Any) -> None:
        """Trigger callbacks for a specific event type.

        Args:
            event_type: Type of event that occurred.
            value: Value associated with the event.
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(value)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")

    def store_state_snapshot(self) -> None:
        """Store current state in history."""
        snapshot = SystemState(
            tick_phase=self.current_state.tick_phase,
            portfolio_shift=self.current_state.portfolio_shift.copy() if self.current_state.portfolio_shift else None,
            state_valid=self.current_state.state_valid,
            timestamp=datetime.now(),
            market_conditions=self.current_state.market_conditions.copy(),
            risk_metrics=self.current_state.risk_metrics.copy(),
            execution_flags=self.current_state.execution_flags.copy(),
        )

        self.state_history.append(snapshot)

        # Maintain history size
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history :]

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state.

        Returns:
            Dictionary containing state summary.
        """
        return {
            "tick_phase": self.current_state.tick_phase,
            "portfolio_shift_available": self.current_state.portfolio_shift is not None,
            "state_valid": self.current_state.state_valid,
            "ready_for_execution": self.is_ready_for_execution(),
            "timestamp": self.current_state.timestamp.isoformat(),
            "market_conditions_count": len(self.current_state.market_conditions),
            "risk_metrics_count": len(self.current_state.risk_metrics),
            "execution_flags": self.current_state.execution_flags,
            "history_size": len(self.state_history),
        }

    def get_state_history(self, limit: Optional[int] = None) -> List[SystemState]:
        """Get state history.

        Args:
            limit: Maximum number of history entries to return.

        Returns:
            List of historical state snapshots.
        """
        if limit is None:
            return self.state_history.copy()
        else:
            return self.state_history[-limit:]

    def clear_history(self) -> None:
        """Clear state history."""
        self.state_history.clear()
        logger.info("State history cleared")

    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about state changes.

        Returns:
            Dictionary containing state statistics.
        """
        if not self.state_history:
            return {
                "total_snapshots": 0,
                "phase_changes": 0,
                "validation_changes": 0,
                "average_time_between_snapshots": 0.0,
            }

        total_snapshots = len(self.state_history)
        phase_changes = sum(
            1
            for i in range(1, len(self.state_history))
            if self.state_history[i].tick_phase != self.state_history[i - 1].tick_phase
        )
        validation_changes = sum(
            1
            for i in range(1, len(self.state_history))
            if self.state_history[i].state_valid != self.state_history[i - 1].state_valid
        )

        # Calculate average time between snapshots
        if len(self.state_history) > 1:
            time_diffs = [
                (self.state_history[i].timestamp - self.state_history[i - 1].timestamp).total_seconds()
                for i in range(1, len(self.state_history))
            ]
            avg_time_between = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_between = 0.0

        return {
            "total_snapshots": total_snapshots,
            "phase_changes": phase_changes,
            "validation_changes": validation_changes,
            "average_time_between_snapshots": avg_time_between,
        }


# Global state tracker instance
state_tracker = StateTracker()


def get_state_tracker() -> StateTracker:
    """Get the global state tracker instance.

    Returns:
        The global StateTracker instance.
    """
    return state_tracker


if __name__ == "__main__":
    # Demo the state tracker
    tracker = StateTracker()

    print("State Tracker Demo")
    print("=" * 30)

    # Update some states
    tracker.update_tick_phase("phase_1")
    tracker.update_portfolio_shift({"btc": 0.5, "usdc": 0.5})
    tracker.update_validation_state(True)

    # Show current state
    current_state = tracker.get_current_state()
    print(f"Current tick phase: {current_state.tick_phase}")
    print(f"Portfolio shift: {current_state.portfolio_shift}")
    print(f"State valid: {current_state.state_valid}")
    print(f"Ready for execution: {tracker.is_ready_for_execution()}")

    # Show summary
    summary = tracker.get_state_summary()
    print(f"State summary: {summary}")
