#!/usr/bin/env python3
"""
Mode Manager - Schwabot Mathematical Framework.

Comprehensive mode management system for mathematical trading operations
supporting different computational and risk modes with seamless transitions.

Operational Modes:
- SAFE_MODE: Conservative operations with strict constraints
- OPTIMIZATION_MODE: Advanced mathematical optimization enabled  
- PRODUCTION_MODE: Full trading capabilities with real-time processing
- DIAGNOSTIC_MODE: Testing and validation operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class OperationalMode(Enum):
    """Enumeration of operational modes."""

    SAFE_MODE = "safe"
    OPTIMIZATION_MODE = "optimization"
    PRODUCTION_MODE = "production"
    DIAGNOSTIC_MODE = "diagnostic"
    EMERGENCY_MODE = "emergency"


@dataclass
class ModeConfiguration:
    """Configuration settings for each operational mode."""

    mode: OperationalMode
    max_position_size: float
    max_leverage: float
    enable_advanced_math: bool
    enable_ai_features: bool
    risk_tolerance: float
    computational_timeout: float
    validation_level: str
    auto_fallback: bool


@dataclass
class ModeTransition:
    """Container for mode transition information."""

    from_mode: OperationalMode
    to_mode: OperationalMode
    reason: str
    timestamp: float
    success: bool
    rollback_available: bool


class ModeManager:
    """Core mode management and transition system."""

    def __init__(self) -> None:
        """Initialize the mode management system."""
        self.version = "1.0.0"
        self.current_mode = OperationalMode.SAFE_MODE
        self.previous_mode: Optional[OperationalMode] = None
        self.transition_history = []
        self.mode_configurations = self._initialize_mode_configurations()
        self.emergency_triggered = False
        logger.info(f"ModeManager v{self.version} initialized in {self.current_mode.value} mode")

    def _initialize_mode_configurations(self: Self) -> Dict[OperationalMode, ModeConfiguration]:
        """Initialize default configurations for all operational modes."""
        return {
            OperationalMode.SAFE_MODE: ModeConfiguration(
                mode=OperationalMode.SAFE_MODE,
                max_position_size=0.1,
                max_leverage=1.0,
                enable_advanced_math=False,
                enable_ai_features=False,
                risk_tolerance=0.05,
                computational_timeout=5.0,
                validation_level="strict",
                auto_fallback=True
            ),
            OperationalMode.OPTIMIZATION_MODE: ModeConfiguration(
                mode=OperationalMode.OPTIMIZATION_MODE,
                max_position_size=0.5,
                max_leverage=1.5,
                enable_advanced_math=True,
                enable_ai_features=True,
                risk_tolerance=0.1,
                computational_timeout=30.0,
                validation_level="normal",
                auto_fallback=True
            ),
            OperationalMode.PRODUCTION_MODE: ModeConfiguration(
                mode=OperationalMode.PRODUCTION_MODE,
                max_position_size=1.0,
                max_leverage=2.0,
                enable_advanced_math=True,
                enable_ai_features=True,
                risk_tolerance=0.15,
                computational_timeout=60.0,
                validation_level="normal",
                auto_fallback=True
            ),
            OperationalMode.DIAGNOSTIC_MODE: ModeConfiguration(
                mode=OperationalMode.DIAGNOSTIC_MODE,
                max_position_size=0.01,
                max_leverage=1.0,
                enable_advanced_math=True,
                enable_ai_features=True,
                risk_tolerance=0.02,
                computational_timeout=120.0,
                validation_level="verbose",
                auto_fallback=False
            ),
            OperationalMode.EMERGENCY_MODE: ModeConfiguration(
                mode=OperationalMode.EMERGENCY_MODE,
                max_position_size=0.0,
                max_leverage=1.0,
                enable_advanced_math=False,
                enable_ai_features=False,
                risk_tolerance=0.0,
                computational_timeout=1.0,
                validation_level="emergency",
                auto_fallback=False
            )
        }

    def get_current_mode(self: Self) -> OperationalMode:
        """Get the current operational mode."""
        return self.current_mode

    def get_current_configuration(self: Self) -> ModeConfiguration:
        """Get the configuration for the current mode."""
        return self.mode_configurations[self.current_mode]

    def is_feature_enabled(self: Self, feature: str) -> bool:
        """
        Check if a specific feature is enabled in the current mode.

        Args:
            feature: Feature name to check

        Returns:
            Boolean indicating if feature is enabled
        """
        config = self.get_current_configuration()

        feature_map = {
            'advanced_math': config.enable_advanced_math,
            'ai_features': config.enable_ai_features,
            'auto_fallback': config.auto_fallback,
            'strict_validation': config.validation_level == "strict",
            'emergency_stop': self.emergency_triggered
        }

        return feature_map.get(feature, False)

    def request_mode_transition(self: Self, target_mode: OperationalMode, reason: str = "") -> bool:
        """
        Request transition to a new operational mode.

        Args:
            target_mode: Desired operational mode
            reason: Reason for mode transition

        Returns:
            Boolean indicating if transition was successful
        """
        if target_mode == self.current_mode:
            logger.info(f"Already in {target_mode.value} mode")
            return True

        # Check if transition is allowed
        if not self._is_transition_allowed(self.current_mode, target_mode):
            logger.warning(f"Transition from {self.current_mode.value} to {target_mode.value} not allowed")
            return False

        # Emergency mode can always be activated
        if target_mode == OperationalMode.EMERGENCY_MODE:
            return self._execute_emergency_transition(reason)

        # Standard mode transition
        return self._execute_mode_transition(target_mode, reason)

    def _is_transition_allowed(self: Self, from_mode: OperationalMode, to_mode: OperationalMode) -> bool:
        """Check if a mode transition is allowed."""
        # Emergency mode can always be activated
        if to_mode == OperationalMode.EMERGENCY_MODE:
            return True

        # Cannot transition from emergency mode without manual override
        if from_mode == OperationalMode.EMERGENCY_MODE:
            return False

        # Safe mode transitions
        if from_mode == OperationalMode.SAFE_MODE:
            return to_mode in [OperationalMode.OPTIMIZATION_MODE, OperationalMode.DIAGNOSTIC_MODE]

        # Optimization mode transitions
        if from_mode == OperationalMode.OPTIMIZATION_MODE:
            return to_mode in [OperationalMode.SAFE_MODE, OperationalMode.PRODUCTION_MODE]

        # Production mode transitions
        if from_mode == OperationalMode.PRODUCTION_MODE:
            return to_mode in [OperationalMode.SAFE_MODE, OperationalMode.OPTIMIZATION_MODE]

        # Diagnostic mode transitions
        if from_mode == OperationalMode.DIAGNOSTIC_MODE:
            return True  # Can transition to any mode

        return False

    def _execute_mode_transition(self: Self, target_mode: OperationalMode, reason: str) -> bool:
        """Execute a standard mode transition."""
        import time

        try:
            logger.info(f"Transitioning from {self.current_mode.value} to {target_mode.value}: {reason}")

            # Store previous mode
            self.previous_mode = self.current_mode

            # Update current mode
            self.current_mode = target_mode

            # Record transition
            transition = ModeTransition(
                from_mode=self.previous_mode,
                to_mode=target_mode,
                reason=reason,
                timestamp=time.time(),
                success=True,
                rollback_available=True
            )
            self.transition_history.append(transition)

            logger.info(f"Successfully transitioned to {target_mode.value} mode")
            return True

        except Exception as e:
            logger.error(f"Mode transition failed: {e}")
            return False

    def _execute_emergency_transition(self: Self, reason: str) -> bool:
        """Execute emergency mode transition."""
        import time

        try:
            logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")

            self.previous_mode = self.current_mode
            self.current_mode = OperationalMode.EMERGENCY_MODE
            self.emergency_triggered = True

            # Record emergency transition
            transition = ModeTransition(
                from_mode=self.previous_mode,
                to_mode=OperationalMode.EMERGENCY_MODE,
                reason=f"EMERGENCY: {reason}",
                timestamp=time.time(),
                success=True,
                rollback_available=False
            )
            self.transition_history.append(transition)

            return True

        except Exception as e:
            logger.critical(f"Emergency transition failed: {e}")
            return False

    def rollback_to_previous_mode(self: Self) -> bool:
        """Rollback to the previous operational mode if possible."""
        if not self.previous_mode:
            logger.warning("No previous mode available for rollback")
            return False

        if self.current_mode == OperationalMode.EMERGENCY_MODE:
            logger.warning("Cannot rollback from emergency mode")
            return False

        logger.info(f"Rolling back from {self.current_mode.value} to {self.previous_mode.value}")
        return self.request_mode_transition(self.previous_mode, "rollback_requested")

    def reset_emergency_mode(self: Self, target_mode: OperationalMode = OperationalMode.SAFE_MODE) -> bool:
        """Reset from emergency mode to specified target mode (manual override required)."""
        if self.current_mode != OperationalMode.EMERGENCY_MODE:
            logger.warning("Not currently in emergency mode")
            return False

        logger.info(f"Resetting emergency mode, transitioning to {target_mode.value}")
        self.emergency_triggered = False
        self.previous_mode = OperationalMode.EMERGENCY_MODE
        self.current_mode = target_mode

        return True

    def get_mode_statistics(self: Self) -> Dict[str, Any]:
        """Get statistics about mode usage and transitions."""
        mode_counts = {}
        for transition in self.transition_history:
            mode = transition.to_mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        return {
            'current_mode': self.current_mode.value,
            'previous_mode': self.previous_mode.value if self.previous_mode else None,
            'total_transitions': len(self.transition_history),
            'mode_usage_counts': mode_counts,
            'emergency_triggered': self.emergency_triggered,
            'last_transition': (self.transition_history[-1].reason if self.transition_history else None)
        }

    def validate_mode_constraints(self: Self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if an operation can be performed in the current mode.

        Args:
            operation: Name of the operation to validate
            parameters: Operation parameters

        Returns:
            Dictionary with validation results
        """
        config = self.get_current_configuration()
        violations = []
        adjustments = {}

        # Check position size constraints
        if 'position_size' in parameters:
            pos_size = parameters['position_size']
            if pos_size > config.max_position_size:
                violations.append(f"Position size {pos_size} exceeds mode limit {config.max_position_size}")
                adjustments['position_size'] = config.max_position_size

        # Check leverage constraints
        if 'leverage' in parameters:
            leverage = parameters['leverage']
            if leverage > config.max_leverage:
                violations.append(f"Leverage {leverage} exceeds mode limit {config.max_leverage}")
                adjustments['leverage'] = config.max_leverage

        # Check feature availability
        if operation in ['ai_optimization', 'advanced_math'] and not config.enable_advanced_math:
            violations.append(f"Operation {operation} not available in {self.current_mode.value} mode")

        return {
            'allowed': len(violations) == 0,
            'violations': violations,
            'adjustments': adjustments,
            'mode': self.current_mode.value,
            'risk_tolerance': config.risk_tolerance
        }


def main() -> None:
    """Demo of mode management system."""
    try:
        mode_manager = ModeManager()
        print(f"✅ ModeManager v{mode_manager.version} initialized")
        print(f"🔧 Current mode: {mode_manager.get_current_mode().value}")

        # Test mode transition
        success = mode_manager.request_mode_transition(OperationalMode.OPTIMIZATION_MODE, "testing")
        print(f"📈 Transition to optimization mode: {'✅' if success else '❌'}")

        # Test feature check
        ai_enabled = mode_manager.is_feature_enabled('ai_features')
        print(f"🤖 AI features enabled: {'✅' if ai_enabled else '❌'}")

        # Test operation validation
        test_params = {'position_size': 0.8, 'leverage': 1.2}
        validation = mode_manager.validate_mode_constraints('trade_execution', test_params)
        print(f"⚖️  Operation allowed: {'✅' if validation['allowed'] else '❌'}")

        # Get statistics
        stats = mode_manager.get_mode_statistics()
        print(f"📊 Total transitions: {stats['total_transitions']}")

        print("🎉 Mode management demo completed!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 