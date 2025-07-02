#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Settings Engine.

Implements Schwabot's advanced settings framework that controls pipeline logic
through bias coefficients and weighted confidence frameworks without disabling
core mathematical operations.

This system enables:
- Echo weight modulation
- AI confidence bias adjustment
- Recursive memory tuning
- Strategy vector weighting
- Temporal filtering controls
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    from schwabot_unified_math import (
        MathematicalValidator,
        RecursionGuard,
        UnifiedMathematicsFramework,
    )
except ImportError:
    # Fallback implementations for testing
    class MathematicalValidator:
        def validate(self, *args, **kwargs):
            return True

    class RecursionGuard:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class UnifiedMathematicsFramework:
        def __init__(self):
            pass

logger = logging.getLogger(__name__)


@dataclass
class SettingDefinition:
    """Definition of an advanced setting parameter."""

    name: str
    display_name: str
    description: str
    setting_type: str  # 'slider', 'toggle', 'select', 'numeric'
    default_value: Union[float, bool, str, int]
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    category: str = "general"
    affects_modules: List[str] = field(default_factory=list)
    validation_func: Optional[Callable] = None


@dataclass
class ConfidenceVector:
    """Multi-dimensional confidence vector for decision weighting."""

    ai_consensus: float = 0.6
    profit_memory: float = 0.3
    strategy_alignment: float = 0.2
    user_bias: float = 0.1
    echo_resonance: float = 0.0
    temporal_decay: float = 0.9

    def normalize(self) -> None:
        """Normalize confidence vector to unit magnitude."""
        total = sum([
            self.ai_consensus,
            self.profit_memory,
            self.strategy_alignment,
            self.user_bias,
        ])
        if total > 0:
            factor = 1.0 / total
            self.ai_consensus *= factor
            self.profit_memory *= factor
            self.strategy_alignment *= factor
            self.user_bias *= factor


@dataclass
class EchoState:
    """Echo signal state for recursive feedback."""

    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    decay_rate: float = 0.1
    stability_threshold: float = 1e-6
    last_update: float = field(default_factory=time.time)


class AdvancedSettingsEngine:
    """
    Advanced settings engine that implements Schwabot's spatial momentum system
    with weighted confidence frameworks and echo-based signal modulation.
    """

    def __init__(
        self,
        config_path: Path = Path("settings/advanced_config.json"),
        math_framework: Optional[UnifiedMathematicsFramework] = None,
    ) -> None:
        """Initialize the advanced settings engine."""
        self.config_path = Path(config_path)
        self.math_framework = math_framework or UnifiedMathematicsFramework()
        self.validator = MathematicalValidator()
        self.recursion_guard = RecursionGuard()

        # Core state management
        self.settings_state: Dict[str, Any] = {}
        self.confidence_vectors: Dict[str, ConfidenceVector] = {}
        self.echo_states: Dict[str, EchoState] = {}
        self.bias_coefficients: Dict[str, float] = {}
        self.memory_weights: Dict[str, float] = {}

        # Performance tracking
        self.profit_feedback: Dict[str, List[float]] = {}
        self.setting_effectiveness: Dict[str, float] = {}
        self.adaptive_memory: Dict[str, Dict] = {}

        # Threading for async updates
        self._update_lock = threading.RLock()
        self._running = False

        # Initialize settings definitions
        self.setting_definitions = self._initialize_setting_definitions()

        # Load configuration
        self._load_configuration()

        logger.info("Advanced Settings Engine initialized")

    def _initialize_setting_definitions(self) -> Dict[str, SettingDefinition]:
        """Initialize all advanced setting definitions."""
        definitions = {}

        # Echo Control Settings
        definitions["echo_delay_sensitivity"] = SettingDefinition(
            name="echo_delay_sensitivity",
            display_name="Echo Delay Sensitivity",
            description="Controls lag window for ghost signal detection",
            setting_type="slider",
            default_value=0.8,
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            category="echo_control",
            affects_modules=["echo_modulator", "drift_engine", "signal_processing"],
        )

        definitions["ghost_relay_threshold"] = SettingDefinition(
            name="ghost_relay_threshold",
            display_name="Ghost Relay Threshold",
            description="Minimum entropy for ghost trade activation",
            setting_type="slider",
            default_value=0.9,
            min_value=0.5,
            max_value=1.5,
            step=0.05,
            category="echo_control",
            affects_modules=["ghost_detector", "execution_engine"],
        )

        # AI Bias Settings
        definitions["ai_consensus_weight"] = SettingDefinition(
            name="ai_consensus_weight",
            display_name="AI Consensus Weight",
            description="Weight given to AI consensus in decision making",
            setting_type="slider",
            default_value=0.6,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            category="ai_control",
            affects_modules=["ai_coordinator", "decision_engine"],
        )

        definitions["r1_preference"] = SettingDefinition(
            name="r1_preference",
            display_name="R1 Model Preference",
            description="Preference weight for R1 model outputs",
            setting_type="slider",
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            category="ai_control",
            affects_modules=["ai_coordinator"],
        )

        definitions["claude_preference"] = SettingDefinition(
            name="claude_preference",
            display_name="Claude Model Preference",
            description="Preference weight for Claude model outputs",
            setting_type="slider",
            default_value=0.3,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            category="ai_control",
            affects_modules=["ai_coordinator"],
        )

        definitions["gpt4_preference"] = SettingDefinition(
            name="gpt4_preference",
            display_name="GPT-4 Model Preference",
            description="Preference weight for GPT-4 model outputs",
            setting_type="slider",
            default_value=0.2,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            category="ai_control",
            affects_modules=["ai_coordinator"],
        )

        # Trading Control Settings
        definitions["buy_wall_aggression"] = SettingDefinition(
            name="buy_wall_aggression",
            display_name="Buy Wall Aggression",
            description="Scale factor for buy wall construction during predictable pumps",
            setting_type="slider",
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            step=0.1,
            category="trading_control",
            affects_modules=["execution_engine", "volume_optimizer"],
        )
        return definitions

    def get_setting_value(self, setting_name: str) -> Any:
        """Get the current value of a setting."""
        with self._update_lock:
            return self.settings_state.get(
                setting_name, self.setting_definitions[setting_name].default_value
            )

    def set_setting_value(self, setting_name: str, value: Any) -> bool:
        """Set the value of a setting, with validation."""
        if setting_name not in self.setting_definitions:
            logger.warning(f"Attempted to set undefined setting: {setting_name}")
            return False

        definition = self.setting_definitions[setting_name]
        if not self._validate_setting_value(definition, value):
            logger.warning(f"Invalid value for setting {setting_name}: {value}")
            return False

        with self._update_lock:
            old_value = self.settings_state.get(setting_name)
            self.settings_state[setting_name] = value
            self._update_bias_coefficients(setting_name, value, old_value)
            self._log_setting_change(setting_name, old_value, value)
        
        logger.info(f"Set {setting_name} to {value}")
        return True

    def apply_bias_to_module(self, module_name: str, base_value: float) -> float:
        """Apply the relevant bias coefficient to a base value for a module."""
        with self._update_lock:
            # Simple bias application, can be expanded for complex interactions
            bias = self.bias_coefficients.get(module_name, 1.0)
            return base_value * bias

    def get_confidence_vector(self, context: str = "default") -> ConfidenceVector:
        """Get the confidence vector for a given context."""
        with self._update_lock:
            if context not in self.confidence_vectors:
                self.confidence_vectors[context] = ConfidenceVector()
            return self.confidence_vectors[context]

    def calculate_unified_signal_score(
        self, echo_signals: List[float], confidence_context: str = "default"
    ) -> float:
        """
        Calculate a unified signal score based on echo signals and confidence.
        """
        if not echo_signals:
            return 0.0

        with self.recursion_guard:
            confidence = self.get_confidence_vector(confidence_context)
            echo_state = self._get_echo_state(confidence_context)

            # Modulate signals with echo state
            modulated_signals = [
                s * echo_state.amplitude * np.sin(echo_state.frequency)
                for s in echo_signals
            ]

            # Weight signals with confidence vector
            weighted_score = (
                np.mean(modulated_signals) * confidence.ai_consensus
                + self.memory_weights.get(confidence_context, 0.5) * confidence.profit_memory
            )

            # Apply temporal decay
            time_since_update = time.time() - echo_state.last_update
            decay_factor = np.exp(-echo_state.decay_rate * time_since_update)
            final_score = weighted_score * decay_factor

        return final_score

    def update_profit_feedback(self, setting_name: str, profit_delta: float) -> None:
        """Update profit feedback for adaptive learning."""
        with self._update_lock:
            if setting_name not in self.profit_feedback:
                self.profit_feedback[setting_name] = []
            self.profit_feedback[setting_name].append(profit_delta)
            
            # Simple effectiveness update
            if len(self.profit_feedback[setting_name]) > 10:
                self.setting_effectiveness[setting_name] = np.mean(
                    self.profit_feedback[setting_name][-10:]
                )

    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """Generate adaptive recommendations for settings."""
        recommendations = {}
        with self._update_lock:
            for name, effectiveness in self.setting_effectiveness.items():
                if effectiveness > 0.01:  # Profitable
                    recommendations[name] = {"suggestion": "increase", "confidence": effectiveness}
                elif effectiveness < -0.01:  # Unprofitable
                    recommendations[name] = {"suggestion": "decrease", "confidence": abs(effectiveness)}
        return recommendations

    def _validate_setting_value(self, definition: SettingDefinition, value: Any) -> bool:
        """Validate a setting value against its definition."""
        if definition.setting_type == "slider" or definition.setting_type == "numeric":
            if not isinstance(value, (int, float)):
                return False
            if definition.min_value is not None and value < definition.min_value:
                return False
            if definition.max_value is not None and value > definition.max_value:
                return False
        elif definition.setting_type == "toggle":
            if not isinstance(value, bool):
                return False
        elif definition.setting_type == "select":
            if value not in (definition.options or []):
                return False
        
        if definition.validation_func and not definition.validation_func(value):
            return False
            
        return True

    def _update_bias_coefficients(self, setting_name: str, new_value: Any, old_value: Any) -> None:
        """Update bias coefficients based on a setting change."""
        # This is a simplified example; a real implementation would be more complex
        definition = self.setting_definitions[setting_name]
        for module in definition.affects_modules:
            # Example: adjust bias based on percentage change from default
            change_ratio = new_value / definition.default_value if definition.default_value != 0 else 1.0
            self.bias_coefficients[module] = self.bias_coefficients.get(module, 1.0) * change_ratio

    def _update_confidence_vectors(self, setting_name: str, value: Any) -> None:
        """Update confidence vectors based on a setting change."""
        # Placeholder for more complex logic
        pass

    def _log_setting_change(self, setting_name: str, old_value: Any, new_value: Any) -> None:
        """Log setting change for performance tracking."""
        timestamp = time.time()
        change_log = {
            "timestamp": timestamp,
            "setting": setting_name,
            "old_value": old_value,
            "new_value": new_value,
        }
        # In a real system, this would be written to a persistent log
        logger.debug(f"Setting changed: {change_log}")


    def _get_echo_state(self, context: str) -> EchoState:
        """Get echo state for a given context."""
        with self._update_lock:
            if context not in self.echo_states:
                self.echo_states[context] = EchoState()
            return self.echo_states[context]

    def _get_strategy_weight(self, index: int) -> float:
        """Get strategy weight based on settings."""
        # Placeholder for logic that uses settings to weight strategies
        return 1.0

    def _calculate_entropy_cost(self, signals: List[float]) -> float:
        """Calculate entropy cost for a set of signals."""
        if not signals:
            return 0.0
        
        hist, _ = np.histogram(signals, bins='auto', density=True)
        # Filter out zero probabilities for log calculation
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def _calculate_expected_deviation(self, signals: List[float]) -> float:
        """Calculate expected deviation of signals."""
        if len(signals) < 2:
            return 0.0
        return np.std(signals)

    def _load_configuration(self) -> None:
        """Load settings from the configuration file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)
                
                with self._update_lock:
                    self.settings_state = config_data.get("settings_state", {})
                    # Load other states like confidence vectors if needed
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load settings from {self.config_path}: {e}")
        else:
            # Apply defaults if no config file
            with self._update_lock:
                for name, definition in self.setting_definitions.items():
                    self.settings_state[name] = definition.default_value


    def save_configuration(self) -> bool:
        """Save the current settings to the configuration file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._update_lock:
                config_data = {
                    "settings_state": self.settings_state,
                    # Save other states if needed
                }
                with open(self.config_path, "w") as f:
                    json.dump(config_data, f, indent=4)
            logger.info(f"Saved settings to {self.config_path}")
            return True
        except IOError as e:
            logger.error(f"Failed to save settings to {self.config_path}: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "settings_count": len(self.settings_state),
            "confidence_vectors_count": len(self.confidence_vectors),
            "echo_states_count": len(self.echo_states),
            "bias_coefficients_count": len(self.bias_coefficients),
            "is_running": self._running
        } 