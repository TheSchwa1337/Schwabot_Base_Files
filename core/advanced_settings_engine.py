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

        definitions["sell_wall_aggression"] = SettingDefinition(
            name="sell_wall_aggression",
            display_name="Sell Wall Aggression",
            description="Scale factor for sell wall construction during predictable dumps",
            setting_type="slider",
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            step=0.1,
            category="trading_control",
            affects_modules=["execution_engine", "volume_optimizer"],
        )

        # Memory and Learning Settings
        definitions["strategy_decay_rate"] = SettingDefinition(
            name="strategy_decay_rate",
            display_name="Strategy Memory Decay",
            description="Rate at which old strategies decay from memory",
            setting_type="slider",
            default_value=0.1,
            min_value=0.01,
            max_value=0.5,
            step=0.01,
            category="memory_control",
            affects_modules=["memory_stack", "strategy_mapper"],
        )

        definitions["profit_memory_window"] = SettingDefinition(
            name="profit_memory_window",
            display_name="Profit Memory Window",
            description="Time window for retaining profitable strategy hashes",
            setting_type="numeric",
            default_value=24,
            min_value=1,
            max_value=168,
            step=1,
            category="memory_control",
            affects_modules=["profit_memory", "ferris_wheel"],
        )

        # Advanced Mathematical Controls
        definitions["fractal_noise_tolerance"] = SettingDefinition(
            name="fractal_noise_tolerance",
            display_name="Fractal Noise Tolerance",
            description="Tolerance for fractal deviation in signal processing",
            setting_type="slider",
            default_value=0.05,
            min_value=0.01,
            max_value=0.2,
            step=0.01,
            category="mathematical_control",
            affects_modules=["fractal_processor", "signal_filter"],
        )

        definitions["entropy_stabilization"] = SettingDefinition(
            name="entropy_stabilization",
            display_name="Entropy Stabilization",
            description="Enable entropy-based signal stabilization",
            setting_type="toggle",
            default_value=True,
            category="mathematical_control",
            affects_modules=["entropy_stabilizer", "recursive_processor"],
        )

        return definitions

    def get_setting_value(self, setting_name: str) -> Any:
        """Get current value of a setting."""
        if setting_name in self.settings_state:
            return self.settings_state[setting_name]

        # Return default if not set
        if setting_name in self.setting_definitions:
            return self.setting_definitions[setting_name].default_value

        logger.warning(f"Unknown setting requested: {setting_name}")
        return None

    def set_setting_value(self, setting_name: str, value: Any) -> bool:
        """Set value of a setting with validation."""
        if setting_name not in self.setting_definitions:
            logger.error(f"Unknown setting: {setting_name}")
            return False

        definition = self.setting_definitions[setting_name]

        # Validate value
        if not self._validate_setting_value(definition, value):
            logger.error(f"Invalid value for setting {setting_name}: {value}")
            return False

        with self._update_lock:
            old_value = self.settings_state.get(setting_name)
            self.settings_state[setting_name] = value

            # Update bias coefficients
            self._update_bias_coefficients(setting_name, value, old_value)

            # Update confidence vectors
            self._update_confidence_vectors(setting_name, value)

            # Log setting change for performance tracking
            self._log_setting_change(setting_name, old_value, value)

        logger.info(f"Setting {setting_name} updated: {old_value} -> {value}")
        return True

    def apply_bias_to_module(self, module_name: str, base_value: float) -> float:
        """Apply settings bias to a module's base value."""
        total_bias = 1.0

        # Find all settings that affect this module
        for setting_name, definition in self.setting_definitions.items():
            if module_name in definition.affects_modules:
                setting_value = self.get_setting_value(setting_name)
                if setting_value is not None:
                    # Apply bias based on setting type
                    if definition.setting_type == "slider":
                        # Normalize setting value to bias coefficient
                        if definition.default_value != 0:
                            bias = setting_value / definition.default_value
                        else:
                            bias = setting_value
                        total_bias *= bias
                    elif definition.setting_type == "toggle":
                        if not setting_value:
                            total_bias *= 0.5  # Reduce effect if disabled

        return base_value * total_bias

    def get_confidence_vector(self, context: str = "default") -> ConfidenceVector:
        """Get confidence vector for a specific context."""
        if context not in self.confidence_vectors:
            self.confidence_vectors[context] = ConfidenceVector()

        # Apply current settings to confidence vector
        cv = self.confidence_vectors[context]

        # Update AI consensus based on preferences
        ai_weight = self.get_setting_value("ai_consensus_weight")
        if ai_weight is not None:
            cv.ai_consensus = ai_weight

        # Update user bias based on settings activity
        user_activity = len([
            s for s in self.settings_state.values()
            if s != self.setting_definitions[
                list(self.settings_state.keys())[0]
            ].default_value
        ])
        cv.user_bias = min(0.3, user_activity * 0.05)

        cv.normalize()
        return cv

    def calculate_unified_signal_score(
        self, echo_signals: List[float], confidence_context: str = "default"
    ) -> float:
        """
        Calculate unified signal activation score based on spatial momentum system.

        Implements: S(t) = Σᵢ εᵢ(t)·Cᵢ(t)·ωᵢ(t) / (λ(t) + ΔE(t))
        """
        if not echo_signals:
            return 0.0

        confidence = self.get_confidence_vector(confidence_context)
        echo_state = self._get_echo_state(confidence_context)

        # Calculate weighted signal sum
        weighted_sum = 0.0
        for i, signal in enumerate(echo_signals):
            # Echo strength εᵢ(t)
            echo_strength = signal * echo_state.amplitude

            # Confidence level Cᵢ(t)
            conf_level = (
                confidence.ai_consensus if i % 4 == 0 else confidence.profit_memory
            )

            # Strategy weight ωᵢ(t)
            strategy_weight = self._get_strategy_weight(i)

            weighted_sum += echo_strength * conf_level * strategy_weight

        # Entropy suppression cost λ(t)
        entropy_cost = self._calculate_entropy_cost(echo_signals)

        # Expected deviation ΔE(t)
        deviation = self._calculate_expected_deviation(echo_signals)

        # Final score calculation
        if entropy_cost + deviation > 0:
            signal_score = weighted_sum / (entropy_cost + deviation)
        else:
            signal_score = weighted_sum

        return float(signal_score)

    def update_profit_feedback(self, setting_name: str, profit_delta: float) -> None:
        """Update profit feedback for a specific setting."""
        if setting_name not in self.profit_feedback:
            self.profit_feedback[setting_name] = []

        self.profit_feedback[setting_name].append(profit_delta)

        # Keep only recent feedback (last 100 trades)
        if len(self.profit_feedback[setting_name]) > 100:
            self.profit_feedback[setting_name] = self.profit_feedback[setting_name][
                -100:
            ]

        # Update effectiveness score
        if len(self.profit_feedback[setting_name]) >= 5:
            recent_profits = self.profit_feedback[setting_name][-5:]
            avg_profit = sum(recent_profits) / len(recent_profits)
            self.setting_effectiveness[setting_name] = avg_profit

    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """Get adaptive setting recommendations based on performance."""
        recommendations = {}

        for setting_name, effectiveness in self.setting_effectiveness.items():
            if setting_name in self.setting_definitions:
                definition = self.setting_definitions[setting_name]
                current_value = self.get_setting_value(setting_name)

                if effectiveness > 0.1:  # Good performance
                    if definition.setting_type == "slider":
                        # Suggest slight increase
                        new_value = min(definition.max_value, current_value * 1.1)
                        recommendations[setting_name] = {
                            "action": "increase",
                            "current": current_value,
                            "suggested": new_value,
                            "reason": f"High effectiveness: {effectiveness:.3f}",
                        }
                elif effectiveness < -0.1:  # Poor performance
                    if definition.setting_type == "slider":
                        # Suggest slight decrease
                        new_value = max(definition.min_value, current_value * 0.9)
                        recommendations[setting_name] = {
                            "action": "decrease",
                            "current": current_value,
                            "suggested": new_value,
                            "reason": f"Low effectiveness: {effectiveness:.3f}",
                        }

        return recommendations

    # Private Methods

    def _validate_setting_value(
        self, definition: SettingDefinition, value: Any
    ) -> bool:
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
            if definition.options and value not in definition.options:
                return False

        # Custom validation function
        if definition.validation_func:
            try:
                return definition.validation_func(value)
            except Exception as e:
                logger.error(f"Validation function failed for {definition.name}: {e}")
                return False

        return True

    def _update_bias_coefficients(
        self, setting_name: str, new_value: Any, old_value: Any
    ) -> None:
        """Update bias coefficients based on setting changes."""
        definition = self.setting_definitions[setting_name]

        if definition.setting_type == "slider":
            # Calculate bias coefficient
            if definition.default_value != 0:
                bias = new_value / definition.default_value
            else:
                bias = new_value
            self.bias_coefficients[setting_name] = bias
        elif definition.setting_type == "toggle":
            self.bias_coefficients[setting_name] = 1.0 if new_value else 0.5

    def _update_confidence_vectors(self, setting_name: str, value: Any) -> None:
        """Update confidence vectors based on setting changes."""
        # Update confidence vectors for all contexts
        for context, cv in self.confidence_vectors.items():
            if setting_name == "ai_consensus_weight":
                cv.ai_consensus = value
            cv.normalize()

    def _log_setting_change(
        self, setting_name: str, old_value: Any, new_value: Any
    ) -> None:
        """Log setting change for performance tracking."""
        timestamp = time.time()
        change_log = {
            "timestamp": timestamp,
            "setting": setting_name,
            "old_value": old_value,
            "new_value": new_value,
        }

        if setting_name not in self.adaptive_memory:
            self.adaptive_memory[setting_name] = {"changes": []}

        self.adaptive_memory[setting_name]["changes"].append(change_log)

        # Keep only recent changes
        if len(self.adaptive_memory[setting_name]["changes"]) > 50:
            self.adaptive_memory[setting_name]["changes"] = self.adaptive_memory[
                setting_name
            ]["changes"][-50:]

    def _get_echo_state(self, context: str) -> EchoState:
        """Get or create echo state for context."""
        if context not in self.echo_states:
            self.echo_states[context] = EchoState()

        # Apply settings to echo state
        echo_state = self.echo_states[context]

        # Update amplitude based on echo delay sensitivity
        sensitivity = self.get_setting_value("echo_delay_sensitivity")
        if sensitivity is not None:
            echo_state.amplitude = sensitivity

        return echo_state

    def _get_strategy_weight(self, index: int) -> float:
        """Get strategy weight for signal index."""
        # Simple strategy weighting - can be enhanced
        base_weight = 1.0

        # Apply strategy decay if relevant
        decay_rate = self.get_setting_value("strategy_decay_rate")
        if decay_rate is not None:
            time_factor = np.exp(-decay_rate * index)
            base_weight *= time_factor

        return base_weight

    def _calculate_entropy_cost(self, signals: List[float]) -> float:
        """Calculate entropy suppression cost."""
        if not signals:
            return 1.0

        # Calculate signal entropy
        signal_array = np.array(signals)
        probabilities = np.abs(signal_array) ** 2
        probabilities = (
            probabilities / np.sum(probabilities)
            if np.sum(probabilities) > 0
            else probabilities
        )

        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Apply entropy stabilization setting
        if self.get_setting_value("entropy_stabilization"):
            entropy *= 0.8  # Reduce cost when stabilization is enabled

        return max(0.1, entropy)

    def _calculate_expected_deviation(self, signals: List[float]) -> float:
        """Calculate expected deviation in signal behavior."""
        if len(signals) < 2:
            return 0.1

        signal_array = np.array(signals)
        deviation = np.std(signal_array)

        # Apply fractal noise tolerance
        tolerance = self.get_setting_value("fractal_noise_tolerance")
        if tolerance is not None:
            deviation = max(tolerance, deviation)

        return deviation

    def _load_configuration(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                for setting_name, value in config.get("settings", {}).items():
                    if setting_name in self.setting_definitions:
                        self.set_setting_value(setting_name, value)

                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Initialize with defaults
                for setting_name, definition in self.setting_definitions.items():
                    self.settings_state[setting_name] = definition.default_value
                logger.info("Initialized with default settings")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to defaults
            for setting_name, definition in self.setting_definitions.items():
                self.settings_state[setting_name] = definition.default_value

    def save_configuration(self) -> bool:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            config = {
                "settings": self.settings_state,
                "timestamp": time.time(),
                "effectiveness_scores": self.setting_effectiveness,
            }

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "settings_loaded": len(self.settings_state),
            "confidence_vectors": len(self.confidence_vectors),
            "echo_states": len(self.echo_states),
            "bias_coefficients": len(self.bias_coefficients),
            "memory_weights": len(self.memory_weights),
            "profit_feedback_entries": sum(
                len(feedback) for feedback in self.profit_feedback.values()
            ),
            "setting_effectiveness": len(self.setting_effectiveness),
            "adaptive_memory_entries": len(self.adaptive_memory),
            "running": self._running,
        }
