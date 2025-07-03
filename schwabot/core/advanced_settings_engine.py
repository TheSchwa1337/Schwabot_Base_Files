# -*- coding: utf-8 -*-
"""
Advanced Settings Engine.

This module provides a robust, thread-safe, and persistent settings engine
for the Schwabot trading system. It allows dynamic registration of settings,
real-time updates, and persistence to a configuration file.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional

# Logging setup
logger = logging.getLogger(__name__)


@dataclass
class SettingDefinition:
    """Defines a setting, its type, validation rules, and default value."""

    setting_type: type
    default_value: Any
    description: str
    validator: Optional[Callable[[Any], bool]] = None


class AdvancedSettingsEngine:
    """Manages dynamic settings for the Schwabot system with persistence."""

    def __init__(self, config_path: str = "config/advanced_settings.json"):
        """Initializes the settings engine."""
        self.config_path = Path(config_path)
        self.setting_definitions: Dict[str, SettingDefinition] = {}
        self.settings_state: Dict[str, Any] = {}
        self._update_lock = Lock()
        self._load_configuration()

    def register_setting(
        self,
        name: str,
        setting_type: type,
        default_value: Any,
        description: str,
        validator: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """Registers a new setting with the engine."""
        if name in self.setting_definitions:
            logger.warning(f"Setting '{name}' is already registered. Overwriting.")

        definition = SettingDefinition(setting_type, default_value, description, validator)
        self.setting_definitions[name] = definition
        if name not in self.settings_state:
            self.settings_state[name] = default_value

    def get_setting(self, name: str) -> Optional[Any]:
        """Retrieves the current value of a setting."""
        return self.settings_state.get(name)

    def set_setting(self, name: str, value: Any) -> bool:
        """Sets the value of a registered setting."""
        if name not in self.setting_definitions:
            logger.error(f"Attempted to set unregistered setting: '{name}'")
            return False

        definition = self.setting_definitions[name]
        if not isinstance(value, definition.setting_type):
            logger.error(
                f"Invalid type for setting '{name}'. Expected {definition.setting_type}, got {type(value)}."
            )
            return False

        if definition.validator and not definition.validator(value):
            logger.error(f"Validation failed for setting '{name}' with value: {value}")
            return False

        with self._update_lock:
            self.settings_state[name] = value
        return True

    def _load_configuration(self) -> None:
        """Load settings from the configuration file."""
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            with self._update_lock:
                self.settings_state = config_data.get("settings_state", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load settings from {self.config_path}: {e}")

    def save_configuration(self) -> bool:
        """Save the current settings to the configuration file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._update_lock:
                config_data = {"settings_state": self.settings_state}
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Saved settings to {self.config_path}")
            return True
        except IOError as e:
            logger.error(f"Failed to save settings to {self.config_path}: {e}")
            return False
