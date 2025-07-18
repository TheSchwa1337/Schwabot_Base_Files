#!/usr/bin/env python3
"""
Configuration Loader
===================

Cross-platform configuration loader for hash_recollection system.
Supports YAML, JSON, and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .exceptions import ConfigurationError


class ConfigLoader:
    """Cross-platform configuration loader with fallback support."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration loader."""
        self.config_path = config_path or self._find_config_file()
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _find_config_file(self) -> str:
        """Find the configuration file in common locations."""
        possible_paths = [
            "config/default.yaml",
            "config/config.yaml",
            "config.yaml",
            "default.yaml",
            "config/default.json",
            "config/config.json",
            "config.json",
            "default.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Return default path if none found
        return "config/default.yaml"

    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Load from file
            if os.path.exists(self.config_path):
                self._load_from_file()
            else:
                # Create default config if file doesn't exist
                self._create_default_config()

            # Override with environment variables
            self._load_from_env()

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_from_file(self) -> None:
        """Load configuration from file."""
        file_path = Path(self.config_path)

        if file_path.suffix.lower() in [".yaml", ".yml"]:
            with open(file_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            raise ConfigurationError(f"Unsupported config file format: {file_path.suffix}")

    def _create_default_config(self) -> None:
        """Create default configuration."""
        default_config = {
            "api_url": "http://localhost:8000",
            "frontend": {"theme": "light", "refresh_interval": 10},
            "backend": {
                "entropy": {
                    "max_history_size": 1000,
                    "signal_confidence_threshold": 0.7,
                },
                "bit_operations": {"max_history_size": 1000, "max_patterns": 500},
                "pattern_utils": {
                    "max_patterns": 500,
                    "trend_confidence_threshold": 0.6,
                },
            },
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Write default config
        with open(self.config_path, "w", encoding="utf-8") as f:
            if self.config_path.endswith(".yaml"):
                yaml.dump(default_config, f, default_flow_style=False)
            else:
                json.dump(default_config, f, indent=2)

        self._config = default_config

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "HASH_RECOLLECTION_API_URL": ("api_url", str),
            "HASH_RECOLLECTION_FRONTEND_THEME": ("frontend.theme", str),
            "HASH_RECOLLECTION_FRONTEND_REFRESH_INTERVAL": (
                "frontend.refresh_interval",
                int,
            ),
            "HASH_RECOLLECTION_ENTROPY_MAX_HISTORY": (
                "backend.entropy.max_history_size",
                int,
            ),
            "HASH_RECOLLECTION_ENTROPY_CONFIDENCE_THRESHOLD": (
                "backend.entropy.signal_confidence_threshold",
                float,
            ),
        }
        for env_var, (config_path, value_type) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = value_type(os.environ[env_var])
                    self._set_nested_value(config_path, value)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Invalid environment variable {env_var}: {e}")

    def _set_nested_value(self, path: str, value: Any) -> None:
        """Set a nested configuration value."""
        keys = path.split(".")
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split(".")
        current = self._config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_api_url(self) -> str:
        """Get the API URL."""
        return self.get("api_url", "http://localhost:8000")

    def get_frontend_config(self) -> Dict[str, Any]:
        """Get frontend configuration."""
        return self.get("frontend", {})

    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend configuration."""
        return self.get("backend", {})

    def get_entropy_config(self) -> Dict[str, Any]:
        """Get entropy module configuration."""
        return self.get("backend.entropy", {})

    def get_bit_operations_config(self) -> Dict[str, Any]:
        """Get bit operations configuration."""
        return self.get("backend.bit_operations", {})

    def get_pattern_utils_config(self) -> Dict[str, Any]:
        """Get pattern utils configuration."""
        return self.get("backend.pattern_utils", {})

    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from file and environment."""
        self._load_config()


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value using the global loader."""
    return get_config().get(key, default)
