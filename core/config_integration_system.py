# -*- coding: utf-8 -*-
"""
Configuration Integration System

This module provides a unified interface for loading and managing YAML/JSON configurations
and integrating them with the existing mathematical engines and trading systems.

Features:
- YAML/JSON configuration loading and validation
- Recursive Unicode pathway configuration
- Mathematical engine parameter management
- Trigger system integration
- Backchannel configuration
- Resource utilization mapping
"""

from __future__ import annotations

import json
import logging
import os
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum

import numpy as np

# Import core components
try:
    from core.unified_math_system import UnifiedMathSystem
    from core.synthesis_engine_system import CoreTensorModulator
    from dual_unicore_handler import DualUnicoreHandler
except ImportError as e:
    logging.warning(f"Could not import core components: {e}")

# Configure logging
logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """Configuration file types."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerConfig:
    """Configuration for a trigger."""
    trigger_id: str
    trigger_type: str
    conditions: Dict[str, Any]
    actions: List[str]
    priority: str
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    """Configuration for a mathematical engine."""
    engine_name: str
    version: str
    enabled: bool
    parameters: Dict[str, Any]
    components: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigurationIntegrationSystem:
    """
    Configuration integration system for Schwabot.

    This class manages the loading, validation, and integration of YAML/JSON
    configurations with the existing mathematical engines and trading systems.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration integration system.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache: Dict[str, Any] = {}
        self.trigger_registry: Dict[str, TriggerConfig] = {}
        self.engine_configs: Dict[str, EngineConfig] = {}

        # Initialize core systems
        self.unicore = DualUnicoreHandler()
        self.math_system = None
        self.synthesis_engine = None

        # Load configurations
        self._load_all_configurations()

        logger.info("🎛️ Configuration Integration System initialized")

    def _load_all_configurations(self) -> None:
        """Load all configuration files."""
        try:
            # Load core configuration
            core_config_path = self.config_dir / "schwabot_core_config.yaml"
            if core_config_path.exists():
                self.config_cache["core"] = self._load_yaml_config(
                    core_config_path)
                logger.info("✅ Core configuration loaded")
            else:
                logger.warning("⚠️ Core configuration not found")

            # Load mathematical triggers
            triggers_config_path = self.config_dir / "mathematical_triggers.json"
            if triggers_config_path.exists():
                self.config_cache["triggers"] = self._load_json_config(
                    triggers_config_path)
                self._process_trigger_configurations()
                logger.info("✅ Mathematical triggers loaded")
            else:
                logger.warning("⚠️ Mathematical triggers not found")

            # Load existing configurations
            self._load_existing_configurations()

        except Exception as e:
            logger.error(f"❌ Error loading configurations: {e}")

    def _load_yaml_config(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading YAML config {file_path}: {e}")
            return {}

    def _load_json_config(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON config {file_path}: {e}")
            return {}

    def _load_existing_configurations(self) -> None:
        """Load existing configuration files."""
        config_files = [
            "strategies.yaml",
            "settings.yaml",
            "gpu_config.yaml",
            "logging.yaml"
        ]

        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                if config_file.endswith('.yaml'):
                    self.config_cache[config_file.replace(
                        '.yaml', '')] = self._load_yaml_config(config_path)
                elif config_file.endswith('.json'):
                    self.config_cache[config_file.replace(
                        '.json', '')] = self._load_json_config(config_path)

    def _process_trigger_configurations(self) -> None:
        """Process and register trigger configurations."""
        triggers_config = self.config_cache.get("triggers", {})
        triggers = triggers_config.get("triggers", {})

        for trigger_category, category_triggers in triggers.items():
            for trigger_name, trigger_data in category_triggers.items():
                if isinstance(trigger_data, dict):
                    trigger_config = TriggerConfig(
                        trigger_id=f"{trigger_category}.{trigger_name}",
                        trigger_type=trigger_data.get(
                            "trigger_type",
                            "default"),
                        conditions=trigger_data.get(
                            "conditions",
                            {}),
                        actions=trigger_data.get(
                            "actions",
                            []),
                        priority=self._get_trigger_priority(trigger_name),
                        enabled=True,
                        metadata=trigger_data)
                    self.trigger_registry[trigger_config.trigger_id] = trigger_config

    def _get_trigger_priority(self, trigger_name: str) -> str:
        """Get trigger priority based on name."""
        priorities = self.config_cache.get(
            "triggers", {}).get(
            "trigger_priorities", {})

        for priority, trigger_list in priorities.items():
            if trigger_name in trigger_list:
                return priority
        return "medium"

    def get_config(self, config_key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config_cache.get(config_key, default)

    def get_trigger_config(self, trigger_id: str) -> Optional[TriggerConfig]:
        """Get trigger configuration by ID."""
        return self.trigger_registry.get(trigger_id)

    def get_all_triggers(self) -> Dict[str, TriggerConfig]:
        """Get all registered triggers."""
        return self.trigger_registry.copy()

    def validate_configuration(
            self, config_name: str) -> ConfigValidationResult:
        """Validate a specific configuration."""
        config = self.config_cache.get(config_name)
        if not config:
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Configuration '{config_name}' not found"]
            )

        result = ConfigValidationResult(is_valid=True)

        # Validate based on configuration type
        if config_name == "core":
            result = self._validate_core_config(config)
        elif config_name == "triggers":
            result = self._validate_triggers_config(config)

        return result

    def _validate_core_config(
            self, config: Dict[str, Any]) -> ConfigValidationResult:
        """Validate core configuration."""
        result = ConfigValidationResult(is_valid=True)

        # Check required sections
        required_sections = [
            "system",
            "mathematical_engines",
            "profit_tier_navigation"]
        for section in required_sections:
            if section not in config:
                result.errors.append(f"Missing required section: {section}")
                result.is_valid = False

        # Validate mathematical engines
        engines = config.get("mathematical_engines", {})
        for engine_name, engine_config in engines.items():
            if not isinstance(engine_config, dict):
                result.errors.append(
                    f"Invalid engine configuration for {engine_name}")
                result.is_valid = False
            elif "enabled" not in engine_config:
                result.warnings.append(
                    f"Engine {engine_name} missing 'enabled' flag")

        return result

    def _validate_triggers_config(
            self, config: Dict[str, Any]) -> ConfigValidationResult:
        """Validate triggers configuration."""
        result = ConfigValidationResult(is_valid=True)

        triggers = config.get("triggers", {})
        if not triggers:
            result.errors.append("No triggers defined")
            result.is_valid = False
            return result

        # Validate trigger structure
        for category, category_triggers in triggers.items():
            if not isinstance(category_triggers, dict):
                result.errors.append(f"Invalid trigger category: {category}")
                result.is_valid = False
                continue

            for trigger_name, trigger_data in category_triggers.items():
                if not isinstance(trigger_data, dict):
                    result.errors.append(
                        f"Invalid trigger data for {trigger_name}")
                    result.is_valid = False
                    continue

                # Check required fields
                required_fields = ["conditions", "actions"]
                for field in required_fields:
                    if field not in trigger_data:
                        result.warnings.append(
                            f"Trigger {trigger_name} missing '{field}' field")

        return result

    def integrate_with_mathematical_systems(self) -> None:
        """Integrate configurations with mathematical systems."""
        try:
            # Initialize mathematical systems if not already done
            if not self.math_system:
                self.math_system = UnifiedMathSystem()

            if not self.synthesis_engine:
                self.synthesis_engine = CoreTensorModulator()

            # Apply configuration to mathematical systems
            self._apply_engine_configurations()
            self._apply_unicode_pathway_configurations()
            self._apply_profit_tier_configurations()

            logger.info("✅ Configuration integrated with mathematical systems")

        except Exception as e:
            logger.error(f"❌ Error integrating configurations: {e}")

    def _apply_engine_configurations(self) -> None:
        """Apply engine configurations to mathematical systems."""
        core_config = self.config_cache.get("core", {})
        engines_config = core_config.get("mathematical_engines", {})

        for engine_name, engine_config in engines_config.items():
            if engine_config.get("enabled", False):
                # Store engine configuration
                self.engine_configs[engine_name] = EngineConfig(
                    engine_name=engine_name,
                    version=engine_config.get("version", "1.0.0"),
                    enabled=True,
                    parameters=engine_config.get("parameters", {}),
                    components=engine_config.get("components", {}),
                    metadata=engine_config
                )

                logger.info(f"✅ Engine configuration applied: {engine_name}")

    def _apply_unicode_pathway_configurations(self) -> None:
        """Apply Unicode pathway configurations."""
        core_config = self.config_cache.get("core", {})
        unicode_config = core_config.get(
            "system", {}).get(
            "unicode_pathways", {})

        if unicode_config.get("enabled", False):
            # Update Unicode handler with configuration
            if hasattr(self.unicore, 'update_configuration'):
                self.unicore.update_configuration(unicode_config)

            logger.info("✅ Unicode pathway configuration applied")

    def _apply_profit_tier_configurations(self) -> None:
        """Apply profit tier navigation configurations."""
        core_config = self.config_cache.get("core", {})
        profit_tier_config = core_config.get("profit_tier_navigation", {})

        if profit_tier_config.get("enabled", False):
            # Store profit tier configuration for use in trading systems
            self.config_cache["profit_tiers"] = profit_tier_config.get(
                "tiers", {})

            logger.info("✅ Profit tier configuration applied")

    def execute_trigger(self, trigger_id: str,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trigger with given context."""
        trigger_config = self.get_trigger_config(trigger_id)
        if not trigger_config:
            return {
                "success": False,
                "error": f"Trigger {trigger_id} not found"}

        if not trigger_config.enabled:
            return {
                "success": False,
                "error": f"Trigger {trigger_id} is disabled"}

        try:
            # Check conditions
            if not self._check_trigger_conditions(trigger_config, context):
                return {
                    "success": False,
                    "error": "Trigger conditions not met"}

            # Execute actions
            results = []
            for action in trigger_config.actions:
                action_result = self._execute_trigger_action(action, context)
                results.append(action_result)

            return {
                "success": True,
                "trigger_id": trigger_id,
                "actions_executed": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error executing trigger {trigger_id}: {e}")
            return {"success": False, "error": str(e)}

    def _check_trigger_conditions(
            self, trigger_config: TriggerConfig, context: Dict[str, Any]) -> bool:
        """Check if trigger conditions are met."""
        conditions = trigger_config.conditions

        for condition_key, condition_value in conditions.items():
            context_value = context.get(condition_key)

            if isinstance(condition_value, (int, float)):
                # Numeric comparison
                if context_value is None or context_value < condition_value:
                    return False
            elif isinstance(condition_value, str):
                # String comparison
                if context_value != condition_value:
                    return False
            elif isinstance(condition_value, bool):
                # Boolean comparison
                if context_value != condition_value:
                    return False

        return True

    def _execute_trigger_action(
            self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trigger action."""
        try:
            # Map actions to functions
            action_mapping = {
                "execute_buy_order": self._action_execute_buy_order,
                "execute_sell_order": self._action_execute_sell_order,
                "update_profit_tier": self._action_update_profit_tier,
                "log_to_backchannel": self._action_log_to_backchannel,
                "store_memory_pattern": self._action_store_memory_pattern,
                "adjust_position_size": self._action_adjust_position_size,
                "trigger_risk_management": self._action_trigger_risk_management,
                "execute_recursive_strategy": self._action_execute_recursive_strategy,
                "update_memory_patterns": self._action_update_memory_patterns,
                "optimize_parameters": self._action_optimize_parameters}

            action_func = action_mapping.get(action)
            if action_func:
                return action_func(context)
            else:
                logger.warning(f"Unknown action: {action}")
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return {"success": False, "error": str(e)}

    # Action implementations
    def _action_execute_buy_order(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute buy order action."""
        # Placeholder implementation
        return {"success": True, "action": "buy_order",
                "timestamp": datetime.now().isoformat()}

    def _action_execute_sell_order(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sell order action."""
        # Placeholder implementation
        return {"success": True, "action": "sell_order",
                "timestamp": datetime.now().isoformat()}

    def _action_update_profit_tier(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update profit tier action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "update_profit_tier",
            "timestamp": datetime.now().isoformat()}

    def _action_log_to_backchannel(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Log to backchannel action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "backchannel_log",
            "timestamp": datetime.now().isoformat()}

    def _action_store_memory_pattern(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store memory pattern action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "store_memory_pattern",
            "timestamp": datetime.now().isoformat()}

    def _action_adjust_position_size(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust position size action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "adjust_position_size",
            "timestamp": datetime.now().isoformat()}

    def _action_trigger_risk_management(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger risk management action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "trigger_risk_management",
            "timestamp": datetime.now().isoformat()}

    def _action_execute_recursive_strategy(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recursive strategy action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "execute_recursive_strategy",
            "timestamp": datetime.now().isoformat()}

    def _action_update_memory_patterns(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update memory patterns action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "update_memory_patterns",
            "timestamp": datetime.now().isoformat()}

    def _action_optimize_parameters(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters action."""
        # Placeholder implementation
        return {
            "success": True,
            "action": "optimize_parameters",
            "timestamp": datetime.now().isoformat()}

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration summary."""
        return {
            "configurations_loaded": len(self.config_cache),
            "triggers_registered": len(self.trigger_registry),
            "engines_configured": len(self.engine_configs),
            "config_cache_keys": list(self.config_cache.keys()),
            "trigger_categories": list(set(t.split('.')[0] for t in self.trigger_registry.keys())),
            "engine_names": list(self.engine_configs.keys()),
            "timestamp": datetime.now().isoformat()
        }

    def reload_configurations(self) -> Dict[str, Any]:
        """Reload all configurations."""
        try:
            self.config_cache.clear()
            self.trigger_registry.clear()
            self.engine_configs.clear()

            self._load_all_configurations()
            self.integrate_with_mathematical_systems()

            return {
                "success": True,
                "message": "Configurations reloaded successfully",
                "status": self.get_system_status()
            }

        except Exception as e:
            logger.error(f"Error reloading configurations: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global configuration system instance
_config_system: Optional[ConfigurationIntegrationSystem] = None


def get_config_system() -> ConfigurationIntegrationSystem:
    """Get the global configuration system instance."""
    global _config_system
    if _config_system is None:
        _config_system = ConfigurationIntegrationSystem()
    return _config_system


def initialize_config_system(
        config_dir: str = "config") -> ConfigurationIntegrationSystem:
    """Initialize the global configuration system."""
    global _config_system
    _config_system = ConfigurationIntegrationSystem(config_dir)
    return _config_system


def main() -> None:
    """Main function for testing the configuration system."""
    try:
        # Initialize configuration system
        config_system = initialize_config_system()

        # Validate configurations
        core_validation = config_system.validate_configuration("core")
        triggers_validation = config_system.validate_configuration("triggers")

        print("Configuration Validation Results:")
        print(f"Core config valid: {core_validation.is_valid}")
        if core_validation.errors:
            print(f"Core config errors: {core_validation.errors}")
        if core_validation.warnings:
            print(f"Core config warnings: {core_validation.warnings}")

        print(f"Triggers config valid: {triggers_validation.is_valid}")
        if triggers_validation.errors:
            print(f"Triggers config errors: {triggers_validation.errors}")
        if triggers_validation.warnings:
            print(f"Triggers config warnings: {triggers_validation.warnings}")

        # Integrate with mathematical systems
        config_system.integrate_with_mathematical_systems()

        # Get system status
        status = config_system.get_system_status()
        print(f"\nSystem Status: {status}")

        # Test trigger execution
        test_context = {
            "profit_threshold": 0.02,
            "volume_threshold": 1500,
            "confidence_minimum": 0.8
        }

        trigger_result = config_system.execute_trigger(
            "unicode_pathway_triggers.profit_trigger", test_context)
        print(f"\nTrigger execution result: {trigger_result}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
