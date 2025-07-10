#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loader for Schwabot
=================================

Provides universal access to configuration keys across all YAML files.
This module centralizes configuration management and provides
type-safe access to configuration values.

Key Features:
• Loads multiple YAML configuration files
• Provides dot-notation access to nested config values
• Type-safe configuration access
• Automatic configuration validation
• Hot-reload capability
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml

logger = logging.getLogger(__name__)

# Type variable for generic configuration types
T = TypeVar('T')

# Configuration file paths
CONFIG_PATHS = {
    "math_core": Path("config/math_core.yaml"),
    "pipeline": Path("config/pipeline.yaml"),
    "pipeline_config": Path("config/pipeline_config.yaml"),
    "schwabot": Path("config/schwabot_config.yaml"),
    "mathematical_framework": Path("config/mathematical_framework_config.py"),
}

# Global configuration cache
_config_cache: Dict[str, Dict[str, Any]] = {}
_config_loaded = False

class ConfigLoader:
    """
    Configuration loader for Schwabot.
    
    Provides centralized configuration management with
    type-safe access and validation capabilities.
    """
    
    def __init__(self, config_paths: Optional[Dict[str, Path]] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_paths: Optional custom configuration paths
        """
        self.logger = logging.getLogger(__name__)
        self.config_paths = config_paths or CONFIG_PATHS.copy()
        self.config_cache = {}
        self.loaded_files = set()
        
        # Load all configurations
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        try:
            self.logger.info("Loading configuration files...")
            
            for config_name, config_path in self.config_paths.items():
                if config_path.exists():
                    self._load_config(config_name, config_path)
                else:
                    self.logger.warning(f"Configuration file not found: {config_path}")
            
            self.logger.info(f"✅ Loaded {len(self.loaded_files)} configuration files")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading configurations: {e}")
    
    def _load_config(self, config_name: str, config_path: Path) -> bool:
        """
        Load a single configuration file.
        
        Args:
            config_name: Name of the configuration
            config_path: Path to the configuration file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.yaml':
                    config_data = yaml.safe_load(f)
                else:
                    # For non-YAML files, try to load as text and parse
                    content = f.read()
                    config_data = self._parse_config_content(content)
                
                self.config_cache[config_name] = config_data
                self.loaded_files.add(config_name)
                
                self.logger.debug(f"✅ Loaded configuration: {config_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error loading {config_name}: {e}")
            return False
    
    def _parse_config_content(self, content: str) -> Dict[str, Any]:
        """
        Parse configuration content for non-YAML files.
        
        Args:
            content: Configuration content
            
        Returns:
            Parsed configuration dictionary
        """
        # This is a simple parser for Python config files
        # In a real implementation, you might want to use ast.literal_eval
        # or a more sophisticated parser
        
        config_data = {}
        try:
            # Try to extract dictionary-like structures
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                if ':' in line and not line.startswith(' '):
                    # Section header
                    key, value = line.split(':', 1)
                    current_section = key.strip()
                    config_data[current_section] = {}
                elif current_section and '=' in line:
                    # Key-value pair
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    # Try to convert value to appropriate type
                    try:
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                    
                    config_data[current_section][key] = value
            
        except Exception as e:
            self.logger.warning(f"Could not parse config content: {e}")
        
        return config_data
    
    def get_config(self, section: str, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            section: Configuration section name
            key_path: Dot-separated key path (e.g., "math_core.drift_range.min")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            if section not in self.config_cache:
                self.logger.warning(f"Configuration section not found: {section}")
                return default
            
            config_data = self.config_cache[section]
            keys = key_path.split('.')
            
            # Navigate through nested dictionary
            current = config_data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    self.logger.debug(f"Configuration key not found: {section}.{key_path}")
                    return default
            
            return current
            
        except Exception as e:
            self.logger.error(f"Error accessing config {section}.{key_path}: {e}")
            return default
    
    def get_config_typed(self, section: str, key_path: str, value_type: Type[T], default: T) -> T:
        """
        Get configuration value with type conversion.
        
        Args:
            section: Configuration section name
            key_path: Dot-separated key path
            value_type: Expected type
            default: Default value
            
        Returns:
            Typed configuration value
        """
        value = self.get_config(section, key_path, default)
        
        try:
            if value is not None:
                return value_type(value)
            return default
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not convert {section}.{key_path} to {value_type.__name__}: {e}")
            return default
    
    def get_math_config(self, key_path: str, default: Any = None) -> Any:
        """
        Get math core configuration value.
        
        Args:
            key_path: Dot-separated key path
            default: Default value
            
        Returns:
            Math configuration value
        """
        return self.get_config("math_core", key_path, default)
    
    def get_pipeline_config(self, key_path: str, default: Any = None) -> Any:
        """
        Get pipeline configuration value.
        
        Args:
            key_path: Dot-separated key path
            default: Default value
            
        Returns:
            Pipeline configuration value
        """
        return self.get_config("pipeline", key_path, default)
    
    def get_schwabot_config(self, key_path: str, default: Any = None) -> Any:
        """
        Get Schwabot configuration value.
        
        Args:
            key_path: Dot-separated key path
            default: Default value
            
        Returns:
            Schwabot configuration value
        """
        return self.get_config("schwabot", key_path, default)
    
    def reload_config(self, section: Optional[str] = None) -> bool:
        """
        Reload configuration files.
        
        Args:
            section: Specific section to reload (None for all)
            
        Returns:
            True if reloaded successfully
        """
        try:
            if section is None:
                # Reload all configurations
                self.config_cache.clear()
                self.loaded_files.clear()
                self._load_all_configs()
                self.logger.info("✅ All configurations reloaded")
            else:
                # Reload specific section
                if section in self.config_paths:
                    config_path = self.config_paths[section]
                    if self._load_config(section, config_path):
                        self.logger.info(f"✅ Configuration {section} reloaded")
                        return True
                    else:
                        self.logger.error(f"❌ Failed to reload configuration {section}")
                        return False
                else:
                    self.logger.error(f"❌ Configuration section not found: {section}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error reloading configuration: {e}")
            return False
    
    def validate_config(self, section: str) -> Dict[str, Any]:
        """
        Validate configuration section.
        
        Args:
            section: Configuration section to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_keys': []
        }
        
        try:
            if section not in self.config_cache:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Configuration section not found: {section}")
                return validation_results
            
            config_data = self.config_cache[section]
            
            # Add validation logic here based on section
            if section == "math_core":
                validation_results.update(self._validate_math_config(config_data))
            elif section == "pipeline":
                validation_results.update(self._validate_pipeline_config(config_data))
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results
    
    def _validate_math_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate math core configuration."""
        results = {'errors': [], 'warnings': [], 'missing_keys': []}
        
        # Check required keys
        required_keys = ['symbolic_aliases', 'drift_range', 'entropy']
        for key in required_keys:
            if key not in config_data:
                results['missing_keys'].append(key)
        
        # Validate drift range
        if 'drift_range' in config_data:
            drift_range = config_data['drift_range']
            if 'min' in drift_range and 'max' in drift_range:
                if drift_range['min'] >= drift_range['max']:
                    results['errors'].append("drift_range.min must be less than drift_range.max")
        
        return results
    
    def _validate_pipeline_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        results = {'errors': [], 'warnings': [], 'missing_keys': []}
        
        # Check required keys
        required_keys = ['hardware', 'features', 'performance']
        for key in required_keys:
            if key not in config_data:
                results['missing_keys'].append(key)
        
        return results
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded configurations.
        
        Returns:
            Dictionary of all configurations
        """
        return self.config_cache.copy()
    
    def get_loaded_sections(self) -> List[str]:
        """
        Get list of loaded configuration sections.
        
        Returns:
            List of loaded section names
        """
        return list(self.loaded_files)
    
    def export_config(self, section: str, file_path: Path) -> bool:
        """
        Export configuration section to file.
        
        Args:
            section: Configuration section to export
            file_path: Output file path
            
        Returns:
            True if exported successfully
        """
        try:
            if section not in self.config_cache:
                self.logger.error(f"Configuration section not found: {section}")
                return False
            
            config_data = self.config_cache[section]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"✅ Configuration {section} exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting configuration {section}: {e}")
            return False

# Global configuration loader instance
config_loader = ConfigLoader()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_config(section: str, key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using global loader.
    
    Args:
        section: Configuration section name
        key_path: Dot-separated key path
        default: Default value
        
    Returns:
        Configuration value
    """
    return config_loader.get_config(section, key_path, default)

def get_math_param(key_path: str, default: Any = None) -> Any:
    """
    Get math core parameter.
    
    Args:
        key_path: Dot-separated key path
        default: Default value
        
    Returns:
        Math parameter value
    """
    return config_loader.get_math_config(key_path, default)

def get_pipeline_param(key_path: str, default: Any = None) -> Any:
    """
    Get pipeline parameter.
    
    Args:
        key_path: Dot-separated key path
        default: Default value
        
    Returns:
        Pipeline parameter value
    """
    return config_loader.get_pipeline_config(key_path, default)

def reload_all_configs() -> bool:
    """
    Reload all configurations.
    
    Returns:
        True if reloaded successfully
    """
    return config_loader.reload_config()

@lru_cache(maxsize=128)
def get_cached_config(section: str, key_path: str, default: Any = None) -> Any:
    """
    Get configuration value with caching.
    
    Args:
        section: Configuration section name
        key_path: Dot-separated key path
        default: Default value
        
    Returns:
        Cached configuration value
    """
    return config_loader.get_config(section, key_path, default)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Common configuration paths
MATH_CORE_CONFIG = "math_core"
PIPELINE_CONFIG = "pipeline"
SCHWABOT_CONFIG = "schwabot"

# Common configuration keys
DRIFT_RANGE_KEY = "drift_range"
ENTROPY_KEY = "entropy"
HARDWARE_KEY = "hardware"
FEATURES_KEY = "features"
PERFORMANCE_KEY = "performance"

# ============================================================================
# EXPORTED FUNCTIONS AND CLASSES
# ============================================================================

__all__ = [
    'ConfigLoader',
    'config_loader',
    'get_config',
    'get_math_param',
    'get_pipeline_param',
    'reload_all_configs',
    'get_cached_config',
    'MATH_CORE_CONFIG',
    'PIPELINE_CONFIG',
    'SCHWABOT_CONFIG',
    'DRIFT_RANGE_KEY',
    'ENTROPY_KEY',
    'HARDWARE_KEY',
    'FEATURES_KEY',
    'PERFORMANCE_KEY'
] 