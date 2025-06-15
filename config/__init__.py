"""
Centralized Configuration Management
===================================

Provides standardized YAML loading, validation, and default generation
for all Schwabot modules. Ensures consistent behavior and prevents
missing file issues regardless of execution directory.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class ConfigManager:
    """Centralized configuration manager for all Schwabot modules"""
    
    def __init__(self):
        # Get the config directory relative to this file
        self.config_dir = Path(__file__).resolve().parent
        self.project_root = self.config_dir.parent
        
        # Cache for loaded configurations
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
    def get_config_path(self, filename: str) -> Path:
        """
        Get standardized config path relative to the config directory.
        
        Args:
            filename: Name of the config file
            
        Returns:
            Path object pointing to the config file
        """
        return self.config_dir / filename
    
    def load_config(self, filename: str, schema: Optional[Dict] = None, 
                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Load YAML configuration with standardized path resolution and validation.
        
        Args:
            filename: Name of the configuration file
            schema: Optional JSON schema for validation
            use_cache: Whether to use cached configuration
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            ConfigError: If config cannot be loaded, parsed, or validated
        """
        # Check cache first
        if use_cache and filename in self._config_cache:
            logger.debug(f"Using cached config for {filename}")
            return self._config_cache[filename].copy()
        
        config_path = self.get_config_path(filename)
        
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # Validate against schema if provided
            if schema:
                try:
                    validate(instance=config, schema=schema)
                    logger.debug(f"Schema validation passed for {filename}")
                except ValidationError as e:
                    raise ConfigError(f"Schema validation failed for {filename}: {e}")
            
            # Cache the configuration
            if use_cache:
                self._config_cache[filename] = config.copy()
            
            logger.info(f"Successfully loaded config from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parsing error in {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading config from {config_path}: {e}")
    
    def save_config(self, filename: str, config_data: Dict[str, Any]) -> None:
        """
        Save configuration data to YAML file.
        
        Args:
            filename: Name of the configuration file
            config_data: Configuration data to save
        """
        config_path = self.get_config_path(filename)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, indent=2, default_flow_style=False)
            
            # Update cache
            self._config_cache[filename] = config_data.copy()
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise ConfigError(f"Failed to save config to {config_path}: {e}")
    
    def ensure_config_exists(self, filename: str, default_config: Dict[str, Any]) -> Path:
        """
        Ensure a configuration file exists, creating it with defaults if necessary.
        
        Args:
            filename: Name of the configuration file
            default_config: Default configuration to write if file doesn't exist
            
        Returns:
            Path to the configuration file
        """
        config_path = self.get_config_path(filename)
        
        if not config_path.exists():
            logger.info(f"Creating default config file: {config_path}")
            self.save_config(filename, default_config)
        
        return config_path
    
    def validate_config(self, config_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration data against a schema.
        
        Args:
            config_data: Configuration data to validate
            schema: JSON schema for validation
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigError: If validation fails
        """
        try:
            validate(instance=config_data, schema=schema)
            return True
        except ValidationError as e:
            raise ConfigError(f"Configuration validation failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        logger.debug("Configuration cache cleared")

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions for backward compatibility
def load_config(filename: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
    """Load configuration using the global config manager."""
    return config_manager.load_config(filename, schema)

def save_config(filename: str, config_data: Dict[str, Any]) -> None:
    """Save configuration using the global config manager."""
    config_manager.save_config(filename, config_data)

def ensure_config_exists(filename: str, default_config: Dict[str, Any]) -> Path:
    """Ensure config exists using the global config manager."""
    return config_manager.ensure_config_exists(filename, default_config)

def get_config_path(filename: str) -> Path:
    """Get config path using the global config manager."""
    return config_manager.get_config_path(filename)

# Export commonly used items
__all__ = [
    'ConfigError',
    'ConfigManager',
    'config_manager',
    'load_config',
    'save_config',
    'ensure_config_exists',
    'get_config_path'
] 