"""
Configuration Management
======================

Centralized configuration loading and validation for the core system.
Handles YAML file loading with proper path resolution and default config generation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Generic
import logging
import os
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when a required config file is not found"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when config validation fails"""
    pass

@dataclass
class ConfigSchema(Generic[T]):
    """Schema for validating configuration"""
    required_fields: Dict[str, type]
    default_values: Dict[str, Any]
    validator: Optional[callable] = None

class ConfigLoader:
    """Centralized configuration loader with repository-relative paths."""
    _instance = None
    _config_dir: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            if cls._config_dir is None:
                # Assumes this file is in core/config/
                cls.repo_root = Path(__file__).resolve().parent.parent
                cls._config_dir = cls.repo_root / 'config'
                cls._config_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    @property
    def config_dir(self) -> Path:
        return self.__class__._config_dir

    def load_yaml(self, filename: str, create_default: bool = True) -> Dict[str, Any]:
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data is None:
                    logger.warning(f"Config file {filename} is empty. Returning empty dict.")
                    return {}
                return config_data
        except FileNotFoundError:
            if create_default:
                logger.warning(f"Config '{filename}' not found, attempting to create default.")
                default_config = self._create_default_config(filename)
                if default_config:
                    self.save_yaml(filename, default_config)
                return default_config
            raise ConfigNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filename}: {e}")
            raise ConfigError(f"Malformed YAML file: {config_path}") from e

    def save_yaml(self, filename: str, content: Dict[str, Any]):
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(content, f, indent=2)
            logger.info(f"Successfully saved default config for '{filename}' at {config_path}")
        except IOError as e:
            logger.error(f"Failed to save config file '{filename}': {e}")
            raise ConfigError(f"Failed to save config file: {config_path}") from e

    def _create_default_config(self, filename: str) -> Dict[str, Any]:
        try:
            from .defaults import DEFAULT_CONFIGS
            default_content = DEFAULT_CONFIGS.get(filename, {})
            if not default_content:
                logger.warning(f"No predefined default content for '{filename}'. Returning empty dict.")
            return default_content
        except ImportError:
            logger.warning(f"defaults.py not found. Returning empty dict for '{filename}'.")
            return {}

def get_config_dir() -> Path:
    """Get the absolute path to the config directory"""
    return Path(__file__).resolve().parent

def validate_config(config: Dict[str, Any], schema: ConfigSchema) -> None:
    """
    Validate configuration against schema
    
    Args:
        config: Configuration to validate
        schema: Schema to validate against
        
    Raises:
        ConfigValidationError: If validation fails
    """
    # Check required fields
    for field, field_type in schema.required_fields.items():
        if field not in config:
            raise ConfigValidationError(f"Missing required field: {field}")
        if not isinstance(config[field], field_type):
            raise ConfigValidationError(
                f"Invalid type for {field}: expected {field_type}, got {type(config[field])}"
            )
    
    # Run custom validator if provided
    if schema.validator:
        schema.validator(config)

def load_yaml_config(config_name: str, schema: Optional[ConfigSchema] = None, 
                    create_default: bool = True) -> Dict[str, Any]:
    """
    Load a YAML configuration file with proper path resolution and validation
    
    Args:
        config_name: Name of the config file (e.g. 'matrix_response_paths.yaml')
        schema: Optional schema for validation
        create_default: Whether to create default config if file doesn't exist
        
    Returns:
        Dict containing the configuration
        
    Raises:
        ConfigNotFoundError: If config file not found and create_default is False
        ConfigValidationError: If config validation fails
        ConfigError: For other configuration errors
    """
    config_path = get_config_dir() / config_name
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate if schema provided
        if schema:
            validate_config(config, schema)
            
        return config
        
    except FileNotFoundError:
        if create_default:
            logger.info(f"Config file {config_name} not found, creating default")
            default_config = get_default_config(config_name)
            if default_config is None:
                raise ConfigNotFoundError(f"No default config available for {config_name}")
            
            # Validate default config if schema provided
            if schema:
                validate_config(default_config, schema)
            
            # Create config directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
        else:
            raise ConfigNotFoundError(f"Config file {config_name} not found")
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing {config_name}: {str(e)}")

def get_default_config(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Get default configuration for a given config file
    
    Args:
        config_name: Name of the config file
        
    Returns:
        Dict containing default configuration or None if no default exists
    """
    defaults = {
        'matrix_response_paths.yaml': {
            'safe': 'hold',
            'warn': 'delay_entry',
            'fail': 'matrix_realign',
            'ZPE-risk': 'cooldown_abort'
        },
        'validation_config.yaml': {
            'validation': {
                'enabled': True,
                'max_retries': 3,
                'timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    }
    
    return defaults.get(config_name)

# Define schemas for known config files
MATRIX_RESPONSE_SCHEMA = ConfigSchema(
    required_fields={
        'safe': str,
        'warn': str,
        'fail': str,
        'ZPE-risk': str
    },
    default_values={
        'safe': 'hold',
        'warn': 'delay_entry',
        'fail': 'matrix_realign',
        'ZPE-risk': 'cooldown_abort'
    }
)

VALIDATION_CONFIG_SCHEMA = ConfigSchema(
    required_fields={
        'validation': dict,
        'logging': dict
    },
    default_values={
        'validation': {
            'enabled': True,
            'max_retries': 3,
            'timeout': 30
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
)

# Initialize a default logger for this module if not already set up by a main logging config
if not logging.getLogger(__name__).handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 