"""
Configuration I/O Utilities
===========================

Centralized YAML configuration loading with schema validation and default generation.
Ensures consistent behavior across all Schwabot modules.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file with error handling.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        ValueError: If config cannot be loaded or parsed
    """
    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if config is None:
            config = {}
            
        logger.info(f"Successfully loaded config from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")

def load_yaml_config(config_path: Path, schema: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load YAML configuration with optional schema validation.
    
    Args:
        config_path: Path to the YAML configuration file
        schema: Optional JSON schema for validation
        
    Returns:
        The loaded configuration data as a dictionary
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ConfigError: If schema validation fails
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    try:
        with open(config_path, "r", encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            
        if schema:
            try:
                validate(instance=data, schema=schema)
                logger.debug(f"Schema validation passed for {config_path}")
            except ValidationError as e:
                raise ConfigError(f"YAML schema validation error for {config_path}: {e}")
                
        return data
        
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parsing error for {config_path}: {e}")

def ensure_config_exists(filename: str, defaults: Optional[Dict] = None) -> Path:
    """
    Ensure a configuration file exists, creating it with defaults if necessary.
    
    Args:
        filename: Name of the configuration file
        defaults: Default configuration to write if file doesn't exist
        
    Returns:
        Path to the configuration file
    """
    # Determine config path relative to this file
    config_dir = Path(__file__).resolve().parent
    config_path = config_dir / filename
    
    if not config_path.exists():
        if defaults:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(defaults, f, indent=2, default_flow_style=False)
            logger.info(f"Created default config file at {config_path}")
        else:
            logger.warning(f"Config file {config_path} does not exist and no defaults provided")
            
    return config_path

def save_config(config_path: Path, config_data: Dict[str, Any]) -> None:
    """
    Save configuration data to YAML file.
    
    Args:
        config_path: Path where to save the configuration
        config_data: Configuration data to save
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_data, f, indent=2, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        raise

def validate_config_schema(config_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
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

def create_default_config(config_path: Path = None) -> None:
    """
    Create default configuration files.
    
    Args:
        config_path: Optional specific path for config file
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / 'matrix_response_paths.yaml'
    
    # Default matrix response configuration
    default_matrix_responses = {
        'matrix_response_paths': {
            'response_template': 'default_response.txt',
            'data_directory': 'data',
            'log_directory': 'logs'
        },
        'render_settings': {
            'resolution': '1080p',
            'background_color': '#000000',
            'frame_rate': 30
        },
        'thresholds': {
            'thermal_state': 0.8,
            'entropy_rate': 0.9,
            'memory_coherence': 0.2,
            'trust_score': 0.3
        }
    }
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    with open(config_path, 'w') as file:
        yaml.safe_dump(default_matrix_responses, file, default_flow_style=False)
    
    logger.info(f"Generated default config: {config_path}")

def create_line_render_config(config_path: Path = None) -> None:
    """
    Create default line render engine configuration.
    
    Args:
        config_path: Optional specific path for config file
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / 'line_render_engine_config.yaml'
    
    default_render_config = {
        'render_settings': {
            'resolution': '1080p',
            'background_color': '#000000',
            'line_thickness': 2,
            'antialiasing': True,
            'frame_rate': 30
        },
        'output_settings': {
            'format': 'mp4',
            'quality': 'high',
            'compression': 'h264'
        }
    }
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    with open(config_path, 'w') as file:
        yaml.safe_dump(default_render_config, file, default_flow_style=False)
    
    logger.info(f"Generated line render config: {config_path}")

def get_config_path(filename: str) -> Path:
    """
    Get standardized config path relative to this module.
    
    Args:
        filename: Name of the config file
        
    Returns:
        Path object pointing to the config file
    """
    return Path(__file__).resolve().parent / filename 