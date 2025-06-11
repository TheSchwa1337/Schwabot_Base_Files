"""
Configuration I/O Utilities
==========================

Provides standardized configuration loading and default generation utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file with error handling.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing configuration data
        
    Raises:
        ValueError: If config file not found or invalid format
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError("YAML format invalid — expected dict.")
            return config
    except FileNotFoundError:
        raise ValueError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

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

def ensure_config_exists(filename: str) -> Path:
    """
    Ensure a config file exists, creating default if necessary.
    
    Args:
        filename: Name of the config file
        
    Returns:
        Path object pointing to the config file
    """
    config_path = get_config_path(filename)
    
    if not config_path.exists():
        if filename == 'matrix_response_paths.yaml':
            create_default_config(config_path)
        elif filename == 'line_render_engine_config.yaml':
            create_line_render_config(config_path)
        else:
            logger.warning(f"No default generator for {filename}")
    
    return config_path 