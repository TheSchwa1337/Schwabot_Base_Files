"""
Configuration Module
==================

Central configuration management for Schwabot.
Provides unified config loading and validation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Raised when there is an error loading or validating config"""
    pass

def get_config_path(config_name: str) -> Path:
    """Get the absolute path to a config file"""
    base_path = Path(__file__).resolve().parent.parent
    return base_path / 'config' / config_name

def ensure_config_exists(config_path: Path) -> None:
    """Ensure config file exists, create default if it doesn't"""
    if not config_path.exists():
        logger.info(f"Creating default config at {config_path}")
        config_path.parent.mkdir(exist_ok=True)
        
        if config_path.suffix == '.yaml':
            create_default_yaml_config(config_path)
        elif config_path.suffix == '.json':
            create_default_json_config(config_path)
        else:
            raise ConfigError(f"Unsupported config format: {config_path.suffix}")

def create_default_yaml_config(config_path: Path) -> None:
    """Create default YAML config file"""
    if config_path.name == 'strategy_config.yaml':
        default_config = {
            'active_strategies': ['default'],
            'default_strategy': {
                'type': 'phase_aware',
                'parameters': {
                    'bit_depth': 64,
                    'trust_threshold': 0.7,
                    'phase_urgency_threshold': 0.5
                }
            },
            'baskets': {
                'BTC_USDC': {
                    'active': True,
                    'fallback_pairs': ['BTC_ETH', 'ETH_USDC']
                }
            }
        }
    else:
        default_config = {}
        
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)

def create_default_json_config(config_path: Path) -> None:
    """Create default JSON config file"""
    if config_path.name == 'phase_config.json':
        default_config = {
            'phase_regions': {
                'STABLE': {
                    'profit_trend_range': [0.001, float('inf')],
                    'stability_range': [0.7, 1.0],
                    'memory_coherence_range': [0.8, 1.0],
                    'paradox_pressure_range': [0.0, 2.0],
                    'entropy_rate_range': [0.0, 0.3],
                    'thermal_state_range': [0.0, 0.6],
                    'bit_depth_range': [16, 81],
                    'trust_score_range': [0.7, 1.0]
                }
            }
        }
    else:
        default_config = {}
        
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)

def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Load YAML config file"""
    config_path = get_config_path(config_name)
    ensure_config_exists(config_path)
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Error loading config {config_name}: {e}")

def load_json_config(config_name: str) -> Dict[str, Any]:
    """Load JSON config file"""
    config_path = get_config_path(config_name)
    ensure_config_exists(config_path)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ConfigError(f"Error loading config {config_name}: {e}")

def save_config(config: Dict[str, Any], config_name: str) -> None:
    """Save config to file"""
    config_path = get_config_path(config_name)
    
    try:
        if config_path.suffix == '.yaml':
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ConfigError(f"Unsupported config format: {config_path.suffix}")
    except Exception as e:
        raise ConfigError(f"Error saving config {config_name}: {e}")

def get_config_value(config_name: str, key_path: str, default: Any = None) -> Any:
    """Get a specific value from config using dot notation"""
    try:
        if config_name.endswith('.yaml'):
            config = load_yaml_config(config_name)
        elif config_name.endswith('.json'):
            config = load_json_config(config_name)
        else:
            raise ConfigError(f"Unsupported config format: {config_name}")
            
        # Navigate key path
        value = config
        for key in key_path.split('.'):
            value = value[key]
        return value
    except (KeyError, ConfigError):
        return default 