"""
Configuration Utilities
======================

Centralized configuration loading and management utilities for Schwabot.
Provides robust YAML loading with path standardization, error handling, and schema validation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, List
import logging
from dataclasses import asdict

# Optional imports
try:
    from .schemas.quantization import QuantizationSchema
    QUANTIZATION_SCHEMA_AVAILABLE = True
except ImportError:
    QuantizationSchema = None
    QUANTIZATION_SCHEMA_AVAILABLE = False

try:
    import pydantic
    PYDANTIC_AVAILABLE = True
except ImportError:
    pydantic = None
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when a configuration file is not found"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails"""
    pass

def get_config_root() -> Path:
    """Get the configuration root directory"""
    return Path(__file__).resolve().parent

def standardize_config_path(config_path: Union[str, Path]) -> Path:
    """
    Standardize configuration path relative to the repository root
    
    Args:
        config_path: Path to configuration file (relative or absolute)
        
    Returns:
        Standardized Path object
    """
    config_path = Path(config_path)
    
    if config_path.is_absolute():
        return config_path
    
    # Make relative to config directory
    config_root = get_config_root()
    return config_root / config_path

def load_yaml_config(config_path: Union[str, Path], 
                    create_default: bool = True,
                    schema: Optional[Type] = None) -> Dict[str, Any]:
    """
    Load YAML configuration with robust error handling and optional schema validation
    
    Args:
        config_path: Path to YAML configuration file
        create_default: Whether to create default config if missing
        schema: Optional Pydantic schema class for validation
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        ConfigNotFoundError: If config file not found and create_default is False
        ConfigValidationError: If configuration validation fails
        ConfigError: For other configuration-related errors
    """
    standardized_path = standardize_config_path(config_path)
    
    try:
        if not standardized_path.exists():
            if create_default:
                logger.info(f"Config file not found, creating default: {standardized_path}")
                create_default_config(standardized_path)
            else:
                raise ConfigNotFoundError(f"Configuration file not found: {standardized_path}")
        
        # Load YAML content
        with open(standardized_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            config_data = {}
            
        # Validate with schema if provided and pydantic is available
        if schema and PYDANTIC_AVAILABLE:
            try:
                validated_config = schema(**config_data)
                logger.debug(f"Configuration validated successfully with {schema.__name__}")
                return asdict(validated_config) if hasattr(validated_config, '__dict__') else config_data
            except Exception as e:
                raise ConfigValidationError(f"Schema validation failed for {standardized_path}: {e}")
        elif schema and not PYDANTIC_AVAILABLE:
            logger.warning("Schema validation requested but Pydantic not available")
        
        logger.info(f"Successfully loaded config: {standardized_path}")
        return config_data
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML file {standardized_path}: {e}")
    except FileNotFoundError:
        raise ConfigNotFoundError(f"Configuration file not found: {standardized_path}")
    except Exception as e:
        raise ConfigError(f"Unexpected error loading config {standardized_path}: {e}")

def save_yaml_config(config_data: Dict[str, Any], 
                    config_path: Union[str, Path],
                    create_dirs: bool = True) -> None:
    """
    Save configuration data to YAML file
    
    Args:
        config_data: Configuration dictionary to save
        config_path: Path where to save the configuration
        create_dirs: Whether to create parent directories if they don't exist
    """
    standardized_path = standardize_config_path(config_path)
    
    try:
        if create_dirs:
            standardized_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(standardized_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to: {standardized_path}")
        
    except Exception as e:
        raise ConfigError(f"Error saving configuration to {standardized_path}: {e}")

def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON configuration file
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Dictionary containing configuration data
    """
    standardized_path = standardize_config_path(config_path)
    
    try:
        with open(standardized_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        logger.info(f"Successfully loaded JSON config: {standardized_path}")
        return config_data
        
    except FileNotFoundError:
        raise ConfigNotFoundError(f"JSON configuration file not found: {standardized_path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Error parsing JSON file {standardized_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Unexpected error loading JSON config {standardized_path}: {e}")

def create_default_config(config_path: Path) -> None:
    """
    Create default configuration based on file name
    
    Args:
        config_path: Path where to create the default configuration
    """
    config_name = config_path.name.lower()
    
    # Default configurations for different file types
    if 'tesseract' in config_name:
        default_config = create_default_tesseract_config()
    elif 'fractal' in config_name:
        default_config = create_default_fractal_config()
    elif 'matrix' in config_name:
        default_config = create_default_matrix_config()
    elif 'risk' in config_name:
        default_config = create_default_risk_config()
    else:
        default_config = create_generic_default_config()
    
    save_yaml_config(default_config, config_path)
    logger.info(f"Created default configuration: {config_path}")

def create_default_tesseract_config() -> Dict[str, Any]:
    """Create default tesseract configuration"""
    return {
        'processing': {
            'baseline_reset_flip_frequency': 100,
            'max_pattern_history': 1000,
            'max_shell_history': 500,
            'profit_blend_alpha': 0.7
        },
        'dimensions': {
            'labels': ['price', 'volume', 'volatility', 'momentum', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        },
        'monitoring': {
            'alerts': {
                'var_threshold': 0.05,
                'var_indexed_threshold': 1.5,
                'coherence_threshold': 0.5,
                'coherence_indexed_threshold': 0.8
            }
        },
        'strategies': {
            'inversion_burst_rebound': {
                'trigger_prefix': 'e1a7'
            },
            'momentum_cascade': {
                'trigger_prefix': 'f2b8'
            },
            'volatility_breakout': {
                'trigger_prefix': 'a3c9'
            }
        },
        'debug': {
            'test_mode': False,
            'verbose_logging': False
        },
        'alert_bus': {
            'enabled': True,
            'channels': ['log', 'console'],
            'severity_levels': {
                'HIGH': 3,
                'MEDIUM': 2,
                'LOW': 1,
                'INFO': 0
            }
        }
    }

def create_default_fractal_config() -> Dict[str, Any]:
    """Create default fractal configuration"""
    return {
        'profile': {
            'name': 'default',
            'type': 'quantization',
            'parameters': {
                'decay_power': 1.5,
                'terms': 12,
                'dimension': 8,
                'epsilon_q': 0.003,
                'precision': 0.001
            }
        },
        'processing': {
            'fft_harmonics': 8,
            'volatility_window': 100,
            'alignment_threshold': 0.8
        }
    }

def create_default_matrix_config() -> Dict[str, Any]:
    """Create default matrix configuration"""
    return {
        'matrix_response_paths': {
            'strategy_a': 'strategies/alt_path_A.json',
            'fallback': 'strategies/fallback_map.json'
        },
        'data_directory': 'data/matrix_logs',
        'response_templates': {
            'safe': 'hold',
            'warn': 'delay_entry',
            'fail': 'matrix_realign',
            'ZPE-risk': 'cooldown_abort'
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

def create_default_risk_config() -> Dict[str, Any]:
    """Create default risk configuration"""
    return {
        'risk_limits': {
            'max_position_size': 0.1,
            'max_drawdown': 0.05,
            'var_threshold': 0.02
        },
        'monitoring': {
            'check_interval': 60,
            'alert_channels': ['log', 'email'],
            'metrics': [
                'total_exposure',
                'portfolio_volatility',
                'current_drawdown',
                'var_95'
            ]
        }
    }

def create_generic_default_config() -> Dict[str, Any]:
    """Create generic default configuration"""
    return {
        'meta': {
            'name': 'default_config',
            'version': '1.0.0',
            'created': 'auto-generated'
        },
        'settings': {
            'enabled': True,
            'debug_mode': False
        }
    }

def get_profile_params_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract profile parameters from YAML configuration
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing profile parameters
    """
    try:
        config = load_yaml_config(config_path)
        
        # Extract profile section if it exists
        if 'profile' in config:
            return config['profile']
        
        # Fallback to extracting parameters section
        if 'parameters' in config:
            return config['parameters']
            
        # Return entire config if no specific profile section
        return config
        
    except Exception as e:
        logger.error(f"Error extracting profile parameters from {config_path}: {e}")
        return {}

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries
    
    Args:
        *configs: Variable number of configuration dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    if not configs:
        return {}
    
    result = configs[0].copy()
    for config in configs[1:]:
        result = deep_merge(result, config)
    
    return result

def validate_config_structure(config: Dict[str, Any], 
                            required_keys: List[str]) -> bool:
    """
    Validate that configuration contains required keys
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys (supports dot notation for nested keys)
        
    Returns:
        True if all required keys are present, False otherwise
    """
    def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = key_path.split('.')
        value = data
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
            
        return value
    
    for key in required_keys:
        if get_nested_value(config, key) is None:
            logger.error(f"Required configuration key missing: {key}")
            return False
    
    return True

# Convenience functions for common configurations
def load_tesseract_config(config_path: str = "tesseract_enhanced.yaml") -> Dict[str, Any]:
    """Load tesseract configuration with defaults"""
    return load_yaml_config(config_path, create_default=True)

def load_fractal_config(config_path: str = "fractal_core.yaml") -> Dict[str, Any]:
    """Load fractal configuration with defaults"""
    schema = QuantizationSchema if QUANTIZATION_SCHEMA_AVAILABLE else None
    return load_yaml_config(config_path, create_default=True, schema=schema)

def load_matrix_config(config_path: str = "matrix_response_paths.yaml") -> Dict[str, Any]:
    """Load matrix configuration with defaults"""
    return load_yaml_config(config_path, create_default=True) 