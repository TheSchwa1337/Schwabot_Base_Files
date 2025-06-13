"""
Configuration Module
===================

Provides standardized configuration loading and management utilities
for the Schwabot trading system.
"""

from .io_utils import (
    load_config,
    create_default_config,
    create_line_render_config,
    get_config_path,
    ensure_config_exists
)

# Import new enhanced configuration utilities
from .config_utils import (
    load_yaml_config,
    save_yaml_config,
    load_json_config,
    standardize_config_path,
    get_profile_params_from_yaml,
    merge_configs,
    validate_config_structure,
    load_tesseract_config,
    load_fractal_config,
    load_matrix_config,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError
)

__all__ = [
    # Legacy io_utils functions
    'load_config',
    'create_default_config', 
    'create_line_render_config',
    'get_config_path',
    'ensure_config_exists',
    
    # Enhanced config_utils functions
    'load_yaml_config',
    'save_yaml_config',
    'load_json_config',
    'standardize_config_path',
    'get_profile_params_from_yaml',
    'merge_configs',
    'validate_config_structure',
    'load_tesseract_config',
    'load_fractal_config',
    'load_matrix_config',
    
    # Exceptions
    'ConfigError',
    'ConfigNotFoundError',
    'ConfigValidationError'
] 