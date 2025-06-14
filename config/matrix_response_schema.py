"""
Matrix Response Schema Definition
================================

Defines JSON schemas and default values for matrix response configuration files.
Used for validation and fallback configuration generation.
"""

from typing import Dict, Any

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

# JSON Schema for matrix_response_paths.yaml
MATRIX_RESPONSE_PATHS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "default_paths": {
            "type": "object",
            "properties": {
                "hold": {"type": "string"},
                "active": {"type": "string"},
                "data_directory": {"type": "string"},
                "log_directory": {"type": "string"}
            },
            "required": ["hold", "active"],
            "additionalProperties": True
        },
        "render_settings": {
            "type": "object",
            "properties": {
                "resolution": {"type": "string"},
                "background_color": {"type": "string"},
                "line_thickness": {"type": "integer", "minimum": 1}
            },
            "additionalProperties": True
        },
        "matrix_response_paths": {
            "type": "object",
            "properties": {
                "data_directory": {"type": "string"},
                "log_directory": {"type": "string"},
                "backup_directory": {"type": "string"}
            },
            "additionalProperties": True
        }
    },
    "additionalProperties": True
}

# JSON Schema for line_render_engine_config.yaml
LINE_RENDER_ENGINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "render_settings": {
            "type": "object",
            "properties": {
                "resolution": {"type": "string"},
                "background_color": {"type": "string"},
                "line_thickness": {"type": "integer", "minimum": 1, "maximum": 10},
                "line_decay_half_life_seconds": {"type": "integer", "minimum": 60}
            },
            "required": ["resolution", "background_color", "line_thickness"],
            "additionalProperties": True
        },
        "performance_settings": {
            "type": "object",
            "properties": {
                "max_workers": {"type": "integer", "minimum": 1, "maximum": 16},
                "memory_check_interval": {"type": "integer", "minimum": 30}
            },
            "additionalProperties": True
        }
    },
    "required": ["render_settings"],
    "additionalProperties": True
}

class MatrixResponseSchema:
    """
    Schema container with validation and default values for matrix response configuration.
    """
    
    def __init__(self):
        self.schema = MATRIX_RESPONSE_PATHS_SCHEMA
        self.default_values = {
            "default_paths": {
                "hold": "/data/matrix/hold",
                "active": "/data/matrix/active",
                "data_directory": "data",
                "log_directory": "logs"
            },
            "render_settings": {
                "resolution": "1080p",
                "background_color": "#121212",
                "line_thickness": 2
            },
            "matrix_response_paths": {
                "data_directory": "data/matrix",
                "log_directory": "logs/matrix",
                "backup_directory": "backups/matrix"
            }
        }

class LineRenderEngineSchema:
    """
    Schema container with validation and default values for line render engine configuration.
    """
    
    def __init__(self):
        self.schema = LINE_RENDER_ENGINE_SCHEMA
        self.default_values = {
            "render_settings": {
                "resolution": "1080p",
                "background_color": "#000000",
                "line_thickness": 2,
                "line_decay_half_life_seconds": 3600
            },
            "performance_settings": {
                "max_workers": 4,
                "memory_check_interval": 60
            }
        }

# Instantiate schemas for easy import
MATRIX_RESPONSE_SCHEMA = MatrixResponseSchema()
LINE_RENDER_SCHEMA = LineRenderEngineSchema()

# Export commonly used schemas
__all__ = [
    'ConfigError',
    'MATRIX_RESPONSE_SCHEMA',
    'LINE_RENDER_SCHEMA',
    'MATRIX_RESPONSE_PATHS_SCHEMA',
    'LINE_RENDER_ENGINE_SCHEMA'
] 