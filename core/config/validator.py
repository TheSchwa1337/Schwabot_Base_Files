from typing import Dict, Any
import jsonschema
from .__init__ import ConfigValidationError

CONFIG_SCHEMAS: Dict[str, Dict[str, Any]] = {
    'matrix_response_paths.yaml': {
        "type": "object",
        "properties": {
            "safe": {"type": "string"},
            "warn": {"type": "string"},
            "fail": {"type": "string"},
            "ZPE-risk": {"type": "string"},
            "thresholds": {
                "type": "object",
                "properties": {
                    "entropy": {"type": "number", "minimum": 0, "maximum": 1},
                    "drift": {"type": "number", "minimum": 0, "maximum": 1},
                    "anomaly": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["entropy", "drift", "anomaly"]
            }
        },
        "required": ["safe", "warn", "fail", "ZPE-risk", "thresholds"],
        "additionalProperties": False
    },
    'tesseract_config.yaml': {
        "type": "object",
        "properties": {
            "coherence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "pattern_decay_rate": {"type": "number", "minimum": 0},
            "memory_window": {"type": "integer", "minimum": 1}
        },
        "required": ["coherence_threshold", "pattern_decay_rate", "memory_window"],
        "additionalProperties": False
    },
    'gpu_offload_config.yaml': {
        "type": "object",
        "properties": {
            "thermal_threshold": {"type": "number"},
            "memory_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "fallback_enabled": {"type": "boolean"}
        },
        "required": ["thermal_threshold", "memory_threshold", "fallback_enabled"],
        "additionalProperties": False
    },
    'logging.yaml': {
        "type": "object",
        "properties": {
            "version": {"type": "integer", "const": 1},
            "disable_existing_loggers": {"type": "boolean"},
            "formatters": {"type": "object"},
            "handlers": {"type": "object"},
            "root": {"type": "object"}
        },
        "required": ["version", "disable_existing_loggers", "formatters", "handlers", "root"],
        "additionalProperties": True
    },
    'validation_config.yaml': {
        "type": "object",
        "properties": {
            "validation": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "max_retries": {"type": "integer", "minimum": 0},
                    "timeout": {"type": "number", "minimum": 0}
                },
                "required": ["enabled", "max_retries", "timeout"]
            }
        },
        "required": ["validation"],
        "additionalProperties": False
    }
}

def validate_config(config: Dict[str, Any], config_name: str) -> None:
    """
    Validate configuration against its predefined JSON schema.
    Args:
        config: The configuration dictionary to validate.
        config_name: The name of the configuration file (e.g., 'settings.yaml').
    Raises:
        ConfigValidationError: If the configuration does not conform to the schema.
    """
    schema = CONFIG_SCHEMAS.get(config_name)
    if not schema:
        return  # Skip validation if no schema is defined for this config
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as e:
        raise ConfigValidationError(f"Invalid config '{config_name}': {e.message} at {'.'.join(str(p) for p in e.path)}") from e
    except Exception as e:
        raise ConfigValidationError(f"An unexpected error occurred during validation of '{config_name}': {e}") from e 