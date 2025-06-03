"""
Configuration Package for Schwabot Trading System
==============================================

This package contains all configuration settings, paths, and environment variables
for the Schwabot trading system. It provides centralized configuration management
and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Export configuration components
__all__ = [
    'load_config'
] 