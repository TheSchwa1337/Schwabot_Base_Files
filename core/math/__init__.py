import os
import yaml

# Expose tensor algebra subpackage
from .tensor_algebra import *

# Optionally expose math infrastructure here

# Registry for math modules
MATH_MODULES_REGISTRY = {}

# Auto-load mathematical_functions_registry.yaml if present
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'mathematical_functions_registry.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        try:
            math_functions_config = yaml.safe_load(f)
            MATH_MODULES_REGISTRY.update(math_functions_config or {})
        except Exception as e:
            MATH_MODULES_REGISTRY['error'] = str(e)

__all__ = ["MATH_MODULES_REGISTRY"] 