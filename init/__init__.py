"""
Initialization Package for Schwabot Trading System
==============================================

This package provides bootstrapping utilities for the Schwabot system,
including environment setup, path configuration, and module initialization.
"""

import sys
from pathlib import Path
from typing import List

def setup_environment():
    """Configure the Python environment for Schwabot"""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def validate_environment() -> List[str]:
    """Validate the runtime environment and return any issues"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8 or higher is required")
    
    # Check required directories
    required_dirs = ['core', 'config', 'engine', 'mathlib']
    for dir_name in required_dirs:
        if not (Path(__file__).parent.parent / dir_name).exists():
            issues.append(f"Required directory '{dir_name}' is missing")
    
    return issues

# Export initialization components
__all__ = [
    'setup_environment',
    'validate_environment'
] 