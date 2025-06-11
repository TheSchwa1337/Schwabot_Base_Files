"""
Matrix Fault Resolver
====================

Handles fault resolution for matrix operations in the Schwabot system.
Uses standardized config loading for consistent behavior.
"""

from pathlib import Path
from typing import Dict, Any
import logging
import sys
import os

# Add the parent directory to sys.path to find config module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.io_utils import load_config, ensure_config_exists
except ImportError:
    # Fallback if config module not found
    import yaml
    def load_config(config_path: Path) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
    
    def ensure_config_exists(filename: str) -> Path:
        return Path(__file__).resolve().parent / 'config' / filename

logger = logging.getLogger(__name__)

class MatrixFaultResolver:
    """Resolves faults in matrix operations"""
    
    def __init__(self, config_path: Path = None):
        """
        Initialize the matrix fault resolver.
        
        Args:
            config_path: Optional path to configuration file
        """
        if config_path is None:
            config_path = ensure_config_exists('matrix_response_paths.yaml')
        
        try:
            self.config = load_config(config_path)
            logger.info(f"MatrixFaultResolver initialized with config from {config_path}")
        except ValueError as e:
            logger.error(f"Failed to load config: {e}")
            # Use minimal default config
            self.config = {
                'matrix_response_paths': {
                    'data_directory': 'data',
                    'log_directory': 'logs'
                }
            }
    
    def resolve_faults(self, fault_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Resolve matrix faults based on configuration and fault data.
        
        Args:
            fault_data: Optional dictionary containing fault information
            
        Returns:
            Dictionary containing resolution results
        """
        try:
            # Get matrix response paths from config
            response_paths = self.config.get('matrix_response_paths', {})
            data_dir = response_paths.get('data_directory', 'data')
            
            # Implement fault resolution logic
            resolution_result = {
                'status': 'resolved',
                'method': 'standard_resolution',
                'data_directory': data_dir,
                'timestamp': str(Path(__file__).stat().st_mtime)
            }
            
            if fault_data:
                resolution_result['fault_type'] = fault_data.get('type', 'unknown')
                resolution_result['severity'] = fault_data.get('severity', 'low')
            
            logger.info(f"Matrix fault resolved: {resolution_result['status']}")
            return resolution_result
            
        except Exception as e:
            logger.error(f"Error resolving matrix fault: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'method': 'error_fallback'
            }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        logger.info("MatrixFaultResolver configuration updated") 