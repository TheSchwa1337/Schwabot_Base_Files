"""
Tests for configuration loading and validation
"""

import unittest
from pathlib import Path
import tempfile
import yaml
import os
import sys

# Add parent directory to path so we can import the core module
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.config import (
    load_yaml_config, 
    ConfigError, 
    ConfigNotFoundError,
    ConfigValidationError,
    MATRIX_RESPONSE_SCHEMA,
    VALIDATION_CONFIG_SCHEMA
)
from core.matrix_fault_resolver import MatrixFaultResolver
from core.line_render_engine import LineRenderEngine

class TestConfigLoading(unittest.TestCase):
    """Test configuration loading and validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name) / 'config'
        self.config_dir.mkdir()
        
        # Create test config files
        self.matrix_config = {
            'safe': 'hold',
            'warn': 'delay_entry',
            'fail': 'matrix_realign',
            'ZPE-risk': 'cooldown_abort'
        }
        
        self.validation_config = {
            'validation': {
                'enabled': True,
                'max_retries': 3,
                'timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # Write test configs
        with open(self.config_dir / 'matrix_response_paths.yaml', 'w') as f:
            yaml.dump(self.matrix_config, f)
            
        with open(self.config_dir / 'validation_config.yaml', 'w') as f:
            yaml.dump(self.validation_config, f)
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_load_valid_config(self):
        """Test loading valid configuration"""
        config = load_yaml_config('matrix_response_paths.yaml', 
                                schema=MATRIX_RESPONSE_SCHEMA)
        self.assertEqual(config, self.matrix_config)
    
    def test_load_invalid_config(self):
        """Test loading invalid configuration"""
        # Write invalid config
        invalid_config = {'invalid': 'config'}
        with open(self.config_dir / 'matrix_response_paths.yaml', 'w') as f:
            yaml.dump(invalid_config, f)
        
        with self.assertRaises(ConfigValidationError):
            load_yaml_config('matrix_response_paths.yaml', 
                           schema=MATRIX_RESPONSE_SCHEMA)
    
    def test_load_missing_config(self):
        """Test loading missing configuration"""
        os.remove(self.config_dir / 'matrix_response_paths.yaml')
        
        # Should create default config
        config = load_yaml_config('matrix_response_paths.yaml',
                                schema=MATRIX_RESPONSE_SCHEMA)
        self.assertEqual(config, MATRIX_RESPONSE_SCHEMA.default_values)
    
    def test_matrix_fault_resolver_config(self):
        """Test MatrixFaultResolver config loading"""
        resolver = MatrixFaultResolver()
        self.assertEqual(resolver.fallback_strategies, self.matrix_config)
    
    def test_line_render_engine_config(self):
        """Test LineRenderEngine config loading"""
        engine = LineRenderEngine()
        self.assertEqual(engine.matrix_paths, self.matrix_config)

if __name__ == '__main__':
    unittest.main() 