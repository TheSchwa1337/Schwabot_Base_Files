"""
Tests for configuration loading functionality
"""

import unittest
from pathlib import Path
import yaml
import os
import shutil
from ..config import load_yaml_config, ConfigError, ConfigNotFoundError

class TestConfigLoading(unittest.TestCase):
    """Test cases for configuration loading"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config_dir = Path(__file__).parent / 'test_configs'
        self.test_config_dir.mkdir(exist_ok=True)
        
        # Create test config file
        self.test_config = {
            'test_key': 'test_value',
            'nested': {
                'key': 'value'
            }
        }
        self.test_config_path = self.test_config_dir / 'test_config.yaml'
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_config_dir.exists():
            shutil.rmtree(self.test_config_dir)
    
    def test_load_existing_config(self):
        """Test loading an existing config file"""
        config = load_yaml_config('test_config.yaml', create_default=False)
        self.assertEqual(config['test_key'], 'test_value')
        self.assertEqual(config['nested']['key'], 'value')
    
    def test_load_nonexistent_config(self):
        """Test loading a nonexistent config file"""
        with self.assertRaises(ConfigNotFoundError):
            load_yaml_config('nonexistent.yaml', create_default=False)
    
    def test_load_with_default(self):
        """Test loading with default config generation"""
        config = load_yaml_config('matrix_response_paths.yaml')
        self.assertIn('safe', config)
        self.assertIn('warn', config)
        self.assertIn('fail', config)
        self.assertIn('ZPE-risk', config)
    
    def test_invalid_yaml(self):
        """Test loading invalid YAML"""
        # Create invalid YAML file
        invalid_path = self.test_config_dir / 'invalid.yaml'
        with open(invalid_path, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with self.assertRaises(ConfigError):
            load_yaml_config('invalid.yaml')

if __name__ == '__main__':
    unittest.main() 