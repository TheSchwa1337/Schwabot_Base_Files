"""
Configuration Loading Tests
==========================

Test suite for validating YAML configuration loading, path resolution,
and integration with core modules.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config_utils import (
    load_yaml_config,
    save_yaml_config,
    standardize_config_path,
    get_profile_params_from_yaml,
    merge_configs,
    validate_config_structure,
    create_default_tesseract_config,
    create_default_fractal_config,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError
)

from dashboard_integration import DashboardBridge, create_dashboard


class TestConfigUtils:
    """Test configuration utilities"""
    
    def test_standardize_config_path_relative(self):
        """Test path standardization for relative paths"""
        relative_path = "test_config.yaml"
        standardized = standardize_config_path(relative_path)
        
        assert standardized.is_absolute()
        assert standardized.name == "test_config.yaml"
    
    def test_standardize_config_path_absolute(self):
        """Test path standardization for absolute paths"""
        absolute_path = Path("/tmp/test_config.yaml")
        standardized = standardize_config_path(absolute_path)
        
        assert standardized == absolute_path
    
    def test_create_default_tesseract_config(self):
        """Test creation of default tesseract configuration"""
        config = create_default_tesseract_config()
        
        # Validate required sections
        assert 'processing' in config
        assert 'dimensions' in config
        assert 'monitoring' in config
        assert 'strategies' in config
        assert 'debug' in config
        assert 'alert_bus' in config
        
        # Validate specific values
        assert config['processing']['baseline_reset_flip_frequency'] == 100
        assert len(config['dimensions']['labels']) == 8
        assert 'inversion_burst_rebound' in config['strategies']
    
    def test_create_default_fractal_config(self):
        """Test creation of default fractal configuration"""
        config = create_default_fractal_config()
        
        # Validate required sections
        assert 'profile' in config
        assert 'processing' in config
        
        # Validate profile structure
        profile = config['profile']
        assert profile['name'] == 'default'
        assert profile['type'] == 'quantization'
        assert 'parameters' in profile
        
        # Validate parameters
        params = profile['parameters']
        assert params['decay_power'] == 1.5
        assert params['terms'] == 12
        assert params['dimension'] == 8
    
    def test_merge_configs(self):
        """Test configuration merging"""
        config1 = {
            'section1': {'key1': 'value1', 'key2': 'value2'},
            'section2': {'key3': 'value3'}
        }
        
        config2 = {
            'section1': {'key2': 'new_value2', 'key4': 'value4'},
            'section3': {'key5': 'value5'}
        }
        
        merged = merge_configs(config1, config2)
        
        # Check merged values
        assert merged['section1']['key1'] == 'value1'  # From config1
        assert merged['section1']['key2'] == 'new_value2'  # Overridden by config2
        assert merged['section1']['key4'] == 'value4'  # From config2
        assert merged['section2']['key3'] == 'value3'  # From config1
        assert merged['section3']['key5'] == 'value5'  # From config2
    
    def test_validate_config_structure(self):
        """Test configuration structure validation"""
        config = {
            'section1': {
                'subsection1': {
                    'key1': 'value1'
                }
            },
            'section2': {
                'key2': 'value2'
            }
        }
        
        # Valid keys
        required_keys = ['section1.subsection1.key1', 'section2.key2']
        assert validate_config_structure(config, required_keys) == True
        
        # Invalid keys
        invalid_keys = ['section1.missing_key', 'missing_section.key']
        assert validate_config_structure(config, invalid_keys) == False


class TestConfigFileOperations:
    """Test file operations for configuration"""
    
    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML configuration"""
        test_config = {
            'test_section': {
                'test_key': 'test_value',
                'test_number': 42,
                'test_list': [1, 2, 3]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save configuration
            save_yaml_config(test_config, temp_path)
            assert temp_path.exists()
            
            # Load configuration
            loaded_config = load_yaml_config(temp_path, create_default=False)
            assert loaded_config == test_config
            
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_yaml_config_missing_file_create_default(self):
        """Test loading missing YAML file with default creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "tesseract_test.yaml"
            
            # Should create default tesseract config
            config = load_yaml_config(temp_path, create_default=True)
            
            assert temp_path.exists()
            assert 'processing' in config
            assert 'strategies' in config
    
    def test_load_yaml_config_missing_file_no_default(self):
        """Test loading missing YAML file without default creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "missing_config.yaml"
            
            with pytest.raises(ConfigNotFoundError):
                load_yaml_config(temp_path, create_default=False)
    
    def test_load_yaml_config_invalid_yaml(self):
        """Test loading invalid YAML file"""
        invalid_yaml = "invalid: yaml: content: [unclosed"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigError):
                load_yaml_config(temp_path, create_default=False)
        finally:
            temp_path.unlink()


class TestEnhancedTesseractProcessor:
    """Test Enhanced Tesseract Processor configuration integration"""
    
    @patch('core.enhanced_tesseract_processor.QuantumCellularRiskMonitor')
    @patch('core.enhanced_tesseract_processor.RiskIndexer')
    @patch('core.enhanced_tesseract_processor.ZygotShell')
    def test_processor_initialization_with_config(self, mock_zygot, mock_indexer, mock_monitor):
        """Test processor initialization with configuration"""
        # Create a temporary config file
        test_config = create_default_tesseract_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Import here to avoid circular imports during test discovery
            from core.enhanced_tesseract_processor import EnhancedTesseractProcessor
            
            # Initialize processor with test config
            processor = EnhancedTesseractProcessor(config_path=temp_config_path)
            
            # Verify configuration was loaded
            assert processor.config == test_config
            assert processor.max_pattern_history == 1000
            assert processor.max_shell_history == 500
            assert processor.profit_blend_alpha == 0.7
            
            # Verify strategy triggers were loaded
            assert 'inversion_burst_rebound' in processor.strategy_triggers
            assert processor.strategy_triggers['inversion_burst_rebound']['trigger_prefix'] == 'e1a7'
            
        finally:
            Path(temp_config_path).unlink()
    
    @patch('core.enhanced_tesseract_processor.QuantumCellularRiskMonitor')
    @patch('core.enhanced_tesseract_processor.RiskIndexer')
    @patch('core.enhanced_tesseract_processor.ZygotShell')
    def test_processor_missing_config_creates_default(self, mock_zygot, mock_indexer, mock_monitor):
        """Test processor creates default config when missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_config_path = Path(temp_dir) / "missing_tesseract.yaml"
            
            # Import here to avoid circular imports during test discovery
            from core.enhanced_tesseract_processor import EnhancedTesseractProcessor
            
            # Initialize processor with missing config
            processor = EnhancedTesseractProcessor(config_path=str(missing_config_path))
            
            # Verify default config was created
            assert missing_config_path.exists()
            assert 'processing' in processor.config
            assert 'strategies' in processor.config


class TestDashboardIntegration:
    """Test dashboard integration functionality"""
    
    def test_dashboard_bridge_initialization(self):
        """Test dashboard bridge initialization"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = create_default_fractal_config()
            yaml.dump(test_config, f)
            temp_config_path = Path(f.name)
        
        try:
            dashboard = DashboardBridge(config_path=temp_config_path)
            
            # Verify profile was loaded
            assert dashboard.profile is not None
            assert 'name' in dashboard.profile
            
        finally:
            temp_config_path.unlink()
    
    def test_dashboard_status_update(self):
        """Test dashboard status update functionality"""
        dashboard = DashboardBridge()
        
        test_status = {
            'tick_counter': 100,
            'active_strategy': 'test_strategy',
            'vault_locked': True
        }
        
        dashboard.update_status(test_status)
        
        # Verify status was updated
        assert dashboard.status_data['tick_counter'] == 100
        assert dashboard.status_data['active_strategy'] == 'test_strategy'
        assert dashboard.status_data['vault_locked'] == True
        assert 'last_update' in dashboard.status_data
    
    def test_dashboard_export_json(self):
        """Test dashboard JSON export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = DashboardBridge()
            
            test_status = {'test_key': 'test_value'}
            dashboard.update_status(test_status)
            
            export_path = Path(temp_dir) / "test_export.json"
            result_path = dashboard.export_dashboard_json(export_path)
            
            assert result_path.exists()
            
            # Verify exported content
            import json
            with open(result_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'profile' in exported_data
            assert 'status' in exported_data
            assert 'timestamp' in exported_data
            assert exported_data['status']['test_key'] == 'test_value'


class TestIntegrationScenarios:
    """Test integration scenarios between components"""
    
    def test_full_config_loading_chain(self):
        """Test complete configuration loading chain"""
        # Create test configuration
        test_config = create_default_tesseract_config()
        test_config['custom_section'] = {'custom_key': 'custom_value'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = Path(f.name)
        
        try:
            # Test config_utils loading
            loaded_config = load_yaml_config(temp_config_path)
            assert loaded_config['custom_section']['custom_key'] == 'custom_value'
            
            # Test dashboard integration
            dashboard = DashboardBridge(config_path=temp_config_path)
            profile = dashboard.get_profile()
            
            # Profile should contain the loaded configuration
            assert profile is not None
            
        finally:
            temp_config_path.unlink()
    
    def test_config_validation_with_missing_keys(self):
        """Test configuration validation with missing required keys"""
        incomplete_config = {
            'processing': {
                'baseline_reset_flip_frequency': 100
                # Missing other required keys
            }
        }
        
        required_keys = [
            'processing.baseline_reset_flip_frequency',
            'processing.max_pattern_history',  # This will be missing
            'dimensions.labels'  # This section will be missing
        ]
        
        assert validate_config_structure(incomplete_config, required_keys) == False
    
    def test_config_path_resolution_from_different_locations(self):
        """Test config path resolution from different working directories"""
        # Create a test config in a known location
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()
            
            test_config_path = config_dir / "test_config.yaml"
            test_config = {'test': 'value'}
            
            with open(test_config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Test absolute path resolution
            standardized = standardize_config_path(test_config_path)
            assert standardized == test_config_path
            
            # Test that the file can be loaded
            loaded = load_yaml_config(test_config_path, create_default=False)
            assert loaded['test'] == 'value'


# Pytest fixtures
@pytest.fixture
def temp_config_file():
    """Fixture providing a temporary configuration file"""
    test_config = create_default_tesseract_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_config_dir():
    """Fixture providing a temporary configuration directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        yield config_dir


# Integration tests using fixtures
def test_processor_with_temp_config(temp_config_file):
    """Test processor initialization with temporary config file"""
    with patch('core.enhanced_tesseract_processor.QuantumCellularRiskMonitor'), \
         patch('core.enhanced_tesseract_processor.RiskIndexer'), \
         patch('core.enhanced_tesseract_processor.ZygotShell'):
        
        from core.enhanced_tesseract_processor import EnhancedTesseractProcessor
        
        processor = EnhancedTesseractProcessor(config_path=str(temp_config_file))
        
        # Verify processor was initialized correctly
        assert processor.config is not None
        assert processor.test_mode == False  # Default value
        assert processor.verbose_logging == False  # Default value


def test_dashboard_with_temp_config(temp_config_file):
    """Test dashboard initialization with temporary config file"""
    dashboard = DashboardBridge(config_path=temp_config_file)
    
    # Verify dashboard was initialized correctly
    assert dashboard.profile is not None
    assert dashboard.config_path == temp_config_file


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"]) 