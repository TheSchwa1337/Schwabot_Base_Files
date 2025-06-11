"""
Configuration Loading Tests
========================

Tests for configuration loading and validation.
"""

import pytest
from pathlib import Path
import yaml
import json
import shutil

from core.config.manager import ConfigManager

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yield config_dir
    shutil.rmtree(config_dir)

def test_config_manager_initialization(temp_config_dir):
    """Test config manager initialization"""
    manager = ConfigManager(temp_config_dir)
    assert manager.root == temp_config_dir
    assert isinstance(manager.configs, dict)
    assert isinstance(manager.config_hashes, dict)

def test_default_config_creation(temp_config_dir):
    """Test creation of default configs"""
    manager = ConfigManager(temp_config_dir)
    
    # Check if default configs were created
    assert (temp_config_dir / "schema.yaml").exists()
    assert (temp_config_dir / "matrix_response_paths.yaml").exists()
    assert (temp_config_dir / "line_render_config.yaml").exists()
    assert (temp_config_dir / "phase_config.json").exists()
    assert (temp_config_dir / "strategy_config.yaml").exists()
    assert (temp_config_dir / "tesseract_config.yaml").exists()
    assert (temp_config_dir / "meta_config.yaml").exists()

def test_config_loading(temp_config_dir):
    """Test loading of configurations"""
    manager = ConfigManager(temp_config_dir)
    
    # Test loading schema config
    schema = manager.get_config("schema.yaml")
    assert isinstance(schema, dict)
    assert "phases" in schema
    
    # Test loading matrix config
    matrix = manager.get_config("matrix_response_paths.yaml")
    assert isinstance(matrix, dict)
    assert "matrix_response_paths" in matrix
    
    # Test loading phase config
    phase = manager.get_config("phase_config.json")
    assert isinstance(phase, dict)
    assert "phase_regions" in phase

def test_config_hash_tracking(temp_config_dir):
    """Test configuration hash tracking"""
    manager = ConfigManager(temp_config_dir)
    
    # Get initial hashes
    initial_hashes = manager.config_hashes.copy()
    
    # Modify a config
    schema_path = temp_config_dir / "schema.yaml"
    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)
    
    schema["phases"]["STABLE"]["profit_trend_range"][0] = 0.002
    
    with open(schema_path, 'w') as f:
        yaml.safe_dump(schema, f)
    
    # Reload config
    manager.reload_config("schema.yaml")
    
    # Check if hash changed
    assert manager.config_hashes["schema.yaml"] != initial_hashes["schema.yaml"]

def test_config_state_tracking(temp_config_dir):
    """Test configuration state tracking"""
    manager = ConfigManager(temp_config_dir)
    
    # Get config state
    state = manager.get_config_state()
    
    assert isinstance(state, dict)
    assert "configs" in state
    assert "timestamp" in state
    
    # Check config entries
    for name, info in state["configs"].items():
        assert "hash" in info
        assert "last_modified" in info

def test_invalid_config_handling(temp_config_dir):
    """Test handling of invalid configurations"""
    # Create invalid YAML
    invalid_path = temp_config_dir / "invalid.yaml"
    with open(invalid_path, 'w') as f:
        f.write("invalid: yaml: content: [")
    
    # Test loading invalid config
    with pytest.raises(Exception):
        manager = ConfigManager(temp_config_dir)

def test_config_reloading(temp_config_dir):
    """Test configuration reloading"""
    manager = ConfigManager(temp_config_dir)
    
    # Get initial config
    initial_config = manager.get_config("schema.yaml")
    
    # Modify config
    schema_path = temp_config_dir / "schema.yaml"
    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)
    
    schema["phases"]["STABLE"]["profit_trend_range"][0] = 0.002
    
    with open(schema_path, 'w') as f:
        yaml.safe_dump(schema, f)
    
    # Reload config
    manager.reload_config("schema.yaml")
    
    # Check if config was updated
    updated_config = manager.get_config("schema.yaml")
    assert updated_config != initial_config
    assert updated_config["phases"]["STABLE"]["profit_trend_range"][0] == 0.002 