"""
Tests for the configuration loader's working directory handling.
"""

import pytest
import importlib
import yaml
from pathlib import Path
from core.config import ConfigLoader, ConfigError

@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Create a temporary config directory with test configuration."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "matrix_response_paths.yaml"
    test_config = {
        "safe": "hold",
        "warn": "delay_entry",
        "fail": "matrix_realign",
        "ZPE-risk": "cooldown_abort",
    }
    cfg_file.write_text(yaml.safe_dump(test_config))

    # Patch the config directory
    monkeypatch.setattr(ConfigLoader, "_config_dir", cfg_dir, raising=False)
    ConfigLoader._instance = None  # Reset singleton
    yield cfg_dir  # Return the directory path, not the config dict

@pytest.fixture
def change_cwd(tmp_path, monkeypatch):
    """Change the current working directory for testing."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path

def test_components_load_from_centralized_loader(change_cwd, temp_config_dir):
    """Test that components load configuration from the centralized loader."""
    # Import components
    MatrixFaultResolver = importlib.import_module("mathlib.matrix_fault_resolver").MatrixFaultResolver
    LineRenderEngine = importlib.import_module("mathlib.line_render_engine").LineRenderEngine
    
    # Initialize components
    resolver = MatrixFaultResolver()
    engine = LineRenderEngine()
    
    # Verify configuration loading - check actual config structure
    assert resolver.config.get("retry_config", {}).get("base_delay") == 1000  # Default value from MatrixFaultResolver
    assert resolver.config.get("retry_config", {}).get("backoff_factor") == 2  # Default value
    
    # Verify LineRenderEngine has matrix_paths loaded
    assert hasattr(engine, 'matrix_paths')
    assert engine.matrix_paths.get("safe") == "hold"  # Default value from LineRenderEngine

def test_config_loader_singleton(change_cwd, temp_config_dir):
    """Test that ConfigLoader maintains singleton pattern."""
    loader1 = ConfigLoader()
    loader2 = ConfigLoader()
    assert loader1 is loader2

def test_config_loader_defaults(change_cwd, temp_config_dir):
    """Test that ConfigLoader provides default values when config is missing."""
    loader = ConfigLoader()
    config = loader.load_yaml("nonexistent.yaml", create_default=True)
    assert isinstance(config, dict)

def test_config_loader_error_handling(change_cwd, temp_config_dir):
    """Test ConfigLoader error handling."""
    loader = ConfigLoader()
    
    # Test loading nonexistent file without defaults
    with pytest.raises(ConfigError):
        loader.load_yaml("nonexistent.yaml", create_default=False)
    
    # Test loading invalid YAML
    invalid_file = temp_config_dir / "invalid.yaml"
    invalid_file.write_text("invalid: yaml: content: [")
    with pytest.raises(ConfigError):
        loader.load_yaml("invalid.yaml") 