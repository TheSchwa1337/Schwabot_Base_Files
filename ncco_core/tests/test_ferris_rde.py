"""
Test suite for Ferris Wheel RDE Integration
=========================================

Tests the integration between RDE engine and Ferris Wheel, including:
- Strategy mapping based on bit modes
- Performance tracking and adaptation
- Spin history and logging
- Mode weight updates
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from ncco_core.ferris_rde import FerrisRDE

@pytest.fixture
def temp_configs():
    """Create temporary config files for testing."""
    rde_config = {
        "axes": {
            "price_metrics": [
                "{asset}_price_delta",
                "{asset}_volatility"
            ],
            "market_state": [
                "{asset}_sentiment_bull",
                "{asset}_sentiment_bear"
            ]
        },
        "assets": ["BTC", "ETH"],
        "scales": {
            "lookback_hours": 1,
            "decay": 0.8,
            "bit_modes": [4, 8, 42]
        },
        "paths": {
            "history_dir": "~/test_spin_history",
            "fig_dpi": 100
        }
    }
    
    ferris_config = {
        "strategies": {
            "hold": {"weight": 1.0},
            "flip": {"weight": 1.0},
            "hedge": {"weight": 1.0},
            "spec": {"weight": 1.0},
            "stable_swap": {"weight": 1.0},
            "aggressive": {"weight": 1.0}
        },
        "settings": {
            "max_strategies": 3,
            "min_weight": 0.1
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_rde.yaml', delete=False) as rde_f:
        yaml.dump(rde_config, rde_f)
        rde_path = rde_f.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='_ferris.yaml', delete=False) as ferris_f:
        yaml.dump(ferris_config, ferris_f)
        ferris_path = ferris_f.name
        
    yield rde_path, ferris_path
    
    os.unlink(rde_path)
    os.unlink(ferris_path)

@pytest.fixture
def ferris_rde(temp_configs):
    """Create a FerrisRDE instance for testing."""
    rde_path, ferris_path = temp_configs
    return FerrisRDE(rde_path, ferris_path)

def test_initialization(ferris_rde):
    """Test FerrisRDE initialization."""
    assert ferris_rde.rde is not None
    assert ferris_rde.ferris is not None
    assert len(ferris_rde._mode_strategy_map) == 3  # 4, 8, 42 bit modes
    assert all(mode in ferris_rde._mode_weights for mode in [4, 8, 42])

def test_strategy_mapping(ferris_rde):
    """Test bit mode to strategy mapping."""
    # Test low precision mode (4-bit)
    strategies_4bit = ferris_rde._mode_strategy_map[4]
    assert "hold" in strategies_4bit
    assert "stable_swap" in strategies_4bit
    
    # Test medium precision mode (8-bit)
    strategies_8bit = ferris_rde._mode_strategy_map[8]
    assert "hedge" in strategies_8bit
    assert "flip" in strategies_8bit
    
    # Test high precision mode (42-bit)
    strategies_42bit = ferris_rde._mode_strategy_map[42]
    assert "aggressive" in strategies_42bit
    assert "spec" in strategies_42bit

def test_market_state_update(ferris_rde):
    """Test market state update and strategy selection."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    ferris_rde.update_market_state(signals)
    
    # Check that a spin was logged
    assert ferris_rde._last_spin is not None
    assert "tag" in ferris_rde._last_spin
    assert "strategies" in ferris_rde._last_spin
    assert "weights" in ferris_rde._last_spin

def test_strategy_performance_update(ferris_rde):
    """Test strategy performance tracking and mode weight updates."""
    # Update performance for some strategies
    ferris_rde.update_strategy_performance("hold", 0.95)
    ferris_rde.update_strategy_performance("flip", 0.82)
    
    # Check strategy performance
    perf = ferris_rde.get_strategy_performance()
    assert perf["hold"] == 0.95
    assert perf["flip"] == 0.82
    
    # Check mode weights were updated
    weights = ferris_rde.get_mode_performance()
    assert all(0 <= w <= 1 for w in weights.values())

def test_spin_history(ferris_rde):
    """Test spin history tracking."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    # Generate multiple spins
    for _ in range(5):
        ferris_rde.update_market_state(signals)
    
    # Check history
    history = ferris_rde.get_spin_history()
    assert len(history) == 5
    
    # Check limited history
    limited = ferris_rde.get_spin_history(limit=3)
    assert len(limited) == 3

def test_mode_weight_adaptation(ferris_rde):
    """Test bit mode weight adaptation based on strategy performance."""
    # Set initial performance
    ferris_rde.update_strategy_performance("hold", 0.95)  # 4-bit strategy
    ferris_rde.update_strategy_performance("flip", 0.82)  # 8-bit strategy
    ferris_rde.update_strategy_performance("spec", 0.75)  # 42-bit strategy
    
    # Get initial weights
    initial_weights = ferris_rde.get_mode_performance()
    
    # Update performance to favor 8-bit mode
    ferris_rde.update_strategy_performance("hedge", 0.98)  # 8-bit strategy
    ferris_rde.update_strategy_performance("flip", 0.96)   # 8-bit strategy
    
    # Get updated weights
    updated_weights = ferris_rde.get_mode_performance()
    
    # 8-bit mode weight should have increased
    assert updated_weights[8] > initial_weights[8]

def test_strategy_weight_calculation(ferris_rde):
    """Test strategy weight calculation based on performance."""
    # Set performance for strategies
    ferris_rde.update_strategy_performance("hold", 0.95)
    ferris_rde.update_strategy_performance("flip", 0.82)
    
    # Get weights for 4-bit strategies
    weights = ferris_rde._get_strategy_weights(["hold", "stable_swap"])
    
    # Check weight properties
    assert len(weights) == 2
    assert all(0 <= w <= 1 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6  # Should sum to 1

def test_reset_performance(ferris_rde):
    """Test performance metrics reset."""
    # Set some performance data
    ferris_rde.update_strategy_performance("hold", 0.95)
    ferris_rde.update_strategy_performance("flip", 0.82)
    
    # Reset performance
    ferris_rde.reset_performance()
    
    # Check that everything was reset
    assert len(ferris_rde._strategy_performance) == 0
    assert all(w == 1.0 for w in ferris_rde._mode_weights.values())
    assert len(ferris_rde._spin_outcomes) == 0
    assert ferris_rde._last_spin is None

def test_log_file_creation(ferris_rde):
    """Test spin log file creation."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    ferris_rde.update_market_state(signals)
    
    # Check that log file was created
    log_dir = Path("~/Schwabot/init/ferris_rde_logs").expanduser()
    log_files = list(log_dir.glob("*_ferris.json"))
    assert len(log_files) > 0
    
    # Check log file contents
    with open(log_files[-1]) as f:
        log_data = json.load(f)
    
    assert "tag" in log_data
    assert "bit_mode" in log_data
    assert "strategies" in log_data
    assert "weights" in log_data 