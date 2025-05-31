"""
Test suite for RDE_core (Radial Dynamic Engine)
=============================================

Tests the core functionality of the RDE engine including:
- Bit mode switching
- Signal processing
- Hash generation
- Ferris Wheel integration
"""

import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from ncco_core.rde_core import RDEEngine

@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    config = {
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def rde_engine(temp_config):
    """Create an RDE engine instance for testing."""
    return RDEEngine(temp_config)

def test_initialization(rde_engine):
    """Test RDE engine initialization."""
    assert rde_engine.assets == ["BTC", "ETH"]
    assert rde_engine.current_bit_mode == 42  # Should default to highest precision
    assert len(rde_engine.axes) == 2  # price_metrics and market_state
    assert all(k in rde_engine._buffers for k in rde_engine.axes)

def test_axis_expansion(rde_engine):
    """Test expansion of {asset} templates in axis names."""
    expanded = rde_engine.axes["price_metrics"]
    assert "BTC_price_delta" in expanded
    assert "ETH_price_delta" in expanded
    assert "BTC_volatility" in expanded
    assert "ETH_volatility" in expanded

def test_signal_update(rde_engine):
    """Test signal update and buffer management."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    rde_engine.update_signals(signals)
    
    # Check that buffers were updated
    for set_name in rde_engine.axes:
        assert len(rde_engine._buffers[set_name]) == 1
        vec = rde_engine._buffers[set_name][0][1]
        assert vec.shape == (len(rde_engine.axes[set_name]),)

def test_bit_mode_switching(rde_engine):
    """Test bit mode switching based on signal stability."""
    # Generate stable signals
    stable_signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": 0.02,
        "ETH_volatility": 0.15
    }
    
    # Generate volatile signals
    volatile_signals = {
        "BTC_price_delta": 0.5,
        "BTC_volatility": 0.8,
        "ETH_price_delta": -0.3,
        "ETH_volatility": 0.9
    }
    
    # Feed stable signals
    for _ in range(10):
        rde_engine.update_signals(stable_signals)
    
    # Should stay in high precision mode
    assert rde_engine.current_bit_mode == 42
    
    # Feed volatile signals
    for _ in range(10):
        rde_engine.update_signals(volatile_signals)
    
    # Should switch to low precision mode
    assert rde_engine.current_bit_mode == 4

def test_biotype_generation(rde_engine):
    """Test biotype (spin ID) generation with different bit modes."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    rde_engine.update_signals(signals)
    
    # Test 4-bit mode
    rde_engine.set_bit_mode(4)
    tag_4bit = rde_engine.compute_biotype()
    assert tag_4bit.startswith("BTC_")
    assert len(tag_4bit.split("_")[1]) == 4
    
    # Test 8-bit mode
    rde_engine.set_bit_mode(8)
    tag_8bit = rde_engine.compute_biotype()
    assert tag_8bit.startswith("BTC_")
    assert len(tag_8bit.split("_")[1]) == 8
    
    # Test 42-bit mode
    rde_engine.set_bit_mode(42)
    tag_42bit = rde_engine.compute_biotype()
    assert tag_42bit.startswith("BTC_")
    assert len(tag_42bit.split("_")[1]) == 11  # 11 hex chars = 44 bits

def test_spin_logging(rde_engine):
    """Test spin logging functionality."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    rde_engine.update_signals(signals)
    tag = rde_engine.compute_biotype()
    
    # Log the spin
    log_path = rde_engine.log_spin(tag)
    assert log_path.exists()
    
    # Verify log contents
    with open(log_path) as f:
        log_data = json.load(f)
    
    assert log_data["tag"] == tag
    assert "utc" in log_data
    assert log_data["bit_mode"] == rde_engine.current_bit_mode
    assert "buffers" in log_data

def test_mode_performance_tracking(rde_engine):
    """Test bit mode performance tracking."""
    # Update performance for each mode
    rde_engine.update_mode_performance(4, 0.8)
    rde_engine.update_mode_performance(8, 0.9)
    rde_engine.update_mode_performance(42, 0.95)
    
    performance = rde_engine.get_mode_performance()
    assert performance[4] == 0.8
    assert performance[8] == 0.9
    assert performance[42] == 0.95

def test_spin_history(rde_engine):
    """Test spin history tracking."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    # Generate multiple spins
    for _ in range(5):
        rde_engine.update_signals(signals)
        tag = rde_engine.compute_biotype()
        rde_engine.log_spin(tag)
    
    # Check history
    history = rde_engine.get_spin_history()
    assert len(history) == 5
    
    # Check limited history
    limited = rde_engine.get_spin_history(limit=3)
    assert len(limited) == 3

def test_bit_mode_history(rde_engine):
    """Test bit mode switch history tracking."""
    # Switch modes a few times
    rde_engine.set_bit_mode(4)
    rde_engine.set_bit_mode(8)
    rde_engine.set_bit_mode(42)
    
    history = rde_engine.get_bit_mode_history()
    assert len(history) == 3
    assert all(isinstance(ts, float) for ts, _ in history)
    assert all(mode in rde_engine.bit_modes for _, mode in history)

def test_ferris_wheel_integration(rde_engine):
    """Test Ferris Wheel integration features."""
    signals = {
        "BTC_price_delta": 0.02,
        "BTC_volatility": 0.15,
        "ETH_price_delta": -0.01,
        "ETH_volatility": 0.12
    }
    
    # Simulate a Ferris Wheel cycle
    rde_engine.update_signals(signals)
    tag = rde_engine.compute_biotype()
    rde_engine.log_spin(tag)
    
    # Verify Ferris Wheel state
    assert rde_engine._last_spin_tag == tag
    assert len(rde_engine._spin_history) == 1
    
    # Update performance based on Ferris Wheel outcome
    rde_engine.update_mode_performance(rde_engine.current_bit_mode, 0.95)
    
    # Verify performance tracking
    performance = rde_engine.get_mode_performance()
    assert performance[rde_engine.current_bit_mode] == 0.95 