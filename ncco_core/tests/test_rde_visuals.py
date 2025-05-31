"""
Test suite for RDE Visualization Module
=====================================

Tests the visualization tools for RDE integration, including:
- Mode weight plots
- Strategy performance plots
- Heatmap generation
- Dashboard creation
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ncco_core.rde_visuals import RDEVisualizer

@pytest.fixture
def sample_history():
    """Create sample spin history for testing."""
    base_time = datetime.now(timezone.utc)
    history = []
    
    # Create 10 spins with different modes and strategies
    for i in range(10):
        spin_time = base_time + timedelta(hours=i)
        mode = [4, 8, 42][i % 3]  # Cycle through modes
        
        # Create different strategy sets based on mode
        if mode == 4:
            strategies = ["hold", "stable_swap"]
            weights = [0.7, 0.3]
        elif mode == 8:
            strategies = ["hedge", "flip"]
            weights = [0.6, 0.4]
        else:  # 42-bit
            strategies = ["aggressive", "spec"]
            weights = [0.5, 0.5]
        
        # Create performance data
        strategy_performance = {
            "hold": 0.8 + 0.1 * np.sin(i),
            "flip": 0.7 + 0.1 * np.cos(i),
            "hedge": 0.75 + 0.1 * np.sin(i + 1),
            "spec": 0.65 + 0.1 * np.cos(i + 1),
            "stable_swap": 0.85 + 0.1 * np.sin(i + 2),
            "aggressive": 0.6 + 0.1 * np.cos(i + 2)
        }
        
        # Create mode performance
        mode_performance = {
            4: 0.8 + 0.1 * np.sin(i),
            8: 0.7 + 0.1 * np.cos(i),
            42: 0.6 + 0.1 * np.sin(i + 1)
        }
        
        spin_data = {
            "tag": f"spin_{i:03d}",
            "utc": spin_time.isoformat(),
            "bit_mode": mode,
            "strategies": strategies,
            "weights": weights,
            "strategy_performance": strategy_performance,
            "mode_performance": mode_performance
        }
        
        history.append(spin_data)
    
    return history

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for test plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def visualizer():
    """Create RDEVisualizer instance for testing."""
    return RDEVisualizer()

def test_initialization(visualizer):
    """Test RDEVisualizer initialization."""
    assert visualizer.log_dir.exists()
    assert len(visualizer.strategy_colors) == 6
    assert visualizer.mode_cmap is not None

def test_plot_mode_weights(visualizer, sample_history, temp_output_dir):
    """Test mode weight plotting."""
    output_path = temp_output_dir / "mode_weights.png"
    visualizer.plot_mode_weights(sample_history, str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_plot_strategy_performance(visualizer, sample_history, temp_output_dir):
    """Test strategy performance plotting."""
    output_path = temp_output_dir / "strategy_performance.png"
    visualizer.plot_strategy_performance(sample_history, str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_plot_strategy_heatmap(visualizer, sample_history, temp_output_dir):
    """Test strategy selection heatmap generation."""
    output_path = temp_output_dir / "strategy_heatmap.png"
    visualizer.plot_strategy_heatmap(sample_history, str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_plot_spin_frequency(visualizer, sample_history, temp_output_dir):
    """Test spin frequency plotting."""
    output_path = temp_output_dir / "spin_frequency.png"
    visualizer.plot_spin_frequency(sample_history, "1H", str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_generate_dashboard(visualizer, sample_history, temp_output_dir):
    """Test complete dashboard generation."""
    visualizer.generate_dashboard(sample_history, temp_output_dir)
    
    # Check that all files were created
    assert (temp_output_dir / "mode_weights.png").exists()
    assert (temp_output_dir / "strategy_performance.png").exists()
    assert (temp_output_dir / "strategy_heatmap.png").exists()
    assert (temp_output_dir / "spin_frequency.png").exists()
    assert (temp_output_dir / "dashboard.html").exists()
    
    # Check HTML content
    html_content = (temp_output_dir / "dashboard.html").read_text()
    assert "RDE Integration Dashboard" in html_content
    assert "Mode Weight Evolution" in html_content
    assert "Strategy Performance" in html_content
    assert "Strategy Selection Heatmap" in html_content
    assert "Spin Frequency" in html_content

def test_empty_history(visualizer, temp_output_dir):
    """Test visualization with empty history."""
    empty_history = []
    
    # Should not raise exceptions
    visualizer.plot_mode_weights(empty_history, str(temp_output_dir / "empty_weights.png"))
    visualizer.plot_strategy_performance(empty_history, str(temp_output_dir / "empty_perf.png"))
    visualizer.plot_strategy_heatmap(empty_history, str(temp_output_dir / "empty_heatmap.png"))
    visualizer.plot_spin_frequency(empty_history, str(temp_output_dir / "empty_freq.png"))
    visualizer.generate_dashboard(empty_history, temp_output_dir)

def test_missing_data(visualizer, sample_history, temp_output_dir):
    """Test visualization with missing data points."""
    # Remove some data points
    incomplete_history = sample_history.copy()
    for spin in incomplete_history[::2]:  # Remove every other spin
        spin["mode_performance"] = {}
        spin["strategy_performance"] = {}
    
    # Should handle missing data gracefully
    visualizer.plot_mode_weights(incomplete_history, str(temp_output_dir / "incomplete_weights.png"))
    visualizer.plot_strategy_performance(incomplete_history, str(temp_output_dir / "incomplete_perf.png"))
    visualizer.plot_strategy_heatmap(incomplete_history, str(temp_output_dir / "incomplete_heatmap.png"))
    visualizer.plot_spin_frequency(incomplete_history, str(temp_output_dir / "incomplete_freq.png"))
    visualizer.generate_dashboard(incomplete_history, temp_output_dir)

def test_custom_colors(visualizer, sample_history, temp_output_dir):
    """Test visualization with custom color schemes."""
    # Create custom color mapping
    custom_colors = {
        "hold": "#FF0000",
        "flip": "#00FF00",
        "hedge": "#0000FF",
        "spec": "#FFFF00",
        "stable_swap": "#FF00FF",
        "aggressive": "#00FFFF"
    }
    
    visualizer.strategy_colors = custom_colors
    
    # Generate plots with custom colors
    output_path = temp_output_dir / "custom_colors.png"
    visualizer.plot_strategy_performance(sample_history, str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0 