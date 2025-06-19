from unittest.mock import patch

"""
Tests for the visualization module
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from schwabot.core.visualization import (
    plot_profit_decay,
    plot_tick_sequence,
    plot_hash_distribution,
    plot_trade_metrics,
    create_interactive_dashboard,
    plot_system_metrics
)
import os
import matplotlib
matplotlib.use("Agg")


def test_plot_profit_decay() -> None:
    """Test profit decay plotting"""
    times = np.linspace(0, 10, 100)
    profits = 0.618 * np.exp(-0.777 * times) + np.random.normal(0, 0.01, 100)
    os.makedirs("tests/output", exist_ok=True)
    fig, ax = plot_profit_decay(
        times,
        profits,
        save_path="tests/output/profit_decay.png"
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1  # Theoretical decay line
    assert len(ax.collections) == 1  # Scatter plot


def test_plot_tick_sequence() -> None:
    """Test tick sequence plotting"""
    ticks = np.random.normal(0, 1, 100).cumsum()
    os.makedirs("tests/output", exist_ok=True)
    fig, ax = plot_tick_sequence(
        ticks,
        save_path="tests/output/tick_sequence.png"
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 2  # Tick values and MA
    assert len(ax.collections) == 1  # Trend zone


def test_plot_hash_distribution() -> None:
    """Test hash distribution plotting"""
    hashes = [f"{np.random.randint(0, 16**8):08x}" for _ in range(100)]
    os.makedirs("tests/output", exist_ok=True)
    fig, ax = plot_hash_distribution(
        hashes,
        save_path="tests/output/hash_distribution.png"
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) > 0  # Histogram bars
    assert len(ax.lines) > 0  # KDE line


def test_plot_trade_metrics() -> None:
    """Test trade metrics plotting"""
    trades = [
        {'profit': float(np.random.normal(0, 1))} for _ in range(100)
    ]
    os.makedirs("tests/output", exist_ok=True)
    fig, ax = plot_trade_metrics(
        trades,
        save_path="tests/output/trade_metrics.png"
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) > 0  # Histogram bars
    assert len(ax.lines) == 1  # Mean line


def test_create_interactive_dashboard() -> None:
    """Test interactive dashboard creation"""
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.randint(1000, 2000, 100)
    }, index=dates)

    fig = create_interactive_dashboard(data)
    assert fig is not None
    assert len(fig.data) == 2  # Candlestick and volume
    # Optionally save Plotly figure
    os.makedirs("tests/output", exist_ok=True)
    fig.write_html("tests/output/interactive_dashboard.html")


def test_plot_system_metrics() -> None:
    """Test system metrics plotting"""
    timestamps = [datetime.now() + timedelta(minutes=i) for i in range(100)]
    metrics = {
        'cpu_temp': list(np.random.normal(60, 5, 100)),
        'cpu_load': list(np.random.normal(50, 10, 100)),
        'gpu_temp': list(np.random.normal(70, 5, 100)),
        'gpu_load': list(np.random.normal(60, 10, 100))
    }
    os.makedirs("tests/output", exist_ok=True)
    fig, (
        ax1,
        ax2) = plot_system_metrics(metrics,
                                   timestamps,
                                   save_path="tests/output/system_metrics.png"
                                   )
    assert fig is not None
    assert ax1 is not None
    assert ax2 is not None
    assert len(ax1.lines) == 2  # CPU metrics
    assert len(ax2.lines) == 2  # GPU metrics