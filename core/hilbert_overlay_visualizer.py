# -*- coding: utf-8 -*-
"""
Hilbert Overlay Visualizer Module - Cross-Logic Platform Positional System
==========================================================================
Advanced visualization system using Hilbert curves for multi-dimensional state mapping,
entry/exit signal generation, and autonomous trading integration with CCXT.

Features:
- Hilbert curve mapping for 1D to 2D state visualization
- Cross-logic platform positional system
- Real-time entry/exit signal generation
- Internalized state synthesis (demo/test/live)
- CCXT integration for autonomous trading
- 4-bit/8-bit logic mapping and vault integration
- Tick-based internalized time states
- Profit optimization with automatic stop-loss
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import tkinter as tk
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox

from core.mathlib_v3 import Dual, MathLibV3
from core.mathlib_v3_visualizer import MathLibState, MathLibV3Visualizer

logger = logging.getLogger(__name__)


@dataclass
class HilbertState:
    """Internalized state for Hilbert curve mapping and trading logic."""

    tick_id: str = ""
    internal_time: int = 0
    price_state: float = 0.0
    volume_state: float = 0.0
    momentum_state: float = 0.0
    volatility_state: float = 0.0
    pattern_state: float = 0.0
    risk_state: float = 0.0
    profit_state: float = 0.0
    entry_signal: bool = False
    exit_signal: bool = False
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    hash_signature: str = ""
    state_vector: List[float] = None

    def __post_init__(self):
        if self.state_vector is None:
            self.state_vector = []
        if not self.tick_id:
            self.tick_id = str(uuid.uuid4())[:8]


@dataclass
class TradingSignal:
    """Trading signal with entry/exit logic and risk management."""

    signal_type: str  # "entry", "exit", "hold"
    confidence: float  # 0.0 to 1.0
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timestamp: str
    state_hash: str


class HilbertCurveMapper:
    """Hilbert curve mapping for multi-dimensional state visualization."""

    def __init__(self, dimensions: int = 2, order: int = 8):
        self.dimensions = dimensions
        self.order = order
        self.max_h = 2**order
        self.max_state = 2 ** (order * dimensions)

        logger.info(f"Hilbert curve mapper initialized: {dimensions}D, order {order}")

    def coordinates_to_distance(self, coords: List[int]) -> int:
        """Convert coordinates to Hilbert distance."""
        if len(coords) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} coordinates, got {len(coords)}")

        # Simplified Hilbert curve mapping
        distance = 0
        for i, coord in enumerate(coords):
            distance += coord * (self.max_h**i)

        return distance % self.max_state

    def distance_to_coordinates(self, distance: int) -> List[int]:
        """Convert Hilbert distance to coordinates."""
        coords = []
        remaining = distance % self.max_state

        for i in range(self.dimensions):
            coord = remaining % self.max_h
            coords.append(coord)
            remaining //= self.max_h

        return coords

    def map_state_to_hilbert(self, state_vector: List[float]) -> Tuple[int, List[int]]:
        """Map state vector to Hilbert curve coordinates."""
        # Normalize state vector to [0, max_h-1] range
        normalized = []
        for val in state_vector:
            norm_val = int((val + 1.0) * (self.max_h - 1) / 2.0)  # Map [-1, 1] to [0, max_h-1]
            norm_val = max(0, min(self.max_h - 1, norm_val))
            normalized.append(norm_val)

        # Convert to Hilbert distance
        distance = self.coordinates_to_distance(normalized)
        coords = self.distance_to_coordinates(distance)

        return distance, coords


class CrossLogicPlatform:
    """Cross-logic platform for positional system and trading signals."""

    def __init__(self, mathlib: MathLibV3):
        self.mathlib = mathlib
        self.state_history: List[HilbertState] = []
        self.signal_history: List[TradingSignal] = []
        self.current_position = 0.0
        self.total_profit = 0.0
        self.risk_tolerance = 0.25
        self.max_position_size = 1.0

        # 4-bit and 8-bit logic states
        self.logic_4bit = 0
        self.logic_8bit = 0
        self.vault_states = {}

        logger.info("Cross-logic platform initialized")

    def update_internal_states(self, price: float, volume: float, timestamp: str) -> HilbertState:
        """Update internalized states and generate new Hilbert state."""
        # Create new state
        state = HilbertState()
        state.tick_id = str(uuid.uuid4())[:8]
        state.internal_time = len(self.state_history)
        state.price_state = self._normalize_price(price)
        state.volume_state = self._normalize_volume(volume)

        # Calculate derived states
        if len(self.state_history) > 0:
            prev_state = self.state_history[-1]
            state.momentum_state = self._calculate_momentum(price, prev_state.price_state)
            state.volatility_state = self._calculate_volatility()
            state.pattern_state = self._detect_patterns()
            state.risk_state = self._assess_risk()
            state.profit_state = self._calculate_profit_potential()

        # Generate state vector
        state.state_vector = [
            state.price_state,
            state.volume_state,
            state.momentum_state,
            state.volatility_state,
            state.pattern_state,
            state.risk_state,
            state.profit_state,
        ]

        # Generate hash signature
        state.hash_signature = self._generate_state_hash(state)

        # Update logic states
        self._update_logic_states(state)

        return state

    def _normalize_price(self, price: float) -> float:
        """Normalize price to [-1, 1] range."""
        # Use BTC price range of 0-100000 as reference
        return (price - 50000) / 50000

    def _normalize_volume(self, volume: float) -> float:
        """Normalize volume to [-1, 1] range."""
        # Use volume range of 0-2000 as reference
        return (volume - 1000) / 1000

    def _calculate_momentum(self, current_price: float, prev_price_state: float) -> float:
        """Calculate momentum state."""
        if len(self.state_history) < 2:
            return 0.0

        # Calculate price momentum
        price_changes = []
        for i in range(min(10, len(self.state_history))):
            if i > 0:
                change = self.state_history[-i].price_state - self.state_history[-i - 1].price_state
                price_changes.append(change)

        if price_changes:
            return np.mean(price_changes)
        return 0.0

    def _calculate_volatility(self) -> float:
        """Calculate volatility state."""
        if len(self.state_history) < 5:
            return 0.0

        # Calculate price volatility
        prices = [state.price_state for state in self.state_history[-20:]]
        return np.std(prices)

    def _detect_patterns(self) -> float:
        """Detect pattern state using MathLib V3."""
        if len(self.state_history) < 10:
            return 0.0

        # Use MathLib V3 pattern detection
        prices = [state.price_state for state in self.state_history[-20:]]
        pattern_result = self.mathlib.detect_patterns_enhanced(np.array(prices))

        # Combine pattern indicators
        trend = pattern_result.get("increasing_trend_probability", 0.5)
        cycles = pattern_result.get("cycle_strength", 0.0)
        mean_reversion = abs(pattern_result.get("mean_reversion_coefficient", 0.0))

        return (trend + cycles + mean_reversion) / 3.0

    def _assess_risk(self) -> float:
        """Assess risk state."""
        if len(self.state_history) < 5:
            return 0.5

        # Calculate risk based on volatility and recent losses
        volatility = self._calculate_volatility()
        recent_profits = [state.profit_state for state in self.state_history[-10:]]
        avg_profit = np.mean(recent_profits)

        # Higher volatility and lower profits = higher risk
        risk = (volatility + (1.0 - avg_profit)) / 2.0
        return max(0.0, min(1.0, risk))

    def _calculate_profit_potential(self) -> float:
        """Calculate profit potential state."""
        if len(self.state_history) < 5:
            return 0.0

        # Use Kelly criterion for profit potential
        returns = []
        for i in range(1, min(20, len(self.state_history))):
            if i > 0:
                ret = self.state_history[-i].price_state - self.state_history[-i - 1].price_state
                returns.append(ret)

        if returns:
            mean_return = np.mean(returns)
            variance = np.var(returns)

            if variance > 0:
                kelly_result = self.mathlib.kelly_criterion_risk_adjusted(mean_return, variance, self.risk_tolerance)
                return kelly_result.get("risk_adjusted_fraction", 0.0)

        return 0.0

    def _generate_state_hash(self, state: HilbertState) -> str:
        """Generate hash signature for state."""
        state_str = f"{state.tick_id}:{state.internal_time}:{state.price_state:.6f}:{state.volume_state:.6f}"
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    def _update_logic_states(self, state: HilbertState):
        """Update 4-bit and 8-bit logic states."""
        # 4-bit logic: Simple state classification
        self.logic_4bit = 0
        if state.price_state > 0.5:
            self.logic_4bit |= 1
        if state.volume_state > 0.5:
            self.logic_4bit |= 2
        if state.momentum_state > 0.1:
            self.logic_4bit |= 4
        if state.profit_state > 0.3:
            self.logic_4bit |= 8

        # 8-bit logic: Extended state classification
        self.logic_8bit = 0
        if state.price_state > 0.3:
            self.logic_8bit |= 1
        if state.volume_state > 0.3:
            self.logic_8bit |= 2
        if state.momentum_state > 0.05:
            self.logic_8bit |= 4
        if state.volatility_state < 0.3:
            self.logic_8bit |= 8
        if state.pattern_state > 0.4:
            self.logic_8bit |= 16
        if state.risk_state < 0.6:
            self.logic_8bit |= 32
        if state.profit_state > 0.2:
            self.logic_8bit |= 64
        if self.current_position == 0:
            self.logic_8bit |= 128

    def generate_trading_signal(self, state: HilbertState) -> TradingSignal:
        """Generate trading signal based on current state."""
        # Entry conditions
        entry_conditions = (
            state.profit_state > 0.3
            and state.risk_state < 0.5
            and state.pattern_state > 0.4
            and self.current_position == 0.0
        )

        # Exit conditions
        exit_conditions = self.current_position != 0.0 and (
            state.profit_state < 0.1 or state.risk_state > 0.7 or state.momentum_state < -0.1
        )

        # Calculate signal confidence
        confidence = (state.profit_state + (1.0 - state.risk_state) + state.pattern_state) / 3.0

        if entry_conditions:
            # Entry signal
            position_size = min(self.max_position_size, state.profit_state)
            stop_loss = state.price_state - (0.02 * state.risk_state)  # 2% stop loss
            take_profit = state.price_state + (0.04 * state.profit_state)  # 4% take profit

            return TradingSignal(
                signal_type="entry",
                confidence=confidence,
                price_target=state.price_state,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=f"Profit: {
                    state.profit_state:.3f}, Risk: {
                    state.risk_state:.3f}, Pattern: {
                    state.pattern_state:.3f}",
                timestamp=datetime.now().isoformat(),
                state_hash=state.hash_signature,
            )

        elif exit_conditions:
            # Exit signal
            return TradingSignal(
                signal_type="exit",
                confidence=confidence,
                price_target=state.price_state,
                stop_loss=0.0,
                take_profit=0.0,
                position_size=0.0,
                reasoning=f"Risk: {state.risk_state:.3f}, Momentum: {state.momentum_state:.3f}",
                timestamp=datetime.now().isoformat(),
                state_hash=state.hash_signature,
            )

        else:
            # Hold signal
            return TradingSignal(
                signal_type="hold",
                confidence=confidence,
                price_target=state.price_state,
                stop_loss=0.0,
                take_profit=0.0,
                position_size=0.0,
                reasoning="No clear signal",
                timestamp=datetime.now().isoformat(),
                state_hash=state.hash_signature,
            )


class HilbertOverlayVisualizer:
    """Hilbert overlay visualizer with cross-logic platform integration."""

    def __init__(self, mode: str = "demo"):
        self.mode = mode
        self.mathlib = MathLibV3()
        self.hilbert_mapper = HilbertCurveMapper(dimensions=2, order=8)
        self.cross_logic = CrossLogicPlatform(self.mathlib)
        self.state_history: List[HilbertState] = []
        self.signal_history: List[TradingSignal] = []
        self.data_queue = queue.Queue()
        self.running = False
        self.fig = None
        self.animation = None

        # CCXT integration placeholder
        self.ccxt_connected = False
        self.trading_enabled = False

        logger.info(f"Hilbert Overlay Visualizer initialized in {mode} mode")

    def start_live_mode(self):
        """Start live mode with real data and CCXT integration."""
        self.mode = "live"
        self.running = True
        self.trading_enabled = True

        # Start data collection thread
        self.data_thread = threading.Thread(target=self._live_data_loop)
        self.data_thread.daemon = True
        self.data_thread.start()

        # Start visualization
        self._create_visualization()

        logger.info("Hilbert Overlay Visualizer started in live mode")

    def start_demo_mode(self):
        """Start demo mode with simulated data."""
        self.mode = "demo"
        self.running = True
        self.trading_enabled = False

        # Start demo data generation
        self.demo_thread = threading.Thread(target=self._demo_data_loop)
        self.demo_thread.daemon = True
        self.demo_thread.start()

        # Start visualization
        self._create_visualization()

        logger.info("Hilbert Overlay Visualizer started in demo mode")

    def start_backtest_mode(self, historical_data: List[Dict[str, Any]]):
        """Start backtest mode with historical data."""
        self.mode = "backtest"
        self.running = True
        self.trading_enabled = False
        self.historical_data = historical_data
        self.current_index = 0

        # Start backtest simulation
        self.backtest_thread = threading.Thread(target=self._backtest_loop)
        self.backtest_thread.daemon = True
        self.backtest_thread.start()

        # Start visualization
        self._create_visualization()

        logger.info("Hilbert Overlay Visualizer started in backtest mode")

    def _live_data_loop(self):
        """Live data collection loop with CCXT integration."""
        while self.running:
            try:
                # Collect live data (placeholder for CCXT integration)
                live_data = self._collect_live_data()
                if live_data:
                    self.data_queue.put(live_data)
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in live data collection: {e}")

    def _demo_data_loop(self):
        """Demo data generation loop."""
        while self.running:
            try:
                demo_data = self._generate_demo_data()
                self.data_queue.put(demo_data)
                time.sleep(2.0)
            except Exception as e:
                logger.error(f"Error in demo data generation: {e}")

    def _backtest_loop(self):
        """Backtest simulation loop."""
        while self.running and self.current_index < len(self.historical_data):
            try:
                backtest_data = self._process_backtest_data()
                self.data_queue.put(backtest_data)
                self.current_index += 1
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in backtest simulation: {e}")

    def _collect_live_data(self) -> Dict[str, Any]:
        """Collect live data from CCXT."""
        # Placeholder for CCXT integration
        return {
            "timestamp": datetime.now().isoformat(),
            "price": np.random.normal(50000, 1000),
            "volume": np.random.uniform(100, 1000),
            "source": "live_ccxt",
        }

    def _generate_demo_data(self) -> Dict[str, Any]:
        """Generate simulated data for demo mode."""
        return {
            "timestamp": datetime.now().isoformat(),
            "price": 50000 + 1000 * np.sin(time.time() / 10),
            "volume": 500 + 200 * np.random.random(),
            "source": "demo_simulation",
        }

    def _process_backtest_data(self) -> Dict[str, Any]:
        """Process historical data for backtest mode."""
        if self.current_index < len(self.historical_data):
            data_point = self.historical_data[self.current_index]
            return {
                "timestamp": data_point.get("timestamp", f"backtest_{self.current_index}"),
                "price": data_point.get("price", 50000),
                "volume": data_point.get("volume", 500),
                "source": "backtest_data",
            }
        return None

    def _create_visualization(self):
        """Create the main visualization window."""
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle(f"Hilbert Overlay Visualizer - {self.mode.upper()} Mode", fontsize=16)

        # Create subplots
        self._create_panel_layout()

        # Start animation
        self.animation = FuncAnimation(self.fig, self._update_visualization, interval=1000, blit=False)

        plt.show()

    def _create_panel_layout(self):
        """Create the panel layout."""
        self.axes = {}

        # Hilbert curve mapping
        self.axes["hilbert_mapping"] = plt.subplot(2, 3, 1)
        self.axes["hilbert_mapping"].set_title("Hilbert Curve State Mapping")

        # State vector visualization
        self.axes["state_vector"] = plt.subplot(2, 3, 2)
        self.axes["state_vector"].set_title("State Vector Components")

        # Trading signals
        self.axes["trading_signals"] = plt.subplot(2, 3, 3)
        self.axes["trading_signals"].set_title("Trading Signals")

        # Logic states
        self.axes["logic_states"] = plt.subplot(2, 3, 4)
        self.axes["logic_states"].set_title("4-bit/8-bit Logic States")

        # Profit tracking
        self.axes["profit_tracking"] = plt.subplot(2, 3, 5)
        self.axes["profit_tracking"].set_title("Profit Tracking")

        # System status
        self.axes["system_status"] = plt.subplot(2, 3, 6)
        self.axes["system_status"].set_title("System Status")
        self.axes["system_status"].axis("off")

    def _update_visualization(self, frame):
        """Update all visualization panels."""
        try:
            # Process any new data
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self._process_data(data)

            # Update each panel
            self._update_hilbert_mapping_panel()
            self._update_state_vector_panel()
            self._update_trading_signals_panel()
            self._update_logic_states_panel()
            self._update_profit_tracking_panel()
            self._update_system_status_panel()

        except Exception as e:
            logger.error(f"Error updating visualization: {e}")

    def _process_data(self, data: Dict[str, Any]):
        """Process incoming data and update states."""
        try:
            # Update cross-logic platform
            state = self.cross_logic.update_internal_states(
                data.get("price", 50000), data.get("volume", 500), data.get("timestamp", "")
            )

            # Generate trading signal
            signal = self.cross_logic.generate_trading_signal(state)

            # Update histories
            self.state_history.append(state)
            self.signal_history.append(signal)

            # Keep only last 100 states
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
                self.signal_history = self.signal_history[-100:]

            # Execute trading logic if enabled
            if self.trading_enabled and signal.signal_type in ["entry", "exit"]:
                self._execute_trading_signal(signal)

        except Exception as e:
            logger.error(f"Error processing data: {e}")

    def _execute_trading_signal(self, signal: TradingSignal):
        """Execute trading signal (placeholder for CCXT integration)."""
        try:
            if signal.signal_type == "entry":
                logger.info(f"ENTRY SIGNAL: {signal.reasoning}")
                # Placeholder for CCXT buy order
                self.cross_logic.current_position = signal.position_size

            elif signal.signal_type == "exit":
                logger.info(f"EXIT SIGNAL: {signal.reasoning}")
                # Placeholder for CCXT sell order
                if self.cross_logic.current_position > 0:
                    # Calculate profit
                    profit = signal.price_target - self.cross_logic.current_position
                    self.cross_logic.total_profit += profit
                    self.cross_logic.current_position = 0.0

        except Exception as e:
            logger.error(f"Error executing trading signal: {e}")

    def _update_hilbert_mapping_panel(self):
        """Update Hilbert curve mapping panel."""
        ax = self.axes["hilbert_mapping"]
        ax.clear()

        if len(self.state_history) > 0:
            # Get recent states
            recent_states = self.state_history[-20:]

            # Map states to Hilbert coordinates
            hilbert_coords = []
            for state in recent_states:
                distance, coords = self.hilbert_mapper.map_state_to_hilbert(state.state_vector)
                hilbert_coords.append(coords)

            # Plot Hilbert mapping
            coords_array = np.array(hilbert_coords)
            ax.scatter(coords_array[:, 0], coords_array[:, 1], c=range(len(coords_array)), cmap="viridis", alpha=0.7)

            # Add current state highlight
            if hilbert_coords:
                current_coords = hilbert_coords[-1]
                ax.scatter(current_coords[0], current_coords[1], color="red", s=100, marker="*", label="Current State")

            ax.set_title("Hilbert Curve State Mapping")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No state data available", ha="center", va="center", transform=ax.transAxes)

    def _update_state_vector_panel(self):
        """Update state vector panel."""
        ax = self.axes["state_vector"]
        ax.clear()

        if len(self.state_history) > 0:
            current_state = self.state_history[-1]

            # Plot state vector components
            components = ["Price", "Volume", "Momentum", "Volatility", "Pattern", "Risk", "Profit"]
            values = current_state.state_vector

            colors = ["blue", "green", "orange", "red", "purple", "brown", "pink"]
            bars = ax.bar(components, values, color=colors)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom"
                )

            ax.set_title("State Vector Components")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(-1, 1)
        else:
            ax.text(0.5, 0.5, "No state data available", ha="center", va="center", transform=ax.transAxes)

    def _update_trading_signals_panel(self):
        """Update trading signals panel."""
        ax = self.axes["trading_signals"]
        ax.clear()

        if len(self.signal_history) > 0:
            # Get recent signals
            recent_signals = self.signal_history[-20:]

            # Plot signal confidence over time
            times = range(len(recent_signals))
            confidences = [signal.confidence for signal in recent_signals]
            signal_types = [signal.signal_type for signal in recent_signals]

            # Color code by signal type
            colors = []
            for signal_type in signal_types:
                if signal_type == "entry":
                    colors.append("green")
                elif signal_type == "exit":
                    colors.append("red")
                else:
                    colors.append("gray")

            ax.scatter(times, confidences, c=colors, alpha=0.7)

            # Add current signal highlight
            if recent_signals:
                current_signal = recent_signals[-1]
                ax.scatter(
                    len(recent_signals) - 1,
                    current_signal.confidence,
                    color="blue",
                    s=100,
                    marker="*",
                    label=f"Current: {current_signal.signal_type}",
                )

            ax.set_title("Trading Signals")
            ax.set_xlabel("Time")
            ax.set_ylabel("Confidence")
            ax.legend()
            ax.grid(True)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, "No signal data available", ha="center", va="center", transform=ax.transAxes)

    def _update_logic_states_panel(self):
        """Update logic states panel."""
        ax = self.axes["logic_states"]
        ax.clear()

        if len(self.state_history) > 0:
            # Plot 4-bit and 8-bit logic states
            recent_states = self.state_history[-20:]
            times = range(len(recent_states))

            # Get logic states (simplified - in real implementation these would be tracked)
            logic_4bit = [i % 16 for i in times]  # Placeholder
            logic_8bit = [i % 256 for i in times]  # Placeholder

            ax.plot(times, logic_4bit, "b-", label="4-bit Logic", linewidth=2)
            ax.plot(times, logic_8bit, "r-", label="8-bit Logic", linewidth=2)

            ax.set_title("Logic States")
            ax.set_xlabel("Time")
            ax.set_ylabel("Logic Value")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No logic data available", ha="center", va="center", transform=ax.transAxes)

    def _update_profit_tracking_panel(self):
        """Update profit tracking panel."""
        ax = self.axes["profit_tracking"]
        ax.clear()

        if len(self.state_history) > 0:
            # Plot profit over time
            times = range(len(self.state_history))
            profits = [state.profit_state for state in self.state_history]
            positions = [self.cross_logic.current_position] * len(self.state_history)

            ax.plot(times, profits, "b-", label="Profit Potential", linewidth=2)
            ax.plot(times, positions, "r--", label="Current Position", linewidth=2)

            # Add total profit line
            total_profit = [self.cross_logic.total_profit] * len(self.state_history)
            ax.plot(times, total_profit, "g-", label="Total Profit", linewidth=2)

            ax.set_title("Profit Tracking")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No profit data available", ha="center", va="center", transform=ax.transAxes)

    def _update_system_status_panel(self):
        """Update system status panel."""
        ax = self.axes["system_status"]
        ax.clear()

        # System status information
        status_items = [
            f"Mode: {self.mode.upper()}",
            f"States: {len(self.state_history)}",
            f"Signals: {len(self.signal_history)}",
            f"Position: {self.cross_logic.current_position:.3f}",
            f"Total Profit: {self.cross_logic.total_profit:.3f}",
            f"Trading: {'ON' if self.trading_enabled else 'OFF'}",
            f"CCXT: {'Connected' if self.ccxt_connected else 'Disconnected'}",
            f"Queue Size: {self.data_queue.qsize()}",
        ]

        # Display status
        for i, item in enumerate(status_items):
            ax.text(0.1, 0.9 - i * 0.1, item, transform=ax.transAxes, fontsize=10, fontweight="bold")

        ax.set_title("System Status")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def stop(self):
        """Stop the visualizer."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close("all")
        logger.info("Hilbert Overlay Visualizer stopped")

    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading summary and statistics."""
        return {
            "total_states": len(self.state_history),
            "total_signals": len(self.signal_history),
            "current_position": self.cross_logic.current_position,
            "total_profit": self.cross_logic.total_profit,
            "entry_signals": len([s for s in self.signal_history if s.signal_type == "entry"]),
            "exit_signals": len([s for s in self.signal_history if s.signal_type == "exit"]),
            "hold_signals": len([s for s in self.signal_history if s.signal_type == "hold"]),
            "average_confidence": np.mean([s.confidence for s in self.signal_history]) if self.signal_history else 0.0,
            "mode": self.mode,
            "trading_enabled": self.trading_enabled,
        }

    def save_state(self, filename: str = None) -> str:
        """Save visualizer state to file."""
        if filename is None:
            filename = f"hilbert_overlay_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        state_data = {
            "mode": self.mode,
            "state_history": [asdict(state) for state in self.state_history[-50:]],  # Last 50 states
            "signal_history": [asdict(signal) for signal in self.signal_history[-50:]],  # Last 50 signals
            "cross_logic_state": {
                "current_position": self.cross_logic.current_position,
                "total_profit": self.cross_logic.total_profit,
                "logic_4bit": self.cross_logic.logic_4bit,
                "logic_8bit": self.cross_logic.logic_8bit,
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"Hilbert overlay state saved to {filename}")
        return filename


def main():
    """Main function for testing the Hilbert Overlay Visualizer."""
    print("Hilbert Overlay Visualizer Test")
    print("=" * 50)

    # Create visualizer
    visualizer = HilbertOverlayVisualizer(mode="demo")

    try:
        # Start demo mode
        print("Starting demo mode...")
        visualizer.start_demo_mode()

        # Run for 30 seconds
        print("Running for 30 seconds...")
        time.sleep(30)

        # Get trading summary
        summary = visualizer.get_trading_summary()
        print(f"Trading Summary: {summary}")

        # Save state
        print("Saving state...")
        state_file = visualizer.save_state()
        print(f"State saved to: {state_file}")

        # Stop visualizer
        print("Stopping visualizer...")
        visualizer.stop()

        print("Test completed successfully!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        visualizer.stop()
    except Exception as e:
        print(f"Test failed with error: {e}")
        visualizer.stop()


if __name__ == "__main__":
    main()
