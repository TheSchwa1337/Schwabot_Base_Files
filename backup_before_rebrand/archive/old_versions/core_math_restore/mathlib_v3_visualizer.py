from __future__ import annotations

import json
import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from core.mathlib_v3 import Dual, MathLibV3

# -*- coding: utf-8 -*-
"""
MathLib V3 Visualizer - Live Mathematical Operations Dashboard
=============================================================
Comprehensive visualization system for MathLib V3 with live panels,
test functionality, and integration hooks for API/CCXT connectivity.

Features:
- Live mathematical operation panels
- Real-time dual number visualization
- Kelly criterion and risk assessment displays
- Pattern detection and market prediction views
- Integration with live/demo/backtest modes
- API and CCXT connectivity hooks
- Internal state management and persistence
"""



logger = logging.getLogger(__name__)


@dataclass
class MathLibState:
    """Internal state management for MathLib V3 operations."""

    current_operation: str = "idle"
    last_calculation: Dict[str, Any] = None
    calculation_history: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    error_log: List[str] = None
    api_connected: bool = False
    ccxt_connected: bool = False
    mode: str = "demo"  # live, demo, backtest

    def __post_init__(self):
        if self.calculation_history is None:
            self.calculation_history = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.error_log is None:
            self.error_log = []


class MathLibV3Visualizer:
    """Comprehensive visualizer for MathLib V3 with live panels and integration."""

    def __init__(self, mode: str = "demo"):
        self.mode = mode  # live, demo, backtest
        self.mathlib = MathLibV3()
        self.state = MathLibState(mode=mode)
        self.data_queue = queue.Queue()
        self.running = False
        self.panels = {}
        self.fig = None
        self.animation = None

        # Initialize panels
        self._initialize_panels()

        logger.info(f"MathLib V3 Visualizer initialized in {mode} mode")

    def _initialize_panels(self):
        """Initialize all visualization panels."""
        self.panels = {
            "dual_operations": {
                "title": "Dual Number Operations",
                "description": "Real-time automatic differentiation",
                "data": {},
                "active": True,
            },
            "kelly_criterion": {
                "title": "Kelly Criterion Analysis",
                "description": "Risk-adjusted portfolio optimization",
                "data": {},
                "active": True,
            },
            "risk_assessment": {
                "title": "Risk Assessment",
                "description": "Portfolio risk metrics and CVaR",
                "data": {},
                "active": True,
            },
            "pattern_detection": {
                "title": "Pattern Detection",
                "description": "AI-enhanced pattern recognition",
                "data": {},
                "active": True,
            },
            "market_prediction": {
                "title": "Market Prediction",
                "description": "Time series forecasting",
                "data": {},
                "active": True,
            },
            "optimization": {
                "title": "Gradient Descent Optimization",
                "description": "Real-time optimization progress",
                "data": {},
                "active": True,
            },
            "performance": {
                "title": "Performance Metrics",
                "description": "System performance and timing",
                "data": {},
                "active": True,
            },
            "integration": {
                "title": "Integration Status",
                "description": "API and CCXT connectivity",
                "data": {},
                "active": True,
            },
        }

    def start_live_mode(self):
        """Start live visualization mode."""
        self.mode = "live"
        self.state.mode = "live"
        self.running = True

        # Start data collection thread
        self.data_thread = threading.Thread(target=self._data_collection_loop)
        self.data_thread.daemon = True
        self.data_thread.start()

        # Start visualization
        self._create_visualization()

        logger.info("MathLib V3 Visualizer started in live mode")

    def start_demo_mode(self):
        """Start demo mode with simulated data."""
        self.mode = "demo"
        self.state.mode = "demo"
        self.running = True

        # Start demo data generation
        self.demo_thread = threading.Thread(target=self._demo_data_loop)
        self.demo_thread.daemon = True
        self.demo_thread.start()

        # Start visualization
        self._create_visualization()

        logger.info("MathLib V3 Visualizer started in demo mode")

    def start_backtest_mode(self, historical_data: np.ndarray):
        """Start backtest mode with historical data."""
        self.mode = "backtest"
        self.state.mode = "backtest"
        self.running = True
        self.historical_data = historical_data
        self.current_index = 0

        # Start backtest simulation
        self.backtest_thread = threading.Thread(target=self._backtest_loop)
        self.backtest_thread.daemon = True
        self.backtest_thread.start()

        # Start visualization
        self._create_visualization()

        logger.info("MathLib V3 Visualizer started in backtest mode")

    def _data_collection_loop(self):
        """Live data collection loop."""
        while self.running:
            try:
                # Collect live data from API/CCXT
                live_data = self._collect_live_data()
                if live_data:
                    self.data_queue.put(live_data)
                time.sleep(1.0)  # 1 second update rate
            except Exception as e:
                logger.error(f"Error in live data collection: {e}")
                self.state.error_log.append(f"Live data error: {e}")

    def _demo_data_loop(self):
        """Demo data generation loop."""
        while self.running:
            try:
                # Generate simulated data
                demo_data = self._generate_demo_data()
                self.data_queue.put(demo_data)
                time.sleep(2.0)  # 2 second update rate for demo
            except Exception as e:
                logger.error(f"Error in demo data generation: {e}")
                self.state.error_log.append(f"Demo data error: {e}")

    def _backtest_loop(self):
        """Backtest simulation loop."""
        while self.running and self.current_index < len(self.historical_data):
            try:
                # Process historical data
                backtest_data = self._process_backtest_data()
                self.data_queue.put(backtest_data)
                self.current_index += 1
                time.sleep(0.5)  # Fast simulation
            except Exception as e:
                logger.error(f"Error in backtest simulation: {e}")
                self.state.error_log.append(f"Backtest error: {e}")

    def _collect_live_data():-> Dict[str, Any]:
        """Collect live data from API/CCXT sources."""
        # Placeholder for live data collection
        # In real implementation, this would connect to your API/CCXT
        return {
            "timestamp": datetime.now().isoformat(),
            "price": np.random.normal(50000, 1000),
            "volume": np.random.uniform(100, 1000),
            "returns": np.random.normal(0.001, 0.02),
            "source": "live_api",
        }

    def _generate_demo_data():-> Dict[str, Any]:
        """Generate simulated data for demo mode."""
        return {
            "timestamp": datetime.now().isoformat(),
            "price": 50000 + 1000 * np.sin(time.time() / 10),
            "volume": 500 + 200 * np.random.random(),
            "returns": np.random.normal(0.001, 0.015),
            "source": "demo_simulation",
        }

    def _process_backtest_data():-> Dict[str, Any]:
        """Process historical data for backtest mode."""
        if self.current_index < len(self.historical_data):
            data_point = self.historical_data[self.current_index]
            return {
                "timestamp": f"backtest_{self.current_index}",
                "price": data_point,
                "volume": np.random.uniform(100, 1000),
                "returns": np.diff(
                    self.historical_data[
                        max(0, self.current_index - 1) : self.current_index + 1
                    ]
                )[0]
                if self.current_index > 0
                else 0,
                "source": "backtest_data",
            }
        return None

    def _create_visualization(self):
        """Create the main visualization window."""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle(
            f"MathLib V3 Visualizer - {self.mode.upper()} Mode", fontsize=16
        )

        # Create subplots for each panel
        self._create_panel_layout()

        # Add control buttons
        self._add_control_buttons()

        # Start animation
        self.animation = FuncAnimation(
            self.fig, self._update_visualization, interval=1000, blit=False
        )

        plt.show()

    def _create_panel_layout(self):
        """Create the panel layout."""
        # 3x3 grid layout
        self.axes = {}

        # Dual Operations Panel
        self.axes["dual_operations"] = plt.subplot(3, 3, 1)
        self.axes["dual_operations"].set_title("Dual Number Operations")

        # Kelly Criterion Panel
        self.axes["kelly_criterion"] = plt.subplot(3, 3, 2)
        self.axes["kelly_criterion"].set_title("Kelly Criterion")

        # Risk Assessment Panel
        self.axes["risk_assessment"] = plt.subplot(3, 3, 3)
        self.axes["risk_assessment"].set_title("Risk Assessment")

        # Pattern Detection Panel
        self.axes["pattern_detection"] = plt.subplot(3, 3, 4)
        self.axes["pattern_detection"].set_title("Pattern Detection")

        # Market Prediction Panel
        self.axes["market_prediction"] = plt.subplot(3, 3, 5)
        self.axes["market_prediction"].set_title("Market Prediction")

        # Optimization Panel
        self.axes["optimization"] = plt.subplot(3, 3, 6)
        self.axes["optimization"].set_title("Optimization")

        # Performance Panel
        self.axes["performance"] = plt.subplot(3, 3, 7)
        self.axes["performance"].set_title("Performance")

        # Integration Panel
        self.axes["integration"] = plt.subplot(3, 3, 8)
        self.axes["integration"].set_title("Integration Status")

        # Control Panel
        self.axes["control"] = plt.subplot(3, 3, 9)
        self.axes["control"].set_title("Controls")
        self.axes["control"].axis("off")

    def _add_control_buttons(self):
        """Add control buttons to the visualization."""
        # Mode selection
        self.mode_var = tk.StringVar(value=self.mode)
        mode_frame = tk.Frame(self.fig.canvas.get_tk_widget())
        mode_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(
            mode_frame,
            text="Live",
            variable=self.mode_var,
            value="live",
            command=self._change_mode,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            mode_frame,
            text="Demo",
            variable=self.mode_var,
            value="demo",
            command=self._change_mode,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            mode_frame,
            text="Backtest",
            variable=self.mode_var,
            value="backtest",
            command=self._change_mode,
        ).pack(side=tk.LEFT)

    def _change_mode(self):
        """Change visualization mode."""
        new_mode = self.mode_var.get()
        if new_mode != self.mode:
            self.stop()
            if new_mode == "live":
                self.start_live_mode()
            elif new_mode == "demo":
                self.start_demo_mode()
            elif new_mode == "backtest":
                # For backtest, we need historical data
                historical_data = np.random.normal(50000, 1000, 1000)
                self.start_backtest_mode(historical_data)

    def _update_visualization(self, frame):
        """Update all visualization panels."""
        try:
            # Process any new data
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self._process_data(data)

            # Update each panel
            self._update_dual_operations_panel()
            self._update_kelly_criterion_panel()
            self._update_risk_assessment_panel()
            self._update_pattern_detection_panel()
            self._update_market_prediction_panel()
            self._update_optimization_panel()
            self._update_performance_panel()
            self._update_integration_panel()

        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            self.state.error_log.append(f"Visualization error: {e}")

    def _process_data(self, data: Dict[str, Any]):
        """Process incoming data and update state."""
        try:
            # Update state
            self.state.last_calculation = data
            self.state.calculation_history.append(data)

            # Keep only last 100 calculations
            if len(self.state.calculation_history) > 100:
                self.state.calculation_history = self.state.calculation_history[-100:]

            # Perform mathematical operations
            self._perform_math_operations(data)

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            self.state.error_log.append(f"Data processing error: {e}")

    def _perform_math_operations(self, data: Dict[str, Any]):
        """Perform mathematical operations on the data."""
        try:
            price = data.get("price", 50000)
            returns = data.get("returns", 0.001)

            # Dual number operations
            dual_result = self._compute_dual_operations(price)
            self.panels["dual_operations"]["data"] = dual_result

            # Kelly criterion
            kelly_result = self.mathlib.kelly_criterion_risk_adjusted(
                returns, returns**2, 0.25
            )
            self.panels["kelly_criterion"]["data"] = kelly_result

            # Risk assessment
            risk_result = self._compute_risk_assessment(data)
            self.panels["risk_assessment"]["data"] = risk_result

            # Pattern detection
            if len(self.state.calculation_history) > 10:
                prices = [
                    calc.get("price", 50000)
                    for calc in self.state.calculation_history[-20:]
                ]
                pattern_result = self.mathlib.detect_patterns_enhanced(np.array(prices))
                self.panels["pattern_detection"]["data"] = pattern_result

            # Market prediction
            if len(self.state.calculation_history) > 20:
                prices = [
                    calc.get("price", 50000)
                    for calc in self.state.calculation_history[-30:]
                ]
                prediction_result = self.mathlib.predict_market_movement(
                    np.array(prices)
                )
                self.panels["market_prediction"]["data"] = prediction_result

            # Optimization
            opt_result = self._compute_optimization(data)
            self.panels["optimization"]["data"] = opt_result

            # Performance metrics
            perf_result = self._compute_performance_metrics()
            self.panels["performance"]["data"] = perf_result

            # Integration status
            integration_result = self._get_integration_status()
            self.panels["integration"]["data"] = integration_result

        except Exception as e:
            logger.error(f"Error performing math operations: {e}")
            self.state.error_log.append(f"Math operations error: {e}")

    def _compute_dual_operations():-> Dict[str, Any]:
        """Compute dual number operations."""
        try:
            # Test function: f(x) = x^2 + 2x + 1
            def test_function():-> Dual:
                return dual_x * dual_x + 2 * dual_x + 1

            val, grad_val = self.mathlib.compute_dual_gradient(test_function, x)

            return {
                "input": x,
                "function_value": val,
                "derivative": grad_val,
                "dual_number": f"{val:.4f} + {grad_val:.4f}ε",
            }
        except Exception as e:
            logger.error(f"Error computing dual operations: {e}")
            return {"error": str(e)}

    def _compute_risk_assessment():-> Dict[str, Any]:
        """Compute risk assessment metrics."""
        try:
            if len(self.state.calculation_history) < 5:
                return {"error": "Insufficient data"}

            returns = [
                calc.get("returns", 0) for calc in self.state.calculation_history[-20:]
            ]
            returns_array = np.array(returns)

            # CVaR calculation
            cvar_95 = self.mathlib.cvar_calculation(returns_array, 0.95)

            # Basic risk metrics
            volatility = np.std(returns_array)
            sharpe_ratio = np.mean(returns_array) / volatility if volatility > 0 else 0

            return {
                "cvar_95": cvar_95,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": np.min(returns_array),
                "var_95": np.percentile(returns_array, 5),
            }
        except Exception as e:
            logger.error(f"Error computing risk assessment: {e}")
            return {"error": str(e)}

    def _compute_optimization():-> Dict[str, Any]:
        """Compute optimization results."""
        try:
            # Simple quadratic optimization
            def objective():-> float:
                return float(
                    np.sum((x - np.array([data.get("price", 50000), 0.1])) ** 2)
                )

            initial_x = np.array([0.0, 0.0])
            opt_result = self.mathlib.gradient_descent_optimization(
                objective, initial_x, learning_rate=0.01, max_iterations=50
            )

            return opt_result
        except Exception as e:
            logger.error(f"Error computing optimization: {e}")
            return {"error": str(e)}

    def _compute_performance_metrics():-> Dict[str, Any]:
        """Compute performance metrics."""
        try:
            if not self.state.calculation_history:
                return {"error": "No data available"}

            # Timing metrics
            timestamps = [
                calc.get("timestamp", "") for calc in self.state.calculation_history
            ]
            processing_times = []

            for i in range(1, len(timestamps)):
                try:
                    t1 = datetime.fromisoformat(timestamps[i - 1])
                    t2 = datetime.fromisoformat(timestamps[i])
                    processing_times.append((t2 - t1).total_seconds())
                except BaseException:
                    pass

            avg_processing_time = np.mean(processing_times) if processing_times else 0

            return {
                "total_calculations": len(self.state.calculation_history),
                "avg_processing_time": avg_processing_time,
                "error_count": len(self.state.error_log),
                "uptime": time.time() - getattr(self, "_start_time", time.time()),
            }
        except Exception as e:
            logger.error(f"Error computing performance metrics: {e}")
            return {"error": str(e)}

    def _get_integration_status():-> Dict[str, Any]:
        """Get integration status."""
        return {
            "api_connected": self.state.api_connected,
            "ccxt_connected": self.state.ccxt_connected,
            "mode": self.state.mode,
            "last_update": datetime.now().isoformat(),
            "data_queue_size": self.data_queue.qsize(),
        }

    def _update_dual_operations_panel(self):
        """Update dual operations panel."""
        ax = self.axes["dual_operations"]
        ax.clear()

        data = self.panels["dual_operations"]["data"]
        if data and "error" not in data:
            # Create dual number visualization
            x = np.linspace(data["input"] - 10, data["input"] + 10, 100)
            y = x**2 + 2 * x + 1
            dy = 2 * x + 2

            ax.plot(x, y, "b-", label="f(x) = x² + 2x + 1")
            ax.plot(x, dy, "r--", label="f'(x) = 2x + 2")
            ax.axvline(
                data["input"],
                color="g",
                linestyle=":",
                label=f"x = {data['input']:.2f}",
            )
            ax.axhline(
                data["function_value"],
                color="g",
                linestyle=":",
                label=f"f(x) = {data['function_value']:.2f}",
            )

            ax.set_title("Dual Number Operations")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_kelly_criterion_panel(self):
        """Update Kelly criterion panel."""
        ax = self.axes["kelly_criterion"]
        ax.clear()

        data = self.panels["kelly_criterion"]["data"]
        if data and "error" not in data:
            # Create Kelly criterion visualization
            fractions = np.linspace(0, 1, 100)
            expected_returns = []

            for f in fractions:
                er = (
                    data.get("expected_utility", 0) * f
                    - 0.5 * data.get("risk_tolerance", 0.25) * f**2
                )
                expected_returns.append(er)

            ax.plot(fractions, expected_returns, "b-", label="Expected Utility")
            ax.axvline(
                data.get("kelly_fraction", 0),
                color="r",
                linestyle="--",
                label=f"Kelly: {data.get('kelly_fraction', 0):.3f}",
            )
            ax.axvline(
                data.get("risk_adjusted_fraction", 0),
                color="g",
                linestyle="--",
                label=f"Risk-Adjusted: {data.get('risk_adjusted_fraction', 0):.3f}",
            )

            ax.set_title("Kelly Criterion Analysis")
            ax.set_xlabel("Allocation Fraction")
            ax.set_ylabel("Expected Utility")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_risk_assessment_panel(self):
        """Update risk assessment panel."""
        ax = self.axes["risk_assessment"]
        ax.clear()

        data = self.panels["risk_assessment"]["data"]
        if data and "error" not in data:
            # Create risk metrics visualization
            metrics = ["CVaR 95%", "Volatility", "Sharpe Ratio", "Max Drawdown"]
            values = [
                abs(data.get("cvar_95", 0)),
                data.get("volatility", 0),
                data.get("sharpe_ratio", 0),
                abs(data.get("max_drawdown", 0)),
            ]
            colors = ["red", "orange", "green", "purple"]
            bars = ax.bar(metrics, values, color=colors)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Risk Assessment Metrics")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_pattern_detection_panel(self):
        """Update pattern detection panel."""
        ax = self.axes["pattern_detection"]
        ax.clear()

        data = self.panels["pattern_detection"]["data"]
        if data and "error" not in data:
            # Create pattern detection visualization
            patterns = ["Trend", "Volatility", "Cycles", "Mean Reversion"]
            values = [
                data.get("increasing_trend_probability", 0),
                data.get("volatility_clustering", 0),
                data.get("cycle_strength", 0),
                abs(data.get("mean_reversion_coefficient", 0)),
            ]
            colors = ["blue", "orange", "green", "red"]
            bars = ax.bar(patterns, values, color=colors)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Pattern Detection")
            ax.set_ylabel("Strength")
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(0, 1)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_market_prediction_panel(self):
        """Update market prediction panel."""
        ax = self.axes["market_prediction"]
        ax.clear()

        data = self.panels["market_prediction"]["data"]
        if data and "error" not in data:
            # Create market prediction visualization
            forecast = data.get("forecast", [])
            if forecast:
                x = range(len(forecast))
                ax.plot(x, forecast, "b-", label="Forecast", linewidth=2)

                # Add confidence intervals
                ci = data.get("confidence_intervals", {})
                if "lower_95" in ci and "upper_95" in ci:
                    ax.fill_between(
                        x,
                        ci["lower_95"],
                        ci["upper_95"],
                        alpha=0.3,
                        color="blue",
                        label="95% CI",
                    )

                ax.set_title("Market Prediction")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No forecast data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_optimization_panel(self):
        """Update optimization panel."""
        ax = self.axes["optimization"]
        ax.clear()

        data = self.panels["optimization"]["data"]
        if data and "error" not in data:
            # Create optimization visualization
            history = data.get("history", [])
            if history:
                iterations = [h.get("iteration", 0) for h in history]
                objectives = [h.get("objective", 0) for h in history]

                ax.plot(iterations, objectives, "b-", label="Objective Value")
                ax.set_title("Gradient Descent Optimization")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Objective Value")
                ax.legend()
                ax.grid(True)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No optimization history",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_performance_panel(self):
        """Update performance panel."""
        ax = self.axes["performance"]
        ax.clear()

        data = self.panels["performance"]["data"]
        if data and "error" not in data:
            # Create performance visualization
            metrics = ["Calculations", "Avg Time (ms)", "Errors", "Uptime (s)"]
            values = [
                data.get("total_calculations", 0),
                data.get("avg_processing_time", 0) * 1000,
                data.get("error_count", 0),
                data.get("uptime", 0),
            ]
            colors = ["green", "blue", "red", "orange"]
            bars = ax.bar(metrics, values, color=colors)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Performance Metrics")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _update_integration_panel(self):
        """Update integration panel."""
        ax = self.axes["integration"]
        ax.clear()

        data = self.panels["integration"]["data"]
        if data:
            # Create integration status visualization
            status_items = ["API Connected", "CCXT Connected", "Mode", "Queue Size"]
            status_values = [
                "✓" if data.get("api_connected", False) else "✗",
                "✓" if data.get("ccxt_connected", False) else "✗",
                data.get("mode", "unknown"),
                data.get("data_queue_size", 0),
            ]
            colors = [
                "green" if v == "✓" else "red" if v == "✗" else "blue"
                for v in status_values
            ]

            y_pos = np.arange(len(status_items))
            ax.barh(y_pos, [1 if v != "✗" else 0 for v in status_values], color=colors)

            # Add labels
            for i, (item, value) in enumerate(zip(status_items, status_values)):
                ax.text(
                    0.5,
                    i,
                    f"{item}: {value}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontweight="bold",
                )

            ax.set_title("Integration Status")
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def stop(self):
        """Stop the visualizer."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close("all")
        logger.info("MathLib V3 Visualizer stopped")

    def save_state():-> str:
        """Save visualizer state to file."""
        if filename is None:
            filename = f"mathlib_v3_visualizer_state_{
                datetime.now().strftime('%Y%m%d_%H%M%S')
            }.json"

        state_data = {
            "visualizer_state": asdict(self.state),
            "panels": self.panels,
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
        }
        with open(filename, "w") as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"Visualizer state saved to {filename}")
        return filename

    def load_state():-> bool:
        """Load visualizer state from file."""
        try:
            with open(filename, "r") as f:
                state_data = json.load(f)

            # Restore state
            self.state = MathLibState(**state_data.get("visualizer_state", {}))
            self.panels = state_data.get("panels", {})
            self.mode = state_data.get("mode", "demo")

            logger.info(f"Visualizer state loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading visualizer state: {e}")
            return False

    def get_integration_hooks():-> Dict[str, Any]:
        """Get integration hooks for external systems."""
        return {
            "mathlib": self.mathlib,
            "state": self.state,
            "panels": self.panels,
            "mode": self.mode,
            "data_queue": self.data_queue,
            "running": self.running,
        }


def main():
    """Main function for testing the MathLib V3 Visualizer."""
    print("MathLib V3 Visualizer Test")
    print("=" * 50)

    # Create visualizer
    visualizer = MathLibV3Visualizer(mode="demo")

    try:
        # Start demo mode
        print("Starting demo mode...")
        visualizer.start_demo_mode()

        # Run for 30 seconds
        print("Running for 30 seconds...")
        time.sleep(30)

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
