import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.brain_trading_engine import BrainSignal, BrainTradingEngine
from core.ccxt_trading_executor import CCXTTradingExecutor
from core.gpu_cpu_calculation_bridge import get_gpu_cpu_bridge
from core.settings_manager import get_settings_manager
from schwabot.speed_lattice_live_panel_system import SpeedLatticeLivePanelSystem
from schwabot.trading_pipeline_integration import TradingPipelineIntegration
from symbolic_profit_router import SymbolicProfitRouter

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Main Entry Point - Enhanced with Brain Trading
======================================================

Main launcher for Schwabot with integrated brain trading functionality.
This replaces placeholder implementations with working brain algorithms.
"""


Schwabot Advanced Trading System
== == == == == == == == == == == == == == == =

Main application entry point with GUI and CLI interfaces.
Supports demo, live, and backtest trading modes across multiple exchanges.

Features:
- Real - time trading interface
- Advanced mathematical processing with GPU acceleration
- Risk management and portfolio tracking
- Multi - exchange support via CCXT
- Professional visualization system
""""""


# GUI imports with fallback
GUI_AVAILABLE = True
    try:
    plt.style.use('dark_background')
    except ImportError as e:
    print(f"GUI libraries not available: {e}")
    GUI_AVAILABLE = False

# Core system imports with fallback
CORE_AVAILABLE = True
    try:
    pass
    except ImportError as e:
    print(f"Core systems not available: {e}")
    CORE_AVAILABLE = False

# Set up logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[]
        logging.FileHandler('logs/schwabot_main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import core components
    try:
    BRAIN_ENGINE_AVAILABLE = True
    except ImportError:
    logger.error("Brain Trading Engine not available")
    BRAIN_ENGINE_AVAILABLE = False

try:
    SYMBOLIC_ROUTER_AVAILABLE = True
    except ImportError:
    logger.error("Symbolic Profit Router not available")
    SYMBOLIC_ROUTER_AVAILABLE = False


class SchwabotGUI:
    """Main GUI application for Schwabot trading system."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Schwabot Advanced Trading System v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        # System components
        self.trading_pipeline = None
        self.visualizer = None
        self.settings_manager = None
        self.is_running = False
        self.system_thread = None

        # Status variables
        self.status_vars = {}
            "system_status": tk.StringVar(value="Initializing..."),
            "trading_mode": tk.StringVar(value="Demo"),
            "api_status": tk.StringVar(value="Disconnected"),
            "gpu_status": tk.StringVar(value="Unknown"),
            "last_update": tk.StringVar(value="Never"),
            "total_profit": tk.StringVar(value="$0.0"),
        }
        self._setup_gui()
        self._initialize_systems()

    def _setup_gui(self):
        """Setup the main GUI interface."""
        # Configure style for dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff")"
        style.configure("TButton", background="#3e3e3e", foreground="#ffffff")"

        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create status frame
        self._create_status_frame(main_container)

        # Create tabbed interface
        self._create_main_tabs(main_container)

    def _create_status_frame(self, parent):
        """Create the system status display frame."""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        status_items = []
            ("System Status", self.status_vars["system_status"], "#00ff00"),
            ("Trading Mode", self.status_vars["trading_mode"], "#ffff00"),
            ("API Status", self.status_vars["api_status"], "#ff0000"),
            ("GPU Status", self.status_vars["gpu_status"], "#00ffff"),"
            ("Last Update", self.status_vars["last_update"], "#ffffff"),"
            ("Total Profit", self.status_vars["total_profit"], "#00ff00"),
        ]
        for i, (label_text, var, color) in enumerate(status_items):
            row = i // 3
            col = i % 3
            frame = ttk.Frame(status_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)

            ttk.Label(frame, text=f"{label_text}:").pack(side=tk.LEFT)
            ttk.Label(frame, textvariable=var, foreground=color).pack(side=tk.RIGHT)

        for i in range(3):
            status_frame.grid_columnconfigure(i, weight=1)

    def _create_main_tabs(self, parent):
        """Create the main tabbed interface."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Trading control tab
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="Trading Control")
        self._create_trading_tab(trading_frame)

        # Visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Live Visualization")
        self._create_visualization_tab(viz_frame)

        # Settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="System Settings")
        self._create_settings_tab(settings_frame)

        # Logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="System Logs")
        self._create_logs_tab(logs_frame)

    def _create_trading_tab(self, parent):
        """Create the trading control tab."""
        # Control section
        control_frame = ttk.LabelFrame(parent, text="Trading Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(mode_frame, text="Trading Mode:").pack(side=tk.LEFT)

        self.mode_var = tk.StringVar(value="demo")
        modes = [("Demo", "demo"), ("Live", "live"), ("Backtest", "backtest")]
        for text, value in modes:
            ttk.Radiobutton()
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self._on_mode_change).pack(
                side=tk.LEFT,
                padx=10)

        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Start Trading",)
                   command=self._start_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Trading",)
                   command=self._stop_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh Status",)
                   command=self._refresh_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Emergency Stop",)
                   command=self._emergency_stop).pack(side=tk.RIGHT, padx=5)

        # Portfolio display
        portfolio_frame = ttk.LabelFrame(parent, text="Portfolio Summary", padding=10)
        portfolio_frame.pack(fill=tk.BOTH, expand=True)

        self.portfolio_fig = Figure(figsize=(12, 6), facecolor="#1e1e1e")
        self.portfolio_canvas = FigureCanvasTkAgg(self.portfolio_fig, portfolio_frame)
        self.portfolio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._init_portfolio_chart()

    def _create_visualization_tab(self, parent):
        """Create the visualization tab."""
        # Visualization controls
        viz_controls = ttk.LabelFrame(parent, text="Visualization Controls", padding=10)
        viz_controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(viz_controls, text="Display Panel:").pack(side=tk.LEFT)
        self.panel_var = tk.StringVar(value="DRIFT_MATRIX")
        panel_options = ["DRIFT_MATRIX", "PROFIT_RESONANCE", "SYSTEM_STATUS",]
                         "TRADING_STATE", "PATTERN_RECOGNITION"]

        panel_combo = ttk.Combobox(viz_controls, textvariable=self.panel_var,)
                                   values=panel_options, state="readonly")
        panel_combo.pack(side=tk.LEFT, padx=10)
        panel_combo.bind("<<ComboboxSelected>>", self._on_panel_change)

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_controls, text="Auto Refresh",)
                        variable=self.auto_refresh_var).pack(side=tk.RIGHT)

        # Visualization area
        self.visualization_frame = ttk.Frame(parent)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True)

    def _create_settings_tab(self, parent):
        """Create the settings configuration tab."""
        # Scrollable settings area
        canvas = tk.Canvas(parent, bg="#1e1e1e")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind()
            "<Configure>", lambda e: canvas.configure()
                scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # API Settings
        api_frame = ttk.LabelFrame()
            scrollable_frame,
            text="API Configuration",
            padding=10)
        api_frame.pack(fill=tk.X, pady=5, padx=10)

        self.api_settings = {}
        api_fields = []
            ("Coinbase API Key", "coinbase_api_key"),
            ("Coinbase Secret", "coinbase_secret"),
            ("Sandbox Mode", "sandbox_mode"),
        ]
        for label_text, var_name in api_fields:
            frame = ttk.Frame(api_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label_text}:").pack(side=tk.LEFT)

            if var_name == "sandbox_mode":
                var = tk.BooleanVar(value=True)
                ttk.Checkbutton(frame, variable=var).pack(side=tk.RIGHT)
            else:
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=40,)
                          show="*").pack(side=tk.RIGHT)

            self.api_settings[var_name] = var

        # Performance Settings
        perf_frame = ttk.LabelFrame()
            scrollable_frame,
            text="Performance Settings",
            padding=10)
        perf_frame.pack(fill=tk.X, pady=5, padx=10)

        self.perf_settings = {}
        perf_fields = []
            ("GPU Acceleration", "gpu_enabled", "boolean"),
            ("CPU Threads", "cpu_threads", "int"),
            ("Memory Limit (MB)", "memory_limit", "int"),
            ("Update Interval (s)", "update_interval", "float"),
        ]
        for label_text, var_name, var_type in perf_fields:
            frame = ttk.Frame(perf_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label_text}:").pack(side=tk.LEFT)

            if var_type == "boolean":
                var = tk.BooleanVar(value=True)
                ttk.Checkbutton(frame, variable=var).pack(side=tk.RIGHT)
            else:
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=20).pack(side=tk.RIGHT)

            self.perf_settings[var_name] = var

        # Settings buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Button(button_frame, text="Save Settings",)
                   command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings",)
                   command=self._load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults",)
                   command=self._reset_settings).pack(side=tk.RIGHT, padx=5)

    def _create_logs_tab(self, parent):
        """Create the system logs tab."""
        log_frame = ttk.LabelFrame(parent, text="System Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        # Log display
        self.log_text = tk.Text(log_frame, bg="#0", fg="#0ff00",)
                                font=("Consolas", 9), wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,)
                                      command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Log controls
        log_controls = ttk.Frame(parent)
        log_controls.pack(fill=tk.X, pady=5)

        ttk.Button(log_controls, text="Clear Logs",)
                   command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Export Logs",)
                   command=self._export_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Refresh",)
                   command=self._refresh_logs).pack(side=tk.LEFT, padx=5)

    def _initialize_systems(self):
        """Initialize all system components."""
        try:
            if not CORE_AVAILABLE:
                self.status_vars["system_status"].set("Core systems unavailable")
                return

            # Initialize components
            self.settings_manager = get_settings_manager()
            self.gpu_cpu_bridge = get_gpu_cpu_bridge()

            self.trading_pipeline = TradingPipelineIntegration()
                enable_gpu=True,
                enable_distributed=False,
                max_concurrent_trades=10,
                risk_management_enabled=True
            )

            if GUI_AVAILABLE:
                self.visualizer = SpeedLatticeLivePanelSystem()

            self.status_vars["system_status"].set("Initialized")
            self.status_vars["gpu_status"].set()
                "Available" if self.gpu_cpu_bridge.gpu_available else "CPU Only")

            self._start_update_thread()

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status_vars["system_status"].set(f"Error: {e}")

    def _init_portfolio_chart(self):
        """Initialize the portfolio chart."""
        self.portfolio_fig.clear()
        ax = self.portfolio_fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        ax.set_title("Portfolio Performance", color="white")
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors="white")
        self.portfolio_fig.tight_layout()
        self.portfolio_canvas.draw()

    def _start_update_thread(self):
        """Start the background update thread."""
        def update_loop():
            while self.is_running:
                try:
                    self._update_status()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Update thread error: {e}")

        self.is_running = True
        self.system_thread = threading.Thread(target=update_loop, daemon=True)
        self.system_thread.start()

    def _update_status(self):
        """Update system status displays."""
        try:
            if self.trading_pipeline:
                performance = self.trading_pipeline.get_pipeline_performance()
                self.status_vars["last_update"].set(datetime.now().strftime("%H:%M:%S"))

                if "pipeline_metrics" in performance:
                    metrics = performance["pipeline_metrics"]
                    total_profit = metrics.get("total_profit", 0.0)
                    self.status_vars["total_profit"].set(f"${total_profit:.2f}")

        except Exception as e:
            logger.error(f"Status update failed: {e}")

    def _on_mode_change(self):
        """Handle trading mode change."""
        mode = self.mode_var.get()
        self.status_vars["trading_mode"].set(mode.title())
        logger.info(f"Trading mode changed to: {mode}")

    def _on_panel_change(self, event=None):
        """Handle visualization panel change."""
        panel_name = self.panel_var.get()
        logger.info(f"Visualization panel changed to: {panel_name}")
        if self.visualizer:
            try:
                self.visualizer.switch_to_panel(panel_name)
            except Exception as e:
                logger.error(f"Panel switch failed: {e}")

    def _start_trading(self):
        """Start trading operations."""
        try:
            if self.trading_pipeline:
                self.trading_pipeline.start_trading()
                self.status_vars["system_status"].set("Trading Active")
                logger.info("Trading started")
            else:
                messagebox.showerror("Error", "Trading pipeline not initialized")
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")

    def _stop_trading(self):
        """Stop trading operations."""
        try:
            if self.trading_pipeline:
                self.trading_pipeline.stop_trading()
                self.status_vars["system_status"].set("Trading Stopped")
                logger.info("Trading stopped")
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")

    def _emergency_stop(self):
        """Emergency stop all operations."""
        try:
            if self.trading_pipeline:
                self.trading_pipeline.emergency_stop()
                self.status_vars["system_status"].set("Emergency Stop")
                logger.warning("Emergency stop activated")
                messagebox.showwarning()
                    "Emergency Stop",
                    "All trading operations stopped immediately")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")

    def _refresh_status(self):
        """Refresh system status."""
        self._update_status()
        logger.info("Status refreshed")

    def _save_settings(self):
        """Save current settings."""
        try:
            if self.settings_manager:
                # Collect settings from GUI
                settings = {}
                for key, var in self.api_settings.items():
                    settings[key] = var.get()
                for key, var in self.perf_settings.items():
                    settings[key] = var.get()

                self.settings_manager.save_settings(settings)
                messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _load_settings(self):
        """Load settings from file."""
        try:
            if self.settings_manager:
                settings = self.settings_manager.load_settings()
                # Update GUI with loaded settings
                for key, value in settings.items():
                    if key in self.api_settings:
                        self.api_settings[key].set(value)
                    elif key in self.perf_settings:
                        self.perf_settings[key].set(value)

                messagebox.showinfo("Success", "Settings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            messagebox.showerror("Error", f"Failed to load settings: {e}")

    def _reset_settings(self):
        """Reset settings to defaults."""
        try:
            if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
                # Reset GUI to defaults
                for var in self.api_settings.values():
                    if isinstance(var, tk.StringVar):
                        var.set("")
                    elif isinstance(var, tk.BooleanVar):
                        var.set(True)

                for var in self.perf_settings.values():
                    if isinstance(var, tk.StringVar):
                        var.set("")
                    elif isinstance(var, tk.BooleanVar):
                        var.set(True)

                messagebox.showinfo("Success", "Settings reset to defaults")
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}")

    def _clear_logs(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)

    def _export_logs(self):
        """Export logs to file."""
        try:
            filename = filedialog.asksaveasfilename()
                defaultextension=".log", filetypes=[]
                    ("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")])
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            messagebox.showerror("Error", f"Failed to export logs: {e}")

    def _refresh_logs(self):
        """Refresh the log display."""
        try:
            # Read from log file
            if Path('schwabot.log').exists():
                with open('schwabot.log', 'r') as f:
                    logs = f.read()
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(1.0, logs)
                self.log_text.see(tk.END)
        except Exception as e:
            logger.error(f"Failed to refresh logs: {e}")

    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

    def cleanup(self):
        """Cleanup resources."""
        self.is_running = False
        if self.system_thread and self.system_thread.is_alive():
            self.system_thread.join(timeout=1)

        if self.trading_pipeline:
            self.trading_pipeline.shutdown()


class SchwabotCLI:
    """Command-line interface for Schwabot."""

    def __init__(self, args):
        self.args = args
        self.trading_pipeline = None
        self.running = True
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.shutdown()

    async def run_demo_mode(self):
        """Run in demonstration mode."""
        logger.info("Starting Schwabot in DEMO mode")

        try:
            if not CORE_AVAILABLE:
                logger.error("Core systems not available")
                return

            # Initialize systems
            settings_manager = get_settings_manager()
            gpu_cpu_bridge = get_gpu_cpu_bridge()

            # Initialize trading pipeline
            self.trading_pipeline = TradingPipelineIntegration()
                enable_gpu=True,
                enable_distributed=False,
                max_concurrent_trades=5,
                risk_management_enabled=True
            )

            # Start demo trading
            logger.info("Demo trading simulation started")

            # Demo loop
            iteration = 0
            while self.running and iteration < 10:  # Limit demo iterations
                iteration += 1

                logger.info(f"Demo iteration {iteration}/10")

                # Simulate trading activity
                if self.trading_pipeline:
                    performance = self.trading_pipeline.get_pipeline_performance()
                    logger.info(f"Pipeline performance: {performance}")

                # Simulate GPU processing
                if gpu_cpu_bridge.gpu_available:
                    test_data = np.random.random((1000, 1000))
                    result = gpu_cpu_bridge.process_matrix_operation()
                        test_data, "multiply", test_data)
                    logger.info(f"GPU processing completed: {result.shape}")

                await asyncio.sleep(2)  # Wait 2 seconds between iterations

            logger.info("Demo mode completed successfully")

        except Exception as e:
            logger.error(f"Demo mode failed: {e}")
        finally:
            if self.trading_pipeline:
                self.trading_pipeline.shutdown()

    async def run_live_mode(self):
        """Run in live trading mode."""
        logger.info("Starting Schwabot in LIVE mode")
        logger.warning("LIVE MODE - Real money trading enabled!")

        try:
            if not CORE_AVAILABLE:
                logger.error("Core systems not available")
                return

            # Initialize for live trading
            settings_manager = get_settings_manager()

            self.trading_pipeline = TradingPipelineIntegration()
                enable_gpu=True,
                enable_distributed=True,
                max_concurrent_trades=20,
                risk_management_enabled=True
            )

            # Start live trading
            self.trading_pipeline.start_trading()
            logger.info("Live trading started")

            # Live trading loop
            while self.running:
                performance = self.trading_pipeline.get_pipeline_performance()
                logger.info(f"Live trading performance: {performance}")
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Live mode failed: {e}")
        finally:
            if self.trading_pipeline:
                self.trading_pipeline.emergency_stop()

    async def run_backtest_mode(self):
        """Run in backtesting mode."""
        logger.info("Starting Schwabot in BACKTEST mode")
        # Implement backtesting logic here
        logger.info("Backtest mode not fully implemented yet")

    def shutdown(self):
        """Shutdown the CLI application."""
        logger.info("Shutting down Schwabot CLI")
        self.running = False
        if self.trading_pipeline:
            self.trading_pipeline.shutdown()


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Schwabot Advanced Trading System")
    parser.add_argument("--mode", choices=["demo", "live", "backtest"],)
                        default="demo", help="Trading mode")
    parser.add_argument("--gui", action="store_true",)
                        help="Start GUI interface")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],)
                        default="INFO", help="Logging level")
    parser.add_argument("--config", type=str,)
                        help="Configuration file path")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Start application
    if args.gui and GUI_AVAILABLE:
        logger.info("Starting Schwabot GUI")
        app = SchwabotGUI()
        try:
            app.run()
        finally:
            app.cleanup()
    else:
        if args.gui and not GUI_AVAILABLE:
            logger.warning("GUI requested but not available, starting CLI mode")

        logger.info("Starting Schwabot CLI")
        cli = SchwabotCLI(args)

        try:
            if args.mode == "demo":
                asyncio.run(cli.run_demo_mode())
            elif args.mode == "live":
                asyncio.run(cli.run_live_mode())
            elif args.mode == "backtest":
                asyncio.run(cli.run_backtest_mode())
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            cli.shutdown()


if __name__ == "__main__":
    main()
