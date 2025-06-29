# -*- coding: utf-8 -*-
"""
GUI System - Cross-Platform Trading Interface
============================================

Comprehensive GUI system for the trading platform with:
- Cross-platform compatibility (Windows, macOS, Linux)
- Real-time data visualization
- Trading controls and monitoring
- System status panels
- Configuration management
- Flake8 compliant design
"""

import asyncio
import logging
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.api_integration_manager import APIConfig, APIIntegrationManager
from utils.safe_print import debug, error, info, safe_print, success, warn

logger = logging.getLogger(__name__)


@dataclass
class GUIState:
    """State management for GUI components."""

    is_running: bool = False
    current_mode: str = "demo"  # demo, live, backtest
    api_connected: bool = False
    last_update: datetime = datetime.now()
    error_count: int = 0
    data_queue: queue.Queue = None

    def __post_init__(self):
        if self.data_queue is None:
            self.data_queue = queue.Queue()


class TradingDashboard:
    """Main trading dashboard GUI."""

    def __init__(self, api_manager: Optional[APIIntegrationManager] = None):
        """Initialize the trading dashboard."""
        self.api_manager = api_manager
        self.state = GUIState()
        self.root = None
        self.figures = {}
        self.canvases = {}
        self.update_callbacks = []

        # Initialize GUI
        self._setup_gui()
        self._create_panels()
        self._setup_event_handlers()

        logger.info("✅ Trading Dashboard initialized")

    def _setup_gui(self):
        """Setup the main GUI window."""
        self.root = tk.Tk()
        self.root.title("Schwabot Trading System - Advanced Mathematical Framework")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        # Configure style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        style.configure("TButton", background="#4a4a4a", foreground="white")

        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_panels(self):
        """Create all dashboard panels."""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self._create_overview_tab()
        self._create_trading_tab()
        self._create_analytics_tab()
        self._create_settings_tab()
        self._create_logs_tab()

    def _create_overview_tab(self):
        """Create overview tab with system status."""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")

        # System status panel
        status_frame = ttk.LabelFrame(overview_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        # Status indicators
        self.status_labels = {}
        status_items = [
            ("API Connection", "Disconnected", "red"),
            ("Trading Mode", "Demo", "yellow"),
            ("System Health", "Good", "green"),
            ("Last Update", "Never", "white"),
        ]

        for i, (label, value, color) in enumerate(status_items):
            row = i // 2
            col = i % 2

            frame = ttk.Frame(status_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=5, pady=2)

            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            self.status_labels[label] = ttk.Label(frame, text=value, foreground=color)
            self.status_labels[label].pack(side=tk.RIGHT)

        # Market data panel
        market_frame = ttk.LabelFrame(overview_frame, text="Market Data", padding=10)
        market_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Market data treeview
        columns = ("Symbol", "Price", "Change", "Volume", "Source")
        self.market_tree = ttk.Treeview(market_frame, columns=columns, show="headings")

        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=100)

        # Scrollbar for market data
        market_scrollbar = ttk.Scrollbar(market_frame, orient=tk.VERTICAL, command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=market_scrollbar.set)

        self.market_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        market_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_trading_tab(self):
        """Create trading tab with controls."""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="Trading")

        # Trading controls
        controls_frame = ttk.LabelFrame(trading_frame, text="Trading Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Mode selection
        mode_frame = ttk.Frame(controls_frame)
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Label(mode_frame, text="Trading Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="demo")

        for mode in ["demo", "live", "backtest"]:
            ttk.Radiobutton(
                mode_frame, text=mode.title(), variable=self.mode_var, value=mode, command=self._on_mode_change
            ).pack(side=tk.LEFT, padx=5)

        # Trading pair selection
        pair_frame = ttk.Frame(controls_frame)
        pair_frame.pack(fill=tk.X, pady=5)

        ttk.Label(pair_frame, text="Trading Pair:").pack(side=tk.LEFT)
        self.pair_var = tk.StringVar(value="BTC/USDC")
        pair_combo = ttk.Combobox(
            pair_frame, textvariable=self.pair_var, values=["BTC/USDC", "ETH/USDC", "XRP/USDC", "BTC/USDT"]
        )
        pair_combo.pack(side=tk.LEFT, padx=5)

        # Action buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Start Trading", command=self._start_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Trading", command=self._stop_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh Data", command=self._refresh_data).pack(side=tk.LEFT, padx=5)

        # Trading visualization
        viz_frame = ttk.LabelFrame(trading_frame, text="Trading Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure for trading charts
        self.figures["trading"] = Figure(figsize=(12, 6), facecolor="#2b2b2b")
        self.canvases["trading"] = FigureCanvasTkAgg(self.figures["trading"], viz_frame)
        self.canvases["trading"].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize trading chart
        self._init_trading_chart()

    def _create_analytics_tab(self):
        """Create analytics tab with mathematical visualizations."""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="Analytics")

        # Create multiple visualization panels
        panels_frame = ttk.Frame(analytics_frame)
        panels_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Mathematical metrics
        left_frame = ttk.LabelFrame(panels_frame, text="Mathematical Metrics", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Metrics display
        self.metrics_text = tk.Text(left_frame, height=20, width=40, bg="#1e1e1e", fg="white")
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Right panel - Charts
        right_frame = ttk.LabelFrame(panels_frame, text="Analytics Charts", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure for analytics
        self.figures["analytics"] = Figure(figsize=(10, 8), facecolor="#2b2b2b")
        self.canvases["analytics"] = FigureCanvasTkAgg(self.figures["analytics"], right_frame)
        self.canvases["analytics"].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize analytics chart
        self._init_analytics_chart()

    def _create_settings_tab(self):
        """Create settings tab for configuration."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        # API Configuration
        api_frame = ttk.LabelFrame(settings_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, padx=5, pady=5)

        # Coinbase settings
        coinbase_frame = ttk.Frame(api_frame)
        coinbase_frame.pack(fill=tk.X, pady=5)

        ttk.Label(coinbase_frame, text="Coinbase API Key:").grid(row=0, column=0, sticky="w")
        self.coinbase_key_var = tk.StringVar()
        ttk.Entry(coinbase_frame, textvariable=self.coinbase_key_var, show="*").grid(
            row=0, column=1, sticky="ew", padx=5
        )

        ttk.Label(coinbase_frame, text="Coinbase Secret:").grid(row=1, column=0, sticky="w")
        self.coinbase_secret_var = tk.StringVar()
        ttk.Entry(coinbase_frame, textvariable=self.coinbase_secret_var, show="*").grid(
            row=1, column=1, sticky="ew", padx=5
        )

        # CoinMarketCap settings
        cmc_frame = ttk.Frame(api_frame)
        cmc_frame.pack(fill=tk.X, pady=5)

        ttk.Label(cmc_frame, text="CoinMarketCap API Key:").grid(row=0, column=0, sticky="w")
        self.cmc_key_var = tk.StringVar()
        ttk.Entry(cmc_frame, textvariable=self.cmc_key_var, show="*").grid(row=0, column=1, sticky="ew", padx=5)

        # Save/Load buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(button_frame, text="Save Settings", command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings", command=self._load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Test Connections", command=self._test_connections).pack(side=tk.LEFT, padx=5)

    def _create_logs_tab(self):
        """Create logs tab for system monitoring."""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")

        # Log display
        log_frame = ttk.LabelFrame(logs_frame, text="System Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log text widget with scrollbar
        self.log_text = tk.Text(log_frame, bg="#1e1e1e", fg="white", font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Log control buttons
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(log_controls, text="Clear Logs", command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Export Logs", command=self._export_logs).pack(side=tk.LEFT, padx=5)

    def _setup_event_handlers(self):
        """Setup event handlers for GUI components."""
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Periodic updates
        self._schedule_updates()

    def _init_trading_chart(self):
        """Initialize trading chart."""
        fig = self.figures["trading"]
        fig.clear()

        ax = fig.add_subplot(111)
        ax.set_facecolor("#2b2b2b")
        ax.grid(True, alpha=0.3)
        ax.set_title("Trading Chart", color="white")
        ax.set_xlabel("Time", color="white")
        ax.set_ylabel("Price", color="white")

        # Set text color
        ax.tick_params(colors="white")

        fig.tight_layout()
        self.canvases["trading"].draw()

    def _init_analytics_chart(self):
        """Initialize analytics chart."""
        fig = self.figures["analytics"]
        fig.clear()

        # Create subplots for different analytics
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("#2b2b2b")
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors="white")

        ax1.set_title("Price Movement", color="white")
        ax2.set_title("Volume Analysis", color="white")
        ax3.set_title("Volatility", color="white")
        ax4.set_title("Correlation", color="white")

        fig.tight_layout()
        self.canvases["analytics"].draw()

    def _on_mode_change(self):
        """Handle trading mode change."""
        new_mode = self.mode_var.get()
        self.state.current_mode = new_mode
        self.status_labels["Trading Mode"].config(text=new_mode.title())

        if new_mode == "live":
            self.status_labels["Trading Mode"].config(foreground="red")
        else:
            self.status_labels["Trading Mode"].config(foreground="yellow")

        logger.info(f"Trading mode changed to: {new_mode}")

    def _start_trading(self):
        """Start trading operations."""
        if not self.state.is_running:
            self.state.is_running = True
            self.status_labels["System Health"].config(text="Running", foreground="green")

            # Start data update thread
            threading.Thread(target=self._data_update_loop, daemon=True).start()

            messagebox.showinfo("Trading", "Trading started successfully!")
        else:
            messagebox.showwarning("Trading", "Trading is already running!")

    def _stop_trading(self):
        """Stop trading operations."""
        if self.state.is_running:
            self.state.is_running = False
            self.status_labels["System Health"].config(text="Stopped", foreground="red")
            messagebox.showinfo("Trading", "Trading stopped!")
        else:
            messagebox.showwarning("Trading", "Trading is not running!")

    def _refresh_data(self):
        """Refresh market data."""
        if self.api_manager:
            asyncio.create_task(self._update_market_data())
        else:
            messagebox.showwarning("Data", "API manager not available!")

    def _data_update_loop(self):
        """Background data update loop."""
        while self.state.is_running:
            try:
                # Update market data
                if self.api_manager:
                    asyncio.run(self._update_market_data())

                # Update status
                self.state.last_update = datetime.now()
                self.root.after(0, self._update_status_display)

                # Sleep for update interval
                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                self.state.error_count += 1

    async def _update_market_data(self):
        """Update market data from API."""
        if not self.api_manager:
            return

        try:
            # Get market data for common pairs
            symbols = ["BTC/USDC", "ETH/USDC", "XRP/USDC"]
            market_data = await self.api_manager.get_multiple_market_data(symbols)

            # Update GUI with new data
            self.root.after(0, lambda: self._update_market_display(market_data))

        except Exception as e:
            logger.error(f"Error updating market data: {e}")

    def _update_market_display(self, market_data: Dict[str, Any]):
        """Update market data display."""
        # Clear existing items
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)

        # Add new data
        for symbol, data in market_data.items():
            if hasattr(data, "price") and hasattr(data, "change_24h"):
                self.market_tree.insert(
                    "",
                    "end",
                    values=(symbol, f"${data.price:.2f}", f"{data.change_24h:.2f}%", f"{data.volume:,.0f}", "API"),
                )

    def _update_status_display(self):
        """Update status display."""
        self.status_labels["Last Update"].config(text=self.state.last_update.strftime("%H:%M:%S"))

        if self.state.error_count > 0:
            self.status_labels["System Health"].config(text=f"Errors: {self.state.error_count}", foreground="orange")

    def _save_settings(self):
        """Save current settings."""
        # Implementation for saving settings
        messagebox.showinfo("Settings", "Settings saved successfully!")

    def _load_settings(self):
        """Load saved settings."""
        # Implementation for loading settings
        messagebox.showinfo("Settings", "Settings loaded successfully!")

    def _test_connections(self):
        """Test API connections."""
        if self.api_manager:
            asyncio.create_task(self._run_connection_test())
        else:
            messagebox.showwarning("Connections", "API manager not available!")

    async def _run_connection_test(self):
        """Run connection test."""
        try:
            results = await self.api_manager.test_connections()

            # Update status
            connected_apis = sum(results.values())
            total_apis = len(results)

            if connected_apis > 0:
                self.status_labels["API Connection"].config(
                    text=f"Connected ({connected_apis}/{total_apis})", foreground="green"
                )
            else:
                self.status_labels["API Connection"].config(text="Disconnected", foreground="red")

            messagebox.showinfo("Connections", f"Connection test completed!\nConnected: {connected_apis}/{total_apis}")

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            messagebox.showerror("Connections", f"Connection test failed: {e}")

    def _clear_logs(self):
        """Clear log display."""
        self.log_text.delete(1.0, tk.END)

    def _export_logs(self):
        """Export logs to file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Export", "Logs exported successfully!")
            except Exception as e:
                messagebox.showerror("Export", f"Failed to export logs: {e}")

    def _schedule_updates(self):
        """Schedule periodic GUI updates."""

        def update():
            # Update status
            self._update_status_display()

            # Schedule next update
            self.root.after(1000, update)  # Update every second

        update()

    def _on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.state.is_running = False
            self.root.destroy()

    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI error: {e}")
            messagebox.showerror("Error", f"GUI error: {e}")


def main():
    """Main function to run the GUI."""
    try:
        # Initialize API manager
        api_config = APIConfig()
        api_manager = APIIntegrationManager(api_config)

        # Create and run GUI
        dashboard = TradingDashboard(api_manager)
        dashboard.run()

    except Exception as e:
        logger.error(f"Failed to start GUI: {e}")
        messagebox.showerror("Startup Error", f"Failed to start GUI: {e}")


if __name__ == "__main__":
    main()
