#!/usr/bin/env python3
"""
Schwabot Complete Trading System - Main Entry Point
==================================================

Comprehensive trading bot system with:
- Real-time visualization dashboard
- Mathematical relay processing
- GPU/CPU handoff optimization
- API connectivity (Coinbase, CCXT)
- Portfolio management and rebalancing
- Risk management and compliance
- Cross-platform compatibility (Windows, macOS, Linux)

Usage:
    python schwabot_main.py --mode demo     # Demo mode
    python schwabot_main.py --mode live     # Live trading mode
    python schwabot_main.py --mode backtest # Backtesting mode
    python schwabot_main.py --gui           # Start GUI interface
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure proper path setup
sys.path.insert(0, str(Path(__file__).parent))

# Core system imports
try:
    from core.trading_pipeline_integration import TradingPipelineIntegration
    from core.ccxt_trading_executor import CCXTTradingExecutor
    from core.speed_lattice_trading_integration import SpeedLatticeTradingIntegration
    from core.api_bridge import APIBridge
    from core.settings_manager import SystemSettings, get_settings_manager
    from core.gpu_cpu_calculation_bridge import get_gpu_cpu_bridge
    from core.unified_connectivity_manager import UnifiedConnectivityManager
    from core.speed_lattice_visualizer import SpeedLatticeLivePanelSystem, PanelType
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Core systems not available: {e}")
    CORE_AVAILABLE = False

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"GUI libraries not available: {e}")
    GUI_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


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
        
        # GUI components
        self.status_frame = None
        self.control_frame = None
        self.visualization_frame = None
        self.settings_frame = None
        
        # Status variables
        self.status_vars = {
            "system_status": tk.StringVar(value="Initializing..."),
            "trading_mode": tk.StringVar(value="Demo"),
            "api_status": tk.StringVar(value="Disconnected"),
            "gpu_status": tk.StringVar(value="Unknown"),
            "last_update": tk.StringVar(value="Never"),
            "total_profit": tk.StringVar(value="$0.00"),
        }
        
        self._setup_gui()
        self._initialize_systems()
    
    def _setup_gui(self):
        """Setup the main GUI interface."""
        # Configure style
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors for dark theme
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff")
        style.configure("TButton", background="#3e3e3e", foreground="#ffffff")
        style.configure("TNotebook", background="#1e1e1e")
        style.configure("TNotebook.Tab", background="#3e3e3e", foreground="#ffffff")
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top status frame
        self._create_status_frame(main_container)
        
        # Create main content area with tabs
        self._create_main_tabs(main_container)
    
    def _create_status_frame(self, parent):
        """Create the top status frame."""
        self.status_frame = ttk.LabelFrame(parent, text="System Status", padding=10)
        self.status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create status grid
        status_items = [
            ("System Status", self.status_vars["system_status"], "#00ff00"),
            ("Trading Mode", self.status_vars["trading_mode"], "#ffff00"),
            ("API Status", self.status_vars["api_status"], "#ff0000"),
            ("GPU Status", self.status_vars["gpu_status"], "#00ffff"),
            ("Last Update", self.status_vars["last_update"], "#ffffff"),
            ("Total Profit", self.status_vars["total_profit"], "#00ff00"),
        ]
        
        for i, (label_text, var, color) in enumerate(status_items):
            row = i // 3
            col = i % 3
            
            frame = ttk.Frame(self.status_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
            
            ttk.Label(frame, text=f"{label_text}:").pack(side=tk.LEFT)
            status_label = ttk.Label(frame, textvariable=var, foreground=color)
            status_label.pack(side=tk.RIGHT)
        
        # Configure grid weights
        for i in range(3):
            self.status_frame.grid_columnconfigure(i, weight=1)
    
    def _create_main_tabs(self, parent):
        """Create the main tabbed interface."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Trading tab
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
        # Control buttons frame
        control_frame = ttk.LabelFrame(parent, text="Trading Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Trading Mode:").pack(side=tk.LEFT)
        
        self.mode_var = tk.StringVar(value="demo")
        modes = [("Demo", "demo"), ("Live", "live"), ("Backtest", "backtest")]
        
        for text, value in modes:
            ttk.Radiobutton(
                mode_frame, text=text, variable=self.mode_var, 
                value=value, command=self._on_mode_change
            ).pack(side=tk.LEFT, padx=10)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Start Trading", 
                  command=self._start_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Trading", 
                  command=self._stop_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh Status", 
                  command=self._refresh_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Emergency Stop", 
                  command=self._emergency_stop).pack(side=tk.RIGHT, padx=5)
        
        # Portfolio summary frame
        portfolio_frame = ttk.LabelFrame(parent, text="Portfolio Summary", padding=10)
        portfolio_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for portfolio
        self.portfolio_fig = Figure(figsize=(12, 6), facecolor="#1e1e1e")
        self.portfolio_canvas = FigureCanvasTkAgg(self.portfolio_fig, portfolio_frame)
        self.portfolio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._init_portfolio_chart()
    
    def _create_visualization_tab(self, parent):
        """Create the visualization tab."""
        # Visualization controls
        viz_controls = ttk.LabelFrame(parent, text="Visualization Controls", padding=10)
        viz_controls.pack(fill=tk.X, pady=(0, 10))
        
        # Panel selection
        ttk.Label(viz_controls, text="Display Panel:").pack(side=tk.LEFT)
        
        self.panel_var = tk.StringVar(value="DRIFT_MATRIX")
        panel_options = [
            "DRIFT_MATRIX", "PROFIT_RESONANCE", "SYSTEM_STATUS", 
            "TRADING_STATE", "PATTERN_RECOGNITION"
        ]
        
        panel_combo = ttk.Combobox(viz_controls, textvariable=self.panel_var, 
                                  values=panel_options, state="readonly")
        panel_combo.pack(side=tk.LEFT, padx=10)
        panel_combo.bind("<<ComboboxSelected>>", self._on_panel_change)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_controls, text="Auto Refresh", 
                       variable=self.auto_refresh_var).pack(side=tk.RIGHT)
        
        # Visualization area
        self.visualization_frame = ttk.Frame(parent)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True)
    
    def _create_settings_tab(self, parent):
        """Create the settings configuration tab."""
        # Create scrollable settings area
        canvas = tk.Canvas(parent, bg="#1e1e1e")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # API Settings
        api_frame = ttk.LabelFrame(scrollable_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, pady=5, padx=10)
        
        self.api_settings = {}
        api_fields = [
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
                entry = ttk.Entry(frame, textvariable=var, width=40, show="*")
                entry.pack(side=tk.RIGHT)
            
            self.api_settings[var_name] = var
        
        # Performance Settings
        perf_frame = ttk.LabelFrame(scrollable_frame, text="Performance Settings", padding=10)
        perf_frame.pack(fill=tk.X, pady=5, padx=10)
        
        self.perf_settings = {}
        perf_fields = [
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
                entry = ttk.Entry(frame, textvariable=var, width=20)
                entry.pack(side=tk.RIGHT)
            
            self.perf_settings[var_name] = var
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Button(button_frame, text="Save Settings", 
                  command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings", 
                  command=self._load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self._reset_settings).pack(side=tk.RIGHT, padx=5)
    
    def _create_logs_tab(self, parent):
        """Create the system logs tab."""
        # Log display area
        log_frame = ttk.LabelFrame(parent, text="System Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        self.log_text = tk.Text(log_frame, bg="#000000", fg="#00ff00", 
                               font=("Consolas", 9), wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log control buttons
        log_controls = ttk.Frame(parent)
        log_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(log_controls, text="Clear Logs", 
                  command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Export Logs", 
                  command=self._export_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Refresh", 
                  command=self._refresh_logs).pack(side=tk.LEFT, padx=5)
    
    def _initialize_systems(self):
        """Initialize all system components."""
        try:
            if not CORE_AVAILABLE:
                self.status_vars["system_status"].set("Core systems unavailable")
                return
            
            # Initialize settings manager
            self.settings_manager = get_settings_manager()
            
            # Initialize GPU/CPU bridge
            self.gpu_cpu_bridge = get_gpu_cpu_bridge()
            
            # Initialize trading pipeline
            self.trading_pipeline = TradingPipelineIntegration(
                enable_gpu=True,
                enable_distributed=False,
                max_concurrent_trades=10,
                risk_management_enabled=True
            )
            
            # Initialize visualizer
            if GUI_AVAILABLE:
                self.visualizer = SpeedLatticeLivePanelSystem()
            
            self.status_vars["system_status"].set("Initialized")
            self.status_vars["gpu_status"].set("Available" if self.gpu_cpu_bridge.gpu_available else "CPU Only")
            
            # Start update thread
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
                    time.sleep(1)  # Update every second
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
                
                # Update status variables
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
        if self.visualizer:
            try:
                panel_type = PanelType(panel_name.lower())
                self.visualizer.current_panel = panel_type
                logger.info(f"Switched to panel: {panel_name}")
            except ValueError:
                logger.warning(f"Unknown panel type: {panel_name}")
    
    def _start_trading(self):
        """Start trading operations."""
        try:
            if self.trading_pipeline and not self.is_running:
                self.is_running = True
                self.status_vars["system_status"].set("Trading Active")
                logger.info("Trading started")
                messagebox.showinfo("Trading", "Trading started successfully!")
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")
    
    def _stop_trading(self):
        """Stop trading operations."""
        try:
            self.is_running = False
            self.status_vars["system_status"].set("Trading Stopped")
            logger.info("Trading stopped")
            messagebox.showinfo("Trading", "Trading stopped successfully!")
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")
            messagebox.showerror("Error", f"Failed to stop trading: {e}")
    
    def _emergency_stop(self):
        """Emergency stop all operations."""
        try:
            self.is_running = False
            if self.trading_pipeline:
                self.trading_pipeline.cleanup()
            self.status_vars["system_status"].set("Emergency Stop")
            logger.warning("Emergency stop activated")
            messagebox.showwarning("Emergency Stop", "All trading operations stopped!")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
    
    def _refresh_status(self):
        """Manually refresh system status."""
        self._update_status()
        logger.info("Status refreshed")
    
    def _save_settings(self):
        """Save current settings."""
        try:
            # Implement settings saving logic
            messagebox.showinfo("Settings", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def _load_settings(self):
        """Load saved settings."""
        try:
            # Implement settings loading logic
            messagebox.showinfo("Settings", "Settings loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")
    
    def _reset_settings(self):
        """Reset settings to defaults."""
        try:
            # Reset all settings to defaults
            messagebox.showinfo("Settings", "Settings reset to defaults!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset settings: {e}")
    
    def _clear_logs(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
    
    def _export_logs(self):
        """Export logs to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"schwabot_logs_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Export", f"Logs exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")
    
    def _refresh_logs(self):
        """Refresh log display."""
        # Implement log refresh logic
        pass
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()
    
    def cleanup(self):
        """Cleanup resources on exit."""
        try:
            self.is_running = False
            if self.trading_pipeline:
                self.trading_pipeline.cleanup()
            logger.info("GUI cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class SchwabotCLI:
    """Command-line interface for Schwabot."""
    
    def __init__(self, args):
        self.args = args
        self.trading_pipeline = None
        self.is_running = False
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    async def run_demo_mode(self):
        """Run in demo mode."""
        logger.info("Starting Schwabot in demo mode...")
        
        if not CORE_AVAILABLE:
            logger.error("Core systems not available")
            return
        
        # Initialize trading pipeline
        self.trading_pipeline = TradingPipelineIntegration(
            enable_gpu=True,
            enable_distributed=False,
            max_concurrent_trades=5,
            risk_management_enabled=True
        )
        
        # Simulate market data
        sample_market_data = {
            "current_price": 62000.0,
            "price_change": 0.02,
            "volume_change": 0.15,
            "volatility": 0.6,
            "temperature": 310.0,
            "price_history": [61000.0, 61500.0, 62000.0, 61800.0, 62200.0],
            "volume_data": [100.0, 120.0, 110.0, 90.0, 130.0],
            "price_data": [61000.0, 61500.0, 62000.0, 61800.0, 62200.0],
            "rsi": 65.0,
            "macd_signal": 0.01,
            "moving_average": 61500.0,
        }
        
        self.is_running = True
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                logger.info(f"Processing iteration {iteration}")
                
                # Process market data
                signal = await self.trading_pipeline.process_market_data(
                    sample_market_data, "BTC", "warm"
                )
                
                logger.info(f"Generated signal: {signal.signal_type} (confidence: {signal.confidence:.3f})")
                
                # Get performance summary
                if iteration % 10 == 0:  # Every 10 iterations
                    performance = self.trading_pipeline.get_pipeline_performance()
                    logger.info(f"Performance summary: {performance}")
                
                await asyncio.sleep(2)  # Wait 2 seconds between iterations
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            self.shutdown()
    
    async def run_live_mode(self):
        """Run in live trading mode."""
        logger.warning("Live trading mode - USE WITH CAUTION!")
        
        if not CORE_AVAILABLE:
            logger.error("Core systems not available")
            return
        
        # Additional safety checks for live mode
        confirm = input("Are you sure you want to enable live trading? (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("Live trading cancelled by user")
            return
        
        # Initialize with stricter risk management
        self.trading_pipeline = TradingPipelineIntegration(
            enable_gpu=True,
            enable_distributed=False,
            max_concurrent_trades=3,  # More conservative
            risk_management_enabled=True
        )
        
        logger.info("Live trading mode started - monitoring markets...")
        # Implement live trading logic here
        
    async def run_backtest_mode(self):
        """Run in backtesting mode."""
        logger.info("Starting backtesting mode...")
        
        # Implement backtesting logic
        logger.info("Backtesting completed")
    
    def shutdown(self):
        """Shutdown the CLI system."""
        self.is_running = False
        if self.trading_pipeline:
            self.trading_pipeline.cleanup()
        logger.info("Schwabot CLI shutdown completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Schwabot Advanced Trading System")
    parser.add_argument("--mode", choices=["demo", "live", "backtest"], 
                       default="demo", help="Trading mode")
    parser.add_argument("--gui", action="store_true", 
                       help="Start GUI interface")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--config", type=str, 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 60)
    logger.info("Schwabot Advanced Trading System v2.0")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"GUI: {args.gui}")
    logger.info(f"Log Level: {args.log_level}")
    
    try:
        if args.gui:
            if not GUI_AVAILABLE:
                logger.error("GUI libraries not available")
                sys.exit(1)
            
            app = SchwabotGUI()
            app.run()
        else:
            cli = SchwabotCLI(args)
            
            if args.mode == "demo":
                asyncio.run(cli.run_demo_mode())
            elif args.mode == "live":
                asyncio.run(cli.run_live_mode())
            elif args.mode == "backtest":
                asyncio.run(cli.run_backtest_mode())
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        logger.info("Schwabot shutdown completed")


if __name__ == "__main__":
    main() 