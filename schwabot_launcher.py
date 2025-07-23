#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Launcher - Professional Trading System Launcher
=======================================================

Provides a comprehensive interface for launching Schwabot with different configurations
and managing system settings including API keys, trading parameters, and system health.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import the API key manager
from api_key_manager import show_api_configuration
# Import the USB manager
from usb_manager import auto_detect_usb, setup_usb_storage, get_usb_status
# Import the Advanced Options GUI
from advanced_options_gui import show_advanced_options
# Import advanced scheduler
try:
    from advanced_scheduler import get_advanced_scheduler, start_advanced_scheduler, stop_advanced_scheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

# Import visual controls
try:
    from visual_controls_gui import show_visual_controls
    VISUAL_CONTROLS_AVAILABLE = True
except ImportError:
    VISUAL_CONTROLS_AVAILABLE = False
    logger.warning("Visual Controls GUI not available")

# Import unified live backtesting
try:
    from unified_live_backtesting_system import BacktestConfig, BacktestMode, start_live_backtest
    UNIFIED_BACKTESTING_AVAILABLE = True
except ImportError:
    UNIFIED_BACKTESTING_AVAILABLE = False
    logger.warning("Unified Live Backtesting not available")

logger = logging.getLogger(__name__)

class SchwabotLauncher:
    """Professional Schwabot launcher with comprehensive configuration options."""
    
    def __init__(self):
        self.root = None
        self.config_file = "AOI_Base_Files_Schwabot/config/launcher_config.json"
        self.config = self._load_config()
        
        # Check if advanced options should be shown
        self.should_show_advanced_options = self.config.get('show_advanced_options_prompt', True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load launcher configuration."""
        default_config = {
            "last_mode": "demo",
            "auto_start": False,
            "check_updates": True,
            "system_health_check": True,
            "api_keys_configured": False,
            "trading_enabled": False,
            "notification_enabled": False,
            "show_advanced_options_prompt": True,
            "compression_enabled": False
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return {**default_config, **json.load(f)}
            return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return default_config
    
    def _save_config(self):
        """Save launcher configuration."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def show_launcher(self):
        """Show the main launcher interface."""
        self.root = tk.Tk()
        self.root.title("🚀 Schwabot Professional Trading System")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        self._configure_styles()
        
        # Create main interface
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
        
        # Start system health check
        if self.config["system_health_check"]:
            self._check_system_health()
        
        # Initialize advanced scheduler
        if SCHEDULER_AVAILABLE:
            self._initialize_advanced_scheduler()
        
        # Center window
        self.root.transient(self.parent) if self.parent else None
        self.root.grab_set()
        self.root.mainloop()
    
    def _configure_styles(self):
        """Configure GUI styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       font=('Arial', 18, 'bold'), 
                       foreground='#00ff00', 
                       background='#1e1e1e')
        
        style.configure('Header.TLabel', 
                       font=('Arial', 14, 'bold'), 
                       foreground='#ffffff', 
                       background='#1e1e1e')
        
        style.configure('Info.TLabel', 
                       font=('Arial', 10), 
                       foreground='#cccccc', 
                       background='#1e1e1e')
        
        style.configure('Status.TLabel', 
                       font=('Arial', 9), 
                       foreground='#888888', 
                       background='#1e1e1e')
        
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', 
                       background='#404040', 
                       foreground='white', 
                       padding=[15, 8])
        
        style.map('TNotebook.Tab',
                 background=[('selected', '#00aa00'), ('active', '#006600')])
    
    def _create_header(self):
        """Create the header section."""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=20, pady=10)
        
        # Title
        title_label = ttk.Label(header_frame, 
                               text="🚀 Schwabot Professional Trading System", 
                               style='Title.TLabel')
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, 
                                  text="Advanced AI-Powered Cryptocurrency Trading with Dynamic Timing", 
                                  style='Info.TLabel')
        subtitle_label.pack(pady=5)
        
        # Version info
        version_label = ttk.Label(header_frame, 
                                 text="Version 2.0 | Dynamic Timing System | USDC-Based Trading", 
                                 style='Status.TLabel')
        version_label.pack()
    
    def _create_main_content(self):
        """Create the main content area."""
        # Create notebook for different sections
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs
        self._create_launch_tab()
        self._create_configuration_tab()
        self._create_api_keys_tab()
        self._create_usb_tab()
        self._create_advanced_options_tab()
        self._create_visual_controls_tab()  # Add visual controls tab
        self._create_system_tab()
        self._create_help_tab()
        
        # Auto-detect USB on startup
        self._auto_detect_usb()
        
        # Show advanced options prompt if needed
        if self.should_show_advanced_options:
            self._show_advanced_options_prompt()
    
    def _auto_detect_usb(self):
        """Automatically detect USB drives and offer setup."""
        try:
            # Run USB detection in a separate thread to avoid blocking the UI
            def detect_thread():
                time.sleep(2)  # Wait for UI to load
                auto_detect_usb()
                # Update USB status after detection
                self.root.after(1000, self._update_usb_status)
            
            threading.Thread(target=detect_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in auto USB detection: {e}")
    
    def _create_launch_tab(self):
        """Create the launch options tab."""
        launch_frame = ttk.Frame(self.notebook)
        self.notebook.add(launch_frame, text="🚀 Launch")
        
        # Title
        title = ttk.Label(launch_frame, text="Launch Options", style='Header.TLabel')
        title.pack(pady=10)
        
        # Launch options frame
        options_frame = ttk.Frame(launch_frame)
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Demo Mode
        demo_frame = self._create_launch_option(
            options_frame, 
            "🎮 Demo Mode", 
            "Safe testing environment with simulated trading",
            "Perfect for learning and testing strategies without real money",
            "Launch demo mode with simulated market data",
            self._launch_demo_mode
        )
        demo_frame.pack(fill='x', pady=5)
        
        # Web Dashboard
        web_frame = self._create_launch_option(
            options_frame, 
            "🌐 Web Dashboard", 
            "Human-friendly web interface with real-time updates",
            "Mobile-responsive dashboard with trading recommendations",
            "Launch web dashboard on localhost:8080",
            self._launch_web_dashboard
        )
        web_frame.pack(fill='x', pady=5)
        
        # GUI Visualizer
        gui_frame = self._create_launch_option(
            options_frame, 
            "📊 GUI Visualizer", 
            "Advanced desktop visualizer with real-time charts",
            "Professional charts, system monitoring, and event logging",
            "Launch advanced desktop visualizer",
            self._launch_gui_visualizer
        )
        gui_frame.pack(fill='x', pady=5)
        
        # CLI Interface
        cli_frame = self._create_launch_option(
            options_frame, 
            "💻 CLI Interface", 
            "Command-line interface for power users",
            "Detailed system output, performance testing, and advanced configuration",
            "Launch command-line interface",
            self._launch_cli_interface
        )
        cli_frame.pack(fill='x', pady=5)
        
        # Live Trading (requires API keys)
        live_frame = self._create_launch_option(
            options_frame, 
            "💰 Live Trading", 
            "Real market execution with actual trading",
            "REQUIRES API KEYS - Execute real trades with real money",
            "Launch live trading system (requires configuration)",
            self._launch_live_trading,
            requires_api_keys=True
        )
        live_frame.pack(fill='x', pady=5)

        # Unified Live Backtesting
        backtest_frame = self._create_launch_option(
            options_frame,
            "🧠 Unified Live Backtesting",
            "Simulate trading with historical data and real-time market updates",
            "Run a live backtest with historical data and real-time market simulation",
            "Launch unified live backtesting system",
            self._launch_unified_backtesting,
            requires_api_keys=True
        )
        backtest_frame.pack(fill='x', pady=5)
    
    def _create_launch_option(self, parent, title: str, description: str, 
                            details: str, button_text: str, command, requires_api_keys: bool = False):
        """Create a launch option frame."""
        frame = ttk.Frame(parent)
        frame.configure(relief='solid', borderwidth=1)
        
        # Header
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = ttk.Label(header_frame, text=title, style='Header.TLabel')
        title_label.pack(side='left')
        
        if requires_api_keys:
            api_warning = ttk.Label(header_frame, text="🔑 Requires API Keys", 
                                  style='Info.TLabel', foreground='#ffaa00')
            api_warning.pack(side='right')
        
        # Description
        desc_label = ttk.Label(frame, text=description, style='Info.TLabel')
        desc_label.pack(anchor='w', padx=10, pady=2)
        
        # Details
        details_label = ttk.Label(frame, text=details, style='Status.TLabel')
        details_label.pack(anchor='w', padx=10, pady=2)
        
        # Button
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        launch_btn = tk.Button(button_frame, text=button_text, command=command,
                             bg='#00aa00', fg='white', font=('Arial', 11, 'bold'),
                             padx=20, pady=5)
        launch_btn.pack(side='left')
        
        return frame
    
    def _create_configuration_tab(self):
        """Create the configuration tab."""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        title = ttk.Label(config_frame, text="System Configuration", style='Header.TLabel')
        title.pack(pady=10)
        
        # Configuration options
        config_options_frame = ttk.Frame(config_frame)
        config_options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Auto-start option
        auto_start_var = tk.BooleanVar(value=self.config["auto_start"])
        auto_start_cb = tk.Checkbutton(config_options_frame, 
                                     text="🚀 Auto-start system on launch",
                                     variable=auto_start_var,
                                     command=lambda: self._update_config("auto_start", auto_start_var.get()),
                                     bg='#1e1e1e', fg='white', selectcolor='#404040',
                                     font=('Arial', 11))
        auto_start_cb.pack(anchor='w', pady=5)
        
        # Check updates option
        updates_var = tk.BooleanVar(value=self.config["check_updates"])
        updates_cb = tk.Checkbutton(config_options_frame, 
                                  text="🔄 Check for system updates",
                                  variable=updates_var,
                                  command=lambda: self._update_config("check_updates", updates_var.get()),
                                  bg='#1e1e1e', fg='white', selectcolor='#404040',
                                  font=('Arial', 11))
        updates_cb.pack(anchor='w', pady=5)
        
        # System health check option
        health_var = tk.BooleanVar(value=self.config["system_health_check"])
        health_cb = tk.Checkbutton(config_options_frame, 
                                 text="🏥 Perform system health checks",
                                 variable=health_var,
                                 command=lambda: self._update_config("system_health_check", health_var.get()),
                                 bg='#1e1e1e', fg='white', selectcolor='#404040',
                                 font=('Arial', 11))
        health_cb.pack(anchor='w', pady=5)
        
        # Trading configuration button
        trading_btn = tk.Button(config_options_frame, 
                              text="📈 Configure Trading Parameters",
                              command=self._configure_trading,
                              bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                              padx=20, pady=5)
        trading_btn.pack(anchor='w', pady=10)
        
        # Notification configuration button
        notification_btn = tk.Button(config_options_frame, 
                                   text="📱 Configure Notifications",
                                   command=self._configure_notifications,
                                   bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                                   padx=20, pady=5)
        notification_btn.pack(anchor='w', pady=5)
    
    def _create_api_keys_tab(self):
        """Create the API keys management tab."""
        api_frame = ttk.Frame(self.notebook)
        self.notebook.add(api_frame, text="🔑 API Keys")
        
        title = ttk.Label(api_frame, text="API Key Management", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(api_frame, 
                        text="Configure API keys for trading exchanges, AI services, and monitoring systems.\nEach service is clearly labeled with setup instructions.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=10)
        
        # API configuration button
        api_btn = tk.Button(api_frame, 
                           text="🔑 Configure API Keys",
                           command=self._configure_api_keys,
                           bg='#00aa00', fg='white', font=('Arial', 14, 'bold'),
                           padx=30, pady=10)
        api_btn.pack(pady=20)
        
        # API status
        status_frame = ttk.Frame(api_frame)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.api_status_label = ttk.Label(status_frame, 
                                         text="API Keys: Not Configured", 
                                         style='Info.TLabel',
                                         foreground='#ff6666')
        self.api_status_label.pack()
        
        # Update API status
        self._update_api_status()
    
    def _create_usb_tab(self):
        """Create the USB management tab."""
        usb_frame = ttk.Frame(self.notebook)
        self.notebook.add(usb_frame, text="💾 USB Storage")
        
        title = ttk.Label(usb_frame, text="USB Storage Management", style='Header.TLabel')
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(usb_frame, 
                        text="Manage external USB drives for data storage and backup.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=10)
        
        # USB status
        status_frame = ttk.Frame(usb_frame)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.usb_status_label = ttk.Label(status_frame, 
                                         text="USB Status: Not Available", 
                                         style='Info.TLabel',
                                         foreground='#ff6666')
        self.usb_status_label.pack()
        
        # USB setup button
        setup_btn = tk.Button(usb_frame, 
                              text="📁 Setup USB Storage",
                              command=self._setup_usb_storage,
                              bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                              padx=20, pady=5)
        setup_btn.pack(anchor='w', pady=10)
    
    def _create_advanced_options_tab(self):
        """Create the advanced options tab."""
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="⚙️ Advanced Options")
        
        title = ttk.Label(advanced_frame, text="Advanced Options", style='Header.TLabel')
        title.pack(pady=10)
        
        # Advanced options frame
        options_frame = ttk.Frame(advanced_frame)
        options_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Compression option
        compression_var = tk.BooleanVar(value=self.config["compression_enabled"])
        compression_cb = tk.Checkbutton(options_frame, 
                                      text="🔗 Enable Compression for Data Transfer",
                                      variable=compression_var,
                                      command=lambda: self._update_config("compression_enabled", compression_var.get()),
                                      bg='#1e1e1e', fg='white', selectcolor='#404040',
                                      font=('Arial', 11))
        compression_cb.pack(anchor='w', pady=5)
        
        # Advanced options button
        advanced_options_btn = tk.Button(options_frame, 
                                        text="🔧 Show Advanced Options GUI",
                                        command=self._show_advanced_options_gui,
                                        bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                                        padx=20, pady=5)
        advanced_options_btn.pack(anchor='w', pady=10)
    
    def _create_visual_controls_tab(self):
        """Create the visual controls tab."""
        visual_frame = ttk.Frame(self.notebook)
        self.notebook.add(visual_frame, text="🎨 Visual Controls")

        title = ttk.Label(visual_frame, text="Visual Controls", style='Header.TLabel')
        title.pack(pady=10)

        # Description
        desc = ttk.Label(visual_frame, 
                        text="Fine-tune the visual appearance of the trading interface.",
                        style='Info.TLabel', justify='center')
        desc.pack(pady=10)

        # Visual controls button
        visual_btn = tk.Button(visual_frame, 
                               text="🎨 Configure Visual Settings",
                               command=self._show_visual_controls,
                               bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                               padx=20, pady=5)
        visual_btn.pack(anchor='w', pady=10)

    def _create_system_tab(self):
        """Create the system information tab."""
        system_frame = ttk.Frame(self.notebook)
        self.notebook.add(system_frame, text="💻 System")
        
        title = ttk.Label(system_frame, text="System Information", style='Header.TLabel')
        title.pack(pady=10)
        
        # System info frame
        info_frame = ttk.Frame(system_frame)
        info_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # System health
        health_frame = ttk.Frame(info_frame)
        health_frame.pack(fill='x', pady=5)
        
        health_label = ttk.Label(health_frame, text="🏥 System Health:", style='Header.TLabel')
        health_label.pack(side='left')
        
        self.health_status_label = ttk.Label(health_frame, text="Checking...", 
                                           style='Info.TLabel', foreground='#ffaa00')
        self.health_status_label.pack(side='right')
        
        # System status button
        status_btn = tk.Button(info_frame, 
                             text="📊 Check System Status",
                             command=self._check_system_status,
                             bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                             padx=20, pady=5)
        status_btn.pack(anchor='w', pady=10)
        
        # Logs button
        logs_btn = tk.Button(info_frame, 
                           text="📋 View System Logs",
                           command=self._view_logs,
                           bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                           padx=20, pady=5)
        logs_btn.pack(anchor='w', pady=5)
        
        # Emergency stop button
        emergency_btn = tk.Button(info_frame, 
                                text="🛑 Emergency Stop",
                                command=self._emergency_stop,
                                bg='#aa0000', fg='white', font=('Arial', 11, 'bold'),
                                padx=20, pady=5)
        emergency_btn.pack(anchor='w', pady=10)
    
    def _create_help_tab(self):
        """Create the help and documentation tab."""
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="❓ Help")
        
        title = ttk.Label(help_frame, text="Help & Documentation", style='Header.TLabel')
        title.pack(pady=10)
        
        # Help content
        help_content_frame = ttk.Frame(help_frame)
        help_content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Quick start guide
        quick_start_btn = tk.Button(help_content_frame, 
                                  text="🚀 Quick Start Guide",
                                  command=self._show_quick_start,
                                  bg='#00aa00', fg='white', font=('Arial', 11, 'bold'),
                                  padx=20, pady=5)
        quick_start_btn.pack(anchor='w', pady=5)
        
        # API setup guide
        api_guide_btn = tk.Button(help_content_frame, 
                                text="🔑 API Setup Guide",
                                command=self._show_api_guide,
                                bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                                padx=20, pady=5)
        api_guide_btn.pack(anchor='w', pady=5)
        
        # Troubleshooting
        troubleshoot_btn = tk.Button(help_content_frame, 
                                   text="🔧 Troubleshooting",
                                   command=self._show_troubleshooting,
                                   bg='#0066aa', fg='white', font=('Arial', 11, 'bold'),
                                   padx=20, pady=5)
        troubleshoot_btn.pack(anchor='w', pady=5)
        
        # About
        about_btn = tk.Button(help_content_frame, 
                            text="ℹ️ About Schwabot",
                            command=self._show_about,
                            bg='#666666', fg='white', font=('Arial', 11, 'bold'),
                            padx=20, pady=5)
        about_btn.pack(anchor='w', pady=10)
    
    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Status.TLabel')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Version info
        version_label = ttk.Label(status_frame, text="Schwabot v2.0", style='Status.TLabel')
        version_label.pack(side='right', padx=10, pady=5)
    
    # Launch methods
    def _launch_demo_mode(self):
        """Launch demo mode."""
        self._update_status("Launching demo mode...")
        try:
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/simple_timing_test.py"])
            self._update_status("Demo mode launched successfully")
        except Exception as e:
            self._update_status(f"Failed to launch demo mode: {e}")
            messagebox.showerror("Error", f"Failed to launch demo mode: {e}")
    
    def _launch_web_dashboard(self):
        """Launch web dashboard."""
        self._update_status("Launching web dashboard...")
        try:
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/web/dynamic_timing_dashboard.py"])
            self._update_status("Web dashboard launched - Open http://localhost:8080")
            messagebox.showinfo("Web Dashboard", "Web dashboard launched successfully!\n\nOpen your browser and go to:\nhttp://localhost:8080")
        except Exception as e:
            self._update_status(f"Failed to launch web dashboard: {e}")
            messagebox.showerror("Error", f"Failed to launch web dashboard: {e}")
    
    def _launch_gui_visualizer(self):
        """Launch GUI visualizer."""
        self._update_status("Launching GUI visualizer...")
        try:
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/visualization/dynamic_timing_visualizer.py"])
            self._update_status("GUI visualizer launched successfully")
        except Exception as e:
            self._update_status(f"Failed to launch GUI visualizer: {e}")
            messagebox.showerror("Error", f"Failed to launch GUI visualizer: {e}")
    
    def _launch_cli_interface(self):
        """Launch CLI interface."""
        self._update_status("Launching CLI interface...")
        try:
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/main.py", "--system-status"])
            self._update_status("CLI interface launched successfully")
        except Exception as e:
            self._update_status(f"Failed to launch CLI interface: {e}")
            messagebox.showerror("Error", f"Failed to launch CLI interface: {e}")
    
    def _launch_live_trading(self):
        """Launch live trading (requires API keys)."""
        if not self.config["api_keys_configured"]:
            result = messagebox.askyesno("API Keys Required", 
                                       "Live trading requires API keys to be configured.\n\nWould you like to configure API keys now?")
            if result:
                self._configure_api_keys()
            return
        
        result = messagebox.askyesno("Live Trading Warning", 
                                   "⚠️ WARNING: Live trading will execute real trades with real money!\n\n"
                                   "Are you sure you want to proceed with live trading?")
        if not result:
            return
        
        self._update_status("Launching live trading system...")
        try:
            subprocess.Popen([sys.executable, "AOI_Base_Files_Schwabot/main.py"])
            self._update_status("Live trading system launched")
        except Exception as e:
            self._update_status(f"Failed to launch live trading: {e}")
            messagebox.showerror("Error", f"Failed to launch live trading: {e}")

    def _launch_unified_backtesting(self):
        """Launch unified live backtesting."""
        if not UNIFIED_BACKTESTING_AVAILABLE:
            messagebox.showerror("Error", "Unified Live Backtesting system is not available")
            return
        
        # Show backtesting configuration dialog
        result = messagebox.askyesno("Unified Live Backtesting",
                                   "🎯 This is the OFFICIAL backtesting system that uses LIVE API DATA\n"
                                   "without placing real trades.\n\n"
                                   "This is what you call 'backtesting' - testing strategies on\n"
                                   "real market data without real money.\n\n"
                                   "Would you like to start a live backtesting session?")
        if not result:
            return
        
        self._update_status("Starting unified live backtesting...")
        
        try:
            # Create backtesting configuration
            config = BacktestConfig(
                mode=BacktestMode.LIVE_API_BACKTEST,
                symbols=["BTCUSDT", "ETHUSDT"],
                exchanges=["binance"],
                initial_balance=10000.0,
                backtest_duration_hours=1,  # 1 hour for safety
                enable_ai_analysis=True,
                enable_risk_management=True,
                enable_performance_optimization=True,
                data_update_interval=5.0,
                min_confidence=0.6
            )
            
            # Start backtesting in a separate thread
            def run_backtest():
                try:
                    import asyncio
                    result = asyncio.run(start_live_backtest(config))
                    
                    # Show results
                    messagebox.showinfo("Backtest Completed",
                                      f"🎯 Backtest Results:\n\n"
                                      f"Total Return: {result.total_return:.2f}%\n"
                                      f"Total Trades: {result.total_trades}\n"
                                      f"Win Rate: {result.win_rate:.1f}%\n"
                                      f"Final Balance: ${result.final_balance:,.2f}\n\n"
                                      f"This system used LIVE API DATA without placing real trades!")
                    
                    self._update_status("Unified backtesting completed successfully")
                    
                except Exception as e:
                    self._update_status(f"Backtesting failed: {e}")
                    messagebox.showerror("Error", f"Backtesting failed: {e}")
            
            # Run in separate thread
            threading.Thread(target=run_backtest, daemon=True).start()
            
        except Exception as e:
            self._update_status(f"Failed to start backtesting: {e}")
            messagebox.showerror("Error", f"Failed to start backtesting: {e}")
    
    # Configuration methods
    def _configure_api_keys(self):
        """Open API key configuration."""
        self._update_status("Opening API key configuration...")
        try:
            show_api_configuration()
            self._update_api_status()
            self._update_status("API key configuration completed")
        except Exception as e:
            self._update_status(f"Failed to open API configuration: {e}")
            messagebox.showerror("Error", f"Failed to open API configuration: {e}")
    
    def _configure_trading(self):
        """Configure trading parameters."""
        messagebox.showinfo("Trading Configuration", 
                           "Trading configuration will be implemented in the next update.\n\n"
                           "This will include:\n"
                           "• Risk management settings\n"
                           "• Position sizing parameters\n"
                           "• Trading strategy configuration\n"
                           "• Market selection options")
    
    def _configure_notifications(self):
        """Configure notifications."""
        messagebox.showinfo("Notification Configuration", 
                           "Notification configuration will be implemented in the next update.\n\n"
                           "This will include:\n"
                           "• Email notifications\n"
                           "• Telegram alerts\n"
                           "• Discord webhooks\n"
                           "• SMS alerts")
    
    def _setup_usb_storage(self):
        """Offer to setup USB storage."""
        result = messagebox.askyesno("Setup USB Storage", 
                                   "Would you like to set up an external USB drive for data storage and backup?")
        if result:
            try:
                setup_usb_storage()
                self._update_status("USB storage setup completed")
                self._update_usb_status()  # Refresh USB status
                messagebox.showinfo("USB Setup", "USB storage setup completed successfully!")
            except Exception as e:
                self._update_status(f"Failed to setup USB storage: {e}")
                messagebox.showerror("Error", f"Failed to setup USB storage: {e}")
    
    # Advanced Options methods
    def _show_advanced_options_prompt(self):
        """Show a prompt to enable advanced options."""
        result = messagebox.askyesno(
            "Enable Advanced Options",
            "Would you like to enable advanced options for more customization?\n\n"
            "This will allow you to fine-tune system parameters and performance settings."
        )
        if result:
            self._update_config("show_advanced_options_prompt", False)
            self._save_config()
            self._create_advanced_options_tab() # Re-create the tab to show options
            self._show_advanced_options_gui() # Show the GUI
        else:
            self._update_config("show_advanced_options_prompt", False)
            self._save_config()
    
    def _show_advanced_options_gui(self):
        """Show the advanced options GUI."""
        if VISUAL_CONTROLS_AVAILABLE:
            show_visual_controls(self.root)
        else:
            messagebox.showwarning("Visual Controls", "Visual controls are not available.")

    def _show_visual_controls(self):
        """Show the visual controls GUI."""
        if VISUAL_CONTROLS_AVAILABLE:
            show_visual_controls(self.root)
        else:
            messagebox.showwarning("Visual Controls", "Visual controls are not available.")

    def _initialize_advanced_scheduler(self):
        """Initialize and start the advanced scheduler if enabled."""
        if not SCHEDULER_AVAILABLE:
            logger.warning("Advanced scheduler not available")
            return
        
        try:
            # Check if scheduler should be auto-started
            if self.config.get("auto_start_scheduler", False):
                logger.info("🕐 Starting advanced scheduler...")
                start_advanced_scheduler()
                logger.info("✅ Advanced scheduler started successfully")
            else:
                logger.info("🕐 Advanced scheduler available but not auto-started")
        except Exception as e:
            logger.error(f"Failed to initialize advanced scheduler: {e}")
    
    # System methods
    def _check_system_health(self):
        """Check system health."""
        def health_check():
            try:
                # Import and test core components
                import sys
                sys.path.append('.')
                
                from AOI_Base_Files_Schwabot.core.risk_manager import RiskManager
                risk_manager = RiskManager()
                status = risk_manager.get_system_status()
                
                if status['system_health'] == 'healthy':
                    self.health_status_label.config(text="✅ Healthy", foreground='#00ff00')
                else:
                    self.health_status_label.config(text="⚠️ Degraded", foreground='#ffaa00')
                
                self._update_status("System health check completed")
                
            except Exception as e:
                self.health_status_label.config(text="❌ Error", foreground='#ff6666')
                self._update_status(f"System health check failed: {e}")
        
        threading.Thread(target=health_check, daemon=True).start()
    
    def _check_system_status(self):
        """Check detailed system status."""
        try:
            subprocess.run([sys.executable, "AOI_Base_Files_Schwabot/main.py", "--system-status"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check system status: {e}")
    
    def _view_logs(self):
        """View system logs."""
        messagebox.showinfo("System Logs", 
                           "System logs viewer will be implemented in the next update.\n\n"
                           "This will show:\n"
                           "• Trading activity logs\n"
                           "• System error logs\n"
                           "• Performance metrics\n"
                           "• API connection logs")
    
    def _emergency_stop(self):
        """Emergency stop all trading."""
        result = messagebox.askyesno("Emergency Stop", 
                                   "🛑 EMERGENCY STOP\n\n"
                                   "This will immediately stop all trading activities.\n"
                                   "Are you sure you want to proceed?")
        if result:
            try:
                # Run emergency stop script
                subprocess.run([sys.executable, "AOI_Base_Files_Schwabot/emergency_stop.py"])
                self._update_status("Emergency stop executed")
                messagebox.showinfo("Emergency Stop", "All trading activities have been stopped.")
            except Exception as e:
                self._update_status(f"Emergency stop failed: {e}")
                messagebox.showerror("Error", f"Emergency stop failed: {e}")
    
    # Help methods
    def _show_quick_start(self):
        """Show quick start guide."""
        messagebox.showinfo("Quick Start Guide", 
                           "🚀 SCHWABOT QUICK START GUIDE\n\n"
                           "1. Configure API Keys (🔑 tab)\n"
                           "2. Choose launch mode (🚀 tab)\n"
                           "3. Start with Demo Mode for testing\n"
                           "4. Monitor system health (💻 tab)\n"
                           "5. Configure notifications as needed\n\n"
                           "For detailed instructions, see the documentation.")
    
    def _show_api_guide(self):
        """Show API setup guide."""
        messagebox.showinfo("API Setup Guide", 
                           "🔑 API SETUP GUIDE\n\n"
                           "1. Click 'Configure API Keys' in the 🔑 tab\n"
                           "2. Follow the step-by-step instructions for each service\n"
                           "3. Required for live trading:\n"
                           "   • Trading exchange API keys\n"
                           "   • AI service API keys (optional)\n"
                           "   • Notification service keys (optional)\n\n"
                           "All keys are encrypted and stored securely.")
    
    def _show_troubleshooting(self):
        """Show troubleshooting guide."""
        messagebox.showinfo("Troubleshooting", 
                           "🔧 TROUBLESHOOTING GUIDE\n\n"
                           "Common Issues:\n"
                           "• API connection errors → Check API keys\n"
                           "• System not starting → Check system health\n"
                           "• Trading not working → Verify exchange API keys\n"
                           "• Performance issues → Check system resources\n\n"
                           "For more help, check the system logs.")
    
    def _show_about(self):
        """Show about information."""
        messagebox.showinfo("About Schwabot", 
                           "ℹ️ ABOUT SCHWABOT\n\n"
                           "Schwabot Professional Trading System v2.0\n"
                           "Advanced AI-Powered Cryptocurrency Trading\n\n"
                           "Features:\n"
                           "• Dynamic timing system\n"
                           "• USDC-based trading pairs\n"
                           "• Multiple interface options\n"
                           "• Comprehensive risk management\n"
                           "• Real-time monitoring\n\n"
                           "© 2024 Schwabot Trading System")
    
    # Utility methods
    def _update_config(self, key: str, value: Any):
        """Update configuration."""
        self.config[key] = value
        self._save_config()
    
    def _update_status(self, message: str):
        """Update status bar."""
        if self.status_label:
            self.status_label.config(text=message)
    
    def _update_api_status(self):
        """Update API status display."""
        try:
            from api_key_manager import api_key_manager
            
            # Check if any API keys are configured
            has_keys = False
            for category in api_key_manager.api_config.values():
                for service_id in category.keys():
                    if api_key_manager.has_api_key(service_id, "api_key"):
                        has_keys = True
                        break
                if has_keys:
                    break
            
            if has_keys:
                self.api_status_label.config(text="API Keys: ✅ Configured", foreground='#00ff00')
                self._update_config("api_keys_configured", True)
            else:
                self.api_status_label.config(text="API Keys: ❌ Not Configured", foreground='#ff6666')
                self._update_config("api_keys_configured", False)
                
        except Exception as e:
            self.api_status_label.config(text="API Keys: ⚠️ Error", foreground='#ffaa00')
    
    def _update_usb_status(self):
        """Update USB status display."""
        try:
            usb_status = get_usb_status()
            
            if usb_status['drives_detected'] > 0:
                if usb_status['has_configured_drive']:
                    self.usb_status_label.config(
                        text=f"USB: ✅ {usb_status['drives_detected']} drive(s), 1 configured", 
                        foreground='#00ff00'
                    )
                else:
                    self.usb_status_label.config(
                        text=f"USB: 🔄 {usb_status['drives_detected']} drive(s) detected", 
                        foreground='#ffaa00'
                    )
            else:
                self.usb_status_label.config(text="USB: ❌ No drives detected", foreground='#ff6666')
                
        except Exception as e:
            self.usb_status_label.config(text="USB: ⚠️ Error", foreground='#ffaa00')
            logger.error(f"Error updating USB status: {e}")


def main():
    """Main launcher function."""
    launcher = SchwabotLauncher()
    launcher.show_launcher()


if __name__ == "__main__":
    main() 